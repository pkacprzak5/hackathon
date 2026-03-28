"""Gemini API integration for spoken coaching feedback.

After each completed rep, sends a compact scoring summary (NOT raw frames
or pose data) to Gemini, which streams back a 1-2 sentence coaching response.

Uses streaming + piped TTS for lowest perceived latency:
  1. API call starts in background thread
  2. Streaming collects the full response (typically < 1s with flash)
  3. TTS starts immediately after

Setup:
  1. Get API key from https://aistudio.google.com/apikey
  2. Either:
     a. Set env var: export GEMINI_API_KEY=your_key_here
     b. Or paste it in squat_coach/config/default.yaml under gemini.api_key
"""
import logging
import os
import subprocess
import threading
from typing import Optional, Callable

from squat_coach.events.schemas import RepSummaryEvent

logger = logging.getLogger("squat_coach.gemini")

# ── TTS (background, never blocks) ──────────────────────────────────────

_say_proc: Optional[subprocess.Popen] = None
_tts_lock = threading.Lock()


def speak(text: str) -> None:
    """Speak text using macOS 'say' with English voice. Non-blocking."""
    global _say_proc

    def _run():
        global _say_proc
        with _tts_lock:
            try:
                # Kill previous speech
                if _say_proc and _say_proc.poll() is None:
                    _say_proc.terminate()
                _say_proc = subprocess.Popen(
                    ["say", "-v", "Samantha", "-r", "190", text],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                _say_proc.wait(timeout=15)
            except Exception as e:
                logger.debug("TTS error: %s", e)

    threading.Thread(target=_run, daemon=True).start()


# ── Gemini client (lazy singleton) ──────────────────────────────────────

_client = None
_client_lock = threading.Lock()

# System instruction — sent once, not per request (saves tokens)
_SYSTEM_INSTRUCTION = (
    "You are a funny squat coach. Reply with exactly ONE short witty sentence. "
    "Include a fun comparison and the coaching tip. Max 15 words.\n\n"
    "Examples:\n"
    "- 'Deeper than my student loans, score 92, keep that chest proud!'\n"
    "- 'Shallow as a puddle, sit deeper next time!'\n"
    "- 'Leaning like the Tower of Pisa, chest up champ!'\n"
    "- 'Smooth as butter, 88 points, chef kiss!'\n"
    "- 'Great depth but your back said goodbye, keep it straight!'"
)


def _get_client(api_key: str):
    global _client
    with _client_lock:
        if _client is None:
            try:
                from google import genai
                _client = genai.Client(api_key=api_key)
                logger.info("Gemini client initialized")
            except Exception as e:
                logger.error("Failed to init Gemini: %s", e)
                return None
    return _client


# ── Payload formatting ──────────────────────────────────────────────────

def format_gemini_payload(event: RepSummaryEvent) -> dict:
    """Format a rep summary into a compact payload."""
    return {
        "exercise": "squat",
        "rep_index": event.rep_index,
        "phase_durations": event.phase_durations,
        "key_features": event.features,
        "faults": event.faults,
        "scores": event.scores,
        "confidence": event.confidence,
        "primary_coaching_cue": event.coaching_cue,
    }


def _build_prompt(payload: dict) -> str:
    """Build prompt with scores, faults, and the system's coaching suggestion."""
    s = payload["scores"]
    lines = [f"Rep {payload['rep_index']} scores:"]
    lines.append(f"  Overall: {s.get('rep_quality', 50):.0f}/100")
    lines.append(f"  Depth: {s.get('depth', 50):.0f}  Trunk: {s.get('trunk_control', 50):.0f}  "
                 f"Posture: {s.get('posture_stability', 50):.0f}  Consistency: {s.get('movement_consistency', 50):.0f}")

    if payload["faults"]:
        fault_strs = [f"{f['type'].replace('_', ' ')} ({f['severity']:.0%})" for f in payload["faults"][:3]]
        lines.append("Faults: " + ", ".join(fault_strs))
    else:
        lines.append("No significant faults detected.")

    d = payload.get("phase_durations", {})
    if d:
        lines.append(f"Timing: descent {d.get('descent_s', 0):.1f}s, ascent {d.get('ascent_s', 0):.1f}s")

    # Include the system's coaching suggestion so Gemini can rephrase it naturally
    cue = payload.get("primary_coaching_cue", "")
    if cue:
        lines.append(f"System coaching suggestion: {cue}")

    return "\n".join(lines)


# ── Async Gemini call ───────────────────────────────────────────────────

_last_feedback: Optional[str] = None
_feedback_lock = threading.Lock()


def get_last_feedback() -> Optional[str]:
    with _feedback_lock:
        return _last_feedback


def send_to_gemini_async(
    payload: dict,
    api_key: str = "",
    model: str = "gemini-2.0-flash",
    speak_enabled: bool = True,
    on_feedback: Optional[Callable[[str], None]] = None,
) -> None:
    """Send rep data to Gemini in background thread. Never blocks.

    What we send: compact scoring summary (~100 tokens).
    NOT raw frames, NOT pose landmarks, NOT video data.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        return

    def _run():
        global _last_feedback
        client = _get_client(key)
        if not client:
            return

        prompt = _build_prompt(payload)
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config={
                    "system_instruction": _SYSTEM_INSTRUCTION,
                    "max_output_tokens": 80,
                    "temperature": 0.9,
                },
            )

            feedback = response.text.strip() if response.text else ""
            if not feedback:
                return

            logger.info("GEMINI: %s", feedback)

            with _feedback_lock:
                _last_feedback = feedback

            if on_feedback:
                on_feedback(feedback)

            if speak_enabled:
                speak(feedback)

        except Exception as e:
            logger.error("Gemini error: %s", e)

    threading.Thread(target=_run, daemon=True).start()
