"""Gemini API integration for spoken coaching feedback.

After each completed rep, sends a structured summary to Gemini
which returns a natural-language coaching response.

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
from typing import Optional

from squat_coach.events.schemas import RepSummaryEvent

logger = logging.getLogger("squat_coach.gemini")

# Background TTS — never blocks the video loop
_tts_lock = threading.Lock()


def speak(text: str) -> None:
    """Speak text using macOS 'say' with English voice in a background thread."""
    def _run():
        with _tts_lock:
            try:
                subprocess.run(["killall", "say"], capture_output=True)
                subprocess.run(
                    ["say", "-v", "Samantha", "-r", "190", text],
                    capture_output=True,
                    timeout=15,
                )
            except Exception as e:
                logger.debug("TTS error: %s", e)

    threading.Thread(target=_run, daemon=True).start()

# Lazy-initialized client
_client = None


def _get_client(api_key: str):
    """Initialize the Gemini client lazily."""
    global _client
    if _client is None:
        try:
            from google import genai
            _client = genai.Client(api_key=api_key)
            logger.info("Gemini client initialized")
        except Exception as e:
            logger.error("Failed to initialize Gemini client: %s", e)
            return None
    return _client


def format_gemini_payload(event: RepSummaryEvent) -> dict:
    """Format a rep summary into a Gemini-ready payload."""
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


def format_gemini_text_prompt(payload: dict) -> str:
    """Format payload as a text prompt for Gemini."""
    scores = payload["scores"]
    cue = payload["primary_coaching_cue"]

    prompt = (
        "You are a supportive squat coach giving brief spoken feedback after a rep. "
        "Keep it to 1-2 short sentences. Be encouraging but honest. "
        "Focus on the most important thing to improve.\n\n"
        f"Rep {payload['rep_index']} data:\n"
        f"- Overall score: {scores.get('rep_quality', 50):.0f}/100\n"
        f"- Depth score: {scores.get('depth', 50):.0f}/100\n"
        f"- Trunk control: {scores.get('trunk_control', 50):.0f}/100\n"
        f"- Posture stability: {scores.get('posture_stability', 50):.0f}/100\n"
        f"- Movement consistency: {scores.get('movement_consistency', 50):.0f}/100\n"
    )

    if payload["faults"]:
        prompt += "Detected faults:\n"
        for fault in payload["faults"]:
            prompt += f"  - {fault['type'].replace('_', ' ')} (severity: {fault['severity']:.0%})\n"

    if payload["phase_durations"]:
        d = payload["phase_durations"]
        prompt += f"- Descent: {d.get('descent_s', 0):.1f}s, Bottom: {d.get('bottom_s', 0):.1f}s, Ascent: {d.get('ascent_s', 0):.1f}s\n"

    prompt += f"\nSystem coaching cue: {cue}\n"
    prompt += "\nGive your brief coaching feedback:"

    return prompt


# Shared state for async Gemini feedback
_last_feedback: Optional[str] = None
_feedback_lock = threading.Lock()


def get_last_feedback() -> Optional[str]:
    """Get the most recent Gemini feedback (thread-safe). Returns None if none yet."""
    with _feedback_lock:
        return _last_feedback


def send_to_gemini_async(
    payload: dict,
    api_key: str = "",
    model: str = "gemini-2.0-flash",
    speak_enabled: bool = True,
    on_feedback: Optional[callable] = None,
) -> None:
    """Send rep summary to Gemini in a background thread. Never blocks.

    The API call + TTS all happen off the main thread.
    Results available via get_last_feedback() or the on_feedback callback.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        return

    def _run():
        global _last_feedback
        client = _get_client(key)
        if client is None:
            return

        prompt = format_gemini_text_prompt(payload)
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
            )
            feedback = response.text.strip()
            logger.info("Gemini: %s", feedback)

            with _feedback_lock:
                _last_feedback = feedback

            if on_feedback:
                on_feedback(feedback)

            if speak_enabled:
                speak(feedback)
        except Exception as e:
            logger.error("Gemini API error: %s", e)

    threading.Thread(target=_run, daemon=True).start()
