"""Gemini-ready compact payload formatter.

This module formats structured rep summaries into compact payloads
designed for the Gemini Live API. The actual API integration is a
future extension -- this module produces the payload format.

FUTURE: Replace the placeholder adapter with actual Gemini Live API calls.
See: https://ai.google.dev/gemini-api/docs/live (when available)
"""
import json
from squat_coach.events.schemas import RepSummaryEvent


def format_gemini_payload(event: RepSummaryEvent) -> dict:
    """Format a rep summary into a Gemini-ready payload.

    This produces a compact, semantic summary suitable for
    natural language generation -- NOT raw frame data.

    Returns:
        Dict ready for JSON serialization and Gemini API submission.
    """
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
    """Format payload as a text prompt for Gemini language generation.

    FUTURE: This would be sent to Gemini Live API for spoken feedback.
    """
    scores = payload["scores"]
    cue = payload["primary_coaching_cue"]

    prompt = (
        f"The user just completed rep {payload['rep_index']} of squats. "
        f"Overall score: {scores.get('rep_quality', 50):.0f}/100. "
    )
    if payload["faults"]:
        top_fault = payload["faults"][0]
        prompt += f"Main issue: {top_fault['type'].replace('_', ' ')} "
        prompt += f"(severity {top_fault['severity']:.0%}). "
    prompt += f"Coaching cue: {cue}"

    return prompt


# PLACEHOLDER: Future Gemini Live API adapter
# async def send_to_gemini(payload: dict) -> None:
#     """Send payload to Gemini Live API for spoken feedback.
#
#     This will use the Gemini Live streaming API to generate
#     real-time spoken coaching feedback from the structured payload.
#     """
#     # TODO: Implement when Gemini Live API is available
#     # client = genai.Client()
#     # session = await client.live.connect(model="gemini-2.0-flash-live")
#     # await session.send(format_gemini_text_prompt(payload))
#     pass
