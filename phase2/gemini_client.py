"""
Gemini API client for batch story explanation.

No model tools (grounding, URL context) are used here — Gemini 3.6 Flash's
free tier doesn't reliably include them, and relying on a model tool whose
free-tier availability is unclear and shifting is fragile anyway. Instead,
the actual article content is fetched explicitly beforehand (see
phase2/retrieval.py) and included directly in the prompt — the model
explains from real, already-retrieved text rather than being asked to go
find it itself.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time

from google import genai

from phase2.models import ExplainedStory, RetrievedStory
from phase2.prompts import VALID_CATEGORIES, build_batch_prompt

MODEL_NAME = "gemini-3.6-flash"
MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 3

logger = logging.getLogger(__name__)

_JSON_ARRAY_PATTERN = re.compile(r"\[.*\]", re.DOTALL)


class GenerationError(Exception):
    """Raised when a batch fails to produce usable output after all retries."""


def _get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to your .env file (local) "
            "or as a repo/host secret (Actions / Render)."
        )
    return genai.Client(api_key=api_key)


def _call_model(client: genai.Client, prompt: str) -> str:
    """Make one API call and return the raw response text."""
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config={"automatic_function_calling": {"disable": True}},
    )
    return response.text or ""


def _extract_json_array(raw_text: str) -> list[dict]:
    """
    Pull a JSON array out of the model's raw response text.

    Even with explicit "no code fences" instructions, models occasionally
    wrap output in markdown fences or add stray whitespace — this strips
    that defensively rather than trusting the instruction to always hold.
    """
    match = _JSON_ARRAY_PATTERN.search(raw_text)
    if not match:
        raise GenerationError(f"No JSON array found in response: {raw_text[:200]!r}")

    parsed = json.loads(match.group(0))
    if not isinstance(parsed, list):
        raise GenerationError("Parsed JSON was not a list")
    return parsed


def _validate_entry(entry: dict) -> bool:
    """Check that a parsed entry has every field we require, non-empty."""
    required_keys = ("url", "category", "explanation", "source_url", "confidence_note")
    if not all(key in entry and entry[key] for key in required_keys):
        return False
    if entry["category"] not in VALID_CATEGORIES:
        logger.warning("Unrecognized category %r, keeping anyway", entry["category"])
    return True


def _match_entries_to_stories(
    retrieved_stories: list[RetrievedStory], entries: list[dict]
) -> dict[str, ExplainedStory]:
    """Match parsed JSON entries back to their originating Story by URL."""
    stories_by_url = {r.story.url: r.story for r in retrieved_stories}
    matched: dict[str, ExplainedStory] = {}

    for entry in entries:
        if not _validate_entry(entry):
            logger.warning("Skipping malformed entry: %s", entry)
            continue

        story = stories_by_url.get(entry["url"])
        if story is None:
            logger.warning("Response URL didn't match any story in this batch: %s", entry["url"])
            continue

        matched[story.url] = ExplainedStory(
            story=story,
            category=entry["category"],
            explanation=entry["explanation"],
            source_url=entry["source_url"],
            confidence_note=entry["confidence_note"],
        )

    return matched


def generate_batch(retrieved_stories: list[RetrievedStory]) -> dict[str, ExplainedStory]:
    """
    Generate explanations for one batch of already-retrieved stories.

    Returns a dict keyed by story URL. Stories whose entries are missing or
    malformed in the response are simply absent from the result — callers
    should treat "not present" as "this one needs to be skipped," not raise.
    """
    client = _get_client()
    prompt = build_batch_prompt(retrieved_stories)

    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw_text = _call_model(client, prompt)
            entries = _extract_json_array(raw_text)
            return _match_entries_to_stories(retrieved_stories, entries)
        except (GenerationError, json.JSONDecodeError) as exc:
            last_error = exc
            logger.warning(
                "Batch generation attempt %d/%d failed: %s", attempt, MAX_RETRIES, exc
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    logger.error(
        "Batch of %d stories failed after %d attempts (%s); skipping this batch",
        len(retrieved_stories), MAX_RETRIES, last_error,
    )
    return {}