"""
Article content retrieval.

Replaces Gemini's native grounding/URL-context tools, which aren't reliably
available on Gemini 3.6 Flash's free tier. Retrieval is handled explicitly
here instead: Firecrawl fetches the real article content from a story's URL
directly (the same path works for editorial and discovery stories, since
Phase 1's Story.url already points at the actual article in both cases).

A hard fetch failure and a thin paywall teaser are treated identically —
both are "insufficient content" and trigger the same fallback chain:
Firecrawl -> Tavily (supplementary sources) -> the RSS description Phase 1
already captured. A story is only left with no content if all three fail.
"""

from __future__ import annotations

import logging
import os

from firecrawl import V1FirecrawlApp
from tavily import TavilyClient

from fetch_news import Story
from phase2.models import RetrievedStory

MIN_WORD_COUNT = 150

logger = logging.getLogger(__name__)


def _word_count(text: str) -> int:
    return len(text.split())


def _get_firecrawl_client() -> V1FirecrawlApp:
    api_key = os.environ.get("FIRECRAWL_API_KEY")
    if not api_key:
        raise RuntimeError("FIRECRAWL_API_KEY is not set.")
    return V1FirecrawlApp(api_key=api_key)


def _get_tavily_client() -> TavilyClient:
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not set.")
    return TavilyClient(api_key=api_key)


def fetch_article_content(url: str) -> str:
    """
    Fetch the real article content from a URL via Firecrawl.

    Returns an empty string on any failure (bad response, network error,
    missing markdown) rather than raising — a failed fetch is treated as
    "zero content," which flows into the same word-count fallback check
    as a paywall teaser, with no separate error-handling path needed.
    """
    try:
        client = _get_firecrawl_client()
        response = client.scrape_url(url, formats=["markdown"])
    except Exception as exc:  # Firecrawl/network errors of any kind.
        logger.warning("Firecrawl fetch failed for %s: %s", url, exc)
        return ""

    if isinstance(response, dict):
        success = response.get("success", True)
        error = response.get("error", "unknown")
        markdown = response.get("data", {}).get("markdown", "") if "data" in response else response.get("markdown", "")
    else:
        success = getattr(response, "success", True)
        error = getattr(response, "error", "unknown")
        markdown = getattr(response, "markdown", "")

    if not success:
        logger.warning("Firecrawl reported failure for %s: %s", url, error)
        return ""

    return (markdown or "").strip()


def fetch_supplementary_context(story: Story) -> str:
    """
    Search for additional sources covering the same story via Tavily, to
    supplement thin or missing Firecrawl content — cross-referencing rather
    than just re-fetching the same URL with a different tool.
    """
    try:
        client = _get_tavily_client()
        response = client.search(
            query=story.title,
            max_results=3,
            include_answer=True,
        )
    except Exception as exc:
        logger.warning("Tavily search failed for %r: %s", story.title, exc)
        return ""

    pieces = []
    answer = response.get("answer")
    if answer:
        pieces.append(answer)
    for result in response.get("results", []):
        content = result.get("content")
        if content:
            pieces.append(content)

    return "\n\n".join(pieces).strip()


def retrieve_source_material(story: Story) -> RetrievedStory:
    """
    Run the full retrieval fallback chain for one story: Firecrawl, then
    Tavily if needed, then the RSS description as a last resort.
    """
    content = fetch_article_content(story.url)

    if _word_count(content) >= MIN_WORD_COUNT:
        return RetrievedStory(story=story, content=content, retrieval_method="firecrawl")

    logger.info(
        "Firecrawl content thin (%d words) for '%s'; trying Tavily",
        _word_count(content), story.title,
    )
    supplement = fetch_supplementary_context(story)
    combined = "\n\n".join(part for part in (content, supplement) if part).strip()

    if _word_count(combined) >= MIN_WORD_COUNT:
        return RetrievedStory(story=story, content=combined, retrieval_method="firecrawl+tavily")

    logger.info(
        "Still thin (%d words) after Tavily for '%s'; falling back to the RSS description",
        _word_count(combined), story.title,
    )
    final_content = "\n\n".join(part for part in (combined, story.description) if part).strip()
    return RetrievedStory(story=story, content=final_content, retrieval_method="rss_description_fallback")