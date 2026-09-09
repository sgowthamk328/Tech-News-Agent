"""
fetch_news.py

Phase 1 of the tech news learning assistant: fetch stories from a fixed set
of reputable tech RSS feeds, keep only recent items, deduplicate near-identical
coverage of the same story across feeds, and return a small, clean list.

This module intentionally does not call any LLM. Its only job is to prove
the input data is trustworthy before Phase 2 spends model calls explaining it.

Usage:
    python fetch_news.py
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from html import unescape

import feedparser
import requests
from dateutil import parser as dateutil_parser

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

from typing import Literal

SourceType = Literal["editorial", "discovery"]

# Editorial sources are the publication itself — their reporting can be
# summarized and explained fairly directly (subject to the accuracy
# safeguards elsewhere in the pipeline, e.g. grounded generation).
#
# Discovery sources surface *links to* other articles/discussions rather
# than publishing original reporting themselves. An item from a discovery
# source should never be treated as the factual source in its own right —
# Phase 2's generation step should follow the item's URL to the underlying
# article and verify/explain based on that, not the discovery post itself.
FEEDS: dict[str, tuple[str, SourceType]] = {
    "Ars Technica": ("https://feeds.arstechnica.com/arstechnica/index", "editorial"),
    "The Register": ("https://www.theregister.com/headlines.atom", "editorial"),
    "TechCrunch": ("https://techcrunch.com/feed/", "editorial"),
    "The Verge": ("https://www.theverge.com/rss/index.xml", "editorial"),
    "Hacker News": ("https://news.ycombinator.com/rss", "discovery"),
}

RECENCY_WINDOW_HOURS = 24        # Strict cutoff: only stories newer than this. No widening —
                                  # this is a "what's genuinely new today" tool, not a quota to fill.
MAX_STORIES = 10                 # Ceiling on the final story count for the day.
MAX_STORIES_PER_SOURCE = 3       # Cap per feed, so one fast-publishing source can't dominate.
DUPLICATE_TITLE_SIMILARITY = 0.75  # 0-1 scale; higher = stricter match required.
REQUEST_TIMEOUT_SECONDS = 10
USER_AGENT = "Mozilla/5.0 (compatible; TechNewsLearningBot/1.0)"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")


def _clean_description(raw: str) -> str:
    """
    Strip HTML tags and unescape entities from a feed's description field.

    Some feeds (notably Hacker News) put an HTML link — e.g. a "Comments"
    anchor tag — in the description instead of a plain-text summary. Left
    unstripped, that HTML would leak into both the printed output and,
    later, the prompt sent to the model in Phase 2.
    """
    unescaped = unescape(raw)
    without_tags = _HTML_TAG_PATTERN.sub(" ", unescaped)
    return " ".join(without_tags.split())


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

@dataclass
class Story:
    """A single news item, normalized across all source feeds."""

    title: str
    url: str
    description: str
    published: datetime
    source: str
    source_type: SourceType
    normalized_title: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.normalized_title = self._normalize(self.title)

    @staticmethod
    def _normalize(text: str) -> str:
        """Lowercase and strip punctuation-heavy noise for similarity comparison."""
        return " ".join(
            "".join(ch for ch in text.lower() if ch.isalnum() or ch.isspace()).split()
        )


# --------------------------------------------------------------------------- #
# Fetching
# --------------------------------------------------------------------------- #

def fetch_feed(source_name: str, feed_url: str, source_type: SourceType) -> list[Story]:
    """
    Fetch and parse a single RSS/Atom feed into a list of Story objects.

    Failures (network errors, malformed feeds, missing fields) are logged and
    result in an empty list for that feed rather than raising — one bad feed
    should never take down the whole run.
    """
    try:
        response = requests.get(
            feed_url,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Could not fetch %s (%s): %s", source_name, feed_url, exc)
        return []

    parsed = feedparser.parse(response.content)
    if parsed.bozo and not parsed.entries:
        logger.warning("Feed for %s appears malformed and had no entries", source_name)
        return []

    stories: list[Story] = []
    for entry in parsed.entries:
        story = _entry_to_story(entry, source_name, source_type)
        if story is not None:
            stories.append(story)

    logger.info("Fetched %d entries from %s", len(stories), source_name)
    return stories


def _entry_to_story(
    entry: feedparser.FeedParserDict, source_name: str, source_type: SourceType
) -> Story | None:
    """Convert one feedparser entry into a Story, or None if it's unusable."""
    raw_title = entry.get("title", "")
    title = unescape(raw_title.strip()) if isinstance(raw_title, str) else ""

    raw_url = entry.get("link", "")
    url = raw_url.strip() if isinstance(raw_url, str) else ""

    if not title or not url:
        return None

    raw_desc = entry.get("summary", "") or entry.get("description", "")
    description = _clean_description(raw_desc if isinstance(raw_desc, str) else "")

    published = _extract_published(entry)
    if published is None:
        # No reliable timestamp means we can't apply the recency filter safely.
        logger.debug("Skipping entry with no parseable date: %s", title)
        return None

    return Story(
        title=title,
        url=url,
        description=description,
        published=published,
        source=source_name,
        source_type=source_type,
    )


def _extract_published(entry: feedparser.FeedParserDict) -> datetime | None: #FeedParserDict - dictionary like object(type)
    """Pull a timezone-aware UTC datetime out of a feed entry, trying multiple fields."""
    for key in ("published_parsed", "updated_parsed"):
        struct_time = entry.get(key)
        if struct_time and isinstance(struct_time, tuple) and len(struct_time) >= 6:
            return datetime(*struct_time[:6], tzinfo=timezone.utc)  # type: ignore

    for key in ("published", "updated"):
        raw = entry.get(key)
        if isinstance(raw, str) and raw:
            try:
                parsed_dt = dateutil_parser.parse(raw)
                if parsed_dt.tzinfo is None:
                    parsed_dt = parsed_dt.replace(tzinfo=timezone.utc)
                return parsed_dt.astimezone(timezone.utc)
            except (ValueError, OverflowError):
                continue

    return None


def fetch_all_feeds() -> list[Story]:
    """Fetch every configured feed and return the combined, unfiltered story list."""
    all_stories: list[Story] = []
    for source_name, (feed_url, source_type) in FEEDS.items():
        all_stories.extend(fetch_feed(source_name, feed_url, source_type))
    return all_stories


# --------------------------------------------------------------------------- #
# Filtering
# --------------------------------------------------------------------------- #

def filter_recent(stories: list[Story], window_hours: int) -> list[Story]:
    """Keep only stories published within the given window, relative to now."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
    return [story for story in stories if story.published >= cutoff]


# --------------------------------------------------------------------------- #
# Deduplication
# --------------------------------------------------------------------------- #

def _title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def deduplicate(stories: list[Story]) -> list[Story]:
    """
    Collapse near-identical stories (the same event covered by multiple
    outlets, or an editorial article surfaced again via a discovery source
    like Hacker News) into one entry.

    When a match is found, an editorial-source story always wins over a
    discovery-source one, regardless of description length — Hacker News
    items point at other people's reporting, so the original reporting is
    the one worth keeping. Between two stories of the same trust tier, the
    one with the longer description wins.

    Stories are processed most-recent-first so that, all else equal, the
    freshest coverage of an event is what's compared first.
    """
    stories_by_recency = sorted(stories, key=lambda s: s.published, reverse=True)
    kept: list[Story] = []

    for candidate in stories_by_recency:
        match_index = _find_duplicate_index(candidate, kept)
        if match_index is None:
            kept.append(candidate)
            continue

        existing = kept[match_index]
        if _should_replace(existing, candidate):
            logger.info(
                "Duplicate merged: kept '%s' (%s) over '%s' (%s)",
                candidate.title, candidate.source, existing.title, existing.source,
            )
            kept[match_index] = candidate
        else:
            logger.info(
                "Duplicate merged: kept '%s' (%s) over '%s' (%s)",
                existing.title, existing.source, candidate.title, candidate.source,
            )

    return kept


def _should_replace(existing: Story, candidate: Story) -> bool:
    """Decide whether a newly-found duplicate should replace the kept one."""
    if existing.source_type != candidate.source_type:
        # Editorial always beats discovery, independent of description length.
        return candidate.source_type == "editorial"
    return len(candidate.description) > len(existing.description)


def _find_duplicate_index(candidate: Story, kept: list[Story]) -> int | None:
    for index, existing in enumerate(kept):
        similarity = _title_similarity(candidate.normalized_title, existing.normalized_title)
        if similarity >= DUPLICATE_TITLE_SIMILARITY:
            return index
    return None


# --------------------------------------------------------------------------- #
# Source balancing
# --------------------------------------------------------------------------- #

def select_balanced(stories: list[Story]) -> list[Story]:
    """
    Choose the final daily list, most-recent-first, without letting any single
    feed exceed MAX_STORIES_PER_SOURCE — so one high-frequency publisher
    (e.g. TechCrunch) can't crowd out slower, deeper sources on a given day.
    """
    stories_by_recency = sorted(stories, key=lambda s: s.published, reverse=True)

    selected: list[Story] = []
    per_source_count: dict[str, int] = {}

    for story in stories_by_recency:
        if len(selected) >= MAX_STORIES:
            break
        count_so_far = per_source_count.get(story.source, 0)
        if count_so_far >= MAX_STORIES_PER_SOURCE:
            continue
        selected.append(story)
        per_source_count[story.source] = count_so_far + 1

    return selected


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

def get_daily_stories() -> list[Story]:
    """
    Run the full Phase 1 pipeline: fetch, filter to the last 24h, deduplicate,
    and balance across sources (max MAX_STORIES_PER_SOURCE each, up to
    MAX_STORIES total).

    The recency window is intentionally strict and never widened — this tool
    reports what's genuinely new in the last day, not a padded-out quota.
    Some days will honestly produce fewer stories than others; that's a
    correct outcome, not a failure to fix.
    """
    raw_stories = fetch_all_feeds()
    logger.info("Fetched %d raw entries across %d feeds", len(raw_stories), len(FEEDS))

    recent_stories = filter_recent(raw_stories, RECENCY_WINDOW_HOURS)
    logger.info("%d stories within the %dh window", len(recent_stories), RECENCY_WINDOW_HOURS)

    deduplicated_stories = deduplicate(recent_stories)
    logger.info("%d stories remain after deduplication", len(deduplicated_stories))

    final_stories = select_balanced(deduplicated_stories)
    logger.info(
        "%d stories selected for the final report (max %d per source)",
        len(final_stories), MAX_STORIES_PER_SOURCE,
    )
    return final_stories


def print_stories(stories: list[Story]) -> None:
    """Print the final story list in a readable format for manual review."""
    if not stories:
        print("\nNo stories found. Check feed URLs and network access.\n")
        return

    print(f"\n{len(stories)} stories for today's report\n{'-' * 60}")
    for i, story in enumerate(stories, start=1):
        age_hours = (datetime.now(timezone.utc) - story.published).total_seconds() / 3600
        tag = " [discovery — verify against linked article]" if story.source_type == "discovery" else ""
        print(f"\n{i}. {story.title}{tag}")
        print(f"   Source: {story.source}  |  {age_hours:.1f}h ago")
        print(f"   URL: {story.url}")
        if story.description:
            snippet = story.description[:160].rsplit(" ", 1)[0]
            print(f"   {snippet}...")
    print()


if __name__ == "__main__":
    daily_stories = get_daily_stories()
    print_stories(daily_stories)