"""
Per-story generation cache.

The cache key is a hash of the story's own content, the instructions that
apply to it, and the model name — deliberately NOT the literal batch prompt
text. Hashing the literal prompt would make a story's cache key depend on
which other story it happened to be batched with, causing spurious cache
misses whenever batch composition shifts between runs (e.g. a neighboring
story ages out of the recency window). Keying on the story's own fields
means a cache hit means exactly what it should: "this story, under these
instructions, on this model, already has a generated explanation."
"""

from __future__ import annotations

import hashlib
import logging

from diskcache import Cache

from fetch_news import Story
from phase2.models import ExplainedStory
from phase2.prompts import instructions_fingerprint

CACHE_DIRECTORY = ".cache"
CACHE_SIZE_LIMIT_BYTES = 50 * 1024 * 1024  # 50MB, LRU eviction beyond this.

logger = logging.getLogger(__name__)


def _cache_key(story: Story, model_name: str) -> str:
    """Build the hash key for a story: its content + instructions + model."""
    fingerprint = "|".join(
        [
            story.url,
            story.title,
            story.description,
            instructions_fingerprint(story),
            model_name,
        ]
    )
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()


class StoryCache:
    """Thin wrapper around diskcache, scoped to per-story explanations."""

    def __init__(self, directory: str = CACHE_DIRECTORY) -> None:
        self._cache = Cache(directory, size_limit=CACHE_SIZE_LIMIT_BYTES)

    def get(self, story: Story, model_name: str) -> ExplainedStory | None:
        key = _cache_key(story, model_name)
        cached = self._cache.get(key)
        if cached is not None:
            logger.debug("Cache hit: %s", story.title)
        return cached

    def set(self, story: Story, model_name: str, result: ExplainedStory) -> None:
        key = _cache_key(story, model_name)
        self._cache.set(key, result)

    def close(self) -> None:
        self._cache.close()