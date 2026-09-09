"""Shared data models for Phase 2's pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from fetch_news import Story

RetrievalMethod = Literal["firecrawl", "firecrawl+tavily", "rss_description_fallback"]


@dataclass
class RetrievedStory:
    """A Story paired with the content retrieved for it before generation."""

    story: Story
    content: str
    retrieval_method: RetrievalMethod


@dataclass
class ExplainedStory:
    """A Story paired with its generated, category-adapted explanation."""

    story: Story
    category: str
    explanation: str
    source_url: str
    confidence_note: str