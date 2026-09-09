"""Renders a list of ExplainedStory objects into the final markdown report."""

from __future__ import annotations

from datetime import date

from phase2.models import ExplainedStory

CATEGORY_LABELS = {
    "technical_release": "Technical Release",
    "funding_ma": "Funding / M&A",
    "research": "Research",
    "policy_security": "Policy / Security",
    "industry_news": "Industry News",
}


def _render_story(explained: ExplainedStory) -> str:
    story = explained.story
    category_label = CATEGORY_LABELS.get(explained.category, explained.category)
    discovery_note = (
        f"\n> Discovered via {story.source}; explanation is based on the verified source below, not the discovery post itself."
        if story.source_type == "discovery"
        else ""
    )
    return (
        f"## {story.title}\n\n"
        f"**Category:** {category_label}  |  **Found via:** {story.source}"
        f"{discovery_note}\n\n"
        f"{explained.explanation}\n\n"
        f"**Source:** {explained.source_url}\n\n"
        f"*{explained.confidence_note}*\n"
    )


def render_markdown(report_date: date, explained_stories: list[ExplainedStory]) -> str:
    """Render the full day's report as a single markdown document."""
    header = (
        f"# Tech News — {report_date.isoformat()}\n\n"
        f"{len(explained_stories)} stories\n\n---\n\n"
    )
    body = "\n---\n\n".join(_render_story(story) for story in explained_stories)
    return header + body + "\n"