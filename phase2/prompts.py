"""
Prompt construction for the batch generation step.

Kept isolated from the API client and orchestration logic so that tuning the
instructions — the thing most likely to change repeatedly — never requires
touching retrieval, caching, batching, or rendering code.
"""

from __future__ import annotations

from fetch_news import Story
from phase2.models import RetrievedStory

VALID_CATEGORIES = (
    "technical_release",
    "funding_ma",
    "research",
    "policy_security",
    "industry_news",
)

CATEGORY_GUIDANCE = """\
Classify the story into exactly one of these categories, then write the \
explanation to match:

- technical_release: explain the underlying mechanism/architecture and what \
problem it actually solves.
- funding_ma: explain the business logic, who's involved, and what it \
signals about the market.
- research: explain the finding, why it's novel, and what it enables next.
- policy_security: explain what changed, who is affected, and what happens \
next.
- industry_news: explain the context and its implications for the industry.
"""

DISCOVERY_SOURCE_NOTE = """\
NOTE: this item was discovered via Hacker News, a community discussion \
feed — not original reporting. The content below has already been \
retrieved from the actual underlying article or project, not from the \
Hacker News post itself. Base your explanation on the retrieved content, \
and don't be misled by how the Hacker News title framed it.
"""

THIN_SOURCE_NOTE = """\
NOTE: the retrieved content for this story is limited (fetch was partial, \
paywalled, or otherwise thin) — the confidence_note for this story should \
reflect that the source material was limited, not just its individual claims.
"""

_OUTPUT_FORMAT_INSTRUCTIONS = """\
Return ONLY a JSON array, one object per story, with exactly these keys:
- "url": the original story URL, copied exactly as given below (used to \
match your response back to the right story — do not alter it)
- "category": one of technical_release, funding_ma, research, \
policy_security, industry_news
- "explanation": a 300-400 word explanation in a teaching tone, for an \
intelligent reader who is new to this specific topic. End it with a short \
"Who's involved:" line naming the key companies, products, or people.
- "source_url": the real article URL this explanation is based on (usually \
the same as "url", unless the retrieved content came from a different page)
- "confidence_note": one short sentence distinguishing confirmed facts from \
reported-but-unconfirmed claims in this story (e.g. exact figures, dates, \
or leadership/ownership claims), or "No unconfirmed claims of note" if none.

No markdown code fences, no commentary before or after — the entire \
response must be valid JSON and nothing else.
"""


def _story_block(retrieved: RetrievedStory, index: int) -> str:
    """Render one retrieved story's fields as plain text for the prompt."""
    story = retrieved.story
    lines = [
        f"URL: {story.url}",
        f"Title: {story.title}",
        f"Source: {story.source}",
        f"Published: {story.published.isoformat()}",
        f"Retrieved content:\n{retrieved.content}",
    ]
    if story.source_type == "discovery":
        lines.append(DISCOVERY_SOURCE_NOTE)
    if retrieved.retrieval_method == "rss_description_fallback":
        lines.append(THIN_SOURCE_NOTE)
    return "\n".join(lines)


def build_batch_prompt(retrieved_stories: list[RetrievedStory]) -> str:
    """Build the full prompt for one batch of retrieved stories."""
    story_blocks = "\n\n".join(
        f"--- Story {i} ---\n{_story_block(retrieved, i)}"
        for i, retrieved in enumerate(retrieved_stories, start=1)
    )
    return (
        f"{CATEGORY_GUIDANCE}\n"
        f"{_OUTPUT_FORMAT_INSTRUCTIONS}\n"
        f"Here are the {len(retrieved_stories)} stories to process:\n\n"
        f"{story_blocks}"
    )


def instructions_fingerprint(story: Story) -> str:
    """
    The portion of the prompt that varies only by category-of-instructions,
    not by which other stories happen to be batched alongside this one, and
    not by the retrieved content itself.

    Used for cache-key hashing so a story's cache entry depends on its own
    identity and the instructions that apply to it — never on which sibling
    story it happened to share a batch with.
    """
    parts = [CATEGORY_GUIDANCE, _OUTPUT_FORMAT_INSTRUCTIONS]
    if story.source_type == "discovery":
        parts.append(DISCOVERY_SOURCE_NOTE)
    return "\n".join(parts)