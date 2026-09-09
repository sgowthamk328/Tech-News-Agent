"""
generate_report.py

Phase 2 of the tech news learning assistant: take Phase 1's clean story
list, retrieve real article content for each one (Firecrawl, with Tavily
and the RSS description as fallbacks), classify and generate an adaptive,
teaching-style explanation via Gemini 3.6 Flash, and render the result as
a markdown report.

Design principles this follows:
- Retrieval only runs for cache-miss stories — a cached story skips both
  the network fetch and the generation call entirely.
- Per-story caching, independent of batch composition (see phase2/cache.py)
  so re-running the script during development never re-generates a story
  it already has a good explanation for.
- Micro-batching (2 stories/call) to keep daily API usage well within the
  free tier's request budget while still giving the model focused attention
  per story.
- Failures are per-story or per-batch, never all-or-nothing: one bad batch,
  one failed fetch, or one malformed entry means fewer stories in today's
  report, not a crashed run.

Usage:
    python generate_report.py                  # fetch live stories via Phase 1
    python generate_report.py --use-fixtures    # use tests/fixtures/ instead,
                                                 # for prompt tuning without a
                                                 # live fetch
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from fetch_news import Story, get_daily_stories
from phase2.cache import StoryCache
from phase2.gemini_client import MODEL_NAME, generate_batch
from phase2.models import ExplainedStory, RetrievedStory
from phase2.render import render_markdown
from phase2.retrieval import retrieve_source_material

BATCH_SIZE = 2
FIXTURES_PATH = Path("tests/fixtures/sample_stories.json")
DEFAULT_OUTPUT_DIR = Path("output")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_fixture_stories(path: Path = FIXTURES_PATH) -> list[Story]:
    """Load a fixed set of stories from disk, for prompt tuning without a live fetch."""
    raw_entries = json.loads(path.read_text())
    stories = []
    for entry in raw_entries:
        stories.append(
            Story(
                title=entry["title"],
                url=entry["url"],
                description=entry.get("description", ""),
                published=datetime.fromisoformat(entry["published"]),
                source=entry["source"],
                source_type=entry["source_type"],
            )
        )
    return stories


def _chunk(items: list, size: int) -> list[list]:
    """Split a list into consecutive chunks of at most `size` items."""
    return [items[i : i + size] for i in range(0, len(items), size)]

def _partition_cached(
    stories: list[Story], cache: StoryCache
) -> tuple[dict[str, ExplainedStory], list[Story]]:
    """Split stories into (already-cached results, still-need-generation)."""
    cached_results: dict[str, ExplainedStory] = {}
    uncached_stories: list[Story] = []

    for story in stories:
        cached = cache.get(story, MODEL_NAME)
        if cached is not None:
            cached_results[story.url] = cached
        else:
            uncached_stories.append(story)

    return cached_results, uncached_stories


def _retrieve_all(stories: list[Story]) -> list[RetrievedStory]:
    """
    Run the Firecrawl -> Tavily -> RSS-description fallback chain for a list
    of stories. Only called for cache-miss stories — a cached story never
    triggers a network fetch, saving both time and Firecrawl/Tavily credits.
    """
    retrieved = []
    for story in stories:
        retrieved.append(retrieve_source_material(story))
    return retrieved


def generate_all(stories: list[Story], cache: StoryCache) -> list[ExplainedStory]:
    """
    Run the full Phase 2 pipeline over a story list: reuse cached
    explanations where available, retrieve content and batch-generate the
    rest, and return the successfully-explained stories in their original
    order.
    """
    cached_results, uncached_stories = _partition_cached(stories, cache)
    logger.info(
        "%d stories already cached, %d need retrieval + generation",
        len(cached_results), len(uncached_stories),
    )

    retrieved_stories = _retrieve_all(uncached_stories)
    for retrieved in retrieved_stories:
        if retrieved.retrieval_method != "firecrawl":
            logger.info(
                "'%s' used fallback retrieval: %s",
                retrieved.story.title, retrieved.retrieval_method,
            )

    new_results: dict[str, ExplainedStory] = {}
    for batch in _chunk(retrieved_stories, BATCH_SIZE):
        batch_results = generate_batch(batch)
        for url, result in batch_results.items():
            cache.set(result.story, MODEL_NAME, result)
        new_results.update(batch_results)

        missing = {r.story.url for r in batch} - batch_results.keys()
        for url in missing:
            logger.warning("No usable explanation for: %s", url)

    all_results = {**cached_results, **new_results}
    logger.info(
        "%d of %d stories have an explanation for today's report",
        len(all_results), len(stories),
    )

    # Preserve the original story order; silently drop anything that failed.
    return [all_results[story.url] for story in stories if story.url in all_results]


def write_report(markdown: str, report_date: date, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{report_date.isoformat()}.md"
    output_path.write_text(markdown)
    return output_path


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Generate today's tech news report.")
    parser.add_argument(
        "--use-fixtures",
        action="store_true",
        help="Use tests/fixtures/sample_stories.json instead of a live Phase 1 fetch.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write the rendered report into.",
    )
    args = parser.parse_args()

    stories = load_fixture_stories() if args.use_fixtures else get_daily_stories()
    if not stories:
        logger.warning("No stories to process; exiting without writing a report.")
        return

    cache = StoryCache()
    try:
        explained_stories = generate_all(stories, cache)
    finally:
        cache.close()

    if not explained_stories:
        logger.warning("No stories were successfully explained; exiting without writing a report.")
        return

    report_date = datetime.now(timezone.utc).date()
    markdown = render_markdown(report_date, explained_stories)
    output_path = write_report(markdown, report_date, args.output_dir)
    logger.info("Report written to %s", output_path)


if __name__ == "__main__":
    main()