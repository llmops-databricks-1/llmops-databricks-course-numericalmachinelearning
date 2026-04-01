import arxiv
from loguru import logger


def search_papers_by_author(name: str, max_results: int = 3) -> list[dict]:
    """Search arXiv for papers by a given author name.

    Uses the arXiv author query syntax to find papers. Returns abstracts
    rather than full PDFs — abstracts are short enough to pass directly
    to the synthesis LLM without chunking.

    Args:
        name: Full name of the author to search for.
        max_results: Maximum number of papers to return. Defaults to 3.

    Returns:
        List of dicts with keys: title, summary, url.
        Returns an empty list if no papers are found or the search fails.
    """
    try:
        client = arxiv.Client()
        search = arxiv.Search(query=f"au:{name}", max_results=max_results)
        results = [
            {
                "title": paper.title,
                "summary": paper.summary,
                "url": paper.pdf_url,
            }
            for paper in client.results(search)
        ]
        logger.debug("Found {} arXiv papers for '{}'", len(results), name)
        return results
    except Exception as e:
        logger.error("arXiv search error for '{}': {}", name, e)
        return []
