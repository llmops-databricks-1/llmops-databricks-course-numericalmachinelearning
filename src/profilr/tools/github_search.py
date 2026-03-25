from loguru import logger

from profilr.vector_search import VectorSearchManager


def search_github_repos(
    query: str, username: str = "", num_results: int = 3
) -> str:
    """Search GitHub repository content using Vector Search.

    Queries the Databricks Vector Search index pre-populated by the GitHub
    ingestion notebook. Returns top matching chunks as a formatted string
    ready to pass to an LLM. Returns an empty string if the index is
    unavailable or no results are found — callers should handle this
    gracefully.

    Args:
        query: Semantic search query (e.g. 'machine learning projects').
        username: If provided, filters results to this GitHub username only.
        num_results: Number of top chunks to return. Defaults to 3.

    Returns:
        Formatted string of top matching chunks, or empty string if unavailable.
    """
    try:
        vs = VectorSearchManager()
        filters = {"username": username} if username else None
        rows = vs.search(query=query, num_results=num_results, filters=filters)
    except Exception as e:
        logger.warning("GitHub Vector Search unavailable: {}", e)
        return ""

    if not rows:
        logger.info("No GitHub chunks found for query: '{}'", query)
        return ""

    chunks = [
        f"[{row.get('repo_name', '')}] {row.get('repo_description', '')}\n"
        f"{row.get('text', '')}"
        for row in rows
    ]
    return "\n\n---\n\n".join(chunks)
