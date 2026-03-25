from tavily import TavilyClient

from profilr.config import get_secret


def get_profile_url_tavily(query: str) -> str:
    """Search the web using Tavily and return raw results as a string.

    Args:
        query: The search query string.

    Returns:
        String representation of the Tavily search results.
    """
    client = TavilyClient(api_key=get_secret("TAVILY_API_KEY"))
    results = client.search(query)
    return str(results)
