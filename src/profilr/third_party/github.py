import base64

import requests
from loguru import logger

_GITHUB_API = "https://api.github.com"
_HEADERS = {"Accept": "application/vnd.github.v3+json"}


def fetch_github_repos(username: str, max_repos: int = 5) -> list[dict]:
    """Fetch public repositories with their README content for a GitHub user.

    Skips forked repositories. Returns an empty list if the username is
    invalid or the API call fails.

    Args:
        username: GitHub username.
        max_repos: Maximum number of repositories to return. Defaults to 5.

    Returns:
        List of dicts with keys: name, description, url, stars, language, readme.
    """
    if not username or username.lower() == "profile not found":
        logger.warning("Skipping GitHub fetch — invalid username: '{}'", username)
        return []

    try:
        resp = requests.get(
            f"{_GITHUB_API}/users/{username}/repos",
            params={"sort": "updated", "per_page": max_repos},
            headers=_HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
    except Exception as e:
        logger.error("Error fetching GitHub repos for '{}': {}", username, e)
        return []

    results = []
    for repo in resp.json():
        if repo.get("fork"):
            continue
        results.append(
            {
                "name": repo["name"],
                "description": repo.get("description") or "",
                "url": repo["html_url"],
                "stars": repo.get("stargazers_count", 0),
                "language": repo.get("language") or "",
                "readme": _fetch_readme(username, repo["name"]),
            }
        )

    logger.debug("Fetched {} repos for '{}'", len(results), username)
    return results


def _fetch_readme(username: str, repo: str) -> str:
    """Fetch and decode the README for a GitHub repository.

    Args:
        username: Repository owner's GitHub username.
        repo: Repository name.

    Returns:
        Decoded README text, or empty string if not found or on error.
    """
    try:
        resp = requests.get(
            f"{_GITHUB_API}/repos/{username}/{repo}/readme",
            headers=_HEADERS,
            timeout=10,
        )
        if resp.status_code == 404:
            return ""
        resp.raise_for_status()
        content = resp.json().get("content", "")
        return base64.b64decode(content).decode("utf-8", errors="replace")
    except Exception as e:
        logger.debug("Could not fetch README for {}/{}: {}", username, repo, e)
        return ""


def search_github_by_topic(topic: str, max_repos: int = 3) -> list[dict]:
    """Search GitHub for the top repositories matching a topic.

    Args:
        topic: Search topic or keyword (e.g. 'RAG retrieval').
        max_repos: Maximum number of repositories to return. Defaults to 3.

    Returns:
        List of dicts with keys: owner, name, description, url, stars, language.
    """
    try:
        resp = requests.get(
            f"{_GITHUB_API}/search/repositories",
            params={"q": topic, "sort": "stars", "per_page": max_repos},
            headers=_HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])
    except Exception as e:
        logger.error("Error searching GitHub for topic '{}': {}", topic, e)
        return []

    results = [
        {
            "owner": repo["owner"]["login"],
            "name": repo["name"],
            "description": repo.get("description") or "",
            "url": repo["html_url"],
            "stars": repo.get("stargazers_count", 0),
            "language": repo.get("language") or "",
        }
        for repo in items
    ]
    logger.debug("Found {} repos for topic '{}'", len(results), topic)
    return results


def get_repo_contributors(
    owner: str, repo: str, max_contributors: int = 3
) -> list[dict]:
    """Fetch the top contributors for a GitHub repository.

    Enriches each contributor with their display name by hitting the user
    profile endpoint. Falls back to login if the profile fetch fails.

    Args:
        owner: Repository owner (GitHub username or org).
        repo: Repository name.
        max_contributors: Maximum number of contributors to return. Defaults to 3.

    Returns:
        List of dicts with keys: login, name.
    """
    try:
        resp = requests.get(
            f"{_GITHUB_API}/repos/{owner}/{repo}/contributors",
            params={"per_page": max_contributors},
            headers=_HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        contributors = resp.json()
    except Exception as e:
        logger.error("Error fetching contributors for {}/{}: {}", owner, repo, e)
        return []

    results = []
    for contributor in contributors:
        login = contributor["login"]
        try:
            profile_resp = requests.get(
                f"{_GITHUB_API}/users/{login}",
                headers=_HEADERS,
                timeout=10,
            )
            name = profile_resp.json().get("name") or login
        except Exception:
            name = login
        results.append({"login": login, "name": name})

    logger.debug("Found {} contributors for {}/{}", len(results), owner, repo)
    return results
