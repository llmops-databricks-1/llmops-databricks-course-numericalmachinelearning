import json

import mlflow
from loguru import logger
from mlflow.entities import SpanType
from openai import OpenAI
from openai.types.chat import ChatCompletion

from profilr.agents.bluesky_lookup import lookup as bluesky_lookup
from profilr.agents.github_lookup import lookup as github_lookup
from profilr.agents.linkedin_lookup import lookup as linkedin_lookup
from profilr.config import ProjectConfig, get_client
from profilr.schemas import Summary
from profilr.third_party.arxiv_search import search_papers_by_author
from profilr.third_party.bluesky import scrape_user_skeets
from profilr.third_party.github import (
    fetch_github_repos,
    get_repo_contributors,
    search_github_by_topic,
)
from profilr.third_party.linkedin import scrape_linkedin_profile
from profilr.tools.github_search import search_github_repos


@mlflow.trace(span_type=SpanType.CHAIN)
def agent_search(name: str, cfg: ProjectConfig) -> Summary:
    """Research a person and return a short summary with interesting facts.

    Looks up LinkedIn, Bluesky, and GitHub profiles, scrapes public data
    from all three, and uses a Databricks foundation model to synthesise a
    structured summary.

    Args:
        name: Full name of the person to research.
        cfg: Project configuration (LLM endpoint, catalog, schema, prompt).

    Returns:
        A Summary containing a short bio and a list of interesting facts.

    Raises:
        ValueError: If the model response cannot be parsed into a valid Summary.
    """
    client = get_client()

    logger.info("Starting research for: '{}'", name)

    linkedin_url = linkedin_lookup(name=name, llm_endpoint=cfg.llm_endpoint)
    logger.info("LinkedIn URL: {}", linkedin_url)
    if linkedin_url == "Profile not found":
        logger.warning("LinkedIn profile not found for '{}', skipping scrape", name)
        linkedin_data = {}
    else:
        linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_url)
        logger.info("LinkedIn data fetched: {} fields", len(linkedin_data))

    bluesky_username = bluesky_lookup(name=name)
    logger.info("Bluesky username: {}", bluesky_username)
    if bluesky_username == "Profile not found":
        logger.warning("Bluesky profile not found for '{}', skipping scrape", name)
        skeets = []
    else:
        skeets = scrape_user_skeets(username=bluesky_username)
        logger.info("Bluesky posts fetched: {}", len(skeets))

    github_username = github_lookup(name=name, llm_endpoint=cfg.llm_endpoint)
    logger.info("GitHub username: {}", github_username)
    valid_username = github_username if github_username != "Profile not found" else ""
    github_context = search_github_repos(query=name, cfg=cfg, username=valid_username)
    logger.info("GitHub context fetched: {} chars", len(github_context))

    # Fallback: hit the live GitHub API when the vector index has no data
    if not github_context and valid_username:
        logger.info(
            "Vector search empty — falling back to GitHub API for '{}'", valid_username
        )
        repos = fetch_github_repos(username=valid_username, max_repos=3)
        if repos:
            github_context = "\n\n---\n\n".join(
                f"[{r['name']}] {r['description']}\n{r['readme'][:500]}" for r in repos
            )
            logger.info("GitHub API fallback fetched: {} chars", len(github_context))

    # arXiv: always search — works well for researchers, empty list otherwise
    papers = search_papers_by_author(name=name)
    papers = [{**p, "summary": p["summary"][:300]} for p in papers]
    logger.info("arXiv papers found: {}", len(papers))

    user_content = (
        f"Research subject: {name}.\n"
        f"LinkedIn information: {linkedin_data}\n"
        f"Bluesky posts: {skeets}\n"
        f"GitHub repositories: {github_context}\n"
        f"arXiv papers: {papers}\n"
        f"Create a short professional summary and two interesting facts about {name}. "
        f"Use the name '{name}' directly — never write 'the individual' or 'the person'. "
        "In the facts, you MUST name at least one specific project, repository, "
        "paper title, or tool from the provided data. "
        f"Respond using EXACTLY this JSON format:\n"
        f'{{"summary": "{name} is ...", "facts": ["...", "..."]}}'
    )
    with mlflow.start_span("synthesize", span_type=SpanType.LLM) as span:
        span.set_inputs({"name": name})
        response = client.chat.completions.create(
            model=cfg.llm_endpoint,
            temperature=0,
            messages=[
                {"role": "system", "content": cfg.system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        span.set_outputs({"usage": response.usage.model_dump() if response.usage else {}})

    return _parse_summary(name=name, response=response)


def discover_contributors(topic: str, cfg: ProjectConfig) -> list[tuple[str, Summary]]:
    """Discover top contributors to GitHub repos on a topic and research each one.

    Searches GitHub for repositories matching the topic, collects unique
    contributors across the top repos, then for each contributor fetches
    their arXiv papers and GitHub repository context to produce a Summary.

    This is the reverse flow: starting from a topic of interest rather than
    a specific person's name.

    Args:
        topic: A topic or technology to search for (e.g. 'RAG retrieval').
        cfg: Project configuration (LLM endpoint, catalog, schema, prompt).

    Returns:
        List of (name, Summary) tuples, one per discovered contributor.
    """
    client = get_client()

    logger.info("Discovering contributors for topic: '{}'", topic)
    repos = search_github_by_topic(topic=topic)
    logger.info("Found {} repos for topic '{}'", len(repos), topic)

    seen: set[str] = set()
    contributors: list[dict] = []
    for repo in repos:
        for contributor in get_repo_contributors(repo["owner"], repo["name"]):
            if contributor["login"] not in seen:
                seen.add(contributor["login"])
                contributors.append(contributor)

    logger.info("Found {} unique contributors", len(contributors))

    results = []
    for contributor in contributors:
        login = contributor["login"]
        name = contributor["name"]
        logger.info("Researching contributor: '{}' ({})", name, login)

        papers = search_papers_by_author(name=name)
        logger.info("arXiv papers found: {}", len(papers))

        github_context = search_github_repos(query=topic, cfg=cfg, username=login)
        logger.info("GitHub context fetched: {} chars", len(github_context))

        summary = _synthesise_contributor(
            client=client,
            cfg=cfg,
            name=name,
            login=login,
            papers=papers,
            github_context=github_context,
            topic=topic,
        )
        results.append((name, summary))

    return results


@mlflow.trace(span_type=SpanType.LLM)
def _synthesise_contributor(
    client: OpenAI,
    cfg: ProjectConfig,
    name: str,
    login: str,
    papers: list[dict],
    github_context: str,
    topic: str,
) -> Summary:
    """Synthesise a Summary for a contributor using arXiv and GitHub data.

    Args:
        client: OpenAI-compatible Databricks client.
        cfg: Project configuration (LLM endpoint, system prompt).
        name: Contributor's display name.
        login: Contributor's GitHub username.
        papers: List of arXiv paper dicts (title, summary, url).
        github_context: Formatted GitHub repo chunks from Vector Search.
        topic: The topic that led to this contributor being discovered.

    Returns:
        A Summary containing a short bio and a list of interesting facts.

    Raises:
        ValueError: If the model response cannot be parsed into a valid Summary.
    """
    response = client.chat.completions.create(
        model=cfg.llm_endpoint,
        temperature=0,
        messages=[
            {"role": "system", "content": cfg.system_prompt},
            {
                "role": "user",
                "content": (
                    f"Research subject: '{name}' (GitHub: {login}), "
                    f"discovered via the topic '{topic}'. "
                    f"Their arXiv papers: {papers}. "
                    f"Their GitHub repository work: {github_context}. "
                    "Create a short summary and two interesting facts about this person "
                    "based on their published work and GitHub activity. "
                    "Respond with raw JSON only."
                ),
            },
        ],
    )

    return _parse_summary(name=name, response=response)


def _parse_summary(name: str, response: ChatCompletion) -> Summary:
    """Parse a raw LLM response into a Summary, stripping any markdown wrapping.

    Args:
        name: Name of the person being researched (used in error messages).
        response: ChatCompletion response from the OpenAI-compatible client.

    Returns:
        A validated Summary instance.

    Raises:
        ValueError: If the response content is not valid JSON or missing fields.
    """
    content = response.choices[0].message.content.strip()
    logger.debug("Model raw response: {}", content[:200])

    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]

    try:
        data = json.loads(content.strip())
        return Summary(**data)
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(
            "Failed to parse model response for '{}': {}\nContent: {}", name, e, content
        )
        raise ValueError(f"Model returned invalid JSON for '{name}': {e}") from e
