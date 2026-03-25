import json

from loguru import logger

from profilr.agents.bluesky_lookup import lookup as bluesky_lookup
from profilr.agents.linkedin_lookup import lookup as linkedin_lookup
from profilr.config import MODEL_NAME, get_client
from profilr.schemas import Summary
from profilr.third_party.bluesky import scrape_user_skeets
from profilr.third_party.linkedin import scrape_linkedin_profile


def agent_search(name: str) -> Summary:
    """Research a person and return a short summary with interesting facts.

    Looks up LinkedIn and Bluesky profiles, scrapes public data from both,
    and uses a Databricks foundation model to synthesise a structured summary.

    Args:
        name: Full name of the person to research.

    Returns:
        A Summary containing a short bio and a list of interesting facts.

    Raises:
        ValueError: If the model response cannot be parsed into a valid Summary.
    """
    client = get_client()

    logger.info("Starting research for: '{}'", name)

    linkedin_url = linkedin_lookup(name=name)
    logger.info("LinkedIn URL: {}", linkedin_url)
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_url)
    logger.info("LinkedIn data fetched: {} fields", len(linkedin_data))

    bluesky_username = bluesky_lookup(name=name)
    logger.info("Bluesky username: {}", bluesky_username)
    skeets = scrape_user_skeets(username=bluesky_username)
    logger.info("Bluesky posts fetched: {}", len(skeets))

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Always respond with raw JSON only, "
                    "no markdown, no code blocks. "
                    "Use exactly this schema: "
                    '{"summary": "...", "facts": ["...", "..."]}'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Given the LinkedIn information about a person: {linkedin_data}, "
                    f"and their Bluesky posts: {skeets}, create a short summary and "
                    "two interesting facts about them. Use both sources. "
                    "Respond with raw JSON only."
                ),
            },
        ],
    )

    content = response.choices[0].message.content.strip()
    logger.debug("Model raw response: {}", content[:200])

    # Strip markdown code block if the model wraps the JSON anyway
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
