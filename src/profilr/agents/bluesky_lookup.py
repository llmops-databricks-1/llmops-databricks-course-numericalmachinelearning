from atproto import Client
from atproto_client.models.app.bsky.actor.defs import ProfileViewBasic
from loguru import logger


def _name_matches(name: str, actor: ProfileViewBasic) -> bool:
    """Check whether an actor's display name or handle matches the search name.

    Splits the search name into parts and checks if any part appears in either
    the actor's display name or handle, to avoid returning wrong profiles.

    Args:
        name: Full name used for the original search.
        actor: A Bluesky actor result to evaluate.

    Returns:
        True if the actor is likely the person being searched.
    """
    name_parts = name.lower().split()
    display_name = (actor.display_name or "").lower()
    handle = actor.handle.lower()
    return any(part in display_name or part in handle for part in name_parts)


def lookup(name: str) -> str:
    """Return the Bluesky handle for the given person name using the Bluesky search API.

    Searches the top 3 results and returns the first one whose display name or
    handle contains any part of the search name.

    Args:
        name: Full name of the person to look up.

    Returns:
        The Bluesky handle (e.g. 'user.bsky.social'), or 'Profile not found'.
    """
    try:
        client = Client(base_url="https://public.api.bsky.app")
        results = client.app.bsky.actor.search_actors({"q": name, "limit": 3})
        for actor in results.actors:
            if _name_matches(name, actor):
                logger.info("Bluesky handle found for '{}': {}", name, actor.handle)
                return actor.handle
        logger.warning("No matching Bluesky profile found for '{}'", name)
    except Exception as e:
        logger.error("Bluesky search error for '{}': {}", name, e)
    return "Profile not found"
