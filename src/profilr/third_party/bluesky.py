from atproto import Client
from loguru import logger


def scrape_user_skeets(username: str, num_posts: int = 2) -> list[dict]:
    """Return the most recent original posts for the given Bluesky username.

    Fetches posts from the public Bluesky API, excluding replies and reposts.
    Returns an empty list if the username is invalid or the API call fails.

    Args:
        username: Bluesky handle (e.g. 'user.bsky.social') or DID.
        num_posts: Maximum number of posts to return. Defaults to 2.

    Returns:
        List of dicts with keys: text, url, time_posted.
    """
    if not username or username.lower() == "profile not found":
        logger.warning("Skipping Bluesky scrape — invalid username: '{}'", username)
        return []

    try:
        client = Client(base_url="https://public.api.bsky.app")
        feed_response = client.app.bsky.feed.get_author_feed(
            {
                "actor": username,
                "limit": num_posts,
                "filter": "posts_no_replies",
            }
        )

        posts = [
            {
                "text": item.post.record.text,
                "url": f"https://bsky.app/profile/{username}/post/{item.post.uri.split('/')[-1]}",
                "time_posted": item.post.record.created_at,
            }
            for item in feed_response.feed
        ]
        logger.debug("Fetched {} posts for '{}'", len(posts), username)
        return posts

    except Exception as e:
        logger.error("Error fetching Bluesky posts for '{}': {}", username, e)
        return []
