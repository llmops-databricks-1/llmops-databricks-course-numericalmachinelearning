import json

from loguru import logger

from profilr.config import MODEL_NAME, get_client
from profilr.tools.tavily_search import get_profile_url_tavily

MAX_ITERATIONS = 5

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_profile_url_tavily",
            "description": "Search the web to find a person's LinkedIn profile URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query, e.g. 'John Smith LinkedIn profile'",
                    }
                },
                "required": ["query"],
            },
        },
    }
]


def lookup(name: str) -> str:
    """Return the LinkedIn profile URL for the given person name.

    Uses an LLM agent with Tavily web search to find the correct profile URL.
    The agent loop is capped at MAX_ITERATIONS to prevent runaway API usage.

    Args:
        name: Full name of the person to look up.

    Returns:
        The LinkedIn profile URL, or a best-effort string if not found.
    """
    client = get_client()
    messages = [
        {
            "role": "user",
            "content": (
                f"Given the full name '{name}', find their LinkedIn profile URL. "
                "Return only the URL, nothing else."
            ),
        }
    ]

    for iteration in range(MAX_ITERATIONS):
        logger.debug(
            "LinkedIn lookup iteration {}/{} for '{}'",
            iteration + 1,
            MAX_ITERATIONS,
            name,
        )
        response = client.chat.completions.create(
            model=MODEL_NAME,
            tools=TOOLS,
            messages=messages,
            temperature=0,
        )
        choice = response.choices[0]

        if choice.finish_reason == "tool_calls":
            messages.append(choice.message)
            for tool_call in choice.message.tool_calls:
                args = json.loads(tool_call.function.arguments)
                logger.debug("LinkedIn agent searching: {}", args["query"])
                result = get_profile_url_tavily(args["query"])
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )
        else:
            url = choice.message.content
            logger.info("LinkedIn URL found for '{}': {}", name, url)
            return url

    logger.warning(
        "LinkedIn lookup reached max iterations ({}) for '{}'", MAX_ITERATIONS, name
    )
    return "Profile not found"
