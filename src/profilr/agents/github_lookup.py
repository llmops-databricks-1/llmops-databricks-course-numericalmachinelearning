import json

from loguru import logger

from profilr.config import get_client
from profilr.tools.tavily_search import get_profile_url_tavily

MAX_ITERATIONS = 5

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_profile_url_tavily",
            "description": "Search the web to find a person's GitHub username.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query, e.g. 'John Smith GitHub profile'",
                    }
                },
                "required": ["query"],
            },
        },
    }
]


def lookup(name: str, llm_endpoint: str) -> str:
    """Return the GitHub username for the given person name.

    Uses an LLM agent with Tavily web search to find the correct GitHub username.
    The agent loop is capped at MAX_ITERATIONS to prevent runaway API usage.

    Args:
        name: Full name of the person to look up.
        llm_endpoint: Databricks model serving endpoint name.

    Returns:
        The GitHub username (without @ or URL prefix),
        or 'Profile not found' if not found.
    """
    client = get_client()
    messages = [
        {
            "role": "user",
            "content": (
                f"Given the full name '{name}', find their GitHub username. "
                "Return only the username (no @ sign, no URL), nothing else."
            ),
        }
    ]

    for iteration in range(MAX_ITERATIONS):
        logger.debug(
            "GitHub lookup iteration {}/{} for '{}'",
            iteration + 1,
            MAX_ITERATIONS,
            name,
        )
        response = client.chat.completions.create(
            model=llm_endpoint,
            tools=TOOLS,
            messages=messages,
            temperature=0,
        )
        choice = response.choices[0]

        if choice.finish_reason == "tool_calls":
            messages.append(choice.message)
            for tool_call in choice.message.tool_calls:
                args = json.loads(tool_call.function.arguments)
                logger.debug("GitHub agent searching: {}", args["query"])
                result = get_profile_url_tavily(args["query"])
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )
        else:
            username = choice.message.content.strip()
            logger.info("GitHub username found for '{}': {}", name, username)
            return username

    logger.warning(
        "GitHub lookup reached max iterations ({}) for '{}'", MAX_ITERATIONS, name
    )
    return "Profile not found"
