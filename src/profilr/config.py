import os

from databricks.sdk import WorkspaceClient
from loguru import logger
from openai import OpenAI

MODEL_NAME = "databricks-llama-4-maverick"
CATALOG = os.getenv("DATABRICKS_CATALOG", "profilr")
SCHEMA = os.getenv("DATABRICKS_SCHEMA", "default")
VS_ENDPOINT = os.getenv("VS_ENDPOINT", "profilr-vs-endpoint")


def get_secret(key: str, scope: str = "profilr") -> str | None:
    """Get a secret from Databricks secrets, falling back to an environment variable.

    Args:
        key: The secret key name (also used as the env var name locally).
        scope: The Databricks secret scope. Defaults to "profilr".

    Returns:
        The secret value, or None if not found in either source.
    """
    try:
        from databricks.sdk.runtime import dbutils  # noqa: PLC0415

        return dbutils.secrets.get(scope=scope, key=key)
    except Exception:
        return os.getenv(key)


def get_client() -> OpenAI:
    """Create an OpenAI-compatible client pointed at Databricks serving endpoints.

    A fresh client is created on each call so the OAuth token is always valid
    and never stale from a previous authentication cycle.

    Returns:
        An OpenAI client configured to call Databricks foundation model endpoints.
    """
    w = WorkspaceClient()
    headers = w.config.authenticate()
    token = headers.get("Authorization", "").replace("Bearer ", "")
    logger.debug("Databricks client initialised for host: {}", w.config.host)
    return OpenAI(
        api_key=token,
        base_url=f"{w.config.host.rstrip('/')}/serving-endpoints",
    )
