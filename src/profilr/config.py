import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from databricks.sdk import WorkspaceClient
from loguru import logger
from openai import OpenAI

# ---------------------------------------------------------------------------
# Structured config
# ---------------------------------------------------------------------------


@dataclass
class ProjectConfig:
    """All environment-specific settings for Profilr."""

    catalog: str
    schema: str
    llm_endpoint: str
    embedding_endpoint: str
    vs_endpoint: str
    experiment_name: str
    genie_space_id: str = ""
    warehouse_id: str = ""
    lakebase_instance_name: str = ""
    system_prompt: str = field(default="", repr=False)

    @classmethod
    def from_yaml(cls, config_path: str = "../project_config.yml") -> "ProjectConfig":
        """Load config from YAML file for the current environment.

        Args:
            config_path: Path to project_config.yml.

        Returns:
            ProjectConfig for the current environment.
        """
        return load_config(config_path)


def get_env() -> str:
    """Detect the current deployment environment.

    Reads the DATABRICKS_ENV environment variable.
    Defaults to 'dev' when running locally or in notebooks without the variable set.

    Returns:
        One of 'dev', 'acc', or 'prd'.
    """
    return os.getenv("DATABRICKS_ENV", "dev")


def load_config(config_path: str = "../project_config.yml") -> ProjectConfig:
    """Load environment-specific configuration from a YAML file.

    Args:
        config_path: Path to the project_config.yml file, relative to the
            notebook or absolute. Defaults to '../project_config.yml' which
            resolves correctly from notebooks/ to the project root.

    Returns:
        A ProjectConfig populated with values for the current environment.
    """
    env = get_env()
    path = Path(config_path).expanduser()
    with path.open() as f:
        raw = yaml.safe_load(f)

    env_cfg = raw.get(env, raw.get("dev", {}))
    system_prompt = raw.get("system_prompt", "")

    logger.debug("Loaded config for environment '{}' from {}", env, path)
    return ProjectConfig(system_prompt=system_prompt, **env_cfg)


# ---------------------------------------------------------------------------
# Backward-compatible module-level constants
# (used by existing agents, tools, and orchestrator)
# ---------------------------------------------------------------------------

MODEL_NAME = os.getenv("LLM_ENDPOINT", "databricks-llama-4-maverick")
CATALOG = os.getenv("DATABRICKS_CATALOG", "mlops_dev")
SCHEMA = os.getenv("DATABRICKS_SCHEMA", "alessand")
VS_ENDPOINT = os.getenv("VS_ENDPOINT", "llmops_course_vs_endpoint")


# ---------------------------------------------------------------------------
# Secrets and client helpers
# ---------------------------------------------------------------------------


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
