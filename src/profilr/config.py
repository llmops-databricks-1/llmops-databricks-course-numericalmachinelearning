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
    def from_yaml(
        cls, config_path: str = "project_config.yml", env: str = "dev"
    ) -> "ProjectConfig":
        """Load config from YAML file for the given environment.

        Args:
            config_path: Path to project_config.yml.
            env: Environment name ('dev', 'acc', or 'prd').

        Returns:
            ProjectConfig for the requested environment.
        """
        return load_config(config_path, env)


def get_env() -> str:
    """Get the current deployment environment.

    Reads from the 'env' Databricks widget, which is injected by the job
    runner via base_parameters. Falls back to 'dev' when running locally
    (where dbutils is not available).

    Returns:
        One of 'dev', 'acc', or 'prd'.
    """
    try:
        from databricks.sdk.runtime import dbutils  # noqa: PLC0415

        return dbutils.widgets.get("env")
    except Exception:
        return "dev"


def load_config(
    config_path: str = "project_config.yml", env: str | None = None
) -> ProjectConfig:
    """Load environment-specific configuration from a YAML file.

    Searches up to 3 parent directories to locate the config file, so it
    resolves correctly whether called from a notebook, a job, or locally.

    Args:
        config_path: Filename or path to project_config.yml.
        env: Environment to load ('dev', 'acc', 'prd'). If None, calls
            get_env() to detect it from the Databricks widget.

    Returns:
        A ProjectConfig populated with values for the requested environment.
    """
    if env is None:
        env = get_env()

    path = Path(config_path)
    if not path.is_absolute():
        current = Path.cwd()
        for _ in range(3):
            candidate = current / config_path
            if candidate.exists():
                path = candidate
                break
            current = current.parent

    with path.open() as f:
        raw = yaml.safe_load(f)

    if env not in raw:
        raise ValueError(
            f"Environment '{env}' not found in {path}. Available: {list(raw.keys())}"
        )

    env_cfg = raw[env]
    system_prompt = raw.get("system_prompt", "")

    logger.debug("Loaded config for environment '{}' from {}", env, path)
    return ProjectConfig(system_prompt=system_prompt, **env_cfg)


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
