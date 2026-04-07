# Databricks notebook source
"""
Profilr — AI-powered people research using Databricks foundation models.

Given a person's name, looks up their LinkedIn and Bluesky profiles
and returns a short summary with interesting facts.

Parameters:
    name: Full name of the person to research (passed as a job base parameter).
"""

from dotenv import load_dotenv
from loguru import logger

# Load .env for local development (no-op on Databricks)
load_dotenv()

# COMMAND ----------

from profilr.config import get_env, load_config
from profilr.orchestrator import agent_search

# COMMAND ----------

cfg = load_config("project_config.yml", env=get_env())

# Use a Databricks widget for parameterisation; fall back to a default for local dev
try:
    name = dbutils.widgets.get("name")  # noqa: F821
except Exception:
    name = "Alessandro Alati"

logger.info("Running profilr for: '{}'", name)

# COMMAND ----------

result = agent_search(name=name, cfg=cfg)
logger.info("Result: {}", result)
