# Databricks notebook source
"""
Discover Contributors — Week 2 (Reverse Flow)

Starting from a topic of interest, finds the top GitHub repositories,
identifies their key contributors, and produces a research summary for
each person using arXiv papers and GitHub Vector Search data.

Parameters:
    topic: Topic to research (passed as a job base parameter).
"""

from dotenv import load_dotenv
from loguru import logger

# Load .env for local development (no-op on Databricks)
load_dotenv()

# COMMAND ----------

from profilr.orchestrator import discover_contributors

# COMMAND ----------

try:
    topic = dbutils.widgets.get("topic")  # noqa: F821
except Exception:
    topic = "retrieval augmented generation"

logger.info("Discovering contributors for topic: '{}'", topic)

# COMMAND ----------

results = discover_contributors(topic=topic)
logger.info("Research complete: {} contributors found", len(results))

# COMMAND ----------

for name, summary in results:
    logger.info("")
    logger.info("=" * 50)
    logger.info("  {}", name)
    logger.info("=" * 50)
    logger.info("  {}", summary.summary)
    for fact in summary.facts:
        logger.info("  • {}", fact)
