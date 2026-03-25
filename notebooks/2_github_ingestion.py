# Databricks notebook source
"""
GitHub Ingestion Pipeline — Week 2

Fetches public GitHub repositories and README files for a given user,
chunks the content, stores it in a Delta table with Change Data Feed
enabled, and syncs a Databricks Vector Search index so that the profilr
agent can perform semantic retrieval at runtime.

Parameters:
    github_username: GitHub username to ingest (passed as a job base parameter).
"""

from loguru import logger

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Fetch GitHub Repositories and READMEs
# COMMAND ----------
from profilr.third_party.github import fetch_github_repos

# COMMAND ----------

try:
    github_username = dbutils.widgets.get("github_username")  # noqa: F821
except Exception:
    github_username = "langchain-ai"

logger.info("Ingesting GitHub repos for: '{}'", github_username)

# COMMAND ----------

repos = fetch_github_repos(username=github_username, max_repos=5)
logger.info("Fetched {} repos for '{}'", len(repos), github_username)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Chunk README Content
# MAGIC
# MAGIC README files can be thousands of tokens — too large to embed whole.
# MAGIC We split them into fixed-size chunks with overlap so each chunk fits
# MAGIC within the embedding model's token limit (512 tokens) while preserving
# MAGIC context across chunk boundaries.

# COMMAND ----------

import re


def _clean_chunk(text: str) -> str:
    """Normalise whitespace and fix hyphenation across line breaks.

    Args:
        text: Raw text to clean.

    Returns:
        Cleaned text with normalised whitespace.
    """
    t = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)  # fix hyphenated line breaks
    t = re.sub(r"\s*\n\s*", " ", t)  # collapse newlines to spaces
    t = re.sub(r"\s+", " ", t)  # collapse repeated whitespace
    result = t.strip()
    logger.debug("Cleaned chunk: {} → {} chars", len(text), len(result))
    return result


def fixed_size_chunking(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into fixed-size character chunks with overlap.

    Args:
        text: Source text to split.
        chunk_size: Characters per chunk. Defaults to 500.
        overlap: Characters of overlap between consecutive chunks. Defaults to 50.

    Returns:
        List of text chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += chunk_size - overlap
    logger.debug(
        "Chunked {} chars into {} chunks (size={}, overlap={})",
        len(text),
        len(chunks),
        chunk_size,
        overlap,
    )
    return chunks


# COMMAND ----------

records = []
for repo in repos:
    source_text = f"{repo['description']}\n\n{repo['readme']}".strip()
    if not source_text:
        continue

    cleaned = _clean_chunk(source_text)
    for idx, chunk_text in enumerate(fixed_size_chunking(cleaned)):
        records.append(
            {
                "id": f"{github_username}_{repo['name']}_{idx}",
                "text": chunk_text,
                "username": github_username,
                "repo_name": repo["name"],
                "repo_description": repo["description"],
                "repo_url": repo["url"],
                "chunk_index": idx,
            }
        )

logger.info("Created {} chunks from {} repos", len(records), len(repos))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Write Chunks to Delta Table
# MAGIC
# MAGIC We enable Change Data Feed (CDF) on the table so that the Vector Search
# MAGIC index can track only the rows that change on each sync, rather than
# MAGIC reprocessing the whole table every time.

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import types as T
from pyspark.sql.functions import current_timestamp

from profilr.config import CATALOG, SCHEMA

spark = SparkSession.builder.getOrCreate()

schema = T.StructType(
    [
        T.StructField("id", T.StringType(), False),
        T.StructField("text", T.StringType(), True),
        T.StructField("username", T.StringType(), True),
        T.StructField("repo_name", T.StringType(), True),
        T.StructField("repo_description", T.StringType(), True),
        T.StructField("repo_url", T.StringType(), True),
        T.StructField("chunk_index", T.IntegerType(), True),
    ]
)
# ingest_ts (TimestampType) is added via withColumn below

chunks_table = f"{CATALOG}.{SCHEMA}.github_chunks"
chunks_df = spark.createDataFrame(records, schema=schema).withColumn(
    "ingest_ts", current_timestamp()
)

# Create table on first run; subsequent runs use MERGE to avoid duplicates
chunks_df.write.format("delta").mode("ignore").saveAsTable(chunks_table)

chunks_df.createOrReplaceTempView("new_chunks")
spark.sql(f"""
    MERGE INTO {chunks_table} target
    USING new_chunks source
    ON target.id = source.id
    WHEN NOT MATCHED THEN INSERT *
""")
logger.info("Written {} chunks to {}", len(records), chunks_table)

spark.sql(f"""
    ALTER TABLE {chunks_table}
    SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")
logger.info("Change Data Feed enabled on {}", chunks_table)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Create Vector Search Index and Sync
# MAGIC
# MAGIC The index is a Delta Sync index: it automatically generates embeddings
# MAGIC from the `text` column using the configured embedding model and keeps
# MAGIC itself in sync with the Delta table via Change Data Feed.

# COMMAND ----------

from profilr.vector_search import VectorSearchManager

vs_manager = VectorSearchManager()
vs_manager.create_or_get_index()
vs_manager.sync_index()

logger.info("✓ Ingestion complete for '{}'", github_username)
