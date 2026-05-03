# Databricks notebook source
"""
Week 6 — Send example requests to the Profilr agent endpoint to populate traces.

Run this once to seed the trace logs table with real requests before
running the monitoring notebook.
"""

import random
import time
from datetime import datetime

from databricks.sdk import WorkspaceClient
from loguru import logger
from openai import OpenAI

from profilr.config import get_widget, load_config

# COMMAND ----------
env = get_widget("env", "dev")
cfg = load_config("project_config.yml", env=env)

w = WorkspaceClient()
headers = w.config.authenticate()
token = headers.get("Authorization", "").replace("Bearer ", "")

endpoint_name = f"profilr-agent-{env}"

client = OpenAI(
    api_key=token,
    base_url=f"{w.config.host.rstrip('/')}/serving-endpoints",
)

# COMMAND ----------
queries = [
    "Who is Linus Torvalds and what are his main contributions?",
    "Tell me about Andrej Karpathy's work in AI.",
    "What has Yann LeCun contributed to deep learning?",
    "Summarise the professional background of Geoffrey Hinton.",
    "Who is Sam Altman and what has he built?",
    "Tell me about Fei-Fei Li's research.",
    "What is Demis Hassabis known for in the AI world?",
    "Summarise Jeff Dean's contributions to Google.",
    "Who is Ilya Sutskever and what is his research focus?",
    "Tell me about Jeremy Howard's work in machine learning.",
    "What has François Chollet built and published?",
    "Summarise the career of Sebastian Thrun.",
    "Who is Chris Lattner and what programming tools has he created?",
    "Tell me about Pieter Abbeel's research in robotics.",
    "What is Yoshua Bengio known for in deep learning?",
    "Summarise the professional work of Andrew Ng.",
    "Who is Ian Goodfellow and what is he famous for?",
    "Tell me about Richard Sutton's contributions to reinforcement learning.",
    "What has Dario Amodei done in AI safety research?",
    "Summarise the career of Lex Fridman.",
]

# COMMAND ----------
for i, query in enumerate(queries):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    session_id = f"s-{timestamp}-{random.randint(100000, 999999)}"

    logger.info("[{}/{}] {}", i + 1, len(queries), query[:60])
    try:
        client.responses.create(
            model=endpoint_name,
            input=[{"role": "user", "content": query}],
            extra_body={"custom_inputs": {
                "session_id": session_id,
            }},
        )
    except Exception as e:
        logger.error("Request failed: {}", e)
    time.sleep(2)

logger.info("Done — {} requests sent to {}", len(queries), endpoint_name)
