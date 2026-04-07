# Databricks notebook source
import os
import random
from datetime import datetime
from uuid import uuid4

import mlflow
from databricks.sdk import WorkspaceClient
from loguru import logger
from mlflow.types.responses import ResponsesAgentRequest

from profilr.agent import ProfilrAgent, log_register_agent
from profilr.config import get_env, load_config
from profilr.evaluation import (
    factual_guideline,
    has_facts_section,
    professional_tone_guideline,
    summary_not_empty,
    under_300_words,
)

# COMMAND ----------
# Setup
cfg = load_config("project_config.yml", env=get_env())
w = WorkspaceClient()
mlflow.set_experiment(cfg.experiment_name)
logger.info("Config loaded: catalog={}, schema={}", cfg.catalog, cfg.schema)

# COMMAND ----------
# Create agent
agent = ProfilrAgent(cfg=cfg)
logger.info("ProfilrAgent created")

# COMMAND ----------
# Test with tracing
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
session_id = f"s-{timestamp}-{random.randint(100000, 999999)}"
request_id = f"req-{timestamp}-{random.randint(100000, 999999)}"

test_request = ResponsesAgentRequest(
    input=[{"role": "user", "content": "Linus Torvalds"}],
    custom_inputs={"session_id": session_id, "request_id": request_id},
)

logger.info("Running agent for: 'Linus Torvalds'")
response = agent.predict(test_request)
if response.output:
    item = response.output[-1]
    content = (
        item.get("content", [])
        if isinstance(item, dict)
        else getattr(item, "content", [])
    )
    text = (
        (
            content[0].get("text", "")
            if isinstance(content[0], dict)
            else getattr(content[0], "text", "")
        )
        if content
        else str(item)
    )
    logger.info("Response:\n{}", text)
else:
    logger.info("Response: No output")

# COMMAND ----------
# Evaluation
with open("../eval_inputs.txt") as f:
    eval_data = [{"inputs": {"question": line.strip()}} for line in f if line.strip()]

logger.info("Loaded {} evaluation inputs", len(eval_data))


def predict_fn(question: str) -> str:
    """Wrap the agent for mlflow.genai.evaluate."""
    req = ResponsesAgentRequest(
        input=[{"role": "user", "content": question}],
        custom_inputs={"request_id": str(uuid4())},
    )
    result = agent.predict(req)
    if result.output:
        item = result.output[-1]
        content = (
            item.get("content", [])
            if isinstance(item, dict)
            else getattr(item, "content", [])
        )
        if content:
            first = content[0]
            return (
                first.get("text", "")
                if isinstance(first, dict)
                else getattr(first, "text", "")
            )
    return ""


results = mlflow.genai.evaluate(
    predict_fn=predict_fn,
    data=eval_data,
    scorers=[
        summary_not_empty,
        has_facts_section,
        under_300_words,
        professional_tone_guideline,
        factual_guideline,
    ],
)
logger.info("Evaluation metrics: {}", results.metrics)

# COMMAND ----------
# Log & register
git_sha = os.getenv("GIT_SHA", "local")
run_id = os.getenv("MLFLOW_RUN_ID", str(uuid4())[:8])
model_name = f"{cfg.catalog}.{cfg.schema}.profilr_agent"

registered_model = log_register_agent(
    cfg=cfg,
    git_sha=git_sha,
    run_id=run_id,
    agent_code_path="../profilr_agent.py",
    model_name=model_name,
    evaluation_metrics=results.metrics,
)

logger.info("Model registered: {} v{}", model_name, registered_model.version)
logger.info("Alias 'latest-model' set")
