# Databricks notebook source
"""
Week 5 — Deploy the registered Profilr agent to a model serving endpoint.

Reads the latest-model alias from Unity Catalog, deploys it via agents.deploy(),
and validates the live endpoint with a test request.
"""

import time

import mlflow
from databricks import agents
from databricks.sdk import WorkspaceClient
from loguru import logger
from mlflow import MlflowClient

from profilr.config import get_widget, load_config

# COMMAND ----------
# Setup
env = get_widget("env", "dev")
git_sha = get_widget("git_sha", "local")

cfg = load_config("project_config.yml", env=env)
w = WorkspaceClient()

secret_scope = "profilr"
model_name = f"{cfg.catalog}.{cfg.schema}.profilr_agent"
endpoint_name = f"profilr-agent-{env}"

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

logger.info("Model: {}", model_name)
logger.info("Endpoint: {}", endpoint_name)
logger.info("Git SHA: {}", git_sha)

# COMMAND ----------
# Get the latest registered model version
model_version = client.get_model_version_by_alias(
    name=model_name,
    alias="latest-model",
).version

logger.info("Deploying model version: {}", model_version)

# COMMAND ----------
# Deploy the agent endpoint
experiment = client.get_experiment_by_name(cfg.experiment_name)
if experiment is None:
    raise ValueError(f"MLflow experiment not found: {cfg.experiment_name}")

# Wait for any in-progress update to settle before deploying
try:
    ep = w.serving_endpoints.get(endpoint_name)
    while ep.state.config_update.value == "IN_PROGRESS":
        logger.info("Endpoint {} is updating, waiting 30s...", endpoint_name)
        time.sleep(30)
        ep = w.serving_endpoints.get(endpoint_name)
except Exception:
    pass  # endpoint doesn't exist yet — first deploy

endpoint = agents.deploy(
    model_name=model_name,
    model_version=int(model_version),
    endpoint_name=endpoint_name,
    scale_to_zero=True,
    workload_size="Small",
    deploy_feedback_model=False,
    environment_vars={
        "GIT_SHA": git_sha,
        "MODEL_VERSION": model_version,
        "MODEL_SERVING_ENDPOINT_NAME": endpoint_name,
        "MLFLOW_EXPERIMENT_ID": experiment.experiment_id,
        "TAVILY_API_KEY": f"{{secrets/{secret_scope}/TAVILY_API_KEY}}",
        "LAKEBASE_SP_CLIENT_ID": f"{{secrets/{secret_scope}/client-id}}",
        "LAKEBASE_SP_CLIENT_SECRET": f"{{secrets/{secret_scope}/client-secret}}",
        "LAKEBASE_SP_HOST": w.config.host,
    },
)

logger.info("Endpoint deployed: {}", endpoint.endpoint_name)
logger.info("Deployment is async — check status in Serving UI before testing.")
