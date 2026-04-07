# Databricks notebook source
"""
Entry point for MLflow model serving.

MLflow loads this file when serving the Profilr agent endpoint.
It reads model_config from the logged model and instantiates ProfilrAgent.
"""

import mlflow

from profilr.agent import ProfilrAgent
from profilr.config import ProjectConfig

model_config = mlflow.models.ModelConfig(development_config="project_config.yml")

cfg = ProjectConfig(
    catalog=model_config.get("catalog"),
    schema=model_config.get("schema"),
    system_prompt=model_config.get("system_prompt") or "",
    llm_endpoint=model_config.get("llm_endpoint"),
    embedding_endpoint=model_config.get("embedding_endpoint") or "",
    vs_endpoint=model_config.get("vs_endpoint") or "",
    experiment_name=model_config.get("experiment_name") or "",
)

mlflow.models.set_model(ProfilrAgent(cfg=cfg))
