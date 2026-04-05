# Databricks notebook source
"""
Entry point for MLflow model serving.

MLflow loads this file when serving the Profilr agent endpoint.
It reads model_config from the logged model and instantiates ProfilrAgent.
"""
import mlflow
from profilr.agent import ProfilrAgent

model_config = mlflow.models.ModelConfig(development_config="../project_config.yml")

mlflow.models.set_model(
    ProfilrAgent(
        llm_endpoint=model_config.get("llm_endpoint"),
        system_prompt=model_config.get("system_prompt"),
        catalog=model_config.get("catalog"),
        schema=model_config.get("schema"),
    )
)
