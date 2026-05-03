# Databricks notebook source
"""
Week 6 — Evaluate unevaluated agent traces and create an aggregated table
for monitoring Profilr agent performance (latency, token usage, quality scores).

Reads traces directly from the MLflow API — no Delta sync required.

Tier 1: cheap code-based scorers run on ALL new traces.
Tier 2: expensive LLM-based scorers run on a 10% sample only.
"""

import json

import mlflow
import pandas as pd
from loguru import logger
from mlflow import MlflowClient
from pyspark.sql import SparkSession

from profilr.config import get_widget, load_config
from profilr.evaluation import (
    factual_guideline,
    has_facts_section,
    professional_tone_guideline,
    summary_not_empty,
    under_300_words,
)

# COMMAND ----------
# Setup
env = get_widget("env", "dev")
cfg = load_config("project_config.yml", env=env)
mlflow.set_experiment(cfg.experiment_name)

spark = SparkSession.builder.getOrCreate()

catalog = cfg.catalog
schema = cfg.schema
endpoint_name = f"profilr-agent-{env}"
aggregated_table = f"{catalog}.{schema}.profilr_traces_aggregated"

client = MlflowClient()
experiment = client.get_experiment_by_name(cfg.experiment_name)
if experiment is None:
    raise ValueError(f"MLflow experiment not found: {cfg.experiment_name}")

logger.info("Experiment: {} ({})", cfg.experiment_name, experiment.experiment_id)
logger.info("Endpoint filter: {}", endpoint_name)
logger.info("Aggregated table: {}", aggregated_table)

# COMMAND ----------
# Helpers to extract fields from MLflow trace objects


def extract_response_text(response: str | dict | None) -> str | None:
    """Extract plain text from the agent response JSON."""
    if response is None:
        return None
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except (json.JSONDecodeError, TypeError):
            return response[:500]
    if isinstance(response, dict):
        # responses API: {"output": [{"type": "message", "content": [{"text": "..."}]}]}
        for item in response.get("output", []):
            if isinstance(item, dict) and item.get("type") == "message":
                content = item.get("content", [])
                if content:
                    return content[0].get("text")
        # chat completions fallback
        choices = response.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content")
    return None


def extract_request_preview(request: str | dict | None) -> str:
    """Extract the user query from the request JSON."""
    if isinstance(request, str):
        try:
            request = json.loads(request)
        except (json.JSONDecodeError, TypeError):
            return str(request)[:200]
    if isinstance(request, dict):
        for msg in reversed(request.get("input", [])):
            if isinstance(msg, dict) and msg.get("role") == "user":
                return str(msg.get("content", ""))[:200]
        for msg in reversed(request.get("messages", [])):
            if isinstance(msg, dict) and msg.get("role") == "user":
                return str(msg.get("content", ""))[:200]
    return str(request)[:200]


def compute_span_metrics(spans: list | None) -> dict:
    """Count LLM calls, tool calls, and total tokens from a trace's spans."""
    call_llm_count = 0
    tool_count = 0
    total_tokens = 0
    for span in spans or []:
        name = getattr(span, "name", None) or (
            span.get("name") if isinstance(span, dict) else None
        )
        if name == "call_llm":
            call_llm_count += 1
            try:
                attrs = getattr(span, "attributes", {}) or {}
                outputs_str = attrs.get("mlflow.spanOutputs", "{}")
                outputs = (
                    json.loads(outputs_str)
                    if isinstance(outputs_str, str)
                    else outputs_str
                )
                total_tokens += outputs.get("usage", {}).get("total_tokens", 0)
            except Exception:
                pass
        elif name == "execute_tool":
            tool_count += 1
    return {
        "call_llm_exec_count": call_llm_count,
        "tool_call_count": tool_count,
        "total_tokens_used": total_tokens,
    }


def get_assessment_value(assessments: list | None, name: str) -> str | None:
    """Extract the feedback value for a named assessment."""
    for a in assessments or []:
        a_name = getattr(a, "name", None) or (
            a.get("name") if isinstance(a, dict) else None
        )
        if a_name == name:
            try:
                return a.feedback.value
            except AttributeError:
                return (a.get("feedback") or {}).get("value")
    return None


# COMMAND ----------
# Load traces directly from MLflow — filter to our endpoint only
all_traces_df = mlflow.search_traces(
    experiment_ids=[experiment.experiment_id],
    filter_string=f"tag.`model_serving_endpoint_name` = '{endpoint_name}'",
    max_results=500,
)

logger.info("Total traces found: {}", len(all_traces_df))

# Keep only traces without assessments yet
unevaluated_df = all_traces_df[
    all_traces_df["assessments"].apply(
        lambda x: len(x) == 0 if isinstance(x, list) else True
    )
].copy()

logger.info("Unevaluated traces: {}", len(unevaluated_df))

# COMMAND ----------
# Build eval DataFrame — extract request/response text per trace
records = []
for _, row in unevaluated_df.iterrows():
    response_text = extract_response_text(row.get("response"))
    if response_text is None:
        continue
    records.append(
        {
            "trace_id": row["trace_id"],
            "inputs": {"query": extract_request_preview(row.get("request"))},
            "outputs": response_text,
        }
    )

eval_pdf = pd.DataFrame(records)
logger.info("Traces ready for evaluation: {}", len(eval_pdf))

# COMMAND ----------
# Tier 1 — run cheap code-based scorers on ALL traces
if len(eval_pdf) > 0:
    tier1_result = mlflow.genai.evaluate(
        data=eval_pdf[["inputs", "outputs"]],
        scorers=[under_300_words, summary_not_empty, has_facts_section],
    )

    for trace_id, assessments in zip(
        eval_pdf["trace_id"],
        tier1_result.result_df["assessments"],
        strict=True,
    ):
        for a in assessments:
            mlflow.log_feedback(
                trace_id=trace_id,
                name=a["assessment_name"],
                value=a["feedback"]["value"],
            )

    logger.info("Logged Tier 1 assessments for {} traces", len(eval_pdf))

    # Tier 2 — LLM-based scorers on a 10% sample only to control cost
    sample_size = max(1, int(len(eval_pdf) * 0.1))
    sampled_pdf = eval_pdf.sample(n=sample_size)
    logger.info("Sampled {} traces for LLM-based evaluation", len(sampled_pdf))

    tier2_result = mlflow.genai.evaluate(
        data=sampled_pdf[["inputs", "outputs"]],
        scorers=[professional_tone_guideline, factual_guideline],
    )

    for trace_id, assessments in zip(
        sampled_pdf["trace_id"],
        tier2_result.result_df["assessments"],
        strict=True,
    ):
        for a in assessments:
            mlflow.log_feedback(
                trace_id=trace_id,
                name=a["assessment_name"],
                value=a["feedback"]["value"],
            )

    logger.info("Logged Tier 2 assessments for {} traces", len(sampled_pdf))
else:
    logger.info("No new traces to evaluate — skipping scoring")

# COMMAND ----------
# Reload all traces now that assessments have been logged, then build
# the aggregated table — one clean row per trace for dashboarding.
all_traces_df = mlflow.search_traces(
    experiment_ids=[experiment.experiment_id],
    filter_string=f"tag.`model_serving_endpoint_name` = '{endpoint_name}'",
    max_results=500,
)


def score(assessments: list | None, name: str, pass_val: str) -> int:
    """Return 1 if the named assessment matches pass_val, else 0."""
    val = get_assessment_value(assessments, name)
    return 1 if val == pass_val else 0


rows = []
for _, row in all_traces_df.iterrows():
    response_text = extract_response_text(row.get("response"))
    span_metrics = compute_span_metrics(row.get("spans"))
    assessments = row.get("assessments") or []
    rows.append(
        {
            "trace_id": str(row.get("trace_id", "")),
            "request_time": row.get("timestamp_ms"),
            "request_preview": extract_request_preview(row.get("request")),
            "response_text": response_text or "",
            "latency_seconds": (row.get("execution_time_ms") or 0) / 1000.0,
            "call_llm_exec_count": span_metrics["call_llm_exec_count"],
            "tool_call_count": span_metrics["tool_call_count"],
            "total_tokens_used": span_metrics["total_tokens_used"],
            "under_300_words": score(assessments, "under_300_words", "true"),
            "summary_not_empty": score(assessments, "summary_not_empty", "true"),
            "has_facts_section": score(assessments, "has_facts_section", "true"),
            "professional_tone": score(assessments, "professional_tone", "Pass"),
            "factual_content": score(assessments, "factual_content", "Pass"),
        }
    )

aggregated_pdf = pd.DataFrame(rows)
logger.info("Aggregated {} traces for dashboard", len(aggregated_pdf))

# COMMAND ----------
# Write aggregated results to a Delta table (overwrite on each run)
(
    spark.createDataFrame(aggregated_pdf)
    .write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(aggregated_table)
)

logger.info("Table {}.{}.profilr_traces_aggregated written", catalog, schema)
