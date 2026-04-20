import os
from collections.abc import Generator
from datetime import datetime
from uuid import uuid4

import mlflow
from loguru import logger
from mlflow import MlflowClient
from mlflow.models.resources import (
    DatabricksServingEndpoint,
    DatabricksVectorSearchIndex,
)
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from profilr.config import ProjectConfig
from profilr.orchestrator import agent_search


class ProfilrAgent(ResponsesAgent):
    """MLflow-compatible agent that wraps the Profilr research pipeline."""

    def __init__(self, cfg: ProjectConfig) -> None:
        self.cfg = cfg

    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Run the research pipeline and stream the result."""
        custom = request.custom_inputs or {}
        session_id = custom.get("session_id")
        request_id = custom.get("request_id")

        mlflow.update_current_trace(
            tags={
                "git_sha": os.getenv("GIT_SHA", "local"),
                "model_serving_endpoint_name": os.getenv(
                    "MODEL_SERVING_ENDPOINT_NAME", "local"
                ),
                "model_version": os.getenv("MODEL_VERSION", "local"),
            },
            metadata=({"mlflow.trace.session": session_id} if session_id else {}),
            client_request_id=request_id,
        )

        # Extract the person's name from the last user message
        name = next(
            (i.content for i in reversed(request.input) if i.role == "user"),
            "",
        )
        logger.info("ProfilrAgent researching: '{}'", name)

        summary = agent_search(name, self.cfg)

        response_text = f"{summary.summary}\n\nInteresting facts:\n" + "\n".join(
            f"- {fact}" for fact in summary.facts
        )

        item = self.create_text_output_item(response_text, str(uuid4()))
        yield ResponsesAgentStreamEvent(type="response.output_item.done", item=item)

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """Run the research pipeline and return the result."""
        events = list(self.predict_stream(request))
        output = [
            e.item if isinstance(e.item, dict) else e.item.model_dump()
            for e in events
            if e.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(
            output=output,
            custom_outputs=request.custom_inputs,
        )


def log_register_agent(
    cfg: ProjectConfig,
    git_sha: str,
    run_id: str,
    agent_code_path: str,
    model_name: str,
    evaluation_metrics: dict | None = None,
) -> mlflow.entities.model_registry.RegisteredModel:
    """Log and register the Profilr agent as an MLflow model in Unity Catalog.

    Args:
        cfg: Project configuration.
        git_sha: Git commit SHA for tracking.
        run_id: Run identifier for tracking.
        agent_code_path: Path to the agent entry-point Python file.
        model_name: Full model name in Unity Catalog (catalog.schema.name).
        evaluation_metrics: Optional dict of metrics to log with the run.

    Returns:
        RegisteredModel object from Unity Catalog.
    """
    resources = [
        DatabricksServingEndpoint(endpoint_name=cfg.llm_endpoint),
        DatabricksServingEndpoint(endpoint_name=cfg.embedding_endpoint),
        DatabricksVectorSearchIndex(
            index_name=f"{cfg.catalog}.{cfg.schema}.github_index"
        ),
    ]

    model_config = {
        "catalog": cfg.catalog,
        "schema": cfg.schema,
        "system_prompt": cfg.system_prompt,
        "llm_endpoint": cfg.llm_endpoint,
        "embedding_endpoint": cfg.embedding_endpoint,
        "vs_endpoint": cfg.vs_endpoint,
        "experiment_name": cfg.experiment_name,
    }

    test_request = {"input": [{"role": "user", "content": "Linus Torvalds"}]}

    mlflow.set_experiment(cfg.experiment_name)
    ts = datetime.now().strftime("%Y-%m-%d")

    with mlflow.start_run(
        run_name=f"profilr-agent-{ts}",
        tags={"git_sha": git_sha, "run_id": run_id},
    ) as run:
        model_info = mlflow.pyfunc.log_model(
            name="agent",
            python_model=agent_code_path,
            resources=resources,
            input_example=test_request,
            model_config=model_config,
        )
        if evaluation_metrics:
            mlflow.log_metrics(evaluation_metrics)
        logger.info("Logged model in run: {}", run.info.run_id)

    logger.info("Registering model: {}", model_name)
    registered_model = mlflow.register_model(
        model_uri=model_info.model_uri,
        name=model_name,
        env_pack="databricks_model_serving",
        tags={"git_sha": git_sha, "run_id": run_id},
    )
    logger.info("Registered version: {}", registered_model.version)

    client = MlflowClient()
    logger.info("Setting alias 'latest-model'")
    client.set_registered_model_alias(
        name=model_name,
        alias="latest-model",
        version=registered_model.version,
    )
    return registered_model
