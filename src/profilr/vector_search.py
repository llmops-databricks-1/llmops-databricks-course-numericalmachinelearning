"""Vector Search management for GitHub repository chunks."""

from databricks.vector_search.client import VectorSearchClient
from loguru import logger

from profilr.config import CATALOG, SCHEMA, VS_ENDPOINT

_EMBEDDING_MODEL = "databricks-gte-large-en"


class VectorSearchManager:
    """Manages the Vector Search endpoint and index for GitHub repo chunks.

    Mirrors the VectorSearchManager pattern from the arxiv_curator project,
    adapted for profilr's GitHub ingestion pipeline.
    """

    def __init__(
        self,
        endpoint_name: str = VS_ENDPOINT,
        embedding_model: str = _EMBEDDING_MODEL,
    ) -> None:
        self.endpoint_name = endpoint_name
        self.embedding_model = embedding_model
        self.catalog = CATALOG
        self.schema = SCHEMA
        self.client = VectorSearchClient()
        self.index_name = f"{self.catalog}.{self.schema}.github_index"
        self.source_table = f"{self.catalog}.{self.schema}.github_chunks"

    def create_endpoint_if_not_exists(self) -> None:
        """Create the Vector Search endpoint if it does not already exist."""
        endpoints_response = self.client.list_endpoints()
        endpoints = (
            endpoints_response.get("endpoints", [])
            if isinstance(endpoints_response, dict)
            else []
        )
        endpoint_exists = any(
            (ep.get("name") if isinstance(ep, dict) else getattr(ep, "name", None))
            == self.endpoint_name
            for ep in endpoints
        )

        if not endpoint_exists:
            logger.info("Creating Vector Search endpoint: {}", self.endpoint_name)
            self.client.create_endpoint_and_wait(
                name=self.endpoint_name, endpoint_type="STANDARD"
            )
            logger.info("✓ Vector Search endpoint created: {}", self.endpoint_name)
        else:
            logger.info("✓ Vector Search endpoint exists: {}", self.endpoint_name)

    def create_or_get_index(self) -> object:
        """Create the Delta Sync index if it does not exist, otherwise return it.

        Returns:
            The Vector Search index object.
        """
        self.create_endpoint_if_not_exists()

        try:
            index = self.client.get_index(index_name=self.index_name)
            logger.info("✓ Vector Search index exists: {}", self.index_name)
            return index
        except Exception:
            logger.info("Index {} not found, creating it", self.index_name)

        try:
            index = self.client.create_delta_sync_index(
                endpoint_name=self.endpoint_name,
                source_table_name=self.source_table,
                index_name=self.index_name,
                pipeline_type="TRIGGERED",
                primary_key="id",
                embedding_source_column="text",
                embedding_model_endpoint_name=self.embedding_model,
            )
            logger.info("✓ Vector Search index created: {}", self.index_name)
            return index
        except Exception as e:
            if "RESOURCE_ALREADY_EXISTS" not in str(e):
                raise
            logger.info("✓ Vector Search index exists: {}", self.index_name)
            return self.client.get_index(index_name=self.index_name)

    def sync_index(self) -> None:
        """Trigger a sync of the Vector Search index with the source Delta table."""
        index = self.create_or_get_index()
        logger.info("Syncing Vector Search index: {}", self.index_name)
        index.sync()  # type: ignore[union-attr]
        logger.info("✓ Index sync triggered")

    def search(
        self,
        query: str,
        num_results: int = 3,
        filters: dict | None = None,
    ) -> list[dict]:
        """Run a similarity search against the GitHub index.

        Args:
            query: Natural language search query.
            num_results: Number of results to return. Defaults to 3.
            filters: Optional metadata filters (e.g. {"username": "jsmith"}).

        Returns:
            List of result dicts with keys matching the index columns.
        """
        index = self.client.get_index(index_name=self.index_name)
        results = index.similarity_search(
            query_text=query,
            columns=["text", "username", "repo_name", "repo_description"],
            num_results=num_results,
            filters=filters,
        )
        columns = [col["name"] for col in results.get("manifest", {}).get("columns", [])]
        rows = results.get("result", {}).get("data_array", [])
        return [dict(zip(columns, row, strict=False)) for row in rows]
