# Databricks notebook source
"""
Week 5 — Grant Service Principal permissions on workspace resources.

Run this once after creating the SPN to give it the required access
to Vector Search, SQL Warehouse, and Genie Space.
Requires the SPN client_id stored in the 'dev_SPN' secret scope.
"""

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.iam import AccessControlRequest, PermissionLevel
from loguru import logger

from profilr.config import get_env, get_secret, load_config

# COMMAND ----------
# Setup
cfg = load_config("project_config.yml", env=get_env())
w = WorkspaceClient()

spn_client_id = get_secret("client_id", scope="dev_SPN")
if not spn_client_id:
    raise ValueError("SPN client_id not found in secret scope 'dev_SPN'")

logger.info("Granting permissions for SPN: {}", spn_client_id)

# COMMAND ----------
# Vector Search endpoint — CAN_USE
if cfg.vs_endpoint:
    vs_endpoint = w.vector_search_endpoints.get_endpoint(cfg.vs_endpoint)
    w.permissions.update(
        request_object_type="vector-search-endpoints",
        request_object_id=vs_endpoint.id,
        access_control_list=[
            AccessControlRequest(
                service_principal_name=spn_client_id,
                permission_level=PermissionLevel.CAN_USE,
            )
        ],
    )
    logger.info("✓ CAN_USE on Vector Search endpoint: {}", cfg.vs_endpoint)
else:
    logger.warning("vs_endpoint not set in config — skipping")

# COMMAND ----------
# SQL Warehouse — CAN_USE
if cfg.warehouse_id:
    w.permissions.update(
        request_object_type="warehouses",
        request_object_id=cfg.warehouse_id,
        access_control_list=[
            AccessControlRequest(
                service_principal_name=spn_client_id,
                permission_level=PermissionLevel.CAN_USE,
            )
        ],
    )
    logger.info("✓ CAN_USE on SQL Warehouse: {}", cfg.warehouse_id)
else:
    logger.warning("warehouse_id not set in config — skipping")

# COMMAND ----------
# Genie Space — CAN_RUN
if cfg.genie_space_id:
    w.permissions.update(
        request_object_type="genie",
        request_object_id=cfg.genie_space_id,
        access_control_list=[
            AccessControlRequest(
                service_principal_name=spn_client_id,
                permission_level=PermissionLevel.CAN_RUN,
            )
        ],
    )
    logger.info("✓ CAN_RUN on Genie Space: {}", cfg.genie_space_id)
else:
    logger.warning("genie_space_id not set in config — skipping")
