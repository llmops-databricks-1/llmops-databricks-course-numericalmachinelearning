"""Basic tests to ensure the package is properly installed."""

import importlib

import pytest


def test_package_import() -> None:
    """Test that the package can be imported."""
    module = importlib.import_module("profilr")
    assert module is not None


def test_version_exists() -> None:
    """Test that the package has a version attribute."""
    module = importlib.import_module("profilr")
    assert hasattr(module, "__version__")
    assert isinstance(module.__version__, str)


@pytest.mark.parametrize(
    "module_path",
    [
        "profilr.agents.github_lookup",
        "profilr.third_party.github",
        "profilr.third_party.arxiv_search",
        "profilr.vector_search",
        "profilr.tools.github_search",
    ],
)
def test_week2_modules_import(module_path: str) -> None:
    """Test that all Week 2 modules can be imported without errors."""
    module = importlib.import_module(module_path)
    assert module is not None


def test_week3_mcp_import() -> None:
    """Test that the Week 3 MCP module can be imported."""
    module = importlib.import_module("profilr.mcp")
    assert hasattr(module, "ToolInfo")
    assert hasattr(module, "create_managed_exec_fn")
    assert hasattr(module, "create_mcp_tools")


def test_week4_agent_import() -> None:
    """Test that the Week 4 agent module can be imported."""
    module = importlib.import_module("profilr.agent")
    assert hasattr(module, "ProfilrAgent")
    assert hasattr(module, "log_register_agent")


def test_week4_evaluation_import() -> None:
    """Test that the Week 4 evaluation module can be imported."""
    module = importlib.import_module("profilr.evaluation")
    assert hasattr(module, "professional_tone_guideline")
    assert hasattr(module, "has_facts_section")
    assert hasattr(module, "summary_not_empty")
