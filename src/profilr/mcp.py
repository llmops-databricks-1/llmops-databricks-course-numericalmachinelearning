from collections.abc import Callable

from databricks.sdk import WorkspaceClient
from databricks_mcp import DatabricksMCPClient
from pydantic import BaseModel


class ToolInfo(BaseModel):
    """Represents an agent tool with its specification and execution function."""

    model_config = {"arbitrary_types_allowed": True}

    name: str
    spec: dict
    exec_fn: Callable


def create_managed_exec_fn(
    server_url: str, tool_name: str, w: WorkspaceClient
) -> Callable:
    """Create a wrapper function that calls an MCP tool.

    Args:
        server_url: MCP server URL.
        tool_name: Name of the tool to call.
        w: Databricks WorkspaceClient.

    Returns:
        Callable that executes the MCP tool with provided kwargs.
    """

    def exec_fn(**kwargs: object) -> str:
        client = DatabricksMCPClient(server_url=server_url, workspace_client=w)
        response = client.call_tool(tool_name, kwargs)
        return "".join([c.text for c in response.content])

    return exec_fn


async def create_mcp_tools(w: WorkspaceClient, url_list: list[str]) -> list[ToolInfo]:
    """Convert MCP server tools to agent-compatible ToolInfo objects.

    Args:
        w: Databricks WorkspaceClient.
        url_list: List of MCP server URLs.

    Returns:
        List of ToolInfo objects with OpenAI-compatible specs.
    """
    tools = []
    for server_url in url_list:
        mcp_client = DatabricksMCPClient(server_url=server_url, workspace_client=w)
        mcp_tools = mcp_client.list_tools()
        for tool in mcp_tools:
            input_schema = tool.inputSchema.copy() if tool.inputSchema else {}
            tool_spec = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "parameters": input_schema,
                    "description": tool.description or f"Tool: {tool.name}",
                },
            }
            exec_fn = create_managed_exec_fn(server_url, tool.name, w)
            tools.append(ToolInfo(name=tool.name, spec=tool_spec, exec_fn=exec_fn))
    return tools
