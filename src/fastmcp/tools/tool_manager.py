"""Refactored ToolManager using BaseManager."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from mcp.types import ToolAnnotations

from fastmcp import settings
from fastmcp.base_manager import BaseManager
from fastmcp.exceptions import NotFoundError, ToolError
from fastmcp.settings import DuplicateBehavior
from fastmcp.tools.tool import Tool, ToolResult
from fastmcp.tools.tool_transform import (
    ToolTransformConfig,
    apply_transformations_to_tools,
)
from fastmcp.utilities.logging import get_logger

if TYPE_CHECKING:
    from fastmcp.server.server import MountedServer

logger = get_logger(__name__)


class ToolManager(BaseManager[Tool]):
    """Manages FastMCP tools."""

    def __init__(
        self,
        duplicate_behavior: DuplicateBehavior | None = None,
        mask_error_details: bool | None = None,
        transformations: dict[str, ToolTransformConfig] | None = None,
    ):
        super().__init__(duplicate_behavior, mask_error_details)
        self.transformations = transformations or {}

    def get_item_type_name(self) -> str:
        return "tool"

    def get_item_type_plural(self) -> str:
        return "tools"

    async def fetch_items_from_mounted_server(
        self, server: Any, via_server: bool
    ) -> list[Tool]:
        if via_server:
            # Use the server-to-server filtered path
            return await server._list_tools()
        else:
            # Use the manager-to-manager unfiltered path
            return await server._tool_manager.list_tools()

    def get_item_key(self, item: Tool) -> str:
        return item.key

    def apply_prefix_to_items(
        self, items: dict[str, Tool], mounted: MountedServer
    ) -> dict[str, Tool]:
        prefixed_tools = {}
        for tool in items.values():
            prefixed_tool = tool.model_copy(
                update={"key": f"{mounted.prefix}_{tool.key}"}
            )
            prefixed_tools[prefixed_tool.key] = prefixed_tool
        return prefixed_tools

    def post_process_items(self, items: dict[str, Tool]) -> dict[str, Tool]:
        if not self.transformations:
            return items

        return apply_transformations_to_tools(
            tools=items,
            transformations=self.transformations,
        )

    @property
    def _tools_transformed(self) -> list[str]:
        """Get the list of transformed tool names."""
        return [
            transformation.name or tool_name
            for tool_name, transformation in self.transformations.items()
        ]

    def add_tool_from_fn(
        self,
        fn: Callable[..., Any],
        name: str | None = None,
        description: str | None = None,
        tags: set[str] | None = None,
        annotations: ToolAnnotations | None = None,
        serializer: Callable[[Any], str] | None = None,
        exclude_args: list[str] | None = None,
    ) -> Tool:
        """Add a tool from a function."""
        if settings.deprecation_warnings:
            warnings.warn(
                "ToolManager.add_tool_from_fn() is deprecated. "
                "Use Tool.from_function() and call add_tool() instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        tool = Tool.from_function(
            fn=fn,
            name=name,
            description=description,
            tags=tags,
            annotations=annotations,
            serializer=serializer,
            exclude_args=exclude_args,
        )
        return self.add_tool(tool)

    def add_tool(self, tool: Tool) -> Tool:
        """Register a tool with the server."""
        return self.add_item(tool)

    def remove_tool(self, key: str) -> None:
        """Remove a tool from the manager."""
        if key not in self._items:
            raise NotFoundError(f"Tool '{key}' not found")
        del self._items[key]

    async def has_tool(self, key: str) -> bool:
        """Check if a tool exists."""
        return await self.has_item(key)

    async def get_tool(self, key: str) -> Tool:
        """Get a tool by key."""
        return await self.get_item(key)

    async def get_tools(self) -> dict[str, Tool]:
        """Get all tools as a dictionary."""
        return await self.get_items()

    async def list_tools(self) -> list[Tool]:
        """List all tools."""
        return await self.list_items()

    async def call_tool(
        self,
        key: str,
        arguments: dict[str, Any] | None = None,
    ) -> ToolResult:
        """
        Call a tool by key with the given arguments.

        This is tool-specific functionality for executing tools.

        Args:
            key: The tool key
            arguments: The arguments to pass to the tool

        Returns:
            The tool execution result

        Raises:
            ToolError: If there's an error calling the tool
        """
        tool = await self.get_tool(key)

        try:
            # Delegate to the tool's run method
            return await tool.run(arguments or {})
        except ToolError:
            # Re-raise ToolError as-is
            raise
        except Exception as e:
            # Wrap other exceptions based on mask_error_details setting
            if self.mask_error_details:
                raise ToolError(f"Error calling tool {key!r}") from e
            else:
                raise ToolError(f"Error calling tool {key!r}: {e}") from e

    async def handle_call_tool_request(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> list[Any]:
        """
        Handle an MCP tools/call request.

        This method is called by the server to process tool calls
        from the MCP protocol.

        Args:
            name: The tool name
            arguments: The tool arguments

        Returns:
            The tool result formatted for MCP response
        """
        # Get the tool first to fail fast if it doesn't exist
        tools = await self.get_tools()

        if name not in tools:
            # Try looking through mounted servers
            for mounted in reversed(self._mounted_servers):
                try:
                    # Check if this server handles the tool
                    result = await mounted.server._mcp_call_tool(
                        key=name,
                        arguments=arguments or {},
                    )
                    return result
                except Exception:
                    continue

            # Tool not found anywhere
            raise ToolError(f"Unknown tool: {name}")

        # Call the local tool
        result = await self.call_tool(key=name, arguments=arguments)

        # Format the result for MCP
        return result.to_mcp_result()
