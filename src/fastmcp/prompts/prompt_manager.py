"""Refactored PromptManager using BaseManager."""

from __future__ import annotations

import warnings
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from mcp import GetPromptResult

from fastmcp import settings
from fastmcp.base_manager import BaseManager
from fastmcp.exceptions import PromptError
from fastmcp.prompts.prompt import FunctionPrompt, Prompt, PromptResult
from fastmcp.utilities.logging import get_logger

if TYPE_CHECKING:
    from fastmcp.server.server import MountedServer

logger = get_logger(__name__)


class PromptManager(BaseManager[Prompt]):
    """Manages FastMCP prompts."""

    def get_item_type_name(self) -> str:
        return "prompt"

    def get_item_type_plural(self) -> str:
        return "prompts"

    async def fetch_items_from_mounted_server(
        self, server: Any, via_server: bool
    ) -> list[Prompt]:
        if via_server:
            # Use the server-to-server filtered path
            return await server._list_prompts()
        else:
            # Use the manager-to-manager unfiltered path
            return await server._prompt_manager.list_prompts()

    def get_item_key(self, item: Prompt) -> str:
        return item.key

    def apply_prefix_to_items(
        self, items: dict[str, Prompt], mounted: MountedServer
    ) -> dict[str, Prompt]:
        prefixed_prompts = {}
        for prompt in items.values():
            prefixed_prompt = prompt.model_copy(
                update={"key": f"{mounted.prefix}_{prompt.key}"}
            )
            prefixed_prompts[prefixed_prompt.key] = prefixed_prompt
        return prefixed_prompts

    async def has_prompt(self, key: str) -> bool:
        """Check if a prompt exists."""
        return await self.has_item(key)

    async def get_prompt(self, key: str) -> Prompt:
        """Get prompt by key."""
        return await self.get_item(key)

    async def get_prompts(self) -> dict[str, Prompt]:
        """Get all registered prompts, keyed by name."""
        return await self.get_items()

    async def list_prompts(self) -> list[Prompt]:
        """List all prompts (used for MCP list requests)."""
        return await self.list_items()

    def add_prompt_from_fn(
        self,
        fn: Callable[..., Awaitable[PromptResult] | PromptResult | str],
        name: str | None = None,
        description: str | None = None,
        tags: set[str] | None = None,
    ) -> Prompt:
        """Create a prompt from a function and add it to the manager."""
        if settings.deprecation_warnings:
            warnings.warn(
                "PromptManager.add_prompt_from_fn() is deprecated. "
                "Use FunctionPrompt.from_function() and call add_prompt() instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        prompt = FunctionPrompt.from_function(
            fn=fn,
            name=name,
            description=description,
            tags=tags,
        )
        return self.add_prompt(prompt)

    def add_prompt(self, prompt: Prompt) -> Prompt:
        """Register a prompt with the server."""
        return self.add_item(prompt)

    async def render_prompt(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> GetPromptResult:
        """Render a prompt (alias for get_prompt_result for compatibility)."""
        return await self.get_prompt_result(name, arguments)

    async def get_prompt_result(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> GetPromptResult:
        """Get a prompt result by name and arguments."""
        prompt = await self.get_prompt(name)

        try:
            messages = await prompt.render(arguments or {})
            return GetPromptResult(
                description=prompt.description,
                messages=messages,
            )
        except PromptError:
            raise
        except Exception as e:
            if self.mask_error_details:
                raise PromptError(f"Error getting prompt {name!r}")
            else:
                raise PromptError(f"Error getting prompt {name!r}: {e}") from e

    async def handle_get_prompt_request(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> GetPromptResult:
        """Handle an MCP prompts/get request."""
        # First try local processing
        prompts = await self.get_prompts()

        if name in prompts:
            return await self.get_prompt_result(name, arguments)

        # If not found locally, try mounted servers
        for mounted in reversed(self._mounted_servers):
            try:
                # Check if this server handles the prompt
                result = await mounted.server._mcp_get_prompt(
                    name=name,
                    arguments=arguments or {},
                )
                return result
            except Exception:
                continue

        # Prompt not found anywhere
        raise PromptError(f"Unknown prompt: {name}")
