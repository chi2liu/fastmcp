"""Refactored ResourceManager using BaseManager."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pydantic import AnyUrl

from fastmcp import settings
from fastmcp.base_manager import BaseManager
from fastmcp.exceptions import NotFoundError, ResourceError
from fastmcp.resources.resource import Resource
from fastmcp.resources.template import (
    ResourceTemplate,
    match_uri_template,
)
from fastmcp.settings import DuplicateBehavior
from fastmcp.utilities.logging import get_logger

if TYPE_CHECKING:
    from fastmcp.server.server import MountedServer

logger = get_logger(__name__)


class ResourceManagerBase(BaseManager[Resource]):
    """Base manager for resources using BaseManager."""

    def get_item_type_name(self) -> str:
        return "resource"

    def get_item_type_plural(self) -> str:
        return "resources"

    async def fetch_items_from_mounted_server(
        self, server: Any, via_server: bool
    ) -> list[Resource] | dict[str, Resource]:
        if via_server:
            # Use the server-to-server filtered path
            child_resources_list = await server._list_resources()
            return {resource.key: resource for resource in child_resources_list}
        else:
            # Use the manager-to-manager unfiltered path
            return await server._resource_manager.get_resources()

    def get_item_key(self, item: Resource) -> str:
        return item.key

    def apply_prefix_to_items(
        self, items: dict[str, Resource], mounted: MountedServer
    ) -> dict[str, Resource]:
        from fastmcp.server.server import add_resource_prefix

        prefixed_resources = {}
        for uri, resource in items.items():
            if mounted.prefix:
                prefixed_uri = add_resource_prefix(
                    uri, mounted.prefix, mounted.resource_prefix_format
                )
                # Create a copy of the resource with the prefixed key and name
                prefixed_resource = resource.model_copy(
                    update={
                        "name": f"{mounted.prefix}_{resource.name}",
                        "key": prefixed_uri,
                    }
                )
                prefixed_resources[prefixed_uri] = prefixed_resource
            else:
                prefixed_resources[uri] = resource
        return prefixed_resources


class ResourceTemplateManager(BaseManager[ResourceTemplate]):
    """Base manager for resource templates using BaseManager."""

    def get_item_type_name(self) -> str:
        return "resource template"

    def get_item_type_plural(self) -> str:
        return "resource templates"

    async def fetch_items_from_mounted_server(
        self, server: Any, via_server: bool
    ) -> list[ResourceTemplate]:
        if via_server:
            # Use the server-to-server filtered path
            return await server._list_resource_templates()
        else:
            # Use the manager-to-manager unfiltered path
            return await server._resource_manager.list_resource_templates()

    def get_item_key(self, item: ResourceTemplate) -> str:
        return item.key

    def apply_prefix_to_items(
        self, items: dict[str, ResourceTemplate], mounted: MountedServer
    ) -> dict[str, ResourceTemplate]:
        from fastmcp.server.server import add_resource_prefix

        prefixed_templates = {}
        for uri_template, template in items.items():
            if mounted.prefix:
                prefixed_uri_template = add_resource_prefix(
                    uri_template, mounted.prefix, mounted.resource_prefix_format
                )
                # Create a copy of the template with the prefixed key and name
                prefixed_template = template.model_copy(
                    update={
                        "name": f"{mounted.prefix}_{template.name}",
                        "key": prefixed_uri_template,
                    }
                )
                prefixed_templates[prefixed_uri_template] = prefixed_template
            else:
                prefixed_templates[uri_template] = template
        return prefixed_templates


class ResourceManager:
    """Manages FastMCP resources and resource templates."""

    def __init__(
        self,
        duplicate_behavior: DuplicateBehavior | None = None,
        mask_error_details: bool | None = None,
    ):
        self.mask_error_details = mask_error_details or settings.mask_error_details

        # Use two BaseManager instances internally
        self._resource_manager = ResourceManagerBase(
            duplicate_behavior=duplicate_behavior, mask_error_details=mask_error_details
        )
        self._template_manager = ResourceTemplateManager(
            duplicate_behavior=duplicate_behavior, mask_error_details=mask_error_details
        )

        # For compatibility, expose the internal storage
        self._resources = self._resource_manager._items
        self._templates = self._template_manager._items
        self._mounted_servers: list[MountedServer] = []

        # Store the duplicate behavior for compatibility
        self.duplicate_behavior = self._resource_manager.duplicate_behavior

    def mount(self, server: MountedServer) -> None:
        """Adds a mounted server as a source for resources and templates."""
        self._mounted_servers.append(server)
        self._resource_manager.mount(server)
        self._template_manager.mount(server)

    async def get_resources(self) -> dict[str, Resource]:
        """Get all registered resources, keyed by URI."""
        return await self._resource_manager.get_items()

    async def get_resource_templates(self) -> dict[str, ResourceTemplate]:
        """Get all registered templates, keyed by URI template."""
        return await self._template_manager.get_items()

    async def _load_resources(self, *, via_server: bool = False) -> dict[str, Resource]:
        """Load resources using the appropriate path."""
        return await self._resource_manager._load_items(via_server=via_server)

    async def _load_resource_templates(
        self, *, via_server: bool = False
    ) -> dict[str, ResourceTemplate]:
        """Load resource templates using the appropriate path."""
        return await self._template_manager._load_items(via_server=via_server)

    async def list_resources(self) -> list[Resource]:
        """Lists all resources, applying protocol filtering."""
        return await self._resource_manager.list_items()

    async def list_resource_templates(self) -> list[ResourceTemplate]:
        """Lists all templates, applying protocol filtering."""
        return await self._template_manager.list_items()

    def add_resource_or_template_from_fn(
        self,
        fn: Callable[..., Any],
        uri: str,
        name: str | None = None,
        description: str | None = None,
        mime_type: str | None = None,
        tags: set[str] | None = None,
    ) -> Resource | ResourceTemplate:
        """Add a resource or template to the manager from a function."""
        from fastmcp.server.context import Context

        # Check if this should be a template
        has_uri_params = "{" in uri and "}" in uri
        # check if the function has any parameters (other than injected context)
        sig = inspect.signature(fn)
        has_fn_params = any(
            param.annotation != Context
            for param in sig.parameters.values()
            if param.name != "self"
        )

        if has_uri_params and not has_fn_params:
            raise ValueError(
                f"Function {fn.__name__} for template {uri!r} must have parameters"
            )

        if has_uri_params or has_fn_params:
            # Create resource template
            template = ResourceTemplate.from_function(
                fn=fn,
                uri_template=uri,
                name=name,
                description=description,
                mime_type=mime_type,
                tags=tags,
            )
            return self.add_resource_template(template)
        else:
            # Create regular resource
            resource = Resource.from_function(
                fn=fn,
                uri=uri,
                name=name,
                description=description,
                mime_type=mime_type,
                tags=tags,
            )
            return self.add_resource(resource)

    def add_resource(self, resource: Resource) -> Resource:
        """Add a resource to the manager."""
        return self._resource_manager.add_item(resource)

    def add_resource_template(self, template: ResourceTemplate) -> ResourceTemplate:
        """Add a resource template to the manager."""
        return self._template_manager.add_item(template)

    def add_template(self, template: ResourceTemplate) -> ResourceTemplate:
        """Add a resource template (alias for backward compatibility)."""
        return self.add_resource_template(template)

    async def has_resource(self, uri: str) -> bool:
        """Check if a resource exists."""
        return await self._resource_manager.has_item(uri)

    async def has_resource_template(self, uri_template: str) -> bool:
        """Check if a resource template exists."""
        return await self._template_manager.has_item(uri_template)

    async def get_resource(self, uri: AnyUrl | str) -> Resource:
        """Get resource by URI, checking concrete resources first, then templates."""
        uri_str = str(uri)

        # First check concrete resources
        try:
            return await self._resource_manager.get_item(uri_str)
        except NotFoundError:
            pass

        # Then check templates
        templates = await self.get_resource_templates()
        for template in templates.values():
            if params := match_uri_template(uri_str, template.uri_template):
                try:
                    return await template.create_resource(uri_str, params=params)
                except ResourceError:
                    raise
                except Exception as e:
                    if self.mask_error_details:
                        raise ValueError("Error creating resource from template") from e
                    else:
                        raise ValueError(
                            f"Error creating resource from template: {e}"
                        ) from e

        raise NotFoundError(f"Unknown resource: {uri_str}")

    async def get_resource_template(self, uri_template: str) -> ResourceTemplate:
        """Get a resource template by URI template."""
        return await self._template_manager.get_item(uri_template)

    async def match_resource_template(self, uri: str) -> ResourceTemplate | None:
        """Match a URI against registered templates."""
        templates = await self.get_resource_templates()
        for template in templates.values():
            if match_uri_template(uri, template.uri_template):
                return template
        return None

    async def read_resource(
        self,
        uri: AnyUrl | str,
        arguments: dict[str, Any] | None = None,
    ) -> Any:
        """Read a resource by URI."""
        uri = str(uri)

        # First, check if there's a direct resource match
        resources = await self.get_resources()
        if uri in resources:
            resource = resources[uri]
            try:
                return await resource.read()
            except ResourceError:
                raise
            except Exception as e:
                if self.mask_error_details:
                    raise ResourceError(f"Error reading resource {uri!r}")
                else:
                    raise ResourceError(f"Error reading resource {uri!r}: {e}") from e

        # If no direct match, check templates
        templates = await self.get_resource_templates()
        for template in templates.values():
            if match_result := match_uri_template(uri, template.uri_template):
                # Merge matched params with provided arguments
                merged_args = {**match_result, **(arguments or {})}
                try:
                    return await template.read(merged_args)
                except ResourceError:
                    raise
                except Exception as e:
                    if self.mask_error_details:
                        raise ResourceError(f"Error reading resource {uri!r}")
                    else:
                        raise ResourceError(
                            f"Error reading resource {uri!r}: {e}"
                        ) from e

        raise NotFoundError(f"Unknown resource: {uri}")

    async def handle_read_resource_request(
        self,
        uri: str | AnyUrl,
        arguments: dict[str, Any] | None = None,
    ) -> Any:
        """Handle an MCP resources/read request."""
        # First try local processing
        try:
            return await self.read_resource(uri, arguments)
        except NotFoundError:
            pass

        # If not found locally, try mounted servers
        if isinstance(uri, AnyUrl):
            uri = str(uri)

        for mounted in reversed(self._mounted_servers):
            try:
                # Try to read from the mounted server
                result = await mounted.server._mcp_read_resource(uri=uri)
                return result
            except Exception:
                continue

        # Resource not found anywhere
        raise NotFoundError(f"Unknown resource: {uri}")
