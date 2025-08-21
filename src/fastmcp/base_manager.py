"""Base manager class for FastMCP component managers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from fastmcp import settings
from fastmcp.exceptions import NotFoundError
from fastmcp.settings import DuplicateBehavior
from fastmcp.utilities.logging import get_logger

if TYPE_CHECKING:
    from fastmcp.server.server import MountedServer

logger = get_logger(__name__)

T = TypeVar("T")


class BaseManager(ABC, Generic[T]):
    """Abstract base class for FastMCP component managers."""

    def __init__(
        self,
        duplicate_behavior: DuplicateBehavior | None = None,
        mask_error_details: bool | None = None,
    ):
        self._items: dict[str, T] = {}
        self._mounted_servers: list[MountedServer] = []
        self.mask_error_details = mask_error_details or settings.mask_error_details

        # Default to "warn" if None is provided
        if duplicate_behavior is None:
            duplicate_behavior = "warn"

        if duplicate_behavior not in DuplicateBehavior.__args__:
            raise ValueError(
                f"Invalid duplicate_behavior: {duplicate_behavior}. "
                f"Must be one of: {', '.join(DuplicateBehavior.__args__)}"
            )

        self.duplicate_behavior = duplicate_behavior

    def mount(self, server: MountedServer) -> None:
        """Add a mounted server as a source for items."""
        self._mounted_servers.append(server)

    @abstractmethod
    def get_item_type_name(self) -> str:
        """Get the singular name of the item type (e.g., 'tool', 'resource', 'prompt')."""
        pass

    @abstractmethod
    def get_item_type_plural(self) -> str:
        """Get the plural name of the item type (e.g., 'tools', 'resources', 'prompts')."""
        pass

    @abstractmethod
    async def fetch_items_from_mounted_server(
        self, server: Any, via_server: bool
    ) -> list[T] | dict[str, T]:
        """
        Fetch items from a mounted server.

        Args:
            server: The mounted server to fetch from
            via_server: Use server-to-server path if True, manager-to-manager if False
        """
        pass

    @abstractmethod
    def get_item_key(self, item: T) -> str:
        """Extract the unique key from an item."""
        pass

    @abstractmethod
    def apply_prefix_to_items(
        self, items: dict[str, T], mounted: MountedServer
    ) -> dict[str, T]:
        """Apply prefix transformation to items from a mounted server."""
        pass

    async def _load_items(self, *, via_server: bool = False) -> dict[str, T]:
        """
        The single, consolidated recursive method for fetching items. The 'via_server'
        parameter determines the communication path.

        - via_server=False: Manager-to-manager path for complete, unfiltered inventory
        - via_server=True: Server-to-server path for filtered MCP requests
        """
        all_items: dict[str, T] = {}

        for mounted in self._mounted_servers:
            try:
                # Get items from the mounted server
                child_results = await self.fetch_items_from_mounted_server(
                    mounted.server, via_server
                )

                # Convert to dict if it's a list
                if isinstance(child_results, list):
                    child_dict = {
                        self.get_item_key(item): item for item in child_results
                    }
                else:
                    child_dict = child_results

                # Apply prefix if needed
                if mounted.prefix:
                    prefixed_items = self.apply_prefix_to_items(child_dict, mounted)
                    all_items.update(prefixed_items)
                else:
                    all_items.update(child_dict)

            except Exception as e:
                # Skip failed mounts silently, matches existing behavior
                logger.warning(
                    f"Failed to get {self.get_item_type_plural()} from server: "
                    f"{mounted.server.name!r}, mounted at: {mounted.prefix!r}: {e}"
                )
                continue

        # Finally, add local items, which always take precedence
        all_items.update(self._items)

        return self.post_process_items(all_items)

    def post_process_items(self, items: dict[str, T]) -> dict[str, T]:
        """Hook for manager-specific post-processing."""
        return items

    async def has_item(self, key: str) -> bool:
        """Check if an item exists."""
        items = await self.get_items()
        return key in items

    async def get_item(self, key: str) -> T:
        """Get a specific item by key."""
        items = await self.get_items()
        if key in items:
            return items[key]
        raise NotFoundError(f"Unknown {self.get_item_type_name()}: {key}")

    async def get_items(self) -> dict[str, T]:
        """Get all registered items, keyed by identifier."""
        return await self._load_items(via_server=False)

    async def list_items(self) -> list[T]:
        """List all items (used for MCP list requests)."""
        items_dict = await self._load_items(via_server=True)
        return list(items_dict.values())

    def add_item(self, item: T) -> T:
        """Add an item to this manager."""
        key = self.get_item_key(item)

        existing = self._items.get(key)
        if existing:
            if self.duplicate_behavior == "warn":
                logger.warning(
                    f"{self.get_item_type_name().capitalize()} already exists: {key}"
                )
                self._items[key] = item
            elif self.duplicate_behavior == "replace":
                self._items[key] = item
            elif self.duplicate_behavior == "error":
                raise ValueError(
                    f"{self.get_item_type_name().capitalize()} already exists: {key}"
                )
            elif self.duplicate_behavior == "ignore":
                return existing
        else:
            self._items[key] = item
        return item
