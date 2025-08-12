from __future__ import annotations
from pydantic import BaseModel, PrivateAttr, AnyUrl
from typing import Any

# MCP imports
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import (
    CallToolResult,
    ListToolsResult,
    ListResourcesResult,
    ReadResourceResult,
    Tool,
    Resource,
    BlobResourceContents,
    TextResourceContents,
)

# ProxyStore imports
from proxystore.connectors.redis import RedisKey, RedisConnector
from proxystore.store import Store
from proxystore.store.utils import get_key
import cloudpickle

import os
import filetype


class RheaMCPClient:
    """
    A client for interacting with the Rhea Model Context Protocol (MCP) service.

    This class provides a high-level interface for connecting to an MCP server,
    discovering available tools, and executing tool calls. It handles connection
    management and provides async context manager support.

    Attributes:
        url (str): The URL of the MCP server.
        available_tools (list[dict]): List of tools available on the server.
        http_client: The HTTP client used for communication.
        read: The read stream for the HTTP client.
        write: The write stream for the HTTP client.
        session: The active ClientSession for the MCP server.
    """

    def __init__(self, url: str = "http://localhost:3001/mcp"):
        """
        Initialize the RheaMCPClient with a server URL.

        Args:
            url (str): The URL of the MCP server.
                       Defaults to "http://localhost:3001/mcp".
        """
        self.url = url
        self.http_client = None
        self.read = self.write = None
        self.session = None

    async def find_tools(self, query: str) -> list[dict]:
        """
        Find available tools on the MCP server that match the query.

        This method searches for tools matching the provided query string
        and returns their descriptions.

        Args:
            query (str): The search query to find relevant tools.

        Returns:
            list[dict]: A list of tool descriptions matching the query.

        Raises:
            RuntimeError: If the client session fails to initialize.
        """
        if not self.session:
            raise RuntimeError("Client session could not be initialized.")

        res: CallToolResult = await self.session.call_tool(
            "find_tools", arguments={"query": query}
        )

        if res.structuredContent is None:
            print(res.content)
            return []

        return res.structuredContent.get("result", [])

    async def list_tools(self) -> list[Tool]:
        if not self.session:
            raise RuntimeError("Client session could not be initialized.")

        res: ListToolsResult = await self.session.list_tools()

        return res.tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict | None:
        """
        Call a specific tool on the MCP server with the given arguments.

        Args:
            name (str): The name of the tool to call.
            arguments (dict[str, Any]): The arguments to pass to the tool.

        Returns:
            dict | None: The structured content of the tool's response,
                         or None if there is no structured content.

        Raises:
            RuntimeError: If the client session fails to initialize.
        """

        if not self.session:
            raise RuntimeError("Client session could not be initialized.")

        res: CallToolResult = await self.session.call_tool(name, arguments)

        if res.isError:
            print(f"Error occured calling tool: {res.content}")

        return res.structuredContent

    async def list_resources(self) -> list[Resource]:
        """
        List all available resources from the Rhea MCP server.
        This asynchronous method retrieves a list of all resources accessible through
        the initialized Rhea client. The client must have an active session before
        calling this method.
        Returns:
            ListResourcesResult: A result object containing the list of available resources.
        Raises:
            RuntimeError: If the client session has not been initialized.
        """
        if not self.session:
            raise RuntimeError("Client session could not be initialized.")

        res: ListResourcesResult = await self.session.list_resources()

        return res.resources

    async def read_resource(
        self, uri: AnyUrl
    ) -> list[TextResourceContents | BlobResourceContents]:
        if not self.session:
            raise RuntimeError("Client session could not be initialized.")

        res: ReadResourceResult = await self.session.read_resource(uri)

        return res.contents

    async def close(self):
        """
        Close all open connections and resources.

        This method properly cleans up all connections to the MCP server
        and resets the client's state. It should be called when done using
        the client to ensure proper resource cleanup.
        """
        if self.session:
            await self.session.__aexit__(None, None, None)
            self.session = None

        if self.http_client:
            await self.http_client.__aexit__(None, None, None)
            self.http_client = None
            self.read = None
            self.write = None
            self.get_session_id_callback = None

    async def __aenter__(self):
        """
        Async context manager entry point.

        Allows using this client as an async context manager with the 'async with' statement.

        Returns:
            RheaMCPClient: The initialized client instance.
        """
        self.http_client = streamablehttp_client(self.url)
        self.read, self.write, _ = await self.http_client.__aenter__()
        self.session = ClientSession(self.read, self.write)
        await self.session.__aenter__()
        await self.session.initialize()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """
        Async context manager exit point.

        Ensures proper cleanup when exiting the 'async with' context.

        Args:
            exc_type: The exception type if an exception was raised in the context.
            exc_val: The exception value if an exception was raised in the context.
            exc_tb: The traceback if an exception was raised in the context.
        """
        if self.session:
            await self.session.__aexit__(exc_type, exc, tb)
        if self.http_client:
            await self.http_client.__aexit__(exc_type, exc, tb)


def get_file_format(buffer: bytes) -> str:
    """
    Small helper function to get the MIME type of file upload
    """
    try:
        import magic

        m = magic.Magic(mime=True)
        format = m.from_buffer(buffer)
    except Exception:
        kind = filetype.guess(buffer)
        format = kind.mime if kind else "application/octet-stream"
    return format


class RheaFileProxy(BaseModel):
    """
    A Pydantic model to represent a file stored in Redis.

    Attributes:
        name (str): Logical (or user provided) name of file.
        format (str): MIME type of file (magic/filetype).
        filename (str): Original filename.
        filesize (int): Size of the file in bytes.
        contents (bytes): Raw file contents.

    """

    name: str
    format: str
    filename: str
    filesize: int
    contents: bytes
    _key: RedisKey | None = PrivateAttr()

    @classmethod
    def from_proxy(cls, key: RedisKey, store: Store) -> RheaFileProxy:
        data = store.get(key, deserializer=cloudpickle.loads)
        if data is None:
            raise ValueError(f"Key '{key}' not in store")
        return cls.model_validate(data)

    @classmethod
    def from_file(cls, path: str) -> RheaFileProxy:
        """
        Constructs a RheaFileProxy object from local file.
        *Does not put in proxy!* Must add to proxy using .to_proxy()
        """
        with open(path, "rb") as f:
            contents: bytes = f.read()

        return cls(
            name=os.path.basename(path),
            format=get_file_format(contents),
            filename=os.path.basename(path),
            filesize=len(contents),
            contents=contents,
        )

    @classmethod
    def from_buffer(cls, name: str, contents: bytes) -> RheaFileProxy:
        return cls(
            name=name,
            format=get_file_format(contents),
            filename=name,
            filesize=len(contents),
            contents=contents,
        )

    def to_proxy(self, store: Store) -> str:
        proxy = store.proxy(self.model_dump(), serializer=cloudpickle.dumps)
        key = get_key(proxy)
        return key.redis_key  # type: ignore


class RheaFileManager:
    """
    A class to manage file operations with Redis storage using RheaFileProxy.

    This class provides methods to upload and download files to/from Redis
    and keeps track of uploaded files.
    """

    def __init__(self, store: Store[RedisConnector]):
        """
        Initialize the RheaFileManager with a Redis store.

        Args:
            store: ProxyStore instance with Redis connector
        """
        self.store = store
        self.uploaded_files = {}  # Dict to track files: {key: filepath}

    def upload_file(self, filepath: str) -> str:
        """
        Upload a file to Redis storage.

        Args:
            filepath: Path to the file to upload

        Returns:
            str: Redis key for the uploaded file
        """
        proxy: RheaFileProxy = RheaFileProxy.from_file(filepath)
        key = proxy.to_proxy(self.store)
        self.uploaded_files[key] = filepath
        return key

    def download_file(self, key: RedisKey, path: str) -> str:
        """
        Download a file from Redis storage.

        Args:
            key: Redis key for the file
            path: Directory path where the file should be downloaded

        Returns:
            str: Path to the downloaded file
        """
        proxy: RheaFileProxy = RheaFileProxy.from_proxy(key, self.store)
        output_path = os.path.realpath(os.path.join(path, proxy.filename))

        with open(output_path, "wb") as f:
            f.write(proxy.contents)

        return output_path

    def get_uploaded_files(self) -> dict:
        """
        Get a dictionary of all uploaded files.

        Returns:
            dict: Dictionary mapping Redis keys to original filepaths
        """
        return self.uploaded_files.copy()

    def clear_uploaded_files(self):
        """Clear the list of tracked uploaded files."""
        self.uploaded_files.clear()
