import asyncio
import argparse
import os
from pprint import pprint

# Import the client classes
from rhea_client import RheaMCPClient, RheaFileManager

# ProxyStore imports needed for file management
from proxystore.connectors.redis import RedisConnector, RedisKey
from proxystore.store import Store
import cloudpickle

from mcp.types import Tool


async def test_rhea_client(
    fasta_path: str, redis_host: str = "localhost", redis_port: int = 6379
):
    """
    Test the RheaMCPClient functionality by:
    1. Finding a tool to convert FASTA to FASTQ
    2. Uploading a FASTA file
    3. Calling the tool and showing the result

    Args:
        fasta_path: Path to the FASTA file to convert
        redis_host: Redis server hostname
        redis_port: Redis server port
    """
    print(f"Testing RheaClient with FASTA file: {fasta_path}")

    # Create Redis connector and store for file handling
    redis_connector = RedisConnector(hostname=redis_host, port=redis_port)
    store = Store(
        name="rhea-input",
        connector=redis_connector,
        register=True,
        serializer=cloudpickle.dumps,
        deserializer=cloudpickle.loads,
    )

    # Initialize the file manager
    file_manager = RheaFileManager(store)

    # Initialize the MCP client
    async with RheaMCPClient() as client:
        print("\n--- 1. Finding tools for FASTA to FASTQ conversion ---")
        tool_list = await client.find_tools("I need a tool to convert FASTA to FASTQ")

        if not tool_list:
            print("No tools found for FASTA to FASTQ conversion")
            return

        # Print found tools
        print(f"Found {len(tool_list)} tools:")
        for i, tool in enumerate(tool_list):
            print(f"\n--- Tool {i+1}: {tool.get('name')} ---")
            print(f"Description: {tool.get('description')}")

        # Get new tools
        tools: list[Tool] = await client.list_tools()

        # Select the first tool (you could implement logic to choose the most appropriate)
        selected_tool = tools[
            1
        ]  # tools[0] is always `find_tools`, first result is index 1

        print(f"Selected tool: {selected_tool.name}")
        print(f"Tool schema: {selected_tool.inputSchema}")

        print("\n--- 2. Uploading FASTA file ---")
        # Upload the FASTA file and get the Redis key
        file_key = file_manager.upload_file(fasta_path)

        print(f"File uploaded with key: {file_key}")

        print("\n--- 3. Calling conversion tool ---")
        # Prepare arguments for the tool call
        tool_args = {"inputfile": file_key, "score": 40}

        try:
            # Call the tool with the prepared arguments
            result = await client.call_tool(selected_tool.name, arguments=tool_args)

            print("\n--- 4. Tool Results ---")
            if result:
                print("Conversion successful:")
                pprint(result)

                # If the result contains a file key, download it
                output_files = result.get("files")
                if output_files:
                    for file in output_files:
                        output_file_key = file.get("key")
                        output_path = file_manager.download_file(
                            RedisKey(redis_key=output_file_key),
                            os.path.dirname(fasta_path),
                        )
                        print(f"\nDownloaded result to: {output_path}")
            else:
                print("Tool returned no result")

        except Exception as e:
            print(f"Error calling tool: {str(e)}")

    # Close the store
    store.close()
    print("\nTest completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test RheaClient with FASTA to FASTQ conversion"
    )
    parser.add_argument("fasta_path", help="Path to the FASTA file to convert")
    parser.add_argument(
        "--redis-host", default="localhost", help="Redis server hostname"
    )
    parser.add_argument(
        "--redis-port", type=int, default=6379, help="Redis server port"
    )

    args = parser.parse_args()

    asyncio.run(test_rhea_client(args.fasta_path, args.redis_host, args.redis_port))
