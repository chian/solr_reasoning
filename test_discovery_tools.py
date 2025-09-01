#!/usr/bin/env python3
"""
Test script for the new discovery tools in P3_TOOLS_DATA_RETRIEVAL.py
"""

import subprocess
import json
import time

def test_discovery_tools():
    """Test the list_available_tools and get_tool_help functions"""
    
    print("Testing discovery tools...")
    
    # Start the MCP server as a subprocess
    server_process = subprocess.Popen(
        ["python", "P3_TOOLS_DATA_RETRIEVAL.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    try:
        # Wait a moment for server to start
        time.sleep(2)
        
        # Step 1: Send initialize message
        print("\n1. Initializing MCP connection...")
        init_msg = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "clientInfo": {
                    "name": "Test-Client",
                    "version": "1.0.0"
                }
            }
        }
        
        server_process.stdin.write(json.dumps(init_msg) + "\n")
        server_process.stdin.flush()
        
        response = server_process.stdout.readline()
        if response:
            result = json.loads(response)
            print("Initialize response:", json.dumps(result, indent=2))
        else:
            print("No response from initialize")
            return
        
        # Step 2: Send initialized notification
        print("\n2. Sending initialized notification...")
        initialized_msg = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }
        
        server_process.stdin.write(json.dumps(initialized_msg) + "\n")
        server_process.stdin.flush()
        
        # Step 3: Test list_available_tools
        print("\n3. Testing list_available_tools...")
        list_tools_msg = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "list_available_tools",
                "arguments": {}
            }
        }
        
        server_process.stdin.write(json.dumps(list_tools_msg) + "\n")
        server_process.stdin.flush()
        
        response = server_process.stdout.readline()
        if response:
            result = json.loads(response)
            print("Response:", json.dumps(result, indent=2))
        else:
            print("No response from list_available_tools")
        
        # Step 4: Test get_tool_help
        print("\n4. Testing get_tool_help for p3_all_genomes...")
        help_msg = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "get_tool_help",
                "arguments": {"tool_name": "p3_all_genomes"}
            }
        }
        
        server_process.stdin.write(json.dumps(help_msg) + "\n")
        server_process.stdin.flush()
        
        response = server_process.stdout.readline()
        if response:
            result = json.loads(response)
            print("Response:", json.dumps(result, indent=2))
        else:
            print("No response from get_tool_help")
            
    finally:
        # Clean up
        server_process.terminate()
        server_process.wait()

if __name__ == "__main__":
    test_discovery_tools()
