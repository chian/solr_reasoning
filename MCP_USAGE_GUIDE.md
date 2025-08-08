# BV-BRC MCP Servers Usage Guide

This guide explains how to properly use the individual MCP servers that were downloaded from the latest repository.

## 🏗️ MCP Server Architecture

Each MCP server is designed to run as a **separate process** and communicate via the MCP protocol. You cannot import functions directly from one MCP server into another.

### Available MCP Servers

1. **`BVBRC_API.py`** - REST API server (~26 tools)
2. **`P3_TOOLS_DATA_RETRIEVAL.py`** - Data retrieval tools (32 tools)
3. **`P3_TOOLS_COMPUTATIONAL.py`** - Computational services (16 tools)
4. **`P3_TOOLS_SPECIALIZED.py`** - Specialized analysis tools
5. **`P3_TOOLS_UTILITIES.py`** - Utility functions

## 🚀 Running Individual MCP Servers

### Method 1: Direct Execution
```bash
# Run REST API server
python BVBRC_API.py

# Run data retrieval server
python P3_TOOLS_DATA_RETRIEVAL.py

# Run computational services server
python P3_TOOLS_COMPUTATIONAL.py

# Run specialized analysis server
python P3_TOOLS_SPECIALIZED.py

# Run utilities server
python P3_TOOLS_UTILITIES.py
```

### Method 2: Using MCP Client
```bash
# Install MCP client if needed
pip install mcp

# Run server with MCP client
mcp run BVBRC_API.py
```

## 🔧 Integration with Your Existing Code

Since your existing `P3_together.py` expects to import functions directly, you have several options:

### Option 1: Use the New MCP Servers (Recommended)
The new MCP servers provide much more comprehensive functionality and better error handling. Your code has been updated to use these servers directly.

### Option 2: Create a Client-Server Architecture
Instead of importing functions, create a client that communicates with the MCP servers:

```python
import subprocess
import json

class BVBRCMCPClient:
    def __init__(self, server_script):
        self.server_script = server_script
    
    def call_tool(self, tool_name, **kwargs):
        """Call a tool on the MCP server"""
        # This would require implementing MCP client protocol
        # For now, you can run the server and communicate via stdin/stdout
        pass
```

### Option 3: Extract Functions (Not Recommended)
You could extract the tool functions from the MCP servers, but this breaks the MCP architecture and loses the benefits of the protocol.

## 📋 What Each Server Provides

### BVBRC_API.py (REST API Server)
- **Async REST API calls** to BV-BRC
- **Fast data retrieval** without P3-tools
- **Direct database access**
- Tools: `get_genomes_by_species`, `get_complete_genomes`, `get_genome_features`, etc.

### P3_TOOLS_DATA_RETRIEVAL.py
- **Data retrieval tools** (p3-all-*, p3-get-*)
- **Batch processing** capabilities
- **Filtering and querying**
- Tools: `list_genomes`, `get_genome_features_p3`, etc.

### P3_TOOLS_COMPUTATIONAL.py
- **Computational services** (p3-submit-*)
- **Job submission and monitoring**
- **BLAST, MSA, phylogenetic analysis**
- Tools: `submit_blast_job`, `check_job_status`, etc.

### P3_TOOLS_SPECIALIZED.py
- **Specialized analysis tools**
- **Comparative genomics**
- **K-mer analysis**
- **Feature analysis**

### P3_TOOLS_UTILITIES.py
- **File management**
- **Authentication**
- **Data processing**
- Tools: `check_bvbrc_auth`, `get_workspace_info`, etc.

## 🎯 Recommended Approach

For your current use case, I recommend:

1. **Use the new MCP servers** - they provide comprehensive functionality and better error handling
2. **The old server has been removed** - your code now uses the new servers directly
3. **All functionality is available** through the new MCP protocol

## 🔄 Migration Strategy

If you want to adopt the new servers:

1. **Start with one server** (e.g., `BVBRC_API.py` for REST API access)
2. **Create a client wrapper** to communicate with it
3. **Gradually migrate** functionality as needed
4. **Keep your existing server** for compatibility

## 📚 Documentation

Each server has its own documentation:
- **`BVBRC_API.md`** - REST API programming guide
- **`P3_TOOLS_GUIDE.md`** - P3-tools programming guide
- **`P3_TOOLS_*.md`** - Individual server documentation

## 🧪 Testing Individual Servers

You can test each server independently:

```bash
# Test REST API server
python BVBRC_API.py

# Test data retrieval server
python P3_TOOLS_DATA_RETRIEVAL.py

# Test computational server
python P3_TOOLS_COMPUTATIONAL.py
```

Each server will start and wait for MCP protocol messages on stdin/stdout.

## 💡 Key Benefits of Individual Servers

1. **Modularity** - Each server has a focused purpose
2. **Scalability** - Can run servers on different machines
3. **Maintainability** - Easier to update individual components
4. **Performance** - Can optimize each server independently
5. **Standards Compliance** - Follows MCP protocol properly

## 🚨 Important Notes

- **Don't import functions** from MCP servers directly
- **Each server runs independently** as a separate process
- **Communication is via MCP protocol** (stdin/stdout)
- **Your existing code works fine** - no need to change immediately
- **New servers provide additional capabilities** but aren't required for current functionality 