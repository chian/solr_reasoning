# BV-BRC MCP Servers Collection

This repository now includes the latest MCP (Model Context Protocol) implementation for BV-BRC (Bacterial and Viral Bioinformatics Resource Center), providing comprehensive access to bioinformatics data and computational services through modular, specialized servers.

## 🚀 New Features

### **Modular MCP Server Architecture**
- **BV-BRC REST API server** - Direct API calls for fast data retrieval
- **P3-Tools data retrieval server** - Comprehensive data access tools
- **P3-Tools computational server** - Advanced analysis services
- **P3-Tools specialized server** - Domain-specific analysis tools
- **P3-Tools utilities server** - File management and processing
- **Async support** - Non-blocking API requests
- **Comprehensive error handling** - Robust error diagnosis and recovery

### **Available Tools**

#### **REST API Tools (Async)**
- `get_genomes_by_species()` - Get genomes for a specific species
- `get_complete_genomes()` - Get only complete genomes
- `get_genome_features()` - Get features for a specific genome
- `search_features_by_product()` - Search features by product name

#### **P3-Tools Data Retrieval**
- `list_genomes()` - List genomes with filtering
- `get_genome_features_p3()` - Get genome features via P3-tools
- `find_similar_genomes()` - Find similar genomes (legacy compatibility)

#### **P3-Tools Computational Services**
- `submit_blast_job()` - Submit BLAST analysis jobs
- `check_job_status()` - Check job status and results

#### **P3-Tools Utilities**
- `check_bvbrc_auth()` - Check authentication status
- `get_workspace_info()` - Get workspace information
- `run_p3_tool_generic()` - Execute any P3-tool command

## 📁 File Structure

```
solr_reasoning/
├── BVBRC_API.py                  # REST API MCP server
├── P3_TOOLS_DATA_RETRIEVAL.py    # Data retrieval MCP server
├── P3_TOOLS_COMPUTATIONAL.py     # Computational services MCP server
├── P3_TOOLS_SPECIALIZED.py       # Specialized analysis MCP server
├── P3_TOOLS_UTILITIES.py         # Utilities MCP server
├── [MCP servers and documentation]
├── requirements.txt               # Dependencies
├── README_NEW_MCP.md             # This file
├── MCP_USAGE_GUIDE.md            # Usage guide for individual servers
├── MCP_README.md                 # Original MCP documentation
├── BVBRC_API.md                  # REST API guide
├── P3_TOOLS_GUIDE.md             # P3-tools guide
├── P3_TOOLS_*.md                 # Individual tool documentation
└── [existing files...]           # Your existing codebase
```

## 🛠️ Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure BV-BRC is installed:**
   - The servers expect BV-BRC to be installed at `/Applications/BV-BRC.app/`
   - If installed elsewhere, update the path in the server files

## 🚀 Usage

### **Running Individual MCP Servers**
Each server runs as a separate process:

```bash
# REST API server
python BVBRC_API.py

# Data retrieval server
python P3_TOOLS_DATA_RETRIEVAL.py

# Computational services server
python P3_TOOLS_COMPUTATIONAL.py

# Specialized analysis server
python P3_TOOLS_SPECIALIZED.py

# Utilities server
python P3_TOOLS_UTILITIES.py
```

### **Your Updated Code**
Your `P3_together.py` has been updated to use the new MCP servers directly. The old server has been removed and replaced with proper MCP client communication.

### **Integration Options**
See `MCP_USAGE_GUIDE.md` for detailed integration strategies.

## 🔧 Key Improvements

### **1. Modular Architecture**
- Separate servers for different functionality
- Unified server for easy integration
- Clear separation of concerns

### **2. Async Support**
- Non-blocking REST API calls
- Better performance for concurrent requests
- Proper timeout handling

### **3. Enhanced Error Handling**
- Comprehensive error diagnosis
- Detailed error messages
- Graceful failure recovery

### **4. Type Safety**
- Full type annotations
- Pydantic models for validation
- Better IDE support

### **5. Documentation**
- Extensive inline documentation
- Complete API reference
- Usage examples

## 📊 Tool Coverage

| Category | Tools | Status |
|----------|-------|--------|
| REST API | 26+ | ✅ Complete |
| Data Retrieval | 32 | ✅ Complete |
| Computational | 16 | ✅ Complete |
| Specialized | 12+ | ✅ Core Complete |
| Utilities | 8+ | ✅ Core Complete |
| **Total** | **94+** | **✅ Comprehensive** |

## 🔄 Migration Guide

### **From Old Server to New Server**

1. **Update imports** (already done):
   ```python
   from bvbrc_mcp_unified import (
       get_blast_results,
       get_workspace_info,
       check_bvbrc_auth,
       find_similar_genomes,
       submit_blast_job,
       check_job_status,
       list_genomes,
       run_p3_tool_generic as run_p3_tool
   )
   ```

2. **New async tools available**:
   ```python
   # New REST API tools (async)
   result = await get_genomes_by_species("Escherichia coli")
   result = await get_complete_genomes("Escherichia coli")
   result = await get_genome_features("83333.111")
   ```

3. **Enhanced P3-tools**:
   ```python
   # Better error handling and parsing
   result = list_genomes(genus="Escherichia", limit=50)
   result = get_genome_features_p3("83333.111", limit=100)
   ```

## 🧪 Testing

Run the comprehensive test suite:
```bash
python test_new_mcp.py
```

This will test:
- REST API functionality
- P3-tools integration
- Legacy compatibility
- Error handling

## 📚 Documentation

- **`MCP_README.md`** - Overview of all MCP servers
- **`BVBRC_API.md`** - REST API programming guide
- **`P3_TOOLS_GUIDE.md`** - P3-tools programming guide
- **Individual `P3_TOOLS_*.md`** - Specific tool documentation

## 🤝 Contributing

The new MCP implementation is based on the latest standards from the [ralphbutler/LLM_misc](https://github.com/ralphbutler/LLM_misc) repository and maintains compatibility with your existing codebase.

## 📄 License

This implementation maintains the same license as your original codebase while incorporating the latest MCP standards and best practices. 