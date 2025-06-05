from fastmcp import FastMCP, Context
import os
import json

# Initialize the MCP server
mcp = FastMCP(
    name="BV-BRC Documentation & Data Server",
    instructions="This server exposes BV-BRC API documentation and example data as MCP resources and tools."
)

API_DOCS_DIR = os.path.join(os.path.dirname(__file__), "api_docs")

@mcp.resource(
    uri="docs://{doc_name}",
    name="Get Documentation",
    description="Retrieve the contents of a BV-BRC documentation file by name.",
    mime_type="text/plain",
    tags={"documentation", "bvbrc"}
)
def get_doc(doc_name: str) -> str:
    """
    Returns the contents of a documentation file from the api_docs directory.
    Args:
        doc_name: The name of the documentation file (e.g., 'genome.txt').
    Returns:
        The contents of the file as a string.
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    # Sanitize input to prevent directory traversal
    safe_name = os.path.basename(doc_name)
    doc_path = os.path.join(API_DOCS_DIR, safe_name)
    if not os.path.isfile(doc_path):
        raise FileNotFoundError(f"Documentation file '{safe_name}' not found.")
    with open(doc_path, "r", encoding="utf-8") as f:
        return f.read()

@mcp.tool()
def list_docs() -> list:
    """
    List all available documentation files in the api_docs directory.
    Returns:
        A list of documentation file names.
    """
    return sorted([f for f in os.listdir(API_DOCS_DIR) if os.path.isfile(os.path.join(API_DOCS_DIR, f))])

@mcp.tool()
def get_example_genome_info() -> dict:
    """
    Return example genome information as a stub for BV-BRC data retrieval.
    This should be replaced with a real API call or data integration later.
    Returns:
        A dictionary with example genome metadata.
    """
    # Example stub data
    return {
        "genome_id": "12345.6",
        "organism": "Escherichia coli",
        "strain": "K12",
        "features": 4321,
        "source": "example stub"
    }

@mcp.resource(
    uri="docs://index",
    name="Documentation Index",
    description="Returns a list of available documentation files with a short description (first line of each file if present).",
    mime_type="application/json",
    tags={"documentation", "index", "bvbrc"}
)
def docs_index() -> list:
    """
    Returns a list of documentation files with their first line as a description (if present).
    Returns:
        A list of dicts: [{"name": ..., "desc": ...}, ...]
    """
    index = []
    for fname in sorted([f for f in os.listdir(API_DOCS_DIR) if os.path.isfile(os.path.join(API_DOCS_DIR, f))]):
        desc = ""
        try:
            with open(os.path.join(API_DOCS_DIR, fname), "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                desc = first_line if first_line else ""
        except Exception:
            desc = ""
        index.append({"name": fname, "desc": desc})
    return index

@mcp.tool()
def search_docs(query: str) -> list:
    """
    Search documentation files by keyword in the filename or first line.
    Args:
        query: The keyword to search for (case-insensitive).
    Returns:
        A list of matching documentation file names.
    """
    matches = []
    q = query.lower()
    for fname in os.listdir(API_DOCS_DIR):
        if not os.path.isfile(os.path.join(API_DOCS_DIR, fname)):
            continue
        if q in fname.lower():
            matches.append(fname)
            continue
        try:
            with open(os.path.join(API_DOCS_DIR, fname), "r", encoding="utf-8") as f:
                first_line = f.readline().strip().lower()
                if q in first_line:
                    matches.append(fname)
        except Exception:
            continue
    return sorted(matches)

if __name__ == "__main__":
    mcp.run() 