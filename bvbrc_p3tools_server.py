# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "mcp",
# ]
# ///

"""
BV-BRC P3-Tools MCP Server (Fixed)

Wraps BV-BRC p3-tools for comprehensive biology analysis support.
Fixed to use correct p3-tools syntax with --attr and --eq parameters.
"""

import subprocess
import json
import os
import tempfile
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# Initialize the MCP server
mcp = FastMCP("BV-BRC P3-Tools Fixed")

# Helper function to run p3 commands with proper environment
def run_p3_command(command, input_data=None):
    """Run a p3-tool command with BV-BRC environment sourced"""
    try:
        # Source the BV-BRC environment before running the command
        full_command = f"source /Applications/BV-BRC.app/user-env.sh && {command}"
        
        if input_data:
            result = subprocess.run(full_command, input=input_data, capture_output=True, text=True, shell=True)
        else:
            result = subprocess.run(full_command, capture_output=True, text=True, shell=True)
        
        if result.returncode != 0:
            return {"status": "error", "message": result.stderr}
        
        return {"status": "success", "output": result.stdout}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def parse_p3_output(output):
    """Parse p3-tools output, filtering out welcome messages and headers"""
    lines = output.strip().split('\n')
    
    # Filter out welcome message and non-data lines
    data_lines = []
    found_header = False
    
    for line in lines:
        # Skip welcome messages
        if "Welcome to the BV-BRC" in line:
            continue
        
        # Skip empty lines
        if not line.strip():
            continue
            
        # Look for TSV header (contains dots like genome.genome_id)
        if not found_header and ('genome.' in line or '\t' in line):
            # This is likely the header row
            headers = line.split('\t')
            found_header = True
            continue
        
        # This is actual data
        if found_header:
            data_lines.append(line)
    
    return data_lines

@mcp.tool()
def get_blast_results(job_name: str) -> str:
    """Retrieve and parse BLAST results from completed job.
    
    Args:
        job_name: Name of the completed BLAST job
    
    Returns:
        JSON with parsed BLAST results, hit summaries, and statistics
        
    Example:
        get_blast_results("mycoplasma_protein_blast")
    """
    # Check the hidden results directory
    json_path = f"/rbutler@bvbrc/home/.{job_name}/blast_out.json"
    headers_path = f"/rbutler@bvbrc/home/.{job_name}/blast_headers.txt"
    
    try:
        # Get the JSON results
        json_result = run_p3_command(f"p3-cat {json_path}")
        if json_result["status"] != "success":
            return json.dumps({
                "status": "error",
                "message": f"Could not retrieve results for job {job_name}",
                "error": json_result.get("message", "Unknown error")
            })
        
        # Parse the BLAST JSON - filter out welcome messages first
        import json as json_lib
        
        # Clean the output by removing welcome messages
        json_text = json_result["output"].strip()
        lines = json_text.split('\n')
        
        # Find the start of actual JSON (starts with [ or {)
        json_start = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('[') or line.strip().startswith('{'):
                json_start = i
                break
        
        # Reconstruct clean JSON
        clean_json = '\n'.join(lines[json_start:])
        
        if not clean_json.strip():
            return json.dumps({
                "status": "error",
                "message": "No valid JSON data found in results file"
            })
        
        try:
            blast_data = json_lib.loads(clean_json)
        except json_lib.JSONDecodeError as e:
            return json.dumps({
                "status": "error", 
                "message": f"JSON parsing failed: {str(e)}",
                "raw_content_preview": json_text[:200]
            })
        
        # Extract key information
        if blast_data and len(blast_data) > 0:
            search_results = blast_data[0].get("report", {}).get("results", {}).get("search", {})
            
            hits = search_results.get("hits", [])
            query_len = search_results.get("query_len", "N/A")
            query_title = search_results.get("query_title", "N/A")
            message = search_results.get("message", "")
            stats = search_results.get("stat", {})
            
            # Summarize hits
            hit_summaries = []
            for hit in hits[:10]:  # Top 10 hits
                hit_summary = {
                    "title": hit.get("description", [{}])[0].get("title", "N/A"),
                    "accession": hit.get("description", [{}])[0].get("accession", "N/A"),
                    "length": hit.get("len", "N/A")
                }
                
                # Get best HSP (alignment)
                hsps = hit.get("hsps", [])
                if hsps:
                    best_hsp = hsps[0]
                    hit_summary.update({
                        "evalue": best_hsp.get("evalue", "N/A"),
                        "bit_score": best_hsp.get("bit_score", "N/A"),
                        "identity": best_hsp.get("identity", "N/A"),
                        "align_length": best_hsp.get("align_len", "N/A"),
                        "query_coverage": f"{best_hsp.get('query_from', 'N/A')}-{best_hsp.get('query_to', 'N/A')}"
                    })
                    
                    # Calculate identity percentage
                    if best_hsp.get("identity") and best_hsp.get("align_len"):
                        identity_pct = round((best_hsp["identity"] / best_hsp["align_len"]) * 100, 1)
                        hit_summary["identity_percent"] = identity_pct
                
                hit_summaries.append(hit_summary)
            
            return json.dumps({
                "status": "success",
                "job_name": job_name,
                "query_length": query_len,
                "query_title": query_title,
                "total_hits": len(hits),
                "message": message if message else "Search completed",
                "database_stats": {
                    "total_sequences": stats.get("db_num", "N/A"),
                    "total_length": stats.get("db_len", "N/A")
                },
                "top_hits": hit_summaries,
                "raw_json_path": json_path
            })
        else:
            return json.dumps({
                "status": "error",
                "message": "Could not parse BLAST results JSON"
            })
            
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Error retrieving BLAST results: {str(e)}"
        })

@mcp.tool()
def get_workspace_info() -> str:
    """Get BV-BRC workspace information.
    
    Returns:
        JSON with workspace paths and structure
    """
    result = run_p3_command("p3-ls /")
    
    if result["status"] == "success":
        return json.dumps({
            "status": "success",
            "workspace_listing": result["output"].strip(),
            "note": "Use these paths for job output directories"
        })
    else:
        return json.dumps(result)

@mcp.tool()
def check_bvbrc_auth() -> str:
    """Check BV-BRC authentication status.
    
    Returns:
        JSON with username and workspace info
    """
    result = run_p3_command("p3-whoami")
    
    if result["status"] == "success":
        return json.dumps({
            "status": "authenticated",
            "user_info": result["output"].strip()
        })
    else:
        return json.dumps({
            "status": "not_authenticated", 
            "message": "Please run p3-login to authenticate",
            "error": result.get("message", "")
        })

@mcp.tool()
def find_similar_genomes(species: str = None, genus: str = None, strain: str = None, limit: int = 20) -> str:
    """Find genomes similar to a specified species/genus/strain.
    
    Args:
        species: Species name (e.g., "Mycoplasma pneumoniae")
        genus: Genus name (e.g., "Mycoplasma") 
        strain: Optional strain name (e.g., "M129")
        limit: Maximum number of results
    
    Returns:
        JSON with similar genomes and metadata
        
    Example:
        find_similar_genomes(genus="Mycoplasma", limit=10)
        find_similar_genomes(species="Mycoplasma pneumoniae", strain="M129")
    """
    # Build command with correct p3-tools syntax
    cmd = "p3-all-genomes --attr genome_id,genome_name,species,genus,strain,genome_status,genome_length"
    
    # Add search constraints using --eq syntax
    if genus:
        cmd += f' --eq genome.genus,"{genus}"'
    
    if species:
        cmd += f' --eq genome.species,"{species}"'
    
    if strain:
        cmd += f' --eq genome.strain,"{strain}"'
    
    # Add limit using head
    cmd += f" | head -n {limit + 1}"  # +1 to account for header
    
    result = run_p3_command(cmd)
    
    if result["status"] == "success":
        data_lines = parse_p3_output(result["output"])
        
        if data_lines:
            # Get headers from the original output
            output_lines = result["output"].strip().split('\n')
            headers = None
            
            for line in output_lines:
                if 'genome.' in line and '\t' in line:
                    headers = line.split('\t')
                    break
            
            if not headers:
                return json.dumps({
                    "status": "error",
                    "message": "Could not parse headers from p3-tools output"
                })
            
            # Parse data rows
            genomes = []
            for line in data_lines:
                if line.strip():
                    values = line.split('\t')
                    genome = {}
                    for i, header in enumerate(headers):
                        if i < len(values):
                            # Clean up header names (remove genome. prefix for readability)
                            clean_header = header.replace('genome.', '')
                            genome[clean_header] = values[i]
                    genomes.append(genome)
            
            query_desc = []
            if genus:
                query_desc.append(f"genus={genus}")
            if species:
                query_desc.append(f"species={species}")
            if strain:
                query_desc.append(f"strain={strain}")
            
            return json.dumps({
                "status": "success",
                "query": ", ".join(query_desc) if query_desc else "all genomes",
                "count": len(genomes),
                "genomes": genomes
            })
        else:
            return json.dumps({
                "status": "success",
                "query": f"genus={genus}, species={species}, strain={strain}",
                "count": 0,
                "genomes": [],
                "message": "No matching genomes found"
            })
    else:
        return json.dumps(result)

@mcp.tool()
def submit_blast_job(sequence: str, sequence_type: str = "aa", target_genomes: list = None,
                    database: str = "Plasmids", db_type: str = "faa", job_name: str = "BLAST_job", 
                    evalue: float = 1e-5, max_hits: int = 20) -> str:
    """Submit BLAST job using p3-submit-BLAST.
    
    Args:
        sequence: Protein or DNA sequence to search
        sequence_type: Input type "aa" (protein) or "dna" (nucleotide)
        target_genomes: List of genome IDs to search against (preferred for targeted analysis)
        database: Precomputed database ("Plasmids", "Phages") - used if target_genomes not specified
        db_type: Database type ("faa" protein, "fna" contig DNA, "ffn" feature DNA, "frn" RNA)
        job_name: Name for the job
        evalue: E-value cutoff
        max_hits: Maximum number of hits to return
    
    Returns:
        JSON with job ID and submission status
        
    Example:
        # Target specific Mycoplasma genomes
        submit_blast_job("MKAILVVL...", "aa", ["40477.31", "40479.20"], job_name="mycoplasma_search")
        
        # Use precomputed database
        submit_blast_job("MKAILVVL...", "aa", database="Plasmids")
    """
    # Create temporary file for sequence in home directory (BV-BRC can't access /tmp/)
    import os
    home_dir = os.path.expanduser('~')
    temp_file_path = os.path.join(home_dir, f"temp_blast_{job_name}.fasta")
    
    with open(temp_file_path, 'w') as f:
        f.write(f">Query\n{sequence}\n")
    seq_file = temp_file_path
    
    try:
        # Build command with correct p3-submit-BLAST syntax
        cmd = f"p3-submit-BLAST --in-type {sequence_type} --in-fasta-file {seq_file} "
        cmd += f"--workspace-upload-path home "
        
        # Use genome list if provided, otherwise use precomputed database
        if target_genomes:
            genome_list = ','.join(target_genomes)
            cmd += f"--db-genome-list '{genome_list}' "
        else:
            cmd += f"--db-database '{database}' "
        
        cmd += f"--db-type {db_type} "
        cmd += f"--evalue-cutoff {evalue} --max-hits {max_hits} "
        cmd += f"--workspace-path-prefix /rbutler@bvbrc "
        cmd += f"home {job_name}"
        
        result = run_p3_command(cmd)
        
        if result["status"] == "success":
            output = result["output"].strip()
            # Try to extract job ID from output
            lines = output.split('\n')
            job_id = None
            
            for line in lines:
                if "Job ID:" in line:
                    job_id = line.split("Job ID:")[-1].strip()
                elif "job" in line.lower() and any(char.isdigit() for char in line):
                    # Extract job ID from submission message
                    import re
                    matches = re.findall(r'\d+', line)
                    if matches:
                        job_id = matches[-1]
                elif line.strip() and line.strip().isdigit():
                    job_id = line.strip()
            
            return json.dumps({
                "status": "success",
                "job_id": job_id if job_id else "check_output",
                "job_name": job_name,
                "sequence_type": sequence_type,
                "target_genomes": target_genomes,
                "database": database if not target_genomes else "genome_list",
                "db_type": db_type,
                "evalue": evalue,
                "max_hits": max_hits,
                "submission_output": output
            })
        else:
            return json.dumps(result)
            
    finally:
        # Clean up temp file
        os.unlink(seq_file)

@mcp.tool()
def check_job_status(job_id: str) -> str:
    """Check status of any BV-BRC job.
    
    Args:
        job_id: Job ID to check
    
    Returns:
        JSON with job status and details
    """
    cmd = f"p3-job-status {job_id}"
    result = run_p3_command(cmd)
    
    if result["status"] == "success":
        status_info = result["output"].strip()
        
        # Parse status from output
        if "completed" in status_info.lower() or "complete" in status_info.lower():
            status = "completed"
        elif "failed" in status_info.lower() or "error" in status_info.lower():
            status = "failed"
        elif "running" in status_info.lower():
            status = "running"
        elif "queued" in status_info.lower():
            status = "queued"
        else:
            status = "unknown"
        
        return json.dumps({
            "status": "success",
            "job_id": job_id,
            "job_status": status,
            "details": status_info
        })
    else:
        return json.dumps(result)

@mcp.tool()
def submit_proteome_comparison(reference_genome: str, target_genomes: list,
                              job_name: str = "proteome_comparison") -> str:
    """Submit proteome comparison job.
    
    Args:
        reference_genome: Reference genome ID 
        target_genomes: List of target genome IDs (max 9)
        job_name: Name for the job
    
    Returns:
        JSON with job submission info
        
    Example:
        submit_proteome_comparison("511145.12", ["99287.12"], "ecoli_vs_salmonella")
    """
    if len(target_genomes) > 9:
        return json.dumps({
            "status": "error",
            "message": "Maximum 9 target genomes allowed"
        })
    
    # Build command
    cmd = f"p3-submit-proteome-comparison --reference-genome-id {reference_genome} "
    
    for target in target_genomes:
        cmd += f"--comparison-genome-id {target} "
    
    cmd += f"/ {job_name}"
    
    result = run_p3_command(cmd)
    
    if result["status"] == "success":
        output = result["output"].strip()
        # Try to extract job ID
        lines = output.split('\n')
        job_id = None
        
        for line in lines:
            if "Job ID:" in line:
                job_id = line.split("Job ID:")[-1].strip()
            elif line.strip() and line.strip().isdigit():
                job_id = line.strip()
        
        return json.dumps({
            "status": "success", 
            "job_id": job_id if job_id else "check_output",
            "reference": reference_genome,
            "targets": target_genomes,
            "job_name": job_name,
            "submission_output": output
        })
    else:
        return json.dumps(result)

@mcp.tool()
def list_genomes(species: str = None, genus: str = None, limit: int = 50) -> str:
    """List genomes from BV-BRC database.
    
    Args:
        species: Species name to filter by
        genus: Genus name to filter by
        limit: Maximum number of results
    
    Returns:
        JSON with genome list
    """
    cmd = "p3-all-genomes --attr genome_id,genome_name,species,genus,strain,genome_status"
    
    if genus:
        cmd += f' --eq genome.genus,"{genus}"'
    
    if species:
        cmd += f' --eq genome.species,"{species}"'
    
    cmd += f" | head -n {limit + 1}"  # +1 for header
    
    result = run_p3_command(cmd)
    
    if result["status"] == "success":
        data_lines = parse_p3_output(result["output"])
        
        if data_lines:
            # Get headers
            output_lines = result["output"].strip().split('\n')
            headers = None
            
            for line in output_lines:
                if 'genome.' in line and '\t' in line:
                    headers = line.split('\t')
                    break
            
            if not headers:
                return json.dumps({
                    "status": "error",
                    "message": "Could not parse headers from p3-tools output"
                })
            
            # Parse data
            genomes = []
            for line in data_lines:
                if line.strip():
                    values = line.split('\t')
                    genome = {}
                    for i, header in enumerate(headers):
                        if i < len(values):
                            clean_header = header.replace('genome.', '')
                            genome[clean_header] = values[i]
                    genomes.append(genome)
            
            return json.dumps({
                "status": "success",
                "count": len(genomes),
                "genomes": genomes
            })
        else:
            return json.dumps({
                "status": "success",
                "count": 0,
                "genomes": []
            })
    else:
        return json.dumps(result)

@mcp.tool()
def run_p3_tool(command: str) -> str:
    """Execute any p3-tool command string using run_p3_command.
    
    Args:
        command: The p3-tool command string to execute (e.g., "p3-get-genome-features --eq genome_id,12345.1 --attr feature_id,product")
    
    Returns:
        JSON with the command result (success/error status and output)
        
    Example:
        run_p3_tool("p3-get-genome-features --eq genome_id,12345.1 --attr feature_id,product")
    """
    result = run_p3_command(command)
    return json.dumps(result)

# Run the server
if __name__ == "__main__":
    mcp.run(transport="stdio")
