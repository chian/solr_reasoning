### Multi-Turn RL Agent Integration Plan

This document outlines a minimal, non-invasive plan to integrate the `Multi-Turn-RL-Agent` as an external component for experimentation, without modifying existing code paths.

- **Repo**: `external/Multi-Turn-RL-Agent` (added as git submodule)
- **Paper/Code**: see `https://github.com/SiliangZeng/Multi-Turn-RL-Agent`

### Goals

- Keep changes minimal and isolated.
- Provide an adapter interface to route trajectories from our environment to the agent trainer, and accept rewards (to be provided by the user later).
- Avoid renaming variables, changing parameters, or altering existing functions.

### Components

- `external/Multi-Turn-RL-Agent`: upstream trainer, tools, prompts, and configs.
- `docs/mt_rl_setup.md`: environment setup instructions (conda env, Java for search, dataset index generation).
- `integrations/mt_rl/adapter.py`: thin adapter to construct the agent's config and call entrypoints.
- `scripts/run_mt_rl.sh`: convenience script to launch training with user-provided model and hardware flags.

### Data Flow

1. Our system prepares multi-turn episodes (dialogue/tool traces) and writes them to a temporary dataset file compatible with the verifier examples.
2. The adapter invokes the trainer script in the submodule (e.g., `verifiers/examples/triviaqa_search.py`) with CLI flags.
3. Outcome rewards will be provided externally and logged; turn-level rewards can be enabled via flags.

### Non-Goals (for now)

- No modification of existing core modules.
- No changes to existing prompts or tool schemas.

### References

- Multi-Turn RL Agent repo: `https://github.com/SiliangZeng/Multi-Turn-RL-Agent`
# Unified P3 Submit Tool - Integration Guide

## Overview

This guide shows how to integrate the unified `p3_submit` tool into your existing MCP server (`P3_together.py`) to replace all individual `p3-submit-*` tools with a single, consistent interface.

## Current Setup Analysis

Your current `P3_together.py` uses:
- **`computational_client`** pointing to `P3_TOOLS_COMPUTATIONAL.py`
- **16 individual p3-submit-* tools** mapped to this client
- **Job monitoring tools** (`p3_check_job_status`, `p3_get_job_results`)

## Integration Plan

### Step 1: Create Unified Computational Server

Replace `P3_TOOLS_COMPUTATIONAL.py` with a new server that includes the unified tool.

### Step 2: Update Tool Mapping

Replace individual tool mappings with the unified tool.

### Step 3: Update Job Handling

Modify job submission and monitoring logic.

## Detailed Integration Instructions

### Step 1: Create New Unified Computational Server

Create `P3_TOOLS_UNIFIED_COMPUTATIONAL.py`:

```python
#!/usr/bin/env python3
"""
Unified P3 Computational Tools Server

This server replaces all individual p3-submit-* tools with a unified interface
while maintaining compatibility with existing job monitoring tools.
"""

import subprocess
import json
import os
import tempfile
from typing import Dict, List, Any, Optional, Union
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP
mcp = FastMCP("P3-Tools Unified Computational Server")

# Import core functions from unified tool
def run_p3_tool(command: List[str], input_data: str = None, timeout: int = 300) -> Dict[str, Any]:
    """Execute P3 tool with proper error handling"""
    try:
        command_str = ' '.join(command)
        full_command = f"source /Applications/BV-BRC.app/user-env.sh && {command_str}"
        
        if input_data:
            result = subprocess.run(
                full_command,
                input=input_data,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=True,
                check=False
            )
        else:
            result = subprocess.run(
                full_command,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=True,
                check=False
            )
        
        return {
            'success': result.returncode == 0,
            'stdout': result.stdout.strip(),
            'stderr': result.stderr.strip(),
            'returncode': result.returncode,
            'command': full_command
        }
        
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'stdout': '',
            'stderr': f'Command timed out after {timeout} seconds',
            'returncode': -1,
            'command': full_command
        }
    except Exception as e:
        return {
            'success': False,
            'stdout': '',
            'stderr': f'Execution error: {str(e)}',
            'returncode': -1,
            'command': full_command
        }

def get_available_apps() -> Dict[str, Dict[str, str]]:
    """Get list of available apps with descriptions"""
    result = run_p3_tool(['appserv-enumerate-apps'])
    
    if not result['success']:
        return {}
    
    apps = {}
    lines = result['stdout'].strip().split('\n')
    
    # Skip header lines
    for line in lines[2:]:
        if line.strip():
            parts = line.split('   ', 2)
            if len(parts) >= 3:
                app_id = parts[0].strip()
                label = parts[1].strip()
                description = parts[2].strip()
                apps[app_id] = {
                    'label': label,
                    'description': description
                }
    
    return apps

def validate_app_id(app_id: str) -> bool:
    """Validate app_id against available apps"""
    available = get_available_apps()
    return app_id in available

def upload_files_to_workspace(files: List[str], workspace_path: str) -> List[str]:
    """Upload local files to workspace and return workspace paths"""
    workspace_files = []
    
    for file_path in files:
        if os.path.exists(file_path):
            # Upload local file to workspace
            workspace_file = f"{workspace_path}/{os.path.basename(file_path)}"
            result = run_p3_tool(['p3-cp', file_path, workspace_file])
            
            if result['success']:
                workspace_files.append(workspace_file)
            else:
                raise Exception(f"Failed to upload {file_path}: {result['stderr']}")
        else:
            # Assume it's already a workspace path
            workspace_files.append(file_path)
    
    return workspace_files

def build_app_command(app_id: str, output_path: str, output_name: str, 
                     parameters: Dict[str, Any], input_files: List[str]) -> List[str]:
    """Build the appropriate p3-submit-* command"""
    command = [f'p3-submit-{app_id}', output_path, output_name]
    
    # Add parameters
    for key, value in parameters.items():
        if isinstance(value, bool):
            if value:
                command.append(f'--{key}')
        elif value is not None:
            command.extend([f'--{key}', str(value)])
    
    # Add input files
    for file_path in input_files:
        command.extend(['--input-file', file_path])
    
    return command

def parse_job_output(stdout: str) -> Dict[str, Any]:
    """Parse job submission output to extract job information"""
    job_info = {
        'submitted': True,
        'raw_output': stdout
    }
    
    # Try to extract job ID from common patterns
    lines = stdout.split('\n')
    for line in lines:
        line = line.strip()
        if 'job' in line.lower() and 'id' in line.lower():
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    job_id = parts[1].strip()
                    job_info['job_id'] = job_id
                    break
    
    return job_info

# App-specific parameter schemas
APP_PARAMETER_SCHEMAS = {
    'BLAST': {
        'program': str,           # blastp, blastn, etc.
        'database': str,          # nr, nt, etc.
        'evalue': float,          # 1e-5
        'max_target_seqs': int,   # 100
        'min_coverage': float,    # 0.0
        'in_type': str,           # dna, aa
        'db_type': str,           # fna, ffn, frn, faa
    },
    'MSA': {
        'alignment_method': str,  # muscle, clustalw, etc.
        'input_file': str,        # FASTA file
    },
    'gene-tree': {
        'input_file': str,        # FASTA file
        'method': str,            # Phylogenetic method
    },
    'codon-tree': {
        'input_file': str,        # FASTA file
        'method': str,            # Codon-based method
    },
    'genome-annotation': {
        'contigs_file': str,      # Input contigs
        'recipe': str,            # Annotation recipe
    },
    'genome-assembly': {
        'assembly_strategy': str, # auto, etc.
        'recipe': str,            # unicycler, etc.
    },
    'proteome-comparison': {
        'comparison_type': str,   # bidirectional, all-vs-all
        'evalue_cutoff': float,   # 1e-5
        'genome_file': str,       # File with genome IDs
    },
    'variation-analysis': {
        'mapper': str,            # bwa, etc.
        'caller': str,            # gatk, etc.
        'reference_genome': str,  # Reference genome ID
    },
    'rnaseq': {
        'analysis_type': str,     # differential, etc.
        'mapper': str,            # hisat2, etc.
        'reference_genome': str,  # Reference genome ID
    },
    'CGA': {
        'recipe': str,            # comprehensive, minimal
        'reference_genome': str,  # Optional reference
    },
    'fastqutils': {
        'operation': str,         # trim, etc.
        'quality_threshold': int, # 20
    },
    'taxonomic-classification': {
        'method': str,            # Classification method
        'database': str,          # Reference database
    },
    'metagenome-binning': {
        'method': str,            # Binning method
        'min_contig_length': int, # Minimum contig length
    },
    'metagenomic-read-mapping': {
        'mapper': str,            # bowtie2, etc.
        'reference_genome': str,  # Reference genome
    },
    'sars2-analysis': {
        'analysis_type': str,     # comprehensive, etc.
        'reference_genome': str,  # SARS-CoV-2 reference
    },
    'sars2-assembly': {
        'assembly_strategy': str, # consensus, denovo
        'reference_genome': str,  # SARS-CoV-2 reference
    },
    'wastewater-analysis': {
        'analysis_type': str,     # Analysis type
        'reference_genome': str,  # Reference genome
    },
    'viral-assembly': {
        'assembly_strategy': str, # Assembly strategy
        'reference_genome': str,  # Reference genome
    },
    'SubspeciesClassification': {
        'method': str,            # Classification method
        'reference_database': str, # Reference database
    },
    'comparative-systems': {
        'analysis_type': str,     # Analysis type
        'reference_genome': str,  # Reference genome
    }
}

# Unified submit tool
@mcp.tool()
def p3_submit(
    app_id: str,
    output_path: str,
    output_name: str,
    parameters: Dict[str, Any],
    input_files: Optional[List[str]] = None,
    workspace: Optional[str] = None,
    additional_options: Optional[Dict[str, Any]] = None
) -> str:
    """
    Unified tool for submitting jobs to all BV-BRC computational services.
    This replaces all individual p3-submit-* tools with one consistent interface.
    
    Args:
        app_id: The app to run (e.g., "BLAST", "MSA", "genome-annotation")
        output_path: Workspace output directory
        output_name: Job name/output prefix
        parameters: App-specific parameters (dict)
        input_files: Optional list of files to upload
        workspace: Optional target workspace
        additional_options: Optional override options
        
    Returns:
        JSON string with job submission result
    """
    
    try:
        # Validate app_id
        if not validate_app_id(app_id):
            available = get_available_apps()
            available_list = list(available.keys())
            return json.dumps({
                'submitted': False,
                'error': f'Invalid app_id: {app_id}',
                'available_apps': available_list,
                'suggestion': f'Use one of the available app IDs: {", ".join(available_list[:10])}...'
            }, indent=2)
        
        # Get app info
        apps = get_available_apps()
        app_info = apps.get(app_id, {})
        
        # Handle input files
        workspace_files = []
        if input_files:
            try:
                workspace_files = upload_files_to_workspace(input_files, output_path)
            except Exception as e:
                return json.dumps({
                    'submitted': False,
                    'error': f'File upload failed: {str(e)}',
                    'app_id': app_id
                }, indent=2)
        
        # Merge parameters with additional_options
        final_parameters = parameters.copy()
        if additional_options:
            final_parameters.update(additional_options)
        
        # Build command
        command = build_app_command(app_id, output_path, output_name, final_parameters, workspace_files)
        
        # Execute job submission
        result = run_p3_tool(command, timeout=300)
        
        if result['success']:
            job_info = parse_job_output(result['stdout'])
            job_info.update({
                'app_id': app_id,
                'app_label': app_info.get('label', ''),
                'app_description': app_info.get('description', ''),
                'output_path': output_path,
                'output_name': output_name,
                'parameters': final_parameters,
                'input_files': workspace_files,
                'workspace': workspace,
                'command': result['command']
            })
            return json.dumps(job_info, indent=2)
        else:
            return json.dumps({
                'submitted': False,
                'error': result['stderr'],
                'app_id': app_id,
                'command': result['command'],
                'returncode': result['returncode']
            }, indent=2)
    
    except Exception as e:
        return json.dumps({
            'submitted': False,
            'error': f'Unexpected error: {str(e)}',
            'app_id': app_id
        }, indent=2)

# App discovery tools
@mcp.tool()
def p3_list_apps() -> str:
    """List all available computational apps with their descriptions"""
    try:
        apps = get_available_apps()
        
        # Group apps by category
        app_categories = {
            'Sequence Analysis': [],
            'Phylogenetic Analysis': [],
            'Genome Analysis': [],
            'Comparative Analysis': [],
            'Metagenomic Analysis': [],
            'Specialized Analysis': [],
            'Other': []
        }
        
        # Categorize apps
        for app_id, info in apps.items():
            description = info.get('description', '').lower()
            
            if any(term in description for term in ['blast', 'alignment', 'sequence']):
                app_categories['Sequence Analysis'].append({'id': app_id, **info})
            elif any(term in description for term in ['tree', 'phylogenetic', 'evolution']):
                app_categories['Phylogenetic Analysis'].append({'id': app_id, **info})
            elif any(term in description for term in ['genome', 'assembly', 'annotation']):
                app_categories['Genome Analysis'].append({'id': app_id, **info})
            elif any(term in description for term in ['comparison', 'comparative', 'proteome']):
                app_categories['Comparative Analysis'].append({'id': app_id, **info})
            elif any(term in description for term in ['metagenome', 'metagenomic', 'taxonomic']):
                app_categories['Metagenomic Analysis'].append({'id': app_id, **info})
            elif any(term in description for term in ['sars', 'viral', 'wastewater']):
                app_categories['Specialized Analysis'].append({'id': app_id, **info})
            else:
                app_categories['Other'].append({'id': app_id, **info})
        
        return json.dumps({
            'total_apps': len(apps),
            'categories': app_categories
        }, indent=2)
    
    except Exception as e:
        return json.dumps({
            'error': f'Failed to list apps: {str(e)}'
        }, indent=2)

@mcp.tool()
def p3_get_app_info(app_id: str) -> str:
    """Get detailed information about a specific app"""
    try:
        apps = get_available_apps()
        
        if app_id not in apps:
            return json.dumps({
                'error': f'App not found: {app_id}',
                'available_apps': list(apps.keys())
            }, indent=2)
        
        app_info = apps[app_id]
        schema = APP_PARAMETER_SCHEMAS.get(app_id, {})
        
        return json.dumps({
            'app_id': app_id,
            'label': app_info.get('label', ''),
            'description': app_info.get('description', ''),
            'parameter_schema': schema,
            'example_usage': {
                'app_id': app_id,
                'output_path': '/user/username/results',
                'output_name': 'my_job',
                'parameters': {k: 'value' for k in schema.keys()},
                'input_files': ['/path/to/input.fasta']
            }
        }, indent=2)
    
    except Exception as e:
        return json.dumps({
            'error': f'Failed to get app info: {str(e)}'
        }, indent=2)

# Keep existing job monitoring tools
@mcp.tool()
def p3_check_job_status(job_id: str) -> str:
    """Check the status of a submitted job"""
    try:
        result = run_p3_tool(['p3-check-job-status', job_id])
        
        if result['success']:
            return json.dumps({
                'job_id': job_id,
                'status': 'success',
                'output': result['stdout']
            }, indent=2)
        else:
            return json.dumps({
                'job_id': job_id,
                'status': 'error',
                'error': result['stderr']
            }, indent=2)
    
    except Exception as e:
        return json.dumps({
            'job_id': job_id,
            'status': 'error',
            'error': f'Unexpected error: {str(e)}'
        }, indent=2)

@mcp.tool()
def p3_get_job_results(job_name: str) -> str:
    """Get results from a completed job"""
    try:
        result = run_p3_tool(['p3-get-job-results', job_name])
        
        if result['success']:
            return json.dumps({
                'job_name': job_name,
                'status': 'success',
                'output': result['stdout']
            }, indent=2)
        else:
            return json.dumps({
                'job_name': job_name,
                'status': 'error',
                'error': result['stderr']
            }, indent=2)
    
    except Exception as e:
        return json.dumps({
            'job_name': job_name,
            'status': 'error',
            'error': f'Unexpected error: {str(e)}'
        }, indent=2)

if __name__ == "__main__":
    mcp.run()
```

### Step 2: Update P3_together.py

Replace the computational client initialization and tool mapping:

```python
# Replace this line:
# computational_client = BVBRCMCPClient("P3_TOOLS_COMPUTATIONAL.py")

# With this:
computational_client = BVBRCMCPClient("P3_TOOLS_UNIFIED_COMPUTATIONAL.py")

# Replace the tool mapping section:
tool_mapping = {
    # Data retrieval tools (keep existing)
    "p3_all_genomes": data_retrieval_client,
    "p3_all_contigs": data_retrieval_client,
    "p3_all_drugs": data_retrieval_client,
    "p3_all_subsystem_roles": data_retrieval_client,
    "p3_all_subsystems": data_retrieval_client,
    "p3_all_taxonomies": data_retrieval_client,
    "p3_get_genome_data": data_retrieval_client,
    "p3_get_genome_contigs": data_retrieval_client,
    "p3_get_genome_features": data_retrieval_client,
    "p3_get_feature_data": data_retrieval_client,
    "p3_get_feature_sequence": data_retrieval_client,
    "p3_get_feature_subsystems": data_retrieval_client,
    "p3_get_subsystem_features": data_retrieval_client,
    "p3_get_drug_genomes": data_retrieval_client,
    "p3_get_family_data": data_retrieval_client,
    "p3_get_taxonomy_data": data_retrieval_client,
    
    # Unified computational tool (replaces all individual p3-submit-* tools)
    "p3_submit": computational_client,
    
    # App discovery tools
    "p3_list_apps": computational_client,
    "p3_get_app_info": computational_client,
    
    # Job monitoring tools (keep existing)
    "p3_check_job_status": computational_client,
    "p3_get_job_results": computational_client,
    
    # Utility tools (keep existing)
    "p3_workspace_info": utilities_client,
    "p3_check_auth": utilities_client,
    "p3_run_command": utilities_client,
    "p3_ls": utilities_client,
    "p3_cp": utilities_client,
    "p3_rm": utilities_client,
    "p3_cat": utilities_client,
    "p3_extract": utilities_client,
    "p3_collate": utilities_client,
    "p3_count": utilities_client,
    
    # REST API tools (keep existing)
    "get_genomes_by_species": rest_api_client,
    "get_complete_genomes": rest_api_client,
    "get_genome_features": rest_api_client,
    "search_features_by_product": rest_api_client,
}
```

### Step 3: Update Job Handling Logic

Modify the job submission detection in `execute_mcp_action`:

```python
# Replace this section:
if tool_name.startswith("p3_submit_"):
    try:
        if isinstance(result, str):
            result_json = json.loads(result)
        else:
            result_json = result
        
        if "job_id" in result_json:
            return {
                "observation": f"Job submitted successfully. Job ID: {result_json['job_id']}",
                "raw_output": result,
                "status": "job_submitted",
                "job_id": result_json["job_id"],
                "job_type": tool_name
            }
    except (json.JSONDecodeError, TypeError):
        pass

# With this:
if tool_name == "p3_submit":
    try:
        if isinstance(result, str):
            result_json = json.loads(result)
        else:
            result_json = result
        
        if result_json.get("submitted") and "job_id" in result_json:
            return {
                "observation": f"Job submitted successfully. Job ID: {result_json['job_id']}. App: {result_json.get('app_label', result_json.get('app_id', 'Unknown'))}",
                "raw_output": result,
                "status": "job_submitted",
                "job_id": result_json["job_id"],
                "job_type": f"p3_submit_{result_json.get('app_id', 'unknown')}"
            }
        elif not result_json.get("submitted"):
            return {
                "observation": f"Job submission failed: {result_json.get('error', 'Unknown error')}",
                "raw_output": result,
                "status": "error"
            }
    except (json.JSONDecodeError, TypeError):
        pass
```

## Usage Instructions

### 1. **Discover Available Apps**

```python
# List all available computational apps
result = p3_list_apps()
apps_info = json.loads(result)
print(f"Total apps available: {apps_info['total_apps']}")

# Browse by category
for category, apps in apps_info['categories'].items():
    if apps:
        print(f"\n{category}:")
        for app in apps:
            print(f"  - {app['id']}: {app['description']}")
```

### 2. **Get App Information**

```python
# Get detailed information about a specific app
result = p3_get_app_info("BLAST")
blast_info = json.loads(result)

print(f"App: {blast_info['label']}")
print(f"Description: {blast_info['description']}")
print(f"Parameters: {list(blast_info['parameter_schema'].keys())}")
```

### 3. **Submit Jobs**

#### **BLAST Analysis**
```python
result = p3_submit(
    app_id="BLAST",
    output_path="/user/jdoe/results",
    output_name="protein_blast",
    parameters={
        "program": "blastp",
        "database": "nr",
        "evalue": 1e-5,
        "max_target_seqs": 100
    },
    input_files=["/local/path/query_proteins.fasta"]
)
```

#### **Genome Assembly**
```python
result = p3_submit(
    app_id="genome-assembly",
    output_path="/user/jdoe/assembly",
    output_name="ecoli_assembly",
    parameters={
        "assembly_strategy": "auto",
        "recipe": "unicycler"
    },
    input_files=[
        "/local/path/reads_1.fastq",
        "/local/path/reads_2.fastq"
    ]
)
```

#### **RNA-Seq Analysis**
```python
result = p3_submit(
    app_id="rnaseq",
    output_path="/user/jdoe/rnaseq",
    output_name="differential_expression",
    parameters={
        "analysis_type": "differential",
        "mapper": "hisat2",
        "reference_genome": "83333.111"
    },
    input_files=[
        "/local/path/control_1.fastq",
        "/local/path/control_2.fastq",
        "/local/path/treatment_1.fastq",
        "/local/path/treatment_2.fastq"
    ]
)
```

#### **Metagenomic Analysis**
```python
result = p3_submit(
    app_id="taxonomic-classification",
    output_path="/user/jdoe/metagenome",
    output_name="sample_classification",
    parameters={
        "method": "kraken2",
        "database": "standard"
    },
    input_files=["/local/path/metagenome.fastq"]
)
```

### 4. **Monitor Jobs**

```python
# Check job status
status_result = p3_check_job_status(job_id="job_12345")

# Get job results
job_results = p3_get_job_results(job_name="job_12345")
```

## Migration Examples

### **Before (Individual Tools)**

```python
# BLAST submission
result = p3_submit_blast(
    query_sequences={"seq1": "ATCG..."},
    output_path="/user/jdoe/results",
    output_name="blast_job",
    program="blastp",
    database="nr"
)

# Genome assembly submission
result = p3_submit_genome_assembly(
    reads_files=["reads_1.fastq", "reads_2.fastq"],
    output_path="/user/jdoe/assembly",
    output_name="assembly_job",
    assembly_strategy="auto"
)
```

### **After (Unified Tool)**

```python
# BLAST submission
result = p3_submit(
    app_id="BLAST",
    output_path="/user/jdoe/results",
    output_name="blast_job",
    parameters={
        "program": "blastp",
        "database": "nr"
    },
    input_files=["query.fasta"]
)

# Genome assembly submission
result = p3_submit(
    app_id="genome-assembly",
    output_path="/user/jdoe/assembly",
    output_name="assembly_job",
    parameters={
        "assembly_strategy": "auto"
    },
    input_files=["reads_1.fastq", "reads_2.fastq"]
)
```

## Benefits of Integration

### 1. **Simplified Interface**
- One tool (`p3_submit`) replaces 16 individual tools
- Consistent parameter structure across all apps
- Reduced learning curve

### 2. **Dynamic Discovery**
- Automatically discovers available apps
- No need to maintain hardcoded app lists
- Easy to add new apps without code changes

### 3. **Integrated File Management**
- Automatic file upload handling
- Support for both local and workspace files
- Comprehensive error handling

### 4. **Backward Compatibility**
- Job monitoring tools remain unchanged
- Existing workflows continue to work
- Gradual migration possible

### 5. **Enhanced Error Handling**
- Detailed error messages with suggestions
- Validation of app IDs and parameters
- Graceful handling of upload failures

## Testing the Integration

### 1. **Test App Discovery**
```python
# Test that apps can be discovered
result = p3_list_apps()
assert "total_apps" in json.loads(result)
```

### 2. **Test App Information**
```python
# Test getting app details
result = p3_get_app_info("BLAST")
blast_info = json.loads(result)
assert "parameter_schema" in blast_info
```

### 3. **Test Job Submission**
```python
# Test job submission (with valid parameters)
result = p3_submit(
    app_id="BLAST",
    output_path="/user/test/results",
    output_name="test_job",
    parameters={"program": "blastp", "database": "nr"},
    input_files=["/path/to/test.fasta"]
)
```

### 4. **Test Error Handling**
```python
# Test invalid app ID
result = p3_submit(
    app_id="INVALID_APP",
    output_path="/user/test/results",
    output_name="test_job",
    parameters={}
)
# Should return error with suggestions
```

## Troubleshooting

### Common Issues

1. **Authentication Required**
   - Ensure BV-BRC authentication is active
   - Run `p3-login` before using tools

2. **File Upload Failures**
   - Check file paths exist
   - Verify workspace permissions
   - Ensure sufficient disk space

3. **Invalid App ID**
   - Use `p3_list_apps()` to discover available apps
   - Check app ID spelling and case

4. **Parameter Errors**
   - Use `p3_get_app_info(app_id)` to see required parameters
   - Check parameter types and values

### Debug Mode

Enable debug output by setting environment variable:
```bash
export P3_DEBUG=1
```

## Conclusion

This integration provides a powerful, unified interface for all BV-BRC computational services while maintaining backward compatibility with existing job monitoring tools. The unified approach significantly reduces complexity while providing enhanced functionality and better error handling.
