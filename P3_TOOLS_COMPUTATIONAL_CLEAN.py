#!/usr/bin/env python3
"""
P3 Computational Tools Server - Cleaned Version

This server contains only the unified p3_submit tool and job monitoring tools.
All individual p3-submit-* tools have been removed and replaced with the unified interface.
"""

import subprocess
import json
import os
import tempfile
import time
from typing import Dict, List, Any, Optional, Union
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP
mcp = FastMCP("P3-Tools Computational Server (Cleaned)")

# Core utility functions
def run_p3_tool(command, input_data=None, timeout=300):
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

def get_available_apps():
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

def validate_app_id(app_id):
    """Validate app_id against available apps"""
    available = get_available_apps()
    return app_id in available

def upload_files_to_workspace(files, workspace_path):
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

def build_app_command(app_id, output_path, output_name, 
                     parameters, input_files):
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

def parse_job_output(stdout):
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
    app_id,
    output_path,
    output_name,
    parameters,
    input_files=None,
    workspace=None,
    additional_options=None
):
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
def p3_list_apps():
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
def p3_get_app_info(app_id):
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

# Job monitoring tools (kept from original)
def monitor_job_completion(job_id, poll_interval=30, max_wait_time=3600):
    """Monitor job until completion or timeout"""
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        status_result = run_p3_tool(['p3-job-status', job_id])
        
        if not status_result['success']:
            return {
                'success': False,
                'job_id': job_id,
                'error': status_result['stderr'],
                'status': 'error'
            }
        
        status_output = status_result['stdout'].lower()
        
        if 'completed' in status_output or 'complete' in status_output:
            return {
                'success': True,
                'job_id': job_id,
                'status': 'completed',
                'duration': time.time() - start_time
            }
        elif 'failed' in status_output or 'error' in status_output:
            return {
                'success': False,
                'job_id': job_id,
                'status': 'failed',
                'duration': time.time() - start_time
            }
        
        time.sleep(poll_interval)
    
    return {
        'success': False,
        'job_id': job_id,
        'status': 'timeout',
        'duration': max_wait_time,
        'message': f'Job monitoring timeout after {max_wait_time} seconds'
    }

@mcp.tool()
def p3_monitor_job(job_id,
                   poll_interval=30,
                   max_wait_time=3600):
    """
    Monitor a computational job until completion or timeout
    
    Args:
        job_id: Job ID to monitor
        poll_interval: Seconds between status checks (default 30)
        max_wait_time: Maximum seconds to wait (default 1 hour)
    
    Returns:
        JSON string with monitoring results
    """
    
    result = monitor_job_completion(job_id, poll_interval, max_wait_time)
    return json.dumps(result, indent=2)

@mcp.tool()
def p3_check_job_status(job_id: str) -> str:
    """
    Check current status of a computational job
    
    Args:
        job_id: Job ID to check
    
    Returns:
        JSON string with job status information
    """
    
    result = run_p3_tool(['p3-job-status', job_id])
    
    if result['success']:
        status_output = result['stdout'].lower()
        
        if 'completed' in status_output or 'complete' in status_output:
            status = 'completed'
        elif 'failed' in status_output or 'error' in status_output:
            status = 'failed'
        elif 'running' in status_output:
            status = 'running'
        elif 'queued' in status_output:
            status = 'queued'
        else:
            status = 'unknown'
        
        return json.dumps({
            'success': True,
            'job_id': job_id,
            'status': status,
            'details': result['stdout']
        }, indent=2)
    else:
        return json.dumps({
            'success': False,
            'job_id': job_id,
            'error': result['stderr']
        }, indent=2)

@mcp.tool()
def p3_get_job_results(job_name: str, result_type: str = 'summary') -> str:
    """
    Retrieve and parse computational job results
    
    Args:
        job_name: Name of the completed job
        result_type: Type of results to retrieve ('summary', 'detailed', 'files')
    
    Returns:
        JSON string with job results
    """
    
    # Check the hidden results directory
    results_base = f"/rbutler@bvbrc/home/.{job_name}"
    
    # List available result files
    list_result = run_p3_tool(['p3-ls', results_base])
    
    if not list_result['success']:
        return json.dumps({
            'success': False,
            'error': f'Could not access results for job {job_name}',
            'details': list_result['stderr']
        }, indent=2)
    
    available_files = list_result['stdout'].strip().split('\n')
    
    results_info = {
        'success': True,
        'job_name': job_name,
        'results_directory': results_base,
        'available_files': available_files,
        'file_count': len(available_files)
    }
    
    if result_type == 'files':
        return json.dumps(results_info, indent=2)
    
    # For summary or detailed, try to read key result files
    key_files = ['output.txt', 'results.json', 'summary.txt', 'report.html']
    file_contents = {}
    
    for filename in key_files:
        if filename in available_files:
            file_path = f"{results_base}/{filename}"
            file_result = run_p3_tool(['p3-cat', file_path])
            
            if file_result['success']:
                content = file_result['stdout']
                if result_type == 'summary':
                    # Truncate for summary
                    file_contents[filename] = content[:1000] + '...' if len(content) > 1000 else content
                else:
                    file_contents[filename] = content
    
    results_info['file_contents'] = file_contents
    return json.dumps(results_info, indent=2)

if __name__ == "__main__":
    mcp.run()
