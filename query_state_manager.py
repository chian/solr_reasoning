#!/usr/bin/env python3
"""
Query State Manager for Concurrent Processing

Handles saving/loading query states when computational jobs are submitted,
allowing the system to pause queries and resume them when jobs complete.
"""

import json
import os
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid

PAUSED_QUERIES_DIR = "paused_queries"
COMPLETED_JOBS_DIR = "completed_jobs"

def ensure_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs(PAUSED_QUERIES_DIR, exist_ok=True)
    os.makedirs(COMPLETED_JOBS_DIR, exist_ok=True)

def save_paused_query(user_query: str, actions: List[Dict], action_index: int, 
                     job_info: Dict, trace_file_path: str) -> str:
    """
    Save a query state when a computational job is submitted
    
    Args:
        user_query: Original user query
        actions: List of all actions in the query
        action_index: Index of the action that submitted the job
        job_info: Information about the submitted job (job_id, job_type, etc.)
        trace_file_path: Path to the current trace file
    
    Returns:
        Query ID for tracking
    """
    ensure_directories()
    
    query_id = str(uuid.uuid4())[:8]  # Short unique ID
    
    query_state = {
        "query_id": query_id,
        "user_query": user_query,
        "actions": actions,
        "action_index": action_index,
        "job_info": job_info,
        "trace_file_path": trace_file_path,
        "paused_at": datetime.now().isoformat(),
        "status": "waiting_for_job"
    }
    
    state_file = os.path.join(PAUSED_QUERIES_DIR, f"query_{query_id}.json")
    with open(state_file, 'w') as f:
        json.dump(query_state, f, indent=2)
    
    print(f"📋 Query paused and saved as {query_id}")
    print(f"💼 Waiting for job {job_info['job_id']} ({job_info['job_type']}) to complete")
    
    return query_id

def get_paused_queries() -> List[Dict[str, Any]]:
    """Get all currently paused queries"""
    ensure_directories()
    
    paused_queries = []
    if not os.path.exists(PAUSED_QUERIES_DIR):
        return paused_queries
    
    for filename in os.listdir(PAUSED_QUERIES_DIR):
        if filename.startswith("query_") and filename.endswith(".json"):
            try:
                with open(os.path.join(PAUSED_QUERIES_DIR, filename), 'r') as f:
                    query_state = json.load(f)
                    paused_queries.append(query_state)
            except (json.JSONDecodeError, FileNotFoundError):
                continue
    
    return paused_queries

def mark_job_completed(job_id: str, job_results: str):
    """Mark a job as completed and save its results"""
    ensure_directories()
    
    job_completion = {
        "job_id": job_id,
        "completed_at": datetime.now().isoformat(),
        "results": job_results,
        "status": "completed"
    }
    
    completion_file = os.path.join(COMPLETED_JOBS_DIR, f"job_{job_id}.json")
    with open(completion_file, 'w') as f:
        json.dump(job_completion, f, indent=2)
    
    print(f"✅ Job {job_id} marked as completed")

def get_job_results(job_id: str) -> Optional[str]:
    """Get results for a completed job"""
    ensure_directories()
    
    completion_file = os.path.join(COMPLETED_JOBS_DIR, f"job_{job_id}.json")
    if os.path.exists(completion_file):
        try:
            with open(completion_file, 'r') as f:
                job_data = json.load(f)
                return job_data.get("results")
        except (json.JSONDecodeError, FileNotFoundError):
            pass
    
    return None

def resume_query(query_id: str) -> Optional[Dict[str, Any]]:
    """Load a paused query for resumption"""
    ensure_directories()
    
    state_file = os.path.join(PAUSED_QUERIES_DIR, f"query_{query_id}.json")
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r') as f:
                query_state = json.load(f)
                
            # Get job results
            job_id = query_state["job_info"]["job_id"]
            job_results = get_job_results(job_id)
            
            if job_results:
                query_state["job_results"] = job_results
                query_state["status"] = "ready_to_resume"
                return query_state
            
        except (json.JSONDecodeError, FileNotFoundError):
            pass
    
    return None

def cleanup_completed_query(query_id: str):
    """Remove a completed query from the paused queries"""
    ensure_directories()
    
    state_file = os.path.join(PAUSED_QUERIES_DIR, f"query_{query_id}.json")
    if os.path.exists(state_file):
        os.remove(state_file)
        print(f"🧹 Cleaned up completed query {query_id}")

def list_active_queries() -> List[str]:
    """List all currently paused queries"""
    queries = get_paused_queries()
    return [f"Query {q['query_id']}: {q['user_query'][:50]}... (Job: {q['job_info']['job_id']})" 
            for q in queries] 