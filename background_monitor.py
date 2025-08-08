#!/usr/bin/env python3
"""
Background Job Monitor for Concurrent Query Processing

This script runs continuously in the background, checking the status of
computational jobs and resuming paused queries when jobs complete.
"""

import time
import subprocess
import sys
import os
from query_state_manager import (
    get_paused_queries, mark_job_completed, cleanup_completed_query,
    get_job_results, resume_query
)
from P3_together import BVBRCMCPClient

# Initialize computational client for job status checking
computational_client = BVBRCMCPClient("P3_TOOLS_COMPUTATIONAL.py")

def check_job_status(job_id: str) -> dict:
    """Check the status of a computational job"""
    try:
        computational_client.start_server()
        result = computational_client.call_tool("p3_check_job_status", job_id=job_id)
        
        if isinstance(result, str):
            import json
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                return {"status": "unknown", "raw_output": result}
        else:
            return result
            
    except Exception as e:
        print(f"❌ Error checking job {job_id}: {e}")
        return {"status": "error", "message": str(e)}

def get_job_results_data(job_id: str, job_type: str) -> str:
    """Get the results of a completed job"""
    try:
        computational_client.start_server()
        
        # For different job types, we might need different result retrieval methods
        # For now, use a generic approach
        result = computational_client.call_tool("p3_get_job_results", job_name=job_id)
        
        if isinstance(result, str):
            return result
        else:
            return str(result)
            
    except Exception as e:
        print(f"❌ Error getting results for job {job_id}: {e}")
        return f"Error retrieving job results: {e}"

def resume_paused_query(query_state: dict):
    """Resume a paused query by restarting P3_together.py with the saved state"""
    try:
        query_id = query_state["query_id"]
        print(f"🔄 Resuming query {query_id}: {query_state['user_query'][:50]}...")
        
        # Prepare resume arguments
        resume_args = [
            "python", "P3_together.py",
            "--resume", query_id
        ]
        
        # Start the resumed query in the background
        process = subprocess.Popen(
            resume_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print(f"✅ Query {query_id} resumed successfully (PID: {process.pid})")
        
        # Clean up the paused query file
        cleanup_completed_query(query_id)
        
    except Exception as e:
        print(f"❌ Error resuming query {query_state['query_id']}: {e}")

def monitor_jobs():
    """Main monitoring loop"""
    print("🔍 Background job monitor started")
    print("🔄 Checking for paused queries and job completions...")
    
    while True:
        try:
            paused_queries = get_paused_queries()
            
            if not paused_queries:
                print("💤 No paused queries found. Waiting...")
                time.sleep(30)
                continue
            
            print(f"📋 Found {len(paused_queries)} paused queries")
            
            for query_state in paused_queries:
                query_id = query_state["query_id"]
                job_id = query_state["job_info"]["job_id"]
                job_type = query_state["job_info"]["job_type"]
                
                print(f"🔍 Checking job {job_id} for query {query_id}...")
                
                # Check if we already have results for this job
                existing_results = get_job_results(job_id)
                if existing_results:
                    print(f"✅ Job {job_id} already completed, resuming query...")
                    resume_paused_query(query_state)
                    continue
                
                # Check job status
                job_status = check_job_status(job_id)
                
                if job_status.get("status") == "completed":
                    print(f"🎉 Job {job_id} completed! Getting results...")
                    
                    # Get job results
                    job_results = get_job_results_data(job_id, job_type)
                    
                    # Mark job as completed
                    mark_job_completed(job_id, job_results)
                    
                    # Resume the paused query
                    resume_paused_query(query_state)
                    
                elif job_status.get("status") == "failed":
                    print(f"❌ Job {job_id} failed: {job_status.get('message', 'Unknown error')}")
                    
                    # Mark job as failed and resume query with error info
                    error_results = f"Job failed: {job_status.get('message', 'Unknown error')}"
                    mark_job_completed(job_id, error_results)
                    resume_paused_query(query_state)
                    
                elif job_status.get("status") == "running":
                    print(f"⏳ Job {job_id} still running...")
                    
                else:
                    print(f"❓ Job {job_id} status unknown: {job_status}")
            
            print(f"😴 Sleeping for 30 seconds before next check...")
            time.sleep(30)
            
        except KeyboardInterrupt:
            print("\n🛑 Background monitor stopped by user")
            break
        except Exception as e:
            print(f"❌ Error in monitoring loop: {e}")
            print("🔄 Continuing monitoring...")
            time.sleep(10)

if __name__ == "__main__":
    print("🚀 Starting BV-BRC Background Job Monitor")
    print("📋 This will check for completed computational jobs and resume paused queries")
    print("🛑 Press Ctrl+C to stop")
    print("-" * 60)
    
    monitor_jobs() 