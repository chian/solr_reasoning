#!/usr/bin/env python3
"""
Batch Query Manager for Concurrent Processing

This wrapper script manages batch processing of queries from queries.json
using the new single-query mode, with proper concurrent job handling.
"""

import json
import subprocess
import sys
import os
import time
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from query_state_manager import list_active_queries, get_paused_queries

def load_queries() -> List[str]:
    """Load queries from queries.json"""
    try:
        with open('queries.json', 'r') as f:
            queries = json.load(f)
        return queries
    except FileNotFoundError:
        print("❌ Error: queries.json not found")
        print("💡 Create this file with a list of queries to process")
        return []
    except json.JSONDecodeError:
        print("❌ Error: queries.json is not valid JSON")
        return []

def run_single_query(query: str, conda_env: str = "py312") -> Dict[str, any]:
    """Run a single query using P3_together.py"""
    print(f"🚀 Starting query: {query[:60]}...")
    
    start_time = time.time()
    
    try:
        # Run the query directly (assuming we're already in the correct conda env)
        cmd = f"python P3_together.py '{query}'"
        
        process = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        duration = time.time() - start_time
        
        if process.returncode == 0:
            # Check if query was paused (job submitted)
            if "Query paused and saved as" in process.stdout:
                # Extract query ID from output
                lines = process.stdout.split('\n')
                query_id = None
                for line in lines:
                    if "Query saved as" in line:
                        query_id = line.split("Query saved as ")[-1].strip()
                        break
                
                return {
                    "query": query,
                    "status": "paused",
                    "query_id": query_id,
                    "duration": duration,
                    "message": "Query paused for computational job"
                }
            else:
                return {
                    "query": query,
                    "status": "completed",
                    "duration": duration,
                    "message": "Query completed successfully"
                }
        else:
            return {
                "query": query,
                "status": "failed",
                "duration": duration,
                "error": process.stderr,
                "message": f"Query failed with return code {process.returncode}"
            }
            
    except subprocess.TimeoutExpired:
        return {
            "query": query,
            "status": "timeout",
            "duration": 3600,
            "message": "Query timed out after 1 hour"
        }
    except Exception as e:
        return {
            "query": query,
            "status": "error",
            "duration": time.time() - start_time,
            "error": str(e),
            "message": f"Unexpected error: {e}"
        }

def check_background_monitor() -> bool:
    """Check if background monitor is running"""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "background_monitor.py"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except:
        return False

def start_background_monitor() -> bool:
    """Start the background monitor if not running"""
    if check_background_monitor():
        print("✅ Background monitor already running")
        return True
    
    try:
        print("🔄 Starting background monitor...")
        subprocess.Popen(
            ["conda", "activate", "py312", "&&", "python", "background_monitor.py"],
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Give it time to start
        time.sleep(3)
        
        if check_background_monitor():
            print("✅ Background monitor started successfully")
            return True
        else:
            print("❌ Failed to start background monitor")
            return False
            
    except Exception as e:
        print(f"❌ Error starting background monitor: {e}")
        return False

def show_system_status():
    """Show current system status"""
    print("\n" + "="*60)
    print("📊 SYSTEM STATUS")
    print("="*60)
    
    # Check background monitor
    if check_background_monitor():
        print("🔍 Background monitor: ✅ Running")
    else:
        print("🔍 Background monitor: ❌ Not running")
    
    # Check paused queries
    active_queries = list_active_queries()
    if active_queries:
        print(f"⏸️  Paused queries: {len(active_queries)}")
        for query_info in active_queries:
            print(f"   • {query_info}")
    else:
        print("⏸️  Paused queries: None")
    
    # Check directories
    if os.path.exists("query_results"):
        result_dirs = [d for d in os.listdir("query_results") if os.path.isdir(os.path.join("query_results", d))]
        print(f"📁 Completed queries: {len(result_dirs)} result directories")
    else:
        print("📁 Completed queries: 0 (query_results directory not found)")

def run_batch_sequential(queries: List[str], conda_env: str = "py312") -> List[Dict]:
    """Run queries sequentially (safer for resource management)"""
    results = []
    
    print(f"🔄 Processing {len(queries)} queries sequentially...")
    print("💡 Computational jobs will pause queries and resume automatically")
    
    for i, query in enumerate(queries, 1):
        print(f"\n📋 Query {i}/{len(queries)}")
        print(f"🎯 {query[:80]}...")
        
        result = run_single_query(query, conda_env)
        results.append(result)
        
        # Show result
        status_emoji = {
            "completed": "✅",
            "paused": "⏸️ ",
            "failed": "❌",
            "timeout": "⏰",
            "error": "💥"
        }
        
        emoji = status_emoji.get(result["status"], "❓")
        print(f"{emoji} {result['message']} ({result['duration']:.1f}s)")
        
        if result["status"] == "paused":
            print(f"   💼 Job submitted, query will resume automatically")
        elif result["status"] in ["failed", "error"]:
            print(f"   ⚠️  Error: {result.get('error', 'Unknown error')}")
        
        # Brief pause between queries
        time.sleep(1)
    
    return results

def run_batch_concurrent(queries: List[str], conda_env: str = "py312", max_workers: int = 3) -> List[Dict]:
    """Run queries concurrently (experimental - use with caution)"""
    results = []
    
    print(f"🚀 Processing {len(queries)} queries with up to {max_workers} concurrent workers...")
    print("⚠️  Experimental mode - monitor resource usage")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all queries
        future_to_query = {
            executor.submit(run_single_query, query, conda_env): query 
            for query in queries
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_query):
            query = future_to_query[future]
            try:
                result = future.result()
                results.append(result)
                
                # Show progress
                status_emoji = {
                    "completed": "✅",
                    "paused": "⏸️ ",
                    "failed": "❌",
                    "timeout": "⏰",
                    "error": "💥"
                }
                
                emoji = status_emoji.get(result["status"], "❓")
                print(f"{emoji} {result['message']} ({result['duration']:.1f}s)")
                
            except Exception as e:
                print(f"💥 Query failed with exception: {e}")
                results.append({
                    "query": query,
                    "status": "error",
                    "error": str(e),
                    "message": f"Exception: {e}"
                })
    
    return results

def print_summary(results: List[Dict]):
    """Print batch processing summary"""
    print("\n" + "="*60)
    print("📊 BATCH PROCESSING SUMMARY")
    print("="*60)
    
    # Count results by status
    status_counts = {}
    total_duration = 0
    
    for result in results:
        status = result["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
        total_duration += result.get("duration", 0)
    
    # Print counts
    for status, count in status_counts.items():
        emoji = {
            "completed": "✅",
            "paused": "⏸️ ",
            "failed": "❌",
            "timeout": "⏰",
            "error": "💥"
        }.get(status, "❓")
        
        print(f"{emoji} {status.title()}: {count}")
    
    print(f"⏱️  Total processing time: {total_duration:.1f} seconds")
    print(f"📊 Average per query: {total_duration/len(results):.1f} seconds")
    
    # Show paused queries
    paused_results = [r for r in results if r["status"] == "paused"]
    if paused_results:
        print(f"\n⏸️  Paused queries ({len(paused_results)}):")
        for result in paused_results:
            print(f"   • {result['query'][:60]}... (ID: {result.get('query_id', 'Unknown')})")
        print("💡 These will resume automatically when jobs complete")

def main():
    print("🚀 BV-BRC Batch Query Manager")
    print("=" * 60)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Batch process queries with concurrent job support")
    parser.add_argument("--concurrent", action="store_true", help="Run queries concurrently (experimental)")
    parser.add_argument("--workers", type=int, default=3, help="Max concurrent workers (default: 3)")
    parser.add_argument("--conda-env", default="py312", help="Conda environment name (default: py312)")
    parser.add_argument("--no-monitor", action="store_true", help="Don't start background monitor")
    parser.add_argument("--status", action="store_true", help="Show system status and exit")
    
    args = parser.parse_args()
    
    # Show status and exit if requested
    if args.status:
        show_system_status()
        return
    
    # Load queries
    queries = load_queries()
    if not queries:
        print("❌ No queries to process")
        return
    
    print(f"📋 Loaded {len(queries)} queries from queries.json")
    
    # Start background monitor unless disabled
    if not args.no_monitor:
        if not start_background_monitor():
            print("⚠️  Background monitor failed to start")
            print("💡 Computational jobs may not resume automatically")
    
    # Show initial status
    show_system_status()
    
    # Process queries
    print(f"\n🔄 Starting batch processing...")
    
    if args.concurrent:
        results = run_batch_concurrent(queries, args.conda_env, args.workers)
    else:
        results = run_batch_sequential(queries, args.conda_env)
    
    # Show summary
    print_summary(results)
    
    # Final status
    show_system_status()
    
    print(f"\n✅ Batch processing complete!")
    print(f"💡 Use 'python batch_query_manager.py --status' to check system status")

if __name__ == "__main__":
    main() 