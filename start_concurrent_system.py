#!/usr/bin/env python3
"""
Concurrent Query Processing System Startup

This script starts the background job monitor and provides instructions
for using the concurrent query processing system.
"""

import subprocess
import sys
import os
import time
from query_state_manager import list_active_queries

def start_background_monitor():
    """Start the background job monitor"""
    try:
        # Start background monitor
        process = subprocess.Popen(
            ["python", "background_monitor.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print(f"🚀 Background monitor started (PID: {process.pid})")
        
        # Give it a moment to start
        time.sleep(2)
        
        # Check if it's still running
        if process.poll() is None:
            print("✅ Background monitor is running successfully")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Background monitor failed to start:")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            return None
            
    except Exception as e:
        print(f"❌ Error starting background monitor: {e}")
        return None

def show_system_status():
    """Show current system status"""
    print("\n" + "="*60)
    print("🔍 SYSTEM STATUS")
    print("="*60)
    
    # Check for paused queries
    active_queries = list_active_queries()
    if active_queries:
        print(f"📋 Active paused queries ({len(active_queries)}):")
        for query_info in active_queries:
            print(f"   • {query_info}")
    else:
        print("💤 No paused queries currently")
    
    # Check directories
    if os.path.exists("paused_queries"):
        paused_count = len([f for f in os.listdir("paused_queries") if f.endswith('.json')])
        print(f"📁 Paused queries directory: {paused_count} files")
    
    if os.path.exists("completed_jobs"):
        completed_count = len([f for f in os.listdir("completed_jobs") if f.endswith('.json')])
        print(f"📁 Completed jobs directory: {completed_count} files")

def show_usage_instructions():
    """Show usage instructions"""
    print("\n" + "="*60)
    print("📖 HOW TO USE THE CONCURRENT SYSTEM")
    print("="*60)
    
    print("""
🔄 NORMAL QUERY PROCESSING:
   python P3_together.py
   
   - Queries run normally until they hit a computational job
   - When a job is submitted, the query pauses automatically
   - You can immediately start new queries while jobs run

⏸️  WHAT HAPPENS WHEN JOBS ARE SUBMITTED:
   - System detects p3_submit_* tools (MSA, BLAST, etc.)
   - Query state is saved with a unique ID
   - Query pauses and exits
   - Background monitor tracks the job
   - Query automatically resumes when job completes

🎯 CONCURRENT PROCESSING:
   Terminal 1: python P3_together.py  # Submit MSA job → pauses
   Terminal 2: python P3_together.py  # Run different query → immediate
   Terminal 3: python P3_together.py  # Another query → immediate
   
   Background: First query resumes automatically when MSA completes

📋 MONITORING:
   - Background monitor shows job status every 30 seconds
   - Paused queries are saved in paused_queries/ directory
   - Completed job results saved in completed_jobs/ directory
   - All activity logged to original trace files

🛑 STOPPING THE SYSTEM:
   - Ctrl+C in background monitor terminal
   - Or: pkill -f background_monitor.py
   
🔧 TROUBLESHOOTING:
   - Check paused_queries/ for stuck queries
   - Check completed_jobs/ for job results
   - Background monitor logs show job status
   - Original trace files show complete query history
""")

def main():
    print("🚀 BV-BRC Concurrent Query Processing System")
    print("=" * 60)
    
    # Show current status
    show_system_status()
    
    # Start background monitor
    print(f"\n🔄 Starting background job monitor...")
    monitor_process = start_background_monitor()
    
    if monitor_process:
        print(f"\n✅ System is ready for concurrent query processing!")
        show_usage_instructions()
        
        print(f"\n🔍 Background monitor is running (PID: {monitor_process.pid})")
        print(f"📋 You can now run queries with: python P3_together.py")
        print(f"🛑 Stop the monitor with: Ctrl+C or kill {monitor_process.pid}")
        
        try:
            # Keep the script running to show status
            while True:
                time.sleep(60)  # Check every minute
                if monitor_process.poll() is not None:
                    print(f"\n❌ Background monitor stopped unexpectedly")
                    break
                    
        except KeyboardInterrupt:
            print(f"\n🛑 Stopping background monitor...")
            monitor_process.terminate()
            print(f"✅ System shutdown complete")
    
    else:
        print(f"\n❌ Failed to start background monitor")
        print(f"💡 You can still run queries normally, but no concurrent processing")

if __name__ == "__main__":
    main() 