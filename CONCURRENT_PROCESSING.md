# Concurrent Query Processing System

This system enables concurrent processing of multiple queries while computational jobs (MSA, BLAST, etc.) run in the background.

## 🚀 Quick Start

### Single Query Mode (Recommended)
```bash
# Activate environment and run a single query
conda activate py312
python P3_together.py "Find hemagglutinin mutations in H1N1 isolates"
```

### Batch Processing Mode
```bash
# Process all queries from queries.json
conda activate py312
python batch_query_manager.py
```

### Background System
```bash
# Start the concurrent system (optional - auto-started by batch manager)
python start_concurrent_system.py
```

## 🔄 How It Works

### Single Query Processing
```bash
conda activate py312
python P3_together.py "your query here"
```

- Each query runs independently in its own process
- When `p3_submit_*` tools are called, the query automatically pauses
- System saves query state and exits
- Background monitor resumes query when job completes
- No query collision - each query is separate

### Automatic Job Handling

1. **Job Submission**: System detects `p3_submit_msa`, `p3_submit_blast`, etc.
2. **Query Pause**: Current query state saved with unique ID  
3. **Process Exit**: Query process exits cleanly
4. **Background Monitoring**: Monitor checks job status every 30 seconds
5. **Automatic Resume**: New process started when job completes

### Concurrent Processing Example

```bash
# Terminal 1
conda activate py312
python P3_together.py "Run MSA on dengue sequences"  # → pauses for job

# Terminal 2 (immediately)
conda activate py312  
python P3_together.py "Find genes in E. coli 83333.111"  # → completes immediately

# Terminal 3 (immediately)
conda activate py312
python P3_together.py "Compare Salmonella proteomes"  # → completes immediately

# Background: Terminal 1 query resumes automatically when MSA completes
```

## 📁 File Structure

```
project/
├── P3_together.py              # Main processor (single query mode)
├── batch_query_manager.py      # Batch processing wrapper
├── background_monitor.py       # Job status monitor
├── query_state_manager.py      # State persistence
├── start_concurrent_system.py  # System startup (optional)
├── queries.json                # Query list for batch mode
├── paused_queries/             # Saved query states
│   ├── query_abc123.json      # Individual paused queries
│   └── query_def456.json
├── completed_jobs/             # Job results
│   ├── job_789.json           # Completed job data
│   └── job_012.json
└── query_results/              # Normal trace files
    ├── query_1_results/
    └── query_2_results/
```

## 🎯 Usage Modes

### 1. Interactive Single Query
```bash
conda activate py312
python P3_together.py "analyze genome features in Bacillus subtilis"
```
**Best for**: Ad-hoc analysis, testing, interactive work

### 2. Batch Processing (Sequential)
```bash
conda activate py312
python batch_query_manager.py
```
**Best for**: Processing queries.json list, production runs

### 3. Batch Processing (Concurrent - Experimental)
```bash
conda activate py312
python batch_query_manager.py --concurrent --workers 3
```
**Best for**: High-throughput processing (use with caution)

### 4. System Status Check
```bash
conda activate py312
python batch_query_manager.py --status
```
**Shows**: Running jobs, paused queries, system health

## 🔧 Command Reference

### P3_together.py
```bash
# Single query
python P3_together.py "your query here"

# Resume paused query  
python P3_together.py --resume abc123

# Legacy batch mode (fallback)
python P3_together.py  # reads queries.json
```

### batch_query_manager.py
```bash
# Sequential processing (default)
python batch_query_manager.py

# Concurrent processing
python batch_query_manager.py --concurrent --workers 3

# Custom conda environment
python batch_query_manager.py --conda-env py311

# Skip background monitor
python batch_query_manager.py --no-monitor

# Status check only
python batch_query_manager.py --status
```

### background_monitor.py
```bash
# Manual start (usually auto-started)
python background_monitor.py
```

## 📋 Monitoring

### Batch Manager Output
```
🚀 BV-BRC Batch Query Manager
📋 Loaded 5 queries from queries.json
✅ Background monitor already running
🔄 Processing 5 queries sequentially...

📋 Query 1/5
🎯 Find hemagglutinin mutations in H1N1 isolates...
⏸️  Query paused for computational job (15.2s)
   💼 Job submitted, query will resume automatically

📋 Query 2/5  
🎯 Analyze E. coli genome features...
✅ Query completed successfully (8.3s)
```

### Background Monitor Output
```
🔍 Background job monitor started
📋 Found 2 paused queries
🔍 Checking job job_123 for query abc123...
⏳ Job job_123 still running...
🎉 Job job_456 completed! Getting results...
✅ Query def456 resumed successfully
```

## 🛑 System Management

### Start System
```bash
# Option 1: Full system startup
python start_concurrent_system.py

# Option 2: Just batch processing (auto-starts monitor)
python batch_query_manager.py
```

### Stop System
```bash
# Stop background monitor
pkill -f background_monitor.py

# Or Ctrl+C in monitor terminal
```

### Check Status
```bash
# Quick status
python batch_query_manager.py --status

# Detailed file check
ls paused_queries/     # Active paused queries
ls completed_jobs/     # Finished job results  
ls query_results/      # Query output directories
```

## 🔧 Troubleshooting

### Environment Issues
```bash
# Check conda environment
conda env list
conda activate py312
which python

# Test single query
python P3_together.py "test query"
```

### Stuck Queries
```bash
# Check paused queries
ls -la paused_queries/

# Manually resume a query
conda activate py312
python P3_together.py --resume abc123

# Clean up stuck queries
rm paused_queries/query_*.json
```

### Job Issues
```bash
# Check job status
ls -la completed_jobs/

# Monitor logs
# Check terminal where background_monitor.py is running

# Restart background monitor
pkill -f background_monitor.py
python background_monitor.py
```

## 🎯 Benefits

1. **No Query Collisions**: Each query runs independently
2. **Proper Environment**: Each process activates conda environment
3. **Automatic Job Handling**: Jobs tracked and resumed automatically
4. **Flexible Usage**: Single queries, batch processing, or mixed
5. **Resource Efficient**: Only active queries consume resources
6. **Full Traceability**: All activity logged to trace files

## ⚡ Performance

- **Single Queries**: Start immediately, no queue conflicts
- **Batch Processing**: Sequential by default, concurrent optional
- **Job Handling**: Automatic pause/resume, no manual intervention
- **Resource Usage**: One process per active query + background monitor

## 🔄 Migration from Old System

### Old Way
```bash
# All queries in one process, blocking on jobs
python P3_together.py  # processes entire queries.json sequentially
```

### New Way
```bash
# Flexible, concurrent-capable system
python batch_query_manager.py              # batch processing
python P3_together.py "single query"       # individual queries
```

## 🚀 Advanced Usage

### Custom Batch Scripts
```python
import subprocess
import sys

queries = [
    "Analyze H1N1 hemagglutinin evolution",
    "Find antibiotic resistance in E. coli",
    "Compare Salmonella virulence factors"
]

for query in queries:
    cmd = f"conda activate py312 && python P3_together.py '{query}'"
    subprocess.run(cmd, shell=True)
```

### Monitoring Integration
```python
from query_state_manager import list_active_queries, get_paused_queries

# Check system status programmatically
active = list_active_queries()
paused = get_paused_queries()

print(f"Active queries: {len(active)}")
print(f"Paused queries: {len(paused)}")
```

This system transforms the monolithic query processor into a flexible, concurrent-capable platform that handles both interactive and batch workloads efficiently. 