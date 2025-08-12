from together import Together
import os
import time
import json
import re

from agent_parser import sanitize_filename
# MCP Client for communicating with BV-BRC MCP servers
import subprocess
import json
import asyncio
from typing import Dict, Any, Optional

class BVBRCMCPClient:
    """Client for communicating with BV-BRC MCP servers"""
    
    def __init__(self, server_script: str):
        self.server_script = server_script
        self.process = None
        self.available_tools = []
    
    def start_server(self):
        """Start the MCP server process and initialize connection"""
        if self.process is None or self.process.poll() is not None:
            self.process = subprocess.Popen(
                ["python", self.server_script],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Initialize MCP connection
            self._send_initialize()
            self._list_tools()
    
    def _send_initialize(self):
        """Send MCP initialization message and wait for response"""
        print("DEBUG: Sending initialize request...")
        init_msg = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "clientInfo": {
                    "name": "BVBRC-Client",
                    "version": "1.0.0"
                }
            }
        }
        response = self._send_message(init_msg)
        print(f"DEBUG: Initialize response: {response}")
        if not response:
            raise Exception("No response from initialize")
        if "error" in response:
            raise Exception(f"Initialize failed: {response['error']}")
        if "result" not in response:
            raise Exception(f"Invalid initialize response: {response}")
        
        # Send initialized notification (required by MCP protocol)
        print("DEBUG: Sending initialized notification...")
        initialized_msg = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }
        # Notifications don't expect responses, so send without waiting
        self.process.stdin.write(json.dumps(initialized_msg) + "\n")
        self.process.stdin.flush()
        print("DEBUG: Initialization complete")
    
    def _list_tools(self):
        """Get list of available tools"""
        print("DEBUG: Requesting tools list...")
        list_msg = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list"
        }
        response = self._send_message(list_msg)
        print(f"DEBUG: Tools list response: {response}")
        if response and "result" in response:
            self.available_tools = response["result"].get("tools", [])
            print(f"DEBUG: Found {len(self.available_tools)} tools")
        elif response and "error" in response:
            raise Exception(f"Failed to list tools: {response}")
    
    def call_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Call a tool on the MCP server"""
        print(f"DEBUG: Calling tool {tool_name} with args: {kwargs}")
        self.start_server()
        
        # Create MCP tool call message
        message = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": kwargs
            }
        }
        
        print(f"DEBUG: Sending tool call message: {message}")
        response = self._send_message(message)
        print(f"DEBUG: Tool call response: {response}")
        if response and "result" in response:
            return response["result"]
        elif response and "error" in response:
            return {"status": "error", "message": response["error"]}
        else:
            return {"status": "error", "message": "No response from server"}
    
    def _send_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send message to server and get response"""
        try:
            self.process.stdin.write(json.dumps(message) + "\n")
            self.process.stdin.flush()
            
            response_line = self.process.stdout.readline()
            if response_line:
                return json.loads(response_line)
            return None
        except Exception as e:
            return {"error": str(e)}

# Initialize MCP clients for different servers
data_retrieval_client = BVBRCMCPClient("P3_TOOLS_DATA_RETRIEVAL.py")
computational_client = BVBRCMCPClient("P3_TOOLS_COMPUTATIONAL.py")
utilities_client = BVBRCMCPClient("P3_TOOLS_UTILITIES.py")
rest_api_client = BVBRCMCPClient("BVBRC_API.py")

# Direct access to MCP server tools
# Use the clients directly to call tools on the MCP servers
from helper_functions import evaluate_solution_with_llm, remove_think_tags, query_llm, fix_failed_command_with_llm, generate_final_solution_with_llm

# Get API keys from environment variables
together_api_key = os.getenv('TOGETHER_API_KEY')
if not together_api_key:
    raise ValueError("Please set the TOGETHER_API_KEY environment variable")

openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

# Configure model clients
together_client = Together(api_key=together_api_key)
import openai
openai_client = openai.OpenAI(api_key=openai_api_key)

# Model configurations
R1_CONFIG = {
    "api_key": together_api_key,
    "endpoint": "https://api.together.xyz/v1",
    "model": "deepseek-ai/DeepSeek-R1",
    "client": together_client,
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "{user_query}"}
    ],
    "max_tokens": 32000,
    "temperature": 0.7
}

# Configuration for simpler parsing tasks
OPENAI_CONFIG = {
    "api_key": openai_api_key,
    "endpoint": "https://api.openai.com/v1",
    "model": "o3-mini",
    "client": openai_client,
    "messages": [
        {"role": "user", "content": "You are a helpful assistant.\n\n{user_query}"}
    ]
}

def execute_mcp_action(action_dict):
    tool_name = action_dict["tool_name"]
    params = action_dict["parameters"]
    
    # Map tool names to appropriate MCP clients
    tool_mapping = {
        # Data retrieval tools
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
        
        # Computational tools
        "p3_submit_blast": computational_client,
        "p3_submit_msa": computational_client,
        "p3_submit_gene_tree": computational_client,
        "p3_submit_codon_tree": computational_client,
        "p3_submit_genome_annotation": computational_client,
        "p3_submit_genome_assembly": computational_client,
        "p3_submit_proteome_comparison": computational_client,
        "p3_submit_variation_analysis": computational_client,
        "p3_submit_rnaseq": computational_client,
        "p3_submit_cga": computational_client,
        "p3_submit_fastqutils": computational_client,
        "p3_submit_taxonomic_classification": computational_client,
        "p3_submit_metagenome_binning": computational_client,
        "p3_submit_metagenomic_read_mapping": computational_client,
        "p3_submit_sars2_analysis": computational_client,
        "p3_submit_sars2_assembly": computational_client,
        "p3_check_job_status": computational_client,
        "p3_get_job_results": computational_client,
        
        # Utility tools
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
        
        # REST API tools
        "get_genomes_by_species": rest_api_client,
        "get_complete_genomes": rest_api_client,
        "get_genome_features": rest_api_client,
        "search_features_by_product": rest_api_client,
    }
    
    # Get the appropriate client
    client = tool_mapping.get(tool_name)
    if not client:
        return {"observation": f"Unknown tool: {tool_name}", "raw_output": f"Unknown tool: {tool_name}", "status": "error"}
    
    try:
        # Call the tool on the MCP server
        result = client.call_tool(tool_name, **params)
        
        # Check if this is a computational job submission
        if tool_name.startswith("p3_submit_"):
            try:
                if isinstance(result, str):
                    result_json = json.loads(result)
                else:
                    result_json = result
                
                if "job_id" in result_json:
                    # This is a job submission - return special status for pause handling
                    return {
                        "observation": f"Job submitted successfully. Job ID: {result_json['job_id']}",
                        "raw_output": result,
                        "status": "job_submitted",
                        "job_id": result_json["job_id"],
                        "job_type": tool_name
                    }
            except (json.JSONDecodeError, TypeError):
                # If parsing fails, continue with normal processing
                pass
        
        # Normal processing for non-job tools
        # Always parse the raw output into natural language
        if isinstance(result, str):
            # Split into lines for parsing
            result_lines = result.split('\n')
            parsed_result = parse_last_command_output_with_llm(result_lines)
        else:
            # If result is not a string, convert to string first
            result_str = str(result)
            result_lines = result_str.split('\n')
            parsed_result = parse_last_command_output_with_llm(result_lines)
        
        # Extract status from the result
        if isinstance(result, str):
            try:
                result_json = json.loads(result)
                status = result_json.get("status", "unknown")
            except json.JSONDecodeError:
                status = "unknown"
        else:
            status = "unknown"
        
        return {"observation": parsed_result, "raw_output": result, "status": status}
    except Exception as e:
        return {"observation": f"Error executing tool {tool_name}: {e}", "raw_output": f"Error executing tool {tool_name}: {e}", "status": "error"}

def get_prompt_template():
    """Load the prompt template from p3_prompt.txt and populate tool reference"""
    try:
        with open("p3_prompt.txt", "r") as f:
            prompt_template = f.read()
        
        # Load complete tool reference if placeholder exists
        if "{COMPLETE_TOOL_REFERENCE}" in prompt_template:
            try:
                with open("complete_tool_reference.txt", "r") as f:
                    complete_tools = f.read()
                prompt_template = prompt_template.replace("{COMPLETE_TOOL_REFERENCE}", complete_tools)
                print(f"✅ Loaded complete tool reference with 93 tools")
            except FileNotFoundError:
                print("Warning: complete_tool_reference.txt not found, placeholder will remain")
        
        return prompt_template
    except FileNotFoundError:
        print("Warning: p3_prompt.txt not found, using default prompt")
        return "You are an expert in bioinformatics and proficient with the BV-BRC P3-Tools MCP server."



def parse_last_command_output_with_llm(terminal_output_lines):
    """
    Takes in the command output and uses an LLM to extract relevant information.

    Parameters:
        terminal_output_lines (list of str): The lines captured from the command output.

    Returns:
        str: The LLM's best guess at the parsed output.
    """
    output_str = "\n".join(terminal_output_lines)

    prompt = f"""You are an expert in bioinformatics. Below is the raw 
        output from a solr query execution:

        {output_str}

        Your task is to extract and return the relevant information from the output.

        Output:
        """
    print("Parsing Command Output with LLM...")
    
    # Create a proper model config
    model_config = {
        "model": OPENAI_CONFIG["model"],
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    }
    
    return query_llm(OPENAI_CONFIG, prompt, model_config)

def update_commands_with_llm(user_query, original_actions, action_index, observation_text, classification):
    """
    Uses the LLM to update future actions based on the current action's output.
    
    Args:
        user_query (str): The original user query
        original_actions (list): List of action objects
        action_index (int): Index of the current action
        observation_text (str): Output from the command execution
        classification (str): The classification of the command output
    
    Returns:
        tuple: (should_update, updated_actions, update_response)
            - should_update (bool): Whether the actions should be updated
            - updated_actions (list): Updated list of actions
            - update_response (str): Response from the LLM about the update
    """
    # Get the current action for context
    current_action = original_actions[action_index]
    
    # Get future actions to update
    future_actions = original_actions[action_index + 1:] if action_index + 1 < len(original_actions) else []
    
    prompt = f"""
    Based on the following MCP tool execution:
    
    User Query: "{user_query}"
    
    Current MCP tool executed:
    {json.dumps(current_action, indent=2)}
    
    Tool execution result:
    ```
    {observation_text}
    ```
    
    Execution status: {classification}
    
    Future MCP tools to potentially update:
    {json.dumps(future_actions, indent=2)}
    
    TASK: Update the future MCP tool calls based on the current execution result.
    
    UPDATE GUIDELINES:
    1. **Replace ALL hardcoded and placeholder values**: 
       - Look for ANY hardcoded genome IDs, feature IDs, or other values in future actions
       - Replace them with actual data from the current step's output
       - If current step returned genome IDs like ["2861289.3", "3003731.3"], use THOSE exact IDs
       - If current step returned feature IDs, use those exact feature IDs
       - NEVER keep hardcoded values like ["511145.12", "511145.15"] when real data is available
    
    2. **Detect placeholder patterns**: 
       - Replace "output from step1", "REPLACE_WITH_X", or similar placeholder text
       - Replace hardcoded arrays that don't match current step output
       - If current step output contains genome_ids, extract them and use in future steps
    
    3. **Handle dependencies**: 
       - If current step succeeded, update ALL future steps that depend on its output
       - If current step failed, remove or modify dependent future actions
    
    4. **Maintain correct parameter formats**: 
       - Genome IDs must be format "number.number" (e.g., "83333.111")
       - Feature IDs must be format "fig|genome_id.peg.number"
       - Workspace paths should start with "/rbutler@bvbrc/"
    
    EXAMPLE OF CORRECT MCP TOOL FORMAT:
    [
    {{
        "tool_name": "p3_all_genomes",
        "parameters": {{
            "species": "Hepacivirus hominis",
            "limit": 10,
            "attributes": ["genome_id", "genome_name"]
        }}
    }},
    {{
        "tool_name": "p3_get_genome_contigs",
        "parameters": {{
            "genome_ids": ["83333.111", "83333.112", "83333.113"]
        }}
    }}
    ]
    
    CRITICAL RULES:
    - NEVER use placeholder text like "REPLACE_WITH_X" 
    - ALWAYS replace hardcoded values that should come from previous steps
    - Use actual data from the current step's output
    - If current step failed, remove dependent future actions
    - Genome IDs must be in "number.number" format
    - Use valid workspace paths starting with "/rbutler@bvbrc/"
    
    EXAMPLE SCENARIO:
    If current step returned genome IDs: ["2861289.3", "3003731.3"]
    And future action has placeholder: ["RESULT_FROM_FIRST_ACTION"]
    Then you MUST return:
    
    <action>
    [
      {{
        "tool_name": "p3_get_genome_features",
        "parameters": {{
          "genome_ids": ["2861289.3", "3003731.3"],
          "attributes": ["feature_id", "genome_id", "product", "start", "end"],
          "feature_type": "CDS"
        }}
      }}
    ]
    </action>
    
    Return the updated actions wrapped in <action> tags as shown above.
    """
    
    # Create a proper model config
    model_config = {
        "model": "deepseek-ai/DeepSeek-R1",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 32000,
        "temperature": 0.7
    }
    
    next_cmd = query_llm(R1_CONFIG, prompt, model_config)
    
    # Use the existing parse_actions_from_response function
    new_actions, action_count_message = parse_actions_from_response(next_cmd)
    
    if not new_actions:
        return False, original_actions, f"No valid actions found: {action_count_message}"
    
    # Create updated actions list starting with original
    updated_actions = original_actions.copy()
    
    # Replace future actions with the new ones from LLM
    for i, new_action in enumerate(new_actions):
        future_index = action_index + 1 + i
        if future_index < len(updated_actions):
            updated_actions[future_index] = new_action
    
    return True, updated_actions, "Future actions updated based on execution result"

def parse_actions_from_response(response_text):
    """Extract and parse action JSON from LLM response"""
    # First, remove all content within <think></think> tags to avoid parsing actions from reasoning
    think_pattern = re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE)
    cleaned_response = think_pattern.sub('', response_text)
    
    action_pattern = re.compile(r'<action>(.*?)</action>', re.DOTALL | re.IGNORECASE)
    action_blocks = action_pattern.findall(cleaned_response)
    
    if not action_blocks:
        return [], "No action blocks found"
    
    actions = []
    for action_block in action_blocks:
        # Strip out comments before parsing JSON
        cleaned_block = re.sub(r'#.*$', '', action_block, flags=re.MULTILINE)  # Remove # comments
        cleaned_block = re.sub(r'//.*$', '', cleaned_block, flags=re.MULTILINE)  # Remove // comments
        cleaned_block = re.sub(r',\s*}', '}', cleaned_block)  # Remove trailing commas before }
        cleaned_block = re.sub(r',\s*]', ']', cleaned_block)  # Remove trailing commas before ]
        
        try:
            action_json = json.loads(cleaned_block.strip())
            if isinstance(action_json, dict) and "tool_name" in action_json and "parameters" in action_json:
                actions.append(action_json)
            elif isinstance(action_json, list):
                # Handle arrays of actions
                for action in action_json:
                    if isinstance(action, dict) and "tool_name" in action and "parameters" in action:
                        actions.append(action)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Cleaned block: {cleaned_block}")
            continue
    
    return actions, f"Found {len(actions)} valid actions"

#----------------------------------
# Main Execution

def resume_paused_query(query_id: str):
    return

def main():
    import sys
    
    # Check for single query argument
    if len(sys.argv) > 1 and sys.argv[1] != "--resume":
        # Single query mode
        user_query = sys.argv[1]
        print(f"🎯 Processing single query: {user_query}")
        process_single_query(user_query)
        return
    
    # Fallback: Batch mode from queries.json (legacy support)
    print("⚠️  No query provided. Falling back to batch mode from queries.json")
    print("💡 For concurrent processing, use: python P3_together.py 'your query here'")
    
    try:
        with open('queries.json','r') as i:
            queries = json.load(i)
    except FileNotFoundError:
        print("Error: queries.json not found.")
        print("Usage: python P3_together.py 'your query here'")
        exit(1)
    except json.JSONDecodeError:
        print("Error: queries.json is not valid JSON. Please check the file format.")
        exit(1)
    
    # Create the main query_results directory if it doesn't exist
    main_results_dir = "query_results"
    os.makedirs(main_results_dir, exist_ok=True)
    print(f"Total queries to process: {len(queries)}")
    
    for user_query in queries:
        print(f"Processing: {user_query}")
        process_single_query(user_query)

def process_single_query(user_query: str):
    """Process a single query (extracted from main for reuse)"""
    # Create a sanitized subfolder name based on the query
    query_subfolder = sanitize_filename(user_query)

    # Create directory for this specific query's results
    main_results_dir = "query_results"
    os.makedirs(main_results_dir, exist_ok=True)
    query_dir = os.path.join(main_results_dir, query_subfolder)
    os.makedirs(query_dir, exist_ok=True)

    # Check if a training file already exists in this directory
    training_files = [f for f in os.listdir(query_dir) if f.startswith("training_")]
    if training_files:
        print(f"Skipping query '{user_query}' - training file already exists")
        return

    # Create the trace file at the beginning
    trace_file_path = os.path.join(query_dir, "complete_trace.txt")

    # Start writing to the trace file
    with open(trace_file_path, "w", encoding="utf-8") as f:
        f.write(f"# User Query\n{user_query}\n\n")

    print(f"Processing query: {user_query}")
    print(f"Saving results to: {trace_file_path}")

    # Load the prompt template
    prompt_template = get_prompt_template()

    # Single reasoning pass with incremental action execution
    current_user_query = user_query

    with open(trace_file_path, "a", encoding="utf-8") as f:
        f.write(f"# Single Reasoning Pass with Incremental Actions\n\n")

    # Build messages for the LLM
    messages = [
        {"role": "system", "content": prompt_template},
        {"role": "user", "content": current_user_query}
    ]

    # Generate initial LLM response with reasoning and actions
    model_config = {
        "model": R1_CONFIG["model"],
        "messages": messages,
        "max_tokens": R1_CONFIG["max_tokens"],
        "temperature": R1_CONFIG["temperature"]
    }
    llm_response = query_llm(R1_CONFIG, "", model_config)

    print(f"\nInitial LLM Response:\n{llm_response}")

    # Write LLM response to trace file
    with open(trace_file_path, "a", encoding="utf-8") as f:
        f.write(f"## Initial LLM Response\n{llm_response}\n\n")

    # Parse actions from the LLM response
    actions, action_count_message = parse_actions_from_response(llm_response)

    print(f"\n{action_count_message}")

    # Write parsed actions to trace file
    with open(trace_file_path, "a", encoding="utf-8") as f:
        f.write(f"## Parsed Actions\n{action_count_message}\n")
        for i, action in enumerate(actions):
            f.write(f"Action {i+1}: {json.dumps(action, indent=2)}\n")
        f.write(f"\n")

    if not actions:
        print(f"\nNo actions found in response")
        with open(trace_file_path, "a", encoding="utf-8") as f:
            f.write(f"### Final Result\nNo actions found. Final response:\n{llm_response}\n\n")
    else:
        # Execute actions one by one
        for action_index, action in enumerate(actions):
            if not action:
                continue
                
            # Execute the MCP tool call
            result = execute_mcp_action(action)
            observation_text = f"Action result: {result['observation']}"
            
            print(f"Action result: {observation_text}")
            
            # Add the parsed observation to the action for structured response
            actions[action_index]['action_output'] = result['observation']
            
            # Add observation to the llm_response for complete evaluation
            observation_block = f"\n<observation>\n{result['observation']}\n</observation>\n"
            llm_response += observation_block
            
            # Write action execution to trace file
            with open(trace_file_path, "a", encoding="utf-8") as f:
                f.write(f"### Action {action_index + 1} Execution\n")
                f.write(f"Action: {json.dumps(action, indent=2)}\n")
                f.write(f"Action result: {observation_text}\n")
                f.write(f"Raw output: {result['raw_output']}\n")
                f.write(f"Status: {result['status']}\n\n")
            
            # Check if this action submitted a computational job
            if result['status'] == 'job_submitted':
                job_id = result['job_id']
                job_type = result['job_type']
                print(f"🔄 Computational job submitted: {job_id} ({job_type})")
                
                # Write submission info to trace
                with open(trace_file_path, "a", encoding="utf-8") as f:
                    f.write(f"### Job Submitted\n")
                    f.write(f"Job ID: {job_id}\n")
                    f.write(f"Job Type: {job_type}\n")
                    f.write(f"Waiting for job to complete...\n\n")
                
                # Poll for completion synchronously
                while True:
                    status_result = computational_client.call_tool("p3_check_job_status", job_id=job_id)
                    if isinstance(status_result, str):
                        status_json = json.loads(status_result)
                    else:
                        status_json = status_result
                    state = status_json.get("status")
                    print(f"⏳ Job {job_id} status: {state}")
                    if state in ["completed", "failed"]:
                        break
                    time.sleep(10)
                
                if state == "completed":
                    job_results = computational_client.call_tool("p3_get_job_results", job_name=job_id)
                    if isinstance(job_results, str):
                        job_results_str = job_results
                    else:
                        job_results_str = str(job_results)
                    
                    # Record completion and results
                    with open(trace_file_path, "a", encoding="utf-8") as f:
                        f.write(f"### Job Completed\n")
                        f.write(f"Job ID: {job_id}\n")
                        f.write(f"Job Results: {job_results_str[:200]}...\n\n")
                    
                    # Attach results to the action for downstream steps
                    actions[action_index]["job_results"] = job_results_str
                    
                    # Continue to next action
                    continue
                else:
                    with open(trace_file_path, "a", encoding="utf-8") as f:
                        f.write(f"### Job Failed\n")
                        f.write(f"Job ID: {job_id}\n")
                        f.write(f"Status: {json.dumps(status_json)}\n\n")
                    print(f"❌ Job {job_id} failed")
                    break
            
            # If the action failed, attempt to fix it with LLM
            if result['status'] == 'error':
                print(f"Command failed: {result['observation']}")
                print("Attempting to fix the command with LLM...")
                
                # Write error recovery attempt to trace file
                with open(trace_file_path, "a", encoding="utf-8") as f:
                    f.write(f"### Error Recovery Attempt\n")
                    f.write(f"Command failed: {result['observation']}\n")
                    f.write(f"Original action: {json.dumps(action, indent=2)}\n")
                    f.write(f"Attempting to fix the command with LLM...\n\n")
                
                # Try to fix the command with LLM
                fixed_action, fix_response = fix_failed_command_with_llm("", action, result['observation'], "error", R1_CONFIG)
                
                with open(trace_file_path, "a", encoding="utf-8") as f:
                    f.write(f"LLM fix response: {fix_response}\n")
                    f.write(f"Suggested fix: {json.dumps(fixed_action, indent=2)}\n\n")
                
                if fixed_action and fixed_action != action:
                    print(f"LLM suggested fix: {fixed_action}")
                    
                    with open(trace_file_path, "a", encoding="utf-8") as f:
                        f.write(f"Applying suggested fix...\n")
                    
                    # Update the action and retry by decrementing the index
                    actions[action_index] = fixed_action
                    
                    # Retry the fixed action
                    result = execute_mcp_action(fixed_action)
                    observation_text = f"Fixed action result: {result['observation']}"
                    
                    print(f"Fixed action result: {observation_text}")
                    
                    # Write retry result to trace file
                    with open(trace_file_path, "a", encoding="utf-8") as f:
                        f.write(f"### Fixed Action Execution\n")
                        f.write(f"Fixed action: {json.dumps(fixed_action, indent=2)}\n")
                        f.write(f"Fixed action result: {observation_text}\n")
                        f.write(f"Raw output: {result['raw_output']}\n")
                        f.write(f"Status: {result['status']}\n\n")
                else:
                    print("LLM could not suggest a fix or suggested the same action")
                    with open(trace_file_path, "a", encoding="utf-8") as f:
                        f.write(f"No fix suggested or same action suggested. Continuing...\n\n")
            
            # Add action result to the action for future reference
            action['action_output'] = result['observation']
            
            with open(trace_file_path, "a", encoding="utf-8") as f:
                f.write(f"Added action_output to action {action_index + 1}\n")
            
            # Check if future actions need to be updated based on this result
            print(f"\nChecking if future actions need updates...")
            
            # Write update check to trace file
            with open(trace_file_path, "a", encoding="utf-8") as f:
                f.write(f"Checking if future actions need updates...\n")
            
            classification = result['status']  # Use the status from the MCP result
            
            should_update, updated_actions, update_response = update_commands_with_llm(
                user_query, actions, action_index, result['raw_output'], classification
            )
            
            # Write detailed update input/output to trace file for debugging
            with open(trace_file_path, "a", encoding="utf-8") as f:
                f.write(f"### ACTION UPDATER DEBUG INFO ###\n")
                f.write(f"Input to action updater:\n")
                f.write(f"- User Query: {user_query}\n")
                f.write(f"- Current Action Index: {action_index}\n")
                f.write(f"- Current Action: {json.dumps(actions[action_index], indent=2)}\n")
                f.write(f"- Action Output: {result['raw_output']}\n")
                f.write(f"- Classification: {classification}\n")
                f.write(f"- Future Actions to Update: {json.dumps(actions[action_index + 1:], indent=2)}\n")
                f.write(f"\nOutput from action updater:\n")
                f.write(f"- Should Update: {should_update}\n")
                f.write(f"- Update Response: {update_response}\n")
                if should_update:
                    f.write(f"- Updated Actions: {json.dumps(updated_actions[action_index + 1:], indent=2)}\n")
                f.write(f"### END ACTION UPDATER DEBUG INFO ###\n\n")
            
            # Exit if there was a JSON parsing error
            if "Could not parse LLM response as JSON" in update_response:
                print("Exiting due to JSON parsing error in update_commands_with_llm")
                exit(1)
            
            if should_update and updated_actions:
                # Update future actions
                for action_idx in range(action_index + 1, len(updated_actions)):
                    if action_idx < len(actions):  # Only update existing actions
                        actions[action_idx] = updated_actions[action_idx]
                        print(f"Updated action {action_idx+1} based on execution result")
                
                print(f"Updated future actions based on execution result: {update_response}")
                
                # Write update results to trace file
                with open(trace_file_path, "a", encoding="utf-8") as f:
                    for action_idx in range(action_index + 1, len(updated_actions)):
                        if action_idx < len(actions):
                            f.write(f"Updated action {action_idx+1} based on execution result\n")
                    f.write(f"Updated future actions based on execution result: {update_response}\n")
            else:
                print("No action updates needed")
                
                # Write no updates to trace file
                with open(trace_file_path, "a", encoding="utf-8") as f:
                    f.write(f"No action updates needed\n")
            
            # Write update response to trace file
            with open(trace_file_path, "a", encoding="utf-8") as f:
                f.write(f"### Updated Response After Action {action_index + 1}\n{update_response}\n\n")
            

            
        # Build structured response for evaluation with updated actions and their outputs
        think_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL | re.IGNORECASE)
        think_match = think_pattern.search(llm_response)
        think_content = think_match.group(1) if think_match else ""
        
        # Build the structured response
        structured_response = f"<think>{think_content}</think>\n\n<action>\n[\n"
        
        for i, action in enumerate(actions):
            if i > 0:
                structured_response += ",\n"
            
            # Add action_output field with the parsed observation
            action_with_output = action.copy()
            if 'action_output' in action:
                # Already has output from execution
                pass
            else:
                # This shouldn't happen, but handle gracefully
                action_with_output['action_output'] = "No output recorded"
            
            structured_response += f"  {json.dumps(action_with_output, indent=2).replace(chr(10), chr(10) + '  ')}"
        
        structured_response += "\n]\n</action>\n\n"
        
        # Generate final solution using existing function
        print("Generating final solution synthesis...")
        successful_commands = []
        for action in actions:
            if 'action_output' in action:
                successful_commands.append({
                    'action_input': f"{action['tool_name']} with {action['parameters']}",
                    'action_output': action['action_output']
                })
        
        # Use OpenAI (non-thinking model like gpt-4o) for final solution synthesis
        final_solution = generate_final_solution_with_llm(user_query, successful_commands, OPENAI_CONFIG)
        
        structured_response += f"<solution>\n{final_solution}\n</solution>"
        
        # Write final result to trace file
        with open(trace_file_path, "a", encoding="utf-8") as f:
            f.write(f"### Final Result\nAll actions completed.\n\n")
            f.write(f"### Structured Response for Evaluation\n{structured_response}\n\n")

        # Evaluate the solution using the structured response
        print(f"\nEvaluating solution...")
        solution_success, evaluation_text = evaluate_solution_with_llm(user_query, structured_response, OPENAI_CONFIG)

        print(f"Solution evaluation: {evaluation_text}")

        # Write evaluation to trace file
        with open(trace_file_path, "a", encoding="utf-8") as f:
            f.write(f"## Solution Evaluation\n{evaluation_text}\n\n")

        # Create training file only if the solution was successful
        if solution_success:
            training_filename = f"training_{query_subfolder}.json"
            training_file_path = os.path.join(query_dir, training_filename)
            
            # Create structured JSON for ML pipelines (PPO/FT)
            training_data = {
                "prompt": user_query,
                "response": structured_response,
                "evaluation": "COMPLETE" if solution_success else "FAILURE",
                "timestamp": time.time()
            }
            
            with open(training_file_path, "w", encoding="utf-8") as f:
                json.dump(training_data, f, indent=2, ensure_ascii=False)
            
            print(f"Training file created: {training_file_path}")
            
            # Write training file creation to trace file
            with open(trace_file_path, "a", encoding="utf-8") as f:
                f.write(f"Training file created: {training_file_path}\n")
        else:
            print("No training file created - solution was not successful")
            
            # Write no training file to trace file
            with open(trace_file_path, "a", encoding="utf-8") as f:
                f.write(f"No training file created - solution was not successful\n")

        print(f"Completed processing query: {user_query}")
        print("-" * 80)


#----------------------------------
# Entry Point

if __name__ == "__main__":
    main()

