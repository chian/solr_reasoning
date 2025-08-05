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
        """Send MCP initialization message"""
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
        self._send_message(init_msg)
    
    def _list_tools(self):
        """Get list of available tools"""
        list_msg = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list"
        }
        response = self._send_message(list_msg)
        if response and "result" in response:
            self.available_tools = response["result"].get("tools", [])
    
    def call_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Call a tool on the MCP server"""
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
        
        response = self._send_message(message)
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
from helper_functions import evaluate_solution_with_llm, remove_think_tags, query_llm, fix_failed_command_with_llm

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

def update_commands_with_llm(user_query, actions, action_index, observation_text, classification):
    """
    Uses the LLM to update future actions based on the current action's output.
    
    Args:
        user_query (str): The original user query
        actions (list): List of action objects
        action_index (int): Index of the current action
        observation_text (str): Output from the command execution
        classification (str): The classification of the command output
        client: The LLM client to use
    
    Returns:
        tuple: (should_update, updated_actions, update_response)
            - should_update (bool): Whether the actions should be updated
            - updated_actions (list): Updated list of actions
            - update_response (str): Response from the LLM about the update
    """
    # Get the current action for context
    current_action = actions[action_index]
    
    # Get future actions to update
    future_actions = actions[action_index + 1:] if action_index + 1 < len(actions) else []
    
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
    1. **Use actual data from current execution**: Replace placeholder values with real data from the current step's output
       - If current step returned genome IDs, use those actual IDs in future steps
       - If current step returned sequences, use those actual sequences in future steps
    
    2. **Handle dependencies**: 
       - If current step succeeded, update future steps with the actual output data
    
    3. **Maintain correct parameter formats**: 
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
    - Use actual data from the current step's output
    - If current step failed, remove dependent future actions
    - Genome IDs must be in "number.number" format
    - Use valid workspace paths starting with "/rbutler@bvbrc/"
    
    Return ONLY the JSON array of updated MCP tool calls, nothing else.
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
    
    # Parse actions from the LLM response
    actions, action_count_message = parse_actions_from_response(next_cmd)
    
    if not actions:
        return False, actions, f"No valid actions found: {action_count_message}"
    
    # Create updated actions list
    updated_actions = actions.copy()
    
    # Update future actions with the new actions
    future_action_count = len(actions) - action_index - 1
    for i, action in enumerate(actions[:future_action_count]):
        if action_index + 1 + i < len(updated_actions):
            updated_actions[action_index + 1 + i] = action
    
    return True, updated_actions, "Future actions updated based on execution result"

def parse_actions_from_response(response_text):
    """Extract and parse action JSON from LLM response"""
    action_pattern = re.compile(r'<action>(.*?)</action>', re.DOTALL | re.IGNORECASE)
    action_blocks = action_pattern.findall(response_text)
    
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

def main():
    # load queries from queries file
    try:
        with open('queries.json','r') as i:
            queries = json.load(i)
    except FileNotFoundError:
        print("Error: queries.json not found. Please create this file with a list of queries.")
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
        
        # Create a sanitized subfolder name based on the query
        query_subfolder = sanitize_filename(user_query)
        
        # Create directory for this specific query's results
        query_dir = os.path.join(main_results_dir, query_subfolder)
        os.makedirs(query_dir, exist_ok=True)
        
        # Check if a training file already exists in this directory
        training_files = [f for f in os.listdir(query_dir) if f.startswith("training_")]
        if training_files:
            print(f"Skipping query '{user_query}' - training file already exists")
            continue
        
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
        
        # Write the response to trace file
        with open(trace_file_path, "a", encoding="utf-8") as f:
            f.write(f"## Initial Response\n\n")
            f.write(f"### LLM Response\n{llm_response}\n\n")
        
        # Remove thinking sections before processing actions
        llm_response_clean = remove_think_tags(llm_response)
        
        # Process actions one at a time, allowing LLM to update after each
        actions, action_count_message = parse_actions_from_response(llm_response_clean)
        print(f"\n{action_count_message}")
        
        # Write action parsing results to trace file
        with open(trace_file_path, "a", encoding="utf-8") as f:
            f.write(f"### Action Parsing Results\n")
            f.write(f"{action_count_message}\n")
            for i, action in enumerate(actions):
                action_info = f"Action {i+1} content: '{json.dumps(action, indent=2)}...' (length: {len(json.dumps(action))})"
                print(action_info)
                f.write(f"{action_info}\n")
            f.write(f"\n")
        
        for i, action in enumerate(actions):
            print(f"Action {i+1} content: '{json.dumps(action, indent=2)}...' (length: {len(json.dumps(action))})")
        
        if not actions:
            # No actions, we're done
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
                
                # Write action execution to trace file
                with open(trace_file_path, "a", encoding="utf-8") as f:
                    f.write(f"### Action {action_index + 1} Execution\n")
                    f.write(f"Action: {json.dumps(action, indent=2)}\n")
                    f.write(f"Action result: {observation_text}\n")
                    f.write(f"Raw output: {result['raw_output']}\n")
                    f.write(f"Status: {result['status']}\n\n")
                
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
                            f.write(f"LLM suggested fix, updating action and retrying...\n\n")
                        
                        # Update the action and retry by decrementing the index
                        actions[action_index] = fixed_action
                        action_index -= 1
                        continue
                    else:
                        print("LLM could not fix the command")
                        
                        with open(trace_file_path, "a", encoding="utf-8") as f:
                            f.write(f"LLM could not fix the command\n\n")
                        
                        # Exit the query since we couldn't fix it
                        print(f"Command failed and could not be fixed. Exiting query: {user_query}")
                        with open(trace_file_path, "a", encoding="utf-8") as f:
                            f.write(f"### Query Failed\nCommand failed and could not be fixed. Exiting query.\n\n")
                        break  # Exit the action loop for this query
                
                # Add action_output to the action for the final response
                action["action_output"] = result['observation']
                
                print(f"Added action_output to action {action_index + 1}")
                
                # Write action_output addition to trace file
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
                

                    
            # Write final result to trace file
            with open(trace_file_path, "a", encoding="utf-8") as f:
                f.write(f"### Final Result\n{llm_response}\n\n")
        
        print(f"\nCompleted reasoning")
        print(f"Final response:\n{llm_response}")
        
        # Evaluate the solution with the LLM
        solution_success, evaluation_text = evaluate_solution_with_llm(user_query, llm_response, OPENAI_CONFIG)
        print(f"\nSolution Evaluation: {evaluation_text}")
        
        # Only create training file for successful solutions
        if solution_success:
            # Create a simple training file
            training_file_path = os.path.join(query_dir, f"training_{int(time.time())}.json")
            
            training_data = {
                "query": user_query,
                "iterations": 1, # Single reasoning pass
                "final_reasoning": llm_response, # Use the final LLM response as reasoning
                "evaluation": evaluation_text, # Include evaluation results
                "timestamp": time.time()
            }
            
            with open(training_file_path, "w", encoding="utf-8") as f:
                json.dump(training_data, f, indent=2)
            
            print(f"Training data saved to: {training_file_path}")
        else:
            print(f"No training file created - solution was not successful")
        
        print(f"Completed processing query: {user_query}")
        print("-" * 80)


#----------------------------------
# Entry Point

if __name__ == "__main__":
    main()

