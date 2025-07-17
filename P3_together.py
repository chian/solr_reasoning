from together import Together
import os
import time
import json
import re

from agent_parser import sanitize_filename
# Import specific MCP tool functions
from bvbrc_p3tools_server import (
    get_blast_results,
    get_workspace_info,
    check_bvbrc_auth,
    find_similar_genomes,
    submit_blast_job,
    check_job_status,
    submit_proteome_comparison,
    list_genomes,
    run_p3_tool
)

# Get API keys from environment variables
together_api_key = os.getenv('TOGETHER_API_KEY')
if not together_api_key:
    raise ValueError("Please set the TOGETHER_API_KEY environment variable")

# Configure model client
together_client = Together(api_key=together_api_key)

# Model configurations
R1_CONFIG = {
    "api_key": together_api_key,
    "endpoint": "https://api.together.xyz/v1",
    "model_id": "deepseek-ai/DeepSeek-R1",
    "client": together_client,
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "{user_query}"}
    ],
    "max_tokens": 32000,
    "temperature": 0.7
}

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

def execute_mcp_action(action_dict):
    tool_name = action_dict["tool_name"]
    params = action_dict["parameters"]
    # Dispatch to the correct MCP tool function
    tool_func = globals().get(tool_name)
    if not tool_func:
        return {"observation": f"Unknown tool: {tool_name}"}
    try:
        result = tool_func(**params)
        return {"observation": result}
    except Exception as e:
        return {"observation": f"Error executing tool {tool_name}: {e}"}

def get_prompt_template():
    """Load the prompt template from p3_prompt.txt"""
    try:
        with open("p3_prompt.txt", "r") as f:
            return f.read()
    except FileNotFoundError:
        print("Warning: p3_prompt.txt not found, using default prompt")
        return "You are an expert in bioinformatics and proficient with the BV-BRC P3-Tools MCP server."

#----------------------------------
# Main Execution

def main():
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
        llm_response = together_client.chat.completions.create(
            model=R1_CONFIG["model_id"],
            messages=messages,
            temperature=R1_CONFIG["temperature"],
            max_tokens=R1_CONFIG["max_tokens"]
        ).choices[0].message.content
        
        print(f"\nInitial LLM Response:\n{llm_response}")
        
        # Write the response to trace file
        with open(trace_file_path, "a", encoding="utf-8") as f:
            f.write(f"## Initial Response\n\n")
            f.write(f"### LLM Response\n{llm_response}\n\n")
        
        # Process actions one at a time, allowing LLM to update after each
        action_pattern = re.compile(r'<action>(.*?)</action>', re.DOTALL | re.IGNORECASE)
        action_blocks = action_pattern.findall(llm_response)
        
        if not action_blocks:
            # No actions, we're done
            print(f"\nNo actions found in response")
            with open(trace_file_path, "a", encoding="utf-8") as f:
                f.write(f"### Final Result\nNo actions found. Final response:\n{llm_response}\n\n")
        else:
            # Execute actions one by one
            for action_index, action_block in enumerate(action_blocks):
                action_block = action_block.strip()
                if not action_block:
                    continue
                
                print(f"\nExecuting action {action_index + 1}/{len(action_blocks)}")
                
                try:
                    action_json = json.loads(action_block)
                    if isinstance(action_json, dict) and "tool_name" in action_json:
                        # Execute the MCP tool call
                        result = execute_mcp_action(action_json)
                        observation_text = f"Action result: {result['observation']}"
                        
                        # Insert observation right after this action block
                        observation_tag = f"\n<observation>\n{observation_text}\n</observation>\n"
                        action_start = llm_response.find(f"<action>\n{action_block}\n</action>")
                        if action_start != -1:
                            action_end = action_start + len(f"<action>\n{action_block}\n</action>")
                            llm_response = llm_response[:action_end] + observation_tag + llm_response[action_end:]
                        
                        print(f"Action result: {observation_text}")
                        
                        # Write action execution to trace file
                        with open(trace_file_path, "a", encoding="utf-8") as f:
                            f.write(f"### Action {action_index + 1} Execution\n")
                            f.write(f"<action>\n{action_block}\n</action>\n")
                            f.write(f"<observation>\n{observation_text}\n</observation>\n\n")
                        
                        # If there are more actions, let the LLM update its response
                        if action_index < len(action_blocks) - 1:
                            print(f"\nUpdating LLM response with action result...")
                            
                            # Send updated response back to LLM to potentially modify future actions
                            update_messages = [
                                {"role": "system", "content": prompt_template},
                                {"role": "user", "content": f"{user_query}\n\nCurrent reasoning and results:\n{llm_response}"}
                            ]
                            
                            updated_llm_response = together_client.chat.completions.create(
                                model=R1_CONFIG["model_id"],
                                messages=update_messages,
                                temperature=R1_CONFIG["temperature"],
                                max_tokens=R1_CONFIG["max_tokens"]
                            ).choices[0].message.content
                            
                            print(f"\nUpdated LLM Response:\n{updated_llm_response}")
                            
                            # Write updated response to trace file
                            with open(trace_file_path, "a", encoding="utf-8") as f:
                                f.write(f"### Updated Response After Action {action_index + 1}\n{updated_llm_response}\n\n")
                            
                            # Update the response and re-extract remaining actions
                            llm_response = updated_llm_response
                            action_blocks = action_pattern.findall(llm_response)
                            
                    else:
                        error_msg = f"Invalid action format: {action_block}"
                        print(f"Error: {error_msg}")
                        
                except json.JSONDecodeError as e:
                    error_msg = f"Invalid JSON in action: {action_block}"
                    print(f"Error: {error_msg}")
            
            # Write final result to trace file
            with open(trace_file_path, "a", encoding="utf-8") as f:
                f.write(f"### Final Result\n{llm_response}\n\n")
        
        print(f"\nCompleted reasoning")
        print(f"Final response:\n{llm_response}")
        
        # Create a simple training file
        training_file_path = os.path.join(query_dir, f"training_{int(time.time())}.json")
        training_data = {
            "query": user_query,
            "iterations": 1, # Single reasoning pass
            "conversation_history": [], # No conversation history in this single trace
            "final_reasoning": llm_response, # Use the final LLM response as reasoning
            "timestamp": time.time()
        }
        
        with open(training_file_path, "w", encoding="utf-8") as f:
            json.dump(training_data, f, indent=2)
        
        print(f"Training data saved to: {training_file_path}")
        print(f"Completed processing query: {user_query}")
        print("-" * 80)


#----------------------------------
# Entry Point

if __name__ == "__main__":
    main()

