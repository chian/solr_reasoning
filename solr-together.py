from together import Together
import os
import pty
import sys
import platform
import subprocess
import time
from langchain_core.prompts import PromptTemplate
import textwrap
import psutil  # Make sure to install psutil using `pip install psutil`
import json
import re
import io
import openai  # Add OpenAI import for o1-pro model

from agent_parser import test_parse_commands, sanitize_filename, parse_commands, extract_commands_with_llm   
from helper_functions import query_llm, derive_solution_with_llm, classify_last_command_output_with_llm, parse_last_command_output_with_llm, generate_training_output, evaluate_solution_with_llm, generate_final_solution_with_llm, update_commands_with_llm, fix_failed_command_with_llm, remove_think_tags

# Get API keys from environment variables
together_api_key = os.getenv('TOGETHER_API_KEY')
if not together_api_key:
    raise ValueError("Please set the TOGETHER_API_KEY environment variable")

openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

# Configure model clients
together_client = Together(api_key=together_api_key)
openai_client = openai.OpenAI(api_key=openai_api_key)

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

OPENAI_CONFIG = {
    "api_key": openai_api_key,
    "endpoint": "https://api.openai.com/v1",
    "model_id": "o1-preview",
    "client": openai_client,
    "messages": [
        {"role": "user", "content": "You are a helpful assistant.\n\n{user_query}"}
    ],
    "max_tokens": 32000,
    "temperature": 0.7
}

# Task-specific model assignments
MODEL_CONFIGS = {
    "superprompt_generation": OPENAI_CONFIG,
    "command_execution": R1_CONFIG
}

#----------------------------------
# Removed BV-BRC client portion as Terminal interaction is no longer needed

# load queries from queries file
with open('queries.json','r') as i:
    queries = json.load(i)

# load template from template file
with open('solr_superprompt.txt','r') as i:
    cot_template_text = i.read()

# Create a template that handles the nested format
CoT_template = PromptTemplate.from_template(textwrap.dedent(cot_template_text))

def summarize_command_output(output_text, command, user_query, client):
    """
    Summarize the output of a command to include only relevant information for planning.
    
    Args:
        output_text (str): The full output of the command
        command (str): The command that was executed
        user_query (str): The original user query
        client: The LLM client to use for summarization
        
    Returns:
        str: A summarized version of the output with only relevant information
    """
    # If output is very small, no need to summarize
    if len(output_text) < 500:
        return output_text
        
    prompt = f"""Summarize the following command output to include ONLY information relevant for planning the next actions.
The original query is: "{user_query}"
The command executed was: "{command}"

Command output:
```
{output_text[:5000] if len(output_text) > 5000 else output_text}
```

If the output is too large, only the first portion is shown above.

Please provide a concise summary that:
1. Extracts key information relevant to the query
2. Mentions any errors or warnings (if present)
3. Highlights any IDs, values, or patterns that might be needed for follow-up commands
4. Omits irrelevant details, formatting, or verbose logging

Your summary should be much shorter than the original output while maintaining all critical information needed to plan next steps.
"""
    
    try:
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
        
        summary = query_llm(client, prompt, model_config)
        # Extract just the summary text, in case the model included other narrative
        summary_clean = re.sub(r'<think>.*?</think>', '', summary, flags=re.DOTALL)
        return summary_clean
    except Exception as e:
        # If summarization fails, return a truncated version of the original with error notice
        return f"SUMMARIZATION FAILED. First 1000 chars of output:\n\n{output_text[:1000]}"

def generate_enhanced_prompt(user_query):
    """
    Use the OpenAI model to generate an enhanced prompt from the superprompt.
    
    Args:
        user_query (str): The user's query
        
    Returns:
        str: The enhanced prompt to send to R1
    """
    # Format the superprompt with the user query
    # The template uses double curly braces for its own variables, which need to be preserved
    formatted_superprompt = CoT_template.format(user_query=user_query)
    
    # Create a specific config for this call
    o1_config = {
        "api_key": openai_api_key,
        "endpoint": "https://api.openai.com/v1",
        "model_id": "o1-preview",
        "client": openai_client,
        "messages": [
            {"role": "user", "content": formatted_superprompt}
        ],
        "max_tokens": 32000,
        "temperature": 0.7
    }
    
    # Use OpenAI to generate an enhanced prompt
    enhanced_prompt = query_llm(openai_client, formatted_superprompt, o1_config)
    
    # Print the enhanced prompt for debugging
    print("\n=== ENHANCED PROMPT FROM o1-preview ===\n")
    print(enhanced_prompt)
    print("\n=== END ENHANCED PROMPT ===\n")
    
    return enhanced_prompt

def llm_for_actionable_commands(client, user_query, previous_attempts=None, enhanced_prompt=None):
    """
    Generates commands using the LLM based on the user query and previous attempts.
    
    Args:
        client: The Together client instance
        user_query (str): The user's query
        previous_attempts (list, optional): List of previous attempts with their results
        enhanced_prompt (str, optional): Pre-generated enhanced prompt for the first attempt
    
    Returns:
        tuple: (LLM response, prompt used)
    """
    if previous_attempts and len(previous_attempts) > 0:
        # Create a detailed summary of previous attempts
        previous_attempts_str = ""
        for i, attempt in enumerate(previous_attempts):
            previous_attempts_str += f"\nAttempt {i+1}:\n"
            previous_attempts_str += f"Command: {attempt['command']}\n"
            previous_attempts_str += f"Result: {attempt['result']}\n"
            previous_attempts_str += f"Classification: {attempt['classification']}\n"
            previous_attempts_str += f"Status: {attempt['status']}\n"
            previous_attempts_str += "-" * 30 + "\n"
        
        prompt = f"""Previous attempts to answer this query have failed. 
                    Here are the previous attempts and their results:

                    {previous_attempts_str}

                    Please provide a new approach to answer the original query:
                    {user_query}

                    Remember to:
                    1. Consider why the previous attempts failed
                    2. Adjust the approach based on the error messages or results
                    3. Try a different strategy if the previous ones weren't successful
                    4. Include your reasoning in <think> tags and your final commands in <action> tags
                """
        
        # For retries, we'll use R1 directly
        # Create a specific config for this call
        r1_config = {
            "api_key": together_api_key,
            "endpoint": "https://api.together.xyz/v1",
            "model_id": "deepseek-ai/DeepSeek-R1",
            "client": together_client,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 32000,
            "temperature": 0.7
        }
        response = query_llm(client, prompt, r1_config)
    else:
        # For the initial attempt:
        # Use the provided enhanced prompt if available, otherwise generate a new one
        if enhanced_prompt:
            prompt = enhanced_prompt
        else:
            # Fallback to generating a new prompt if none was provided
            prompt = generate_enhanced_prompt(user_query)
        
        # Create a specific config for this call
        r1_config = {
            "api_key": together_api_key,
            "endpoint": "https://api.together.xyz/v1",
            "model_id": "deepseek-ai/DeepSeek-R1",
            "client": together_client,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 32000,
            "temperature": 0.7
        }
        
        # Now use the enhanced prompt with R1 to generate commands
        response = query_llm(client, prompt, r1_config)
        
    return response, prompt

def execute_command(command, llm_response_file=None):
    """
    Executes a command and returns the output.

    Args:
        command (str): The command to execute.
        llm_response_file (str, optional): Path to the LLM response file to append observations.

    Returns:
        dict: Dictionary containing execution results.
    """
    try:
        print(f"Executing command: {command}")
        
        # Execute the command and redirect output to the temporary file
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = process.communicate()
        
        # If there's an error, include it in the output
        error_message = None
        if stderr:
            error_message = f"Error executing command: {stderr}"
        
        # Append the observation to the LLM response file if provided
        if llm_response_file:
            with open(llm_response_file, 'a') as f:
                f.write(f"\n\n<observation>\n{stdout}")
                if stderr:
                    f.write(f"\n\nErrors:\n{stderr}")
                f.write("\n</observation>\n")
        
        # Return a dictionary with the structured results
        return {
            "observation": {
                "stdout": stdout,
                "stderr": stderr,
                "return_code": process.returncode,
                "output_files": []
            },
            "action_input": command
        }
    
    except Exception as e:
        error_message = f"Error executing command: {str(e)}"
        
        # Append the error observation to the LLM response file if provided
        if llm_response_file:
            with open(llm_response_file, 'a') as f:
                f.write(f"\n\n<observation>\n{error_message}\n</observation>\n")
        
        # Return a dictionary with the error information
        return {
            "observation": {
                "stdout": "",
                "stderr": error_message,
                "return_code": 1,
                "output_files": []
            },
            "action_input": command
        }

def is_valid_json(json_str):
    """
    Validates if the provided string is a valid JSON.

    Args:
        json_str (str): The JSON string to validate.

    Returns:
        bool: True if valid JSON, False otherwise.
    """
    try:
        json.loads(json_str)
        return True
    except json.JSONDecodeError:
        print(f"Invalid JSON: {json_str}")
        exit()
        return False

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
        
        # Create the trace file at the beginning
        trace_file_path = os.path.join(query_dir, "complete_trace.txt")
        
        # Start writing to the trace file
        with open(trace_file_path, "w", encoding="utf-8") as f:
            # Write the user query
            f.write(f"# User Query\n{user_query}\n\n")
        
        print(f"Processing query: {user_query}")
        print(f"Saving results to: {trace_file_path}")
        
        # Initialize tracking for attempts
        previous_attempts = []
        max_attempts = 3
        success = False
        current_attempt = 1
        
        # Try multiple attempts if needed
        for attempt_num in range(max_attempts):
            print(f"Attempt {attempt_num + 1} for query: {user_query}")
            
            # Append the attempt header to the trace file
            with open(trace_file_path, "a", encoding="utf-8") as f:
                f.write(f"# Attempt {current_attempt}\n\n")
            
            # Format the superprompt with the user query and write it to trace file immediately
            if current_attempt == 1:
                formatted_superprompt = CoT_template.format(user_query=user_query)
                with open(trace_file_path, "a", encoding="utf-8") as f:
                    f.write(f"## Original Superprompt (before enhancement)\n{formatted_superprompt}\n\n")
                
                # Generate an enhanced prompt using O1
                enhanced_prompt = generate_enhanced_prompt(user_query)
                
                # Write enhanced prompt to trace file
                with open(trace_file_path, "a", encoding="utf-8") as f:
                    f.write(f"## Enhanced Prompt from o1-preview\n{enhanced_prompt}\n\n")
                
                # Use the enhanced prompt with R1 to generate commands
                response, prompt = llm_for_actionable_commands(together_client, user_query, previous_attempts, enhanced_prompt)
            else:
                # Generate commands using LLM with knowledge of previous attempts
                response, prompt = llm_for_actionable_commands(together_client, user_query, previous_attempts)
            
            # Check if response has action tags, if not retry same call up to 3 times
            retry_count = 0
            max_retries = 3
            while ("<action>" not in response or "</action>" not in response) and retry_count < max_retries:
                # Write to trace file about retrying due to missing action tags
                with open(trace_file_path, "a", encoding="utf-8") as f:
                    f.write(f"## Retry for missing action tags (attempt {retry_count + 1})\n")
                    f.write("Response missing required <action> and </action> tags. Retrying same call.\n\n")
                    # Write the raw response that's missing tags
                    f.write(f"## Raw LLM Response with missing tags:\n```\n{response}\n```\n\n")
                
                # LLM extraction before retry
                extracted_commands = extract_commands_with_llm(response, together_client)
                if extracted_commands:
                    # Use actual commands directly without JSON formatting
                    formatted_commands = ""
                    for cmd in extracted_commands:
                        formatted_commands += f'{{"action": "{cmd["action"]}", "action_input": "{cmd["action_input"]}"}}\n'
                    
                    # Replace the original response with properly formatted commands
                    response = f"<action>\n{formatted_commands}\n</action>"
                    break
                
                # Retry the same call
                response, prompt = llm_for_actionable_commands(together_client, user_query, previous_attempts, enhanced_prompt)
                retry_count += 1
            
            # If we still don't have action tags after retries, write the final response
            if "<action>" not in response or "</action>" not in response:
                with open(trace_file_path, "a", encoding="utf-8") as f:
                    f.write("## Final attempt still missing action tags\n")
                    f.write(f"## Raw LLM Response with missing tags:\n```\n{response}\n```\n\n")
            
            # Write the prompt and response to the trace file
            with open(trace_file_path, "a", encoding="utf-8") as f:
                if current_attempt == 1:  # First attempt - prompts already written in generate_enhanced_prompt
                    pass
                else:
                    f.write(f"## Initial Prompt\n{prompt}\n\n")
                f.write(f"## LLM Response\n{response}\n\n")
            
            # Parse commands from LLM response
            commands = parse_commands(response, together_client)
            if not commands:
                print("No valid commands generated")
                
                # Get detailed parsing error information
                parsing_error = "Command parsing failed"
                if "<action>" not in response:
                    parsing_error += ": No <action> tags found in the response"
                elif "</action>" not in response:
                    parsing_error += ": Missing closing </action> tag in response"
                elif "action" not in response and "action_input" not in response:
                    parsing_error += ": Response has <action> tags but missing 'action' and 'action_input' fields"
                elif response.count("{") == 0 or response.count("}") == 0:
                    parsing_error += ": No JSON object found in <action> tags"
                else:
                    parsing_error += ": Invalid JSON format or structure in <action> tags"
                
                # Create a failed attempt record so we can try again
                attempt = {
                    "command": "No valid command generated",
                    "result": parsing_error,
                    "classification": "FAILURE: No valid command could be parsed",
                    #"justification": "",  # Keeping the key but not using it
                    "solution": "",
                    "status": "failure"
                }
                previous_attempts.append(attempt)
                
                # Write the failure to the trace file
                with open(trace_file_path, "a", encoding="utf-8") as f:
                    f.write("## Command Execution 1\n\n")
                    f.write("<action>\nNo valid command generated\n</action>\n\n")
                    f.write(f"<observation>\n{parsing_error}\n</observation>\n\n")
                    f.write("<classification>\nFAILURE: No valid command could be parsed\n</classification>\n\n")
                    f.write("---\n\n")
                
                current_attempt += 1
                continue
            
            # Process commands
            attempt_had_success = False
            solution = ""
            all_observations = []
            successful_commands = []  # Track successful commands specifically
            
            # Execute commands one by one, allowing for updates after each execution
            command_index = 0
            while command_index < len(commands):
                # Get the current command
                cmd = commands[command_index]
                
                print(f"Executing command {command_index + 1}/{len(commands)}: {cmd['action_input']}")
                
                # Write the command to the trace file
                with open(trace_file_path, "a", encoding="utf-8") as f:
                    f.write(f"## Command Execution {command_index + 1}\n\n")
                    f.write(f"<action>\n{cmd['action_input']}\n</action>\n\n")
                
                # Execute the command
                execution_result = execute_command(cmd['action_input'])
                has_error = execution_result["observation"]["stderr"].strip() != ""
                
                # Collect the observation
                observation_text = execution_result["observation"]["stdout"].strip()
                if has_error:
                    observation_text += f"\n{execution_result['observation']['stderr'].strip()}"
                all_observations.append(observation_text)
                
                # Write the observation to the trace file
                with open(trace_file_path, "a", encoding="utf-8") as f:
                    f.write(f"<observation>\n{observation_text}\n</observation>\n\n")
                
                # Only proceed with classification if execution produced output
                if observation_text:
                    classification = classify_last_command_output_with_llm(user_query, cmd['action_input'], observation_text, together_client)
                    # Remove the thinking process from classification
                    classification = remove_think_tags(classification)
                    
                    # Write classification to trace file
                    with open(trace_file_path, "a", encoding="utf-8") as f:
                        f.write(f"<classification>\n{classification}\n</classification>\n\n")

                    # Check if command was technically successful
                    cmd_success = "success" in classification.lower() or "correct" in classification.lower()
                    
                    # Record this attempt details
                    command_result = {
                        "cmd": cmd['action_input'],
                        "observation_text": observation_text,
                        "classification": classification,
                        "success": cmd_success
                    }
                    
                    if cmd_success:
                        # Store successful command for training output
                        successful_commands.append({
                            "action": commands[command_index]["action"],
                            "action_input": commands[command_index]["action_input"],
                            "action_output": observation_text
                        })
                        
                        # For successful commands, derive a solution
                        solution_raw = derive_solution_with_llm(classification, together_client)
                        solution = remove_think_tags(solution_raw)
                        
                        # Write solution to trace file
                        with open(trace_file_path, "a", encoding="utf-8") as f:
                            f.write(f"<intermediate_solution>\n{solution}\n</intermediate_solution>\n\n")
                        
                        # Mark that we found a technically successful command
                        command_result["solution"] = solution
                        
                        # Ask if we should update remaining commands based on this output
                        if command_index < len(commands) - 1:
                            # Use helper function to determine if commands should be updated
                            should_update, updated_commands, update_response = update_commands_with_llm(
                                user_query, commands, command_index, cmd['action_input'], 
                                observation_text, classification, together_client
                            )
                            
                            if should_update:
                                commands = updated_commands
                                
                                # Write the updated commands to the trace file
                                with open(trace_file_path, "a", encoding="utf-8") as f:
                                    f.write(f"<commands_updated>\nRemaining commands updated based on execution result.\n")
                                    f.write(f"New command list: {json.dumps(commands)}\n</commands_updated>\n\n")
                            else:
                                with open(trace_file_path, "a", encoding="utf-8") as f:
                                    f.write("<commands_updated>No updates needed.</commands_updated>\n\n")
                        
                        # Add separator in trace file
                        with open(trace_file_path, "a", encoding="utf-8") as f:
                            f.write("---\n\n")
                        
                        # Move to the next command
                        command_index += 1
                    else:
                        # For failed commands, try to fix the command
                        # Write failure to trace file
                        with open(trace_file_path, "a", encoding="utf-8") as f:
                            f.write("<command_failed>Command classified as technical failure. Attempting to fix.</command_failed>\n\n")
                        
                        # Use helper function to fix the failed command
                        fixed_cmd, fix_response = fix_failed_command_with_llm(
                            user_query, cmd, observation_text, classification, together_client
                        )
                        
                        # Write the fix attempt to the trace file
                        with open(trace_file_path, "a", encoding="utf-8") as f:
                            f.write(f"<fix_attempt>\n{fix_response}\n</fix_attempt>\n\n")
                            f.write(f"<fixed_command>\n{fixed_cmd['action_input']}\n</fixed_command>\n\n")
                        
                        # Update the command in the list and retry it
                        commands[command_index] = fixed_cmd
                        
                        # Add separator in trace file
                        with open(trace_file_path, "a", encoding="utf-8") as f:
                            f.write("---\n\n")
                        
                        # Do not increment command_index - we'll retry this command
                        continue
                else: #no observation text - go here (not sure this is needed)
                    # Write failure to trace file
                    with open(trace_file_path, "a", encoding="utf-8") as f:
                        f.write("<no_output>Command execution failed or produced no output.</no_output>\n\n")
                        f.write("<command_failed>Command produced no output. Attempting to fix.</command_failed>\n\n")
                        
                    # Try to fix the command that produced no output
                    fixed_cmd, fix_response = fix_failed_command_with_llm(
                        user_query, cmd, "Command produced no output", "FAILURE", together_client
                    )
                    
                    # Write the fix attempt to the trace file
                    with open(trace_file_path, "a", encoding="utf-8") as f:
                        f.write(f"<fix_attempt>\n{fix_response}\n</fix_attempt>\n\n")
                        f.write(f"<fixed_command>\n{fixed_cmd['action_input']}\n</fixed_command>\n\n")
                        f.write("---\n\n")
                    
                    # Update the command in the list and retry it
                    commands[command_index] = fixed_cmd
                    
                    # Note: We don't add this to successful_commands since it failed
                    
                    # Do not increment command_index - we'll retry this command
                    continue
            
            # After all commands are executed, evaluate the attempt as a whole
           
            # Get only successful command outputs from the existing successful_commands list
            successful_observations = [cmd['action_output'] for cmd in successful_commands]
            
            # Derive a comprehensive solution from successful command outputs only
            print("Deriving final solution from successful command outputs...")
            final_solution = generate_final_solution_with_llm(user_query, successful_observations, together_client)
            
            # Write the final solution to the trace file
            with open(trace_file_path, "a", encoding="utf-8") as f:
                f.write(f"<answer>\n{final_solution}\n</answer>\n\n")
            
            # Evaluate if the solution answers the user query
            print("Evaluating if solution answers user query...")
            attempt_had_success, solution_evaluation = evaluate_solution_with_llm(user_query, final_solution, together_client)
            success = attempt_had_success
            
            print(f"Solution answers user query: {attempt_had_success}")
            
            # Write evaluation to trace file
            with open(trace_file_path, "a", encoding="utf-8") as f:
                f.write(f"Solution Evaluation:\n{solution_evaluation}\n\n")
            
            # Generate training output if the attempt was successful
            if attempt_had_success:
                # Extract think content from the response
                think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
                think_content = think_match.group(1).strip() if think_match else "No reasoning process provided"
                
                # Format action content with ONLY successful commands that led to the solution
                action_content = json.dumps(successful_commands, indent=2)
                
                # Generate training output in the query's directory
                training_file = generate_training_output(
                    user_query=user_query,
                    think_content=think_content,
                    action_content=action_content,
                    answer_content=final_solution,
                    output_dir=query_dir
                )
                print(f"Generated training output: {training_file}")

            # Break if successful
            if attempt_had_success:
                print(f"Query succeeded on attempt {attempt_num + 1}")
                break
            elif attempt_num < max_attempts - 1:
                print(f"Attempt {attempt_num + 1} failed. Trying another attempt with adjusted approach.")
                current_attempt += 1
            else:
                print(f"All {max_attempts} attempts failed for query: {user_query}")
        
        print(f"Complete trace saved to {trace_file_path}")
        print(f"Finished processing query: {user_query}")
    

#----------------------------------
# Entry Point

if __name__ == "__main__":
    #test_parse_commands()
    main()

def summarize_observations(observations, client):
    """
    Uses the LLM to summarize a list of observations.
    
    Args:
        observations (list): List of observation strings
        client: The LLM client to use
    
    Returns:
        str: A summary of the observations
    """
    # Combine all observations
    combined_observations = "\n\n".join(observations)
    
    # Create a prompt for summarization
    prompt = f"""
    Please summarize the following observations from command executions:
    
    {combined_observations}
    
    Focus on the key findings and any patterns or trends.
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
    
    # Get the summary from the LLM
    summary = query_llm(client, prompt, model_config)
    
    return summary

