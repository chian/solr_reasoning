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
import logging
import io
import openai  # Add OpenAI import for o1-pro model

from agent_parser import test_parse_commands, sanitize_filename, parse_commands   
from helper_functions import query_local_llm, derive_solution_with_llm, justify_solution_with_llm, classify_last_command_output_with_llm, parse_last_command_output_with_llm  

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
    "client": together_client
}

OPENAI_CONFIG = {
    "api_key": openai_api_key,
    "endpoint": "https://api.openai.com/v1",
    "model_id": "o1-preview",
    "client": openai_client
}

# Task-specific model assignments
MODEL_CONFIGS = {
    "superprompt_generation": OPENAI_CONFIG,
    "command_execution": R1_CONFIG
}

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler("together-commands.log"),
        logging.StreamHandler()
    ]
)

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

def remove_think_tags(text: str) -> str:
    #"""<think>...</think>"""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

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
        summary = query_local_llm(client, prompt)
        # Extract just the summary text, in case the model included other narrative
        summary_clean = re.sub(r'<think>.*?</think>', '', summary, flags=re.DOTALL)
        return summary_clean
    except Exception as e:
        logging.exception(f"Error summarizing command output: {e}")
        # If summarization fails, return a truncated version of the original with error notice
        return f"SUMMARIZATION FAILED. First 1000 chars of output:\n\n{output_text[:1000]}"

def query_openai_model(prompt, model_config=OPENAI_CONFIG):
    """
    Query the OpenAI model with a given prompt.
    
    Args:
        prompt (str): The prompt to send to the model
        model_config (dict): Configuration for the model
        
    Returns:
        str: The model's response
    """
    try:
        logging.info(f"Querying OpenAI model: {model_config['model_id']}")
        
        response = model_config['client'].chat.completions.create(
            model=model_config['model_id'],
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logging.exception(f"Error querying OpenAI model: {e}")
        return f"Error querying model: {str(e)}"

def generate_enhanced_prompt(user_query):
    """
    Use the OpenAI model to generate an enhanced prompt from the superprompt.
    
    Args:
        user_query (str): The user's query
        
    Returns:
        str: The enhanced prompt to send to R1
    """
    # Format the superprompt with the user query
    formatted_superprompt = CoT_template.format(user_query=user_query)
    
    # Use OpenAI to generate an enhanced prompt
    logging.info("Generating enhanced prompt using OpenAI model")
    enhanced_prompt = query_openai_model(formatted_superprompt, MODEL_CONFIGS["superprompt_generation"])
    
    # Log the enhanced prompt for debugging
    logging.info(f"Enhanced prompt: {enhanced_prompt}")
    print("\n=== ENHANCED PROMPT FROM o1-preview ===\n")
    print(enhanced_prompt)
    print("\n=== END ENHANCED PROMPT ===\n")
    
    return enhanced_prompt

def llm_for_actionable_commands(client, user_query, previous_attempts=None):
    """
    Generates commands using the LLM based on the user query and previous attempts.
    
    Args:
        client: The Together client instance
        user_query (str): The user's query
        previous_attempts (list, optional): List of previous attempts with their results
    
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
            previous_attempts_str += f"Justification: {attempt['justification']}\n"
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
        response = query_local_llm(client, prompt)
    else:
        # For the initial attempt:
        # 1. Use OpenAI's o1-pro model to generate a better prompt from the superprompt
        # 2. Then feed that enhanced prompt to R1 for command generation
        prompt = generate_enhanced_prompt(user_query)
        
        # Now use the enhanced prompt with R1 to generate commands
        logging.info("Generating commands using enhanced prompt with R1")
        response = query_local_llm(client, prompt)
        
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
        logging.info(f"Executing command: {command}")
        
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
            logging.error(error_message)
        
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
        logging.exception(error_message)
        
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
        try:
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
            
            logging.info(f"Processing query: {user_query}")
            logging.info(f"Saving results to: {trace_file_path}")
            
            # Initialize tracking for attempts
            previous_attempts = []
            max_attempts = 3
            success = False
            current_attempt = 1
            
            # Try multiple attempts if needed
            for attempt_num in range(max_attempts):
                logging.info(f"Attempt {attempt_num + 1} for query: {user_query}")
                
                # Append the attempt header to the trace file
                with open(trace_file_path, "a", encoding="utf-8") as f:
                    f.write(f"# Attempt {current_attempt}\n\n")
                
                # Generate commands using LLM with knowledge of previous attempts
                response, prompt = llm_for_actionable_commands(together_client, user_query, previous_attempts)
                
                # Write the prompt and response to the trace file
                with open(trace_file_path, "a", encoding="utf-8") as f:
                    if attempt_num == 0:  # First attempt - this is the enhanced prompt
                        f.write(f"## Original Superprompt (before enhancement)\n{CoT_template.format(user_query=user_query)}\n\n")
                        f.write(f"## Enhanced Prompt from o1-preview\n{prompt}\n\n")
                    else:
                        f.write(f"## Initial Prompt\n{prompt}\n\n")
                    f.write(f"## LLM Response\n{response}\n\n")
                
                # Parse commands from LLM response
                commands = parse_commands(response, logging)
                if not commands:
                    logging.warning("No valid commands generated")
                    
                    # Get detailed parsing error information
                    parsing_error = "Command parsing failed"
                    if "<action>" not in response:
                        parsing_error += ": No <action> tags found in the response"
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
                        "justification": "The LLM response did not contain a valid command format",
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
                        f.write("<justification>\nThe LLM response did not contain a valid command format\n</justification>\n\n")
                        f.write("---\n\n")
                    
                    current_attempt += 1
                    continue
                
                # For the commands in this attempt, run them in turn
                attempt_had_success = False
                all_observations = []
                
                # Execute commands one by one, allowing for updates after each execution
                command_index = 0
                while command_index < len(commands):
                    # Get the current command
                    cmd = commands[command_index]
                    
                    # Write the command to the trace file
                    with open(trace_file_path, "a", encoding="utf-8") as f:
                        f.write(f"## Command Execution {command_index + 1}\n\n")
                        f.write(f"<action>\n{cmd}\n</action>\n\n")
                    
                    # Execute the command
                    logging.info(f"Executing command: {cmd}")
                    execution_result = execute_command(cmd)
                    
                    # Check if there was an error in stderr
                    has_error = bool(execution_result["observation"]["stderr"])
                    
                    # The raw output lines from the command
                    observation_lines = execution_result["observation"]["stdout"].splitlines()
                    
                    # Collect the observation
                    observation_text = execution_result["observation"]["stdout"]
                    if has_error:
                        observation_text += f"\n\nErrors:\n{execution_result['observation']['stderr']}"
                    all_observations.append(observation_text)
                    
                    # Write the observation to the trace file
                    with open(trace_file_path, "a", encoding="utf-8") as f:
                        f.write("<observation>\n")
                        f.write(observation_text)
                        f.write("\n</observation>\n\n")
                    
                    # 1) Classify the result - even if there was an error
                    logging.info("Classifying command output...")
                    
                    # If there was an error, we'll classify it as a failure directly
                    if has_error:
                        classification = f"FAILURE: Command execution error - {execution_result['observation']['stderr']}"
                        cmd_success = False
                    else:
                        # Normal classification through LLM
                        classification_raw = classify_last_command_output_with_llm(
                            observation_lines,
                            user_query,
                            together_client
                        )
                        classification = remove_think_tags(classification_raw)
                        
                        # 3) Decide if success based on classification
                        cmd_success = ("SUCCESS" in classification.upper() 
                                      and "FAILURE" not in classification.upper())
                    
                    # Write the classification to the trace file
                    with open(trace_file_path, "a", encoding="utf-8") as f:
                        f.write(f"<classification>\n{classification}\n</classification>\n\n")
                    
                    # 2) Justify the result - even for errors
                    logging.info("Justifying command output...")
                    if has_error:
                        justification = f"The command failed to execute properly due to: {execution_result['observation']['stderr']}"
                    else:
                        justification_raw = justify_solution_with_llm(
                            observation_lines,
                            user_query,
                            response,
                            together_client
                        )
                        justification = remove_think_tags(justification_raw)
                    
                    # Write the justification to the trace file
                    with open(trace_file_path, "a", encoding="utf-8") as f:
                        f.write(f"<justification>\n{justification}\n</justification>\n\n")
                    
                    # 4) Only derive a solution if this is successful
                    if cmd_success:
                        logging.info("Command successful! Deriving solution...")
                        solution_raw = derive_solution_with_llm(classification, justification, together_client)
                        solution = remove_think_tags(solution_raw)
                        
                        # Write the solution to the trace file
                        with open(trace_file_path, "a", encoding="utf-8") as f:
                            f.write(f"<solution>\n{solution}\n</solution>\n\n")
                    else:
                        logging.info("Command classified as failure.")
                        solution = ""
                    
                    # Add separator in the trace file
                    with open(trace_file_path, "a", encoding="utf-8") as f:
                        f.write("---\n\n")
                    
                    # Record this attempt
                    attempt = {
                        "command": cmd,
                        "result": execution_result["observation"]["stdout"],
                        "error": execution_result["observation"]["stderr"] if has_error else "",
                        "classification": classification,
                        "justification": justification,
                        "solution": solution,
                        "status": "success" if cmd_success else "failure"
                    }
                    previous_attempts.append(attempt)
                    
                    # If successful, mark this attempt as successful
                    if cmd_success:
                        attempt_had_success = True
                        success = True
                    
                    command_index += 1
                
                # Write all observations together after all commands are executed
                with open(trace_file_path, "a", encoding="utf-8") as f:
                    # First write the raw observations
                    f.write("<observation>\n")
                    f.write("\n\n".join(all_observations))
                    f.write("\n</observation>\n\n")
                    
                    # Then summarize the observations for classification
                    summarized_observations = []
                    for obs in all_observations:
                        summary = summarize_command_output(obs, cmd, user_query, together_client)
                        summarized_observations.append(summary)
                    
                    # Write the summarized observations
                    f.write("<summarized_observation>\n")
                    f.write("\n\n".join(summarized_observations))
                    f.write("\n</summarized_observation>\n\n")
                    
                    # Use summarized observations for classification
                    classification_raw = classify_last_command_output_with_llm(
                        "\n".join(summarized_observations).splitlines(),
                        user_query,
                        together_client
                    )
                    classification = remove_think_tags(classification_raw)
                    
                    # Write the classification
                    f.write(f"<classification>\n{classification}\n</classification>\n\n")
                    
                    # Use summarized observations for justification
                    justification_raw = justify_solution_with_llm(
                        "\n".join(summarized_observations).splitlines(),
                        user_query,
                        response,
                        together_client
                    )
                    justification = remove_think_tags(justification_raw)
                    
                    # Write the justification
                    f.write(f"<justification>\n{justification}\n</justification>\n\n")
                    
                    # Only derive a solution if this is successful
                    cmd_success = ("SUCCESS" in classification.upper() 
                                  and "FAILURE" not in classification.upper())
                    
                    if cmd_success:
                        logging.info("Command successful! Deriving solution...")
                        solution_raw = derive_solution_with_llm(classification, justification, together_client)
                        solution = remove_think_tags(solution_raw)
                        
                        # Write the solution
                        f.write(f"<solution>\n{solution}\n</solution>\n\n")
                    else:
                        logging.info("Command classified as failure.")
                        solution = ""
                    
                    # Add separator
                    f.write("---\n\n")
                
                # Break if successful
                if attempt_had_success:
                    logging.info(f"Query succeeded on attempt {attempt_num + 1}")
                    break
                elif attempt_num < max_attempts - 1:
                    logging.info(f"Attempt {attempt_num + 1} failed. Trying another attempt with adjusted approach.")
                    current_attempt += 1
                else:
                    print(f"All {max_attempts} attempts failed for query: {user_query}")
                    logging.warning(f"All {max_attempts} attempts failed")
            
            logging.info(f"Complete trace saved to {trace_file_path}")
            print(f"Finished processing query: {user_query}")
        
        except Exception as e:
            logging.exception(f"Error processing query: {user_query}")
            continue

#----------------------------------
# Entry Point

if __name__ == "__main__":
    #test_parse_commands(logging)
    main()

