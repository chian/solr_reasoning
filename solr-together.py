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

from agent_parser import test_parse_commands, sanitize_filename, parse_commands   
from helper_functions import query_local_llm, derive_solution_with_llm, justify_solution_with_llm, classify_last_command_output_with_llm, parse_last_command_output_with_llm  

# Get API key from environment variable
api_key = os.getenv('TOGETHER_API_KEY')
if not api_key:
    raise ValueError("Please set the TOGETHER_API_KEY environment variable")

client = Together(api_key=api_key)

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
with open('cot_template_text.txt','r') as i:
    cot_template_text = i.read()

CoT_template = PromptTemplate.from_template(textwrap.dedent(cot_template_text))

def remove_think_tags(text: str) -> str:
    #"""<think>...</think>"""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

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
    else:
        prompt = CoT_template.format(user_query=user_query)
        
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
        return False

#----------------------------------
# Main Execution

def main():
    # Create the main query_results directory if it doesn't exist
    main_results_dir = "query_results"
    os.makedirs(main_results_dir, exist_ok=True)
    count = 0
    for user_query in queries:
        if count == 1:
            sys.exit()
        count += 1
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
                response, prompt = llm_for_actionable_commands(client, user_query, previous_attempts)
                
                # Write the prompt and response to the trace file
                with open(trace_file_path, "a", encoding="utf-8") as f:
                    f.write(f"## Initial Prompt\n{prompt}\n\n")
                    f.write(f"## LLM Response\n{response}\n\n")
                
                # Parse commands from LLM response
                commands = parse_commands(response, logging)
                if not commands:
                    logging.warning("No valid commands generated")
                    # Create a failed attempt record so we can try again
                    attempt = {
                        "command": "No valid command generated",
                        "result": "Command parsing failed",
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
                        f.write("<observation>\nCommand parsing failed\n</observation>\n\n")
                        f.write("<classification>\nFAILURE: No valid command could be parsed\n</classification>\n\n")
                        f.write("<justification>\nThe LLM response did not contain a valid command format\n</justification>\n\n")
                        f.write("---\n\n")
                    
                    current_attempt += 1
                    continue
                
                # For the commands in this attempt, run them in turn
                attempt_had_success = False
                for cmd_index, cmd in enumerate(commands):
                    # Write the command to the trace file
                    with open(trace_file_path, "a", encoding="utf-8") as f:
                        f.write(f"## Command Execution {cmd_index + 1}\n\n")
                        f.write(f"<action>\n{cmd}\n</action>\n\n")
                    
                    # Execute the command
                    logging.info(f"Executing command: {cmd}")
                    # TODO: needs llm_response_file passed after cmd
                    execution_result = execute_command(cmd)
                    
                    # Check if there was an error in stderr
                    has_error = bool(execution_result["observation"]["stderr"])
                    
                    # The raw output lines from the command
                    observation_lines = execution_result["observation"]["stdout"].splitlines()
                    
                    # Write the observation to the trace file
                    with open(trace_file_path, "a", encoding="utf-8") as f:
                        observation_text = execution_result["observation"]["stdout"]
                        if has_error:
                            observation_text += f"\n\nErrors:\n{execution_result['observation']['stderr']}"
                        f.write(f"<observation>\n{observation_text}\n</observation>\n\n")
                    
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
                            client
                        )
                        classification = remove_think_tags(classification_raw)
                        import pdb
                        pdb.set_trace()
                        
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
                            client
                        )
                        justification = remove_think_tags(justification_raw)
                    
                    # Write the justification to the trace file
                    with open(trace_file_path, "a", encoding="utf-8") as f:
                        f.write(f"<justification>\n{justification}\n</justification>\n\n")
                    
                    # 4) Only derive a solution if this is successful
                    if cmd_success:
                        logging.info("Command successful! Deriving solution...")
                        solution_raw = derive_solution_with_llm(classification, justification, client)
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
                    
                    # If successful, break out of the commands loop for this attempt
                    if cmd_success:
                        attempt_had_success = True
                        success = True
                        break
                
                # Break if successful
                if attempt_had_success:
                    logging.info(f"Query succeeded on attempt {attempt_num + 1}")
                    break
                elif attempt_num < max_attempts - 1:
                    logging.info(f"Attempt {attempt_num + 1} failed. Trying another attempt with adjusted approach.")
                    current_attempt += 1
                else:
                    logging.warning(f"All {max_attempts} attempts failed")
            
            logging.info(f"Complete trace saved to {trace_file_path}")
        
        except Exception as e:
            logging.exception(f"Error processing query: {user_query}")
            continue

#----------------------------------
# Entry Point

if __name__ == "__main__":
    #test_parse_commands(logging)
    main()

