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
from helper_functions import query_llm, derive_solution_with_llm, classify_last_command_output_with_llm, parse_last_command_output_with_llm, generate_training_output, evaluate_attempt_with_llm, generate_final_solution_with_llm, update_commands_with_llm, remove_think_tags, CommandClassification, CommandStatus, SolutionClassification, SolutionStatus, save_mcts_trace, visualize_mcts_tree, calculate_node_value, calculate_chain_value
from mcts_thinking import MCTSThinkingManager, NodeType, CommandStatus as MCTSCommandStatus, ActionState

# Initialize MCTS manager
mcts_manager = MCTSThinkingManager()

def parse_classification_text(classification_text):
    """
    Parse a classification text into a CommandClassification object.
    
    Args:
        classification_text (str): The classification text to parse
        
    Returns:
        CommandClassification: A structured classification object
    """
    # Try to find the classification and reason in the text
    status_match = re.search(r'Classification:\s*(SUCCESS|FAILURE)|1\.\s*Classification\s*\((SUCCESS|FAILURE)\)', classification_text, re.IGNORECASE)
    reason_match = re.search(r'Brief explanation:\s*(.*?)(?:\n\n|\Z)|2\.\s*Brief explanation\s*:\s*(.*?)(?:\n\n|\Z)', classification_text, re.DOTALL)
    
    if status_match:
        # Extract status from either format
        status = status_match.group(1) or status_match.group(2)
        status = status.upper()
        
        # Extract reason if available
        reason = ""
        if reason_match:
            reason = reason_match.group(1) or reason_match.group(2)
            reason = reason.strip()
        
        # Create a CommandClassification object
        return CommandClassification(
            status=CommandStatus(status),
            reason=reason
        )
    else:
        # If we can't extract the structured data, try to infer from the text
        if "SUCCESS" in classification_text.upper():
            return CommandClassification(
                status=CommandStatus.SUCCESS,
                reason="Success inferred from text"
            )
        else:
            return CommandClassification(
                status=CommandStatus.FAILURE,
                reason="Failure inferred from text"
            )

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
    "max_completion_tokens": 32000,
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

def initialize_mcts_state(user_query):
    """
    Initialize the MCTS state with the user query.
    
    Args:
        user_query (str): The user's query
        
    Returns:
        dict: The initial MCTS state
    """
    # Create initial state
    initial_state = {
        'thought': f"Initial state for query: {user_query}",
        'thoughtNumber': 1,
        'totalThoughts': 10,  # Estimate
        'nextThoughtNeeded': True,
        'nodeType': NodeType.THOUGHT.value,
        'workingDir': os.getcwd(),
        'environment': dict(os.environ),
        'dependencies': [],
        'prerequisites': []
    }
    
    # Process the initial thought
    return mcts_manager.process_thought(initial_state)

def get_next_action_from_mcts(current_state, possible_actions):
    """
    Get the next action using MCTS.
    
    Args:
        current_state (dict): The current MCTS state
        possible_actions (list): List of possible actions
        
    Returns:
        tuple: (best_action, action_node)
    """
    # Get recommendations from MCTS
    recommendations = mcts_manager.get_recommended_nodes()
    
    # Get the current node
    current_node_id = current_state['currentNodeId']
    current_node = mcts_manager.node_map.get(current_node_id)
    
    if not current_node:
        return None, None
        
    # If we have recommendations, evaluate them
    if recommendations:
        best_recommendation = recommendations[0]
        best_node_id = best_recommendation['nodeId']
        best_node = mcts_manager.node_map.get(best_node_id)
        
        # If the best node is an action node with an action state, use it
        if best_node and best_node.get('nodeType') == NodeType.COMMAND.value and best_node.get('actionState'):
            return best_node['actionState'].command, best_node
            
        # If the best node is a thought node, we can either:
        # 1. Generate a new thought
        # 2. Create an action from this thought
        if best_node and best_node.get('nodeType') == NodeType.THOUGHT.value:
            if possible_actions:
                # Create a new action node from this thought
                action_data = {
                    'thought': f"Executing command: {possible_actions[0]}",
                    'thoughtNumber': current_state['thoughtNumber'] + 1,
                    'totalThoughts': current_state['totalThoughts'],
                    'nextThoughtNeeded': True,
                    'nodeType': NodeType.COMMAND.value,
                    'action': possible_actions[0],
                    'workingDir': os.getcwd(),
                    'environment': dict(os.environ),
                    'dependencies': [],
                    'prerequisites': [],
                    'parentId': best_node_id  # Connect to the thought node
                }
                
                new_state = mcts_manager.process_thought(action_data)
                new_node = mcts_manager.node_map.get(new_state['currentNodeId'])
                
                if new_node and new_node.get('actionState'):
                    return new_node['actionState'].command, new_node
    
    # If no suitable recommendation found and we have possible actions,
    # create a new action node from the current node
    if possible_actions:
        action_data = {
            'thought': f"Executing command: {possible_actions[0]}",
            'thoughtNumber': current_state['thoughtNumber'] + 1,
            'totalThoughts': current_state['totalThoughts'],
            'nextThoughtNeeded': True,
            'nodeType': NodeType.COMMAND.value,
            'action': possible_actions[0],
            'workingDir': os.getcwd(),
            'environment': dict(os.environ),
            'dependencies': [],
            'prerequisites': [],
            'parentId': current_node_id
        }
        
        new_state = mcts_manager.process_thought(action_data)
        new_node = mcts_manager.node_map.get(new_state['currentNodeId'])
        
        if new_node and new_node.get('actionState'):
            return new_node['actionState'].command, new_node
    
    return None, None

def update_mcts_with_command_result(mcts_manager, command, result, classification):
    """
    Update MCTS with command execution results.
    
    Args:
        mcts_manager: The MCTS manager
        command (str): The command that was executed
        result (str): The command output
        classification (CommandClassification): The classification of the command result
        
    Returns:
        dict: The updated MCTS state
    """
    # Calculate value based on classification
    value = 1.0 if classification.status == CommandStatus.SUCCESS else 0.0
    
    # Find the node with this command
    node = None
    for node_id, node_data in mcts_manager.node_map.items():
        if node_data.get('actionState') and node_data['actionState'].command == command:
            node = node_data
            break
    
    if node:
        # Update node's value estimate
        current_value = node.get('valueEstimate', 0.0)
        visits = node.get('visits', 0)
        node['valueEstimate'] = (current_value * visits + value) / (visits + 1)
        node['visits'] = visits + 1
        
        # Propagate value up the tree
        parent_id = node.get('parentId')
        while parent_id:
            parent = mcts_manager.node_map.get(parent_id)
            if parent:
                # For thought nodes, use max value
                if parent.get('nodeType') == NodeType.THOUGHT.value:
                    parent['valueEstimate'] = max(parent.get('valueEstimate', 0.0), value)
                # For action nodes, use average
                else:
                    parent_visits = parent.get('visits', 0)
                    parent_value = parent.get('valueEstimate', 0.0)
                    parent['valueEstimate'] = (parent_value * parent_visits + value) / (parent_visits + 1)
                    parent['visits'] = parent_visits + 1
                parent_id = parent.get('parentId')
            else:
                break

        # Update node type
        node['nodeType'] = NodeType.COMMAND.value

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
        "max_completion_tokens": 32000,
        "temperature": 0.7
    }
    
    # Use OpenAI to generate an enhanced prompt
    enhanced_prompt = query_llm(openai_client, formatted_superprompt, o1_config)
    
    # Print the enhanced prompt for debugging
    print("\n=== ENHANCED PROMPT FROM o1-preview ===\n")
    print(enhanced_prompt)
    print("\n=== END ENHANCED PROMPT ===\n")
    
    return enhanced_prompt

def llm_for_actionable_commands(client, user_query, previous_attempts=None, enhanced_prompt=None, mcts_state=None):
    """
    Generates commands using the LLM based on the user query and previous attempts.
    
    Args:
        client: The Together client instance
        user_query (str): The user's query
        previous_attempts (list, optional): List of previous attempts with their results
        enhanced_prompt (str, optional): Pre-generated enhanced prompt for the first attempt
        mcts_state (dict, optional): Current MCTS state with history
    
    Returns:
        tuple: (LLM response, prompt used)
    """
    if mcts_state and 'concatenated_context' in mcts_state:
        # Use the rich history from MCTS state
        prompt = f"""Previous attempts to answer this query have failed. 
                    Here is the history of attempts and their results:

                    {mcts_state['concatenated_context']}

                    Please provide a new approach to answer the original query:
                    {user_query}

                    Remember to:
                    1. Consider why the previous attempts failed
                    2. Adjust the approach based on the error messages or results
                    3. Try a different strategy if the previous ones weren't successful
                    4. Include your reasoning in <think> tags and your final commands in <action> tags
                    
                    Specifically, analyze the errors from previous attempts and explain how your new approach addresses them.
                """
    elif previous_attempts and len(previous_attempts) > 0:
        # Fallback to old format if no MCTS state
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
    else:
        # First attempt: use the enhanced prompt
        if enhanced_prompt:
            prompt = enhanced_prompt
        else:
            # Format the superprompt with the user query
            formatted_superprompt = CoT_template.format(user_query=user_query)
            prompt = formatted_superprompt
    
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
    
    # Get response from R1
    response = query_llm(client, prompt, r1_config)
    return response, prompt

def execute_command(command):
    """
    Execute a command and return its output.
    
    Args:
        command (str): The command to execute
        
    Returns:
        str: The command output
    """
    try:
        # Execute the command
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        # Return the output
        if result.returncode == 0:
            return result.stdout
        else:
            return f"Error: {result.stderr}\nExit code: {result.returncode}"
    except Exception as e:
        return f"Exception: {str(e)}"

def extract_think_content(response):
    """
    Extract the content within <think> tags from a response.
    
    Args:
        response (str): The response containing <think> tags
        
    Returns:
        str: The content within <think> tags
    """
    think_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
    think_matches = think_pattern.findall(response)
    
    if think_matches:
        return "\n\n".join(think_matches)
    else:
        return ""

def format_action_content(command_attempts):
    """
    Format the command attempts into a string for training data.
    
    Args:
        command_attempts (list): List of command attempts
        
    Returns:
        str: Formatted action content
    """
    action_content = ""
    
    for attempt in command_attempts:
        action_content += f"Command: {attempt['command']}\n"
        action_content += f"Output: {attempt['result']}\n"
        action_content += f"Status: {attempt['status']}\n"
        action_content += "-" * 50 + "\n"
    
    return action_content

def main():
    # Process each query
    for user_query in queries:
        print(f"\n=== Processing Query: {user_query} ===")
        
        # Initialize MCTS
        initial_state = initialize_mcts_state(user_query)
        current_state = initial_state
        
        # Track command attempts
        command_attempts = []
        
        # Main loop
        max_iterations = 10
        for iteration in range(max_iterations):
            print(f"\n=== Iteration {iteration+1}/{max_iterations} ===")
            
            # Generate possible actions
            if iteration == 0:
                # First iteration: use enhanced prompt
                enhanced_prompt = generate_enhanced_prompt(user_query)
                llm_response, _ = llm_for_actionable_commands(together_client, user_query, enhanced_prompt=enhanced_prompt)
            else:
                # Subsequent iterations: use MCTS state history
                llm_response, _ = llm_for_actionable_commands(together_client, user_query, mcts_state=current_state)
            
            # Extract commands
            commands = parse_commands(llm_response, together_client)
            
            if not commands:
                print("No valid commands found. Trying again...")
                continue
            
            # Convert commands to action format
            possible_actions = []
            for cmd in commands:
                if isinstance(cmd, dict) and 'action_input' in cmd:
                    possible_actions.append(cmd['action_input'])
                elif isinstance(cmd, str):
                    possible_actions.append(cmd)
            
            # Get next action from MCTS
            next_action, action_node = get_next_action_from_mcts(current_state, possible_actions)
            
            if not next_action:
                print("MCTS could not select a next action. Trying again...")
                continue
            
            # Execute command
            print(f"Executing command: {next_action}")
            result = execute_command(next_action)
            
            # Get command classification from LLM
            classification_text = classify_last_command_output_with_llm(user_query, next_action, result, together_client)
            classification = parse_classification_text(classification_text)
            
            # Calculate node and chain values
            node_value = calculate_node_value(action_node)
            chain_value = calculate_chain_value(action_node['nodeId'], mcts_manager)
            
            # Update MCTS with result
            update_mcts_with_command_result(mcts_manager, next_action, result, classification)
            
            # Store classification in the node
            action_node['classification'] = classification
            
            # Record attempt
            command_attempts.append({
                'command': next_action,
                'result': result,
                'classification': classification,
                'status': classification.status.value
            })
            
            # Evaluate if we've reached a solution
            solution_classification = evaluate_attempt_with_llm(user_query, command_attempts, together_client)
            
            # Check if we found a complete solution
            if solution_classification.status == SolutionStatus.COMPLETE:
                print("\nFound complete solution!")
                print(f"\nSolution: {generate_final_solution_with_llm(user_query, command_attempts, together_client)}")
                print(f"\nReason: {solution_classification.reason}")
                break
   
        print("\n=== Final Results ===")
        print(f"Query: {user_query}")
        print(f"Total attempts: {len(command_attempts)}")
        print("\nCommand history:")
        print(format_action_content(command_attempts))
        
        # Exit after first query
        break

if __name__ == "__main__":
    main()

