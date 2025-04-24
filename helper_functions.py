from together import Together
import re
import os
from pydantic import BaseModel, Field
from enum import Enum
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Union
import openai
from mcts_thinking import NodeType

# Define the status enum
class CommandStatus(Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    PENDING = "PENDING"

# Define the Pydantic model for command classification
class CommandClassification(BaseModel):
    status: CommandStatus = Field(..., description="The status of the command execution")
    reason: str = Field(..., description="Brief explanation of the classification")
    
    def is_successful(self) -> bool:
        """Helper method to check if the command was successful"""
        return self.status == CommandStatus.SUCCESS

class SolutionStatus(Enum):
    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"

class SolutionClassification(BaseModel):
    status: SolutionStatus = Field(..., description="The status of the solution")
    reason: str = Field(..., description="Brief explanation of the classification")

def remove_think_tags(text: str) -> str:
    """
    Removes <think> tags and their contents from text.
    
    Args:
        text (str): The text to process
    
    Returns:
        str: Text with <think> tags and their contents removed
    """
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

def query_llm(client, user_query, model_config):
    """
    Query an LLM with a user query and model configuration.
    
    Args:
        client: The client to use for the API call
        user_query (str): The user's query
        model_config (dict): The model configuration
        
    Returns:
        str: The model's response
    """
    # Get the model ID from the config
    model_id = model_config.get("model_id", model_config.get("model"))
    
    # Get the messages from the config
    messages = model_config.get("messages", [])
    
    # Create the parameters dictionary for the API call
    parameters = {
        "model": model_id,
        "messages": messages
    }
    
    # Add any additional parameters from the config
    api_params = model_config.get('api_params', {})
    parameters.update(api_params)
    
    # Make the API call
    response = client.chat.completions.create(**parameters)
    
    # Return the response content
    return response.choices[0].message.content

def parse_last_command_output_with_llm(output: str, client) -> Dict[str, Any]:
    """
    Parse the last command output using the LLM.
    
    Args:
        output (str): The command output
        client: The LLM client to use
        
    Returns:
        dict: Parsed output data
    """
    prompt = f"""
    Parse the following command output into structured data:
    
    {output[:1000]}...
    
    Please extract:
    1. Key information
    2. Error messages (if any)
    3. Important values or IDs
    4. Status indicators
    """
    
    model_config = {
        "model_id": "deepseek-ai/DeepSeek-R1",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 32000,
        "temperature": 0.7
    }
    
    response = query_llm(client, prompt, model_config)
    
    # Try to parse as JSON
    try:
        return json.loads(response)
    except:
        # If not JSON, return as text
        return {"text": response}

def classify_last_command_output_with_llm(user_query: str, command: str, output: str, client) -> str:
    """
    Classify the last command output using the LLM.
    
    Args:
        user_query (str): The original user query
        command (str): The command that was executed
        output (str): The command output
        client: The LLM client to use
        
    Returns:
        str: The classification text
    """
    prompt = f"""
    Classify the following command output for the user query:
    
    User Query: "{user_query}"
    Command: "{command}"
    Output: "{output[:1000]}..."
    
    Please provide:
    1. Classification (SUCCESS or FAILURE)
    2. Brief explanation
    3. Any relevant details or patterns
    """
    
    model_config = {
        "model": "deepseek-ai/DeepSeek-R1",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 32000,
        "temperature": 0.7
    }
    
    return query_llm(client, prompt, model_config)

def derive_solution_with_llm(classification_text: str, client) -> str:
    """
    Derive a solution from the classification text.
    
    Args:
        classification_text (str): The classification text
        client: The LLM client to use
        
    Returns:
        str: The derived solution
    """
    prompt = f"""
    Based on the following classification, derive a solution:
    
    {classification_text}
    
    Please provide a clear and concise solution that addresses the original query.
    """
    
    model_config = {
        "model": "deepseek-ai/DeepSeek-R1",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 32000,
        "temperature": 0.7
    }
    
    return query_llm(client, prompt, model_config)

def generate_training_output(user_query: str, think_content: str, action_content: str, answer_content: str, output_dir: Optional[str] = None) -> str:
    """
    Generate training output from the collected data.
    
    Args:
        user_query (str): The original user query
        think_content (str): The thinking process
        action_content (str): The actions taken
        answer_content (str): The final answer
        output_dir (str, optional): Directory to save the output
        
    Returns:
        str: Path to the generated training file
    """
    # Create output directory if not provided
    if not output_dir:
        output_dir = "training_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename
    filename = f"training_{int(time.time())}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Create training data
    training_data = {
        "query": user_query,
        "thinking_process": think_content,
        "actions": action_content,
        "answer": answer_content
    }
    
    # Write to file
    with open(filepath, "w") as f:
        json.dump(training_data, f, indent=2)
    
    return filepath

def save_mcts_trace(mcts_manager, user_query: str, command_attempts: List[Dict[str, Any]], output_dir: Optional[str] = None) -> str:
    """
    Save the complete MCTS trace to a JSON file.
    
    Args:
        mcts_manager: The MCTSThinkingManager instance
        user_query (str): The original user query
        command_attempts (list): List of command attempts
        output_dir (str, optional): Directory to save the output
        
    Returns:
        str: Path to the generated trace file
    """
    # Create output directory if not provided
    if not output_dir:
        output_dir = "mcts_traces"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename
    filename = f"mcts_trace_{int(time.time())}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Create a serializable version of the node map
    serializable_node_map = {}
    for node_id, node in mcts_manager.node_map.items():
        # Convert ActionState to dict if present
        action_state = None
        if node.get('actionState'):
            action_state = {
                'command': node['actionState'].command,
                'working_dir': node['actionState'].working_dir,
                'environment': node['actionState'].environment,
                'dependencies': node['actionState'].dependencies,
                'prerequisites': node['actionState'].prerequisites,
                'command_output': node['actionState'].command_output,
                'exit_code': node['actionState'].exit_code,
                'execution_time': node['actionState'].execution_time,
                'resource_usage': node['actionState'].resource_usage,
                'error_message': node['actionState'].error_message,
                'status': node['actionState'].status.value if node['actionState'].status else None
            }
        
        # Create serializable node
        serializable_node = {
            'thought': node.get('thought'),
            'thoughtNumber': node.get('thoughtNumber'),
            'totalThoughts': node.get('totalThoughts'),
            'nextThoughtNeeded': node.get('nextThoughtNeeded'),
            'nodeId': node.get('nodeId'),
            'parentId': node.get('parentId'),
            'visits': node.get('visits'),
            'valueEstimate': node.get('valueEstimate'),
            'childNodes': node.get('childNodes', []),
            'depth': node.get('depth'),
            'action': node.get('action'),
            'explorationConstant': node.get('explorationConstant'),
            'nodeType': node.get('nodeType'),
            'actionState': action_state
        }
        serializable_node_map[node_id] = serializable_node
    
    # Create trace data
    trace_data = {
        'query': user_query,
        'node_map': serializable_node_map,
        'root_nodes': mcts_manager.root_nodes,
        'thought_history': [
            {
                'thought': thought.get('thought'),
                'thoughtNumber': thought.get('thoughtNumber'),
                'nodeId': thought.get('nodeId'),
                'nodeType': thought.get('nodeType')
            }
            for thought in mcts_manager.thought_history
        ],
        'command_attempts': command_attempts
    }
    
    # Write to file
    with open(filepath, "w") as f:
        json.dump(trace_data, f, indent=2)
    
    print(f"Saved complete MCTS trace to {filepath}")
    return filepath

def visualize_mcts_tree(mcts_manager, output_dir: Optional[str] = None) -> str:
    """
    Generate a text-based visualization of the MCTS tree structure.
    
    Args:
        mcts_manager: The MCTSThinkingManager instance
        output_dir (str, optional): Directory to save the visualization
        
    Returns:
        str: Path to the generated visualization file
    """
    # Create output directory if not provided
    if not output_dir:
        output_dir = "mcts_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename
    filename = f"mcts_tree_{int(time.time())}.txt"
    filepath = os.path.join(output_dir, filename)
    
    # Start with root nodes
    lines = ["MCTS Tree Visualization", "=====================", ""]
    
    # Helper function to recursively build the tree visualization
    def build_tree_visualization(node_id, depth=0, prefix=""):
        node = mcts_manager.node_map.get(node_id)
        if not node:
            return []
        
        # Create node line
        node_type = node.get('nodeType', 'unknown')
        thought_num = node.get('thoughtNumber', 0)
        value = node.get('valueEstimate', 0.0)
        visits = node.get('visits', 0)
        
        # Truncate thought text
        thought = node.get('thought', '')
        if len(thought) > 50:
            thought = thought[:47] + "..."
        
        # Add action info if it's an action node
        action_info = ""
        if node_type == "action" and node.get('action'):
            action_info = f" | Action: {node.get('action')}"
        
        # Create the node line
        node_line = f"{prefix}{'└── ' if depth > 0 else ''}[{node_type.upper()}] {thought_num} | Value: {value:.2f} | Visits: {visits}{action_info}"
        lines.append(node_line)
        
        # Process child nodes
        child_nodes = node.get('childNodes', [])
        for i, child_id in enumerate(child_nodes):
            is_last = i == len(child_nodes) - 1
            new_prefix = prefix + ("    " if depth == 0 else "│   " if not is_last else "    ")
            build_tree_visualization(child_id, depth + 1, new_prefix)
    
    # Process each root node
    for root_id in mcts_manager.root_nodes:
        build_tree_visualization(root_id)
    
    # Write to file
    with open(filepath, "w") as f:
        f.write("\n".join(lines))
    
    print(f"Saved MCTS tree visualization to {filepath}")
    return filepath

def evaluate_solution_with_llm(user_query: str, solution: str, client) -> str:
    """
    Evaluate if the solution answers the user query.
    
    Args:
        user_query (str): The original user query
        solution (str): The proposed solution
        client: The LLM client to use
        
    Returns:
        str: Evaluation result
    """
    prompt = f"""
    Evaluate if the following solution answers the user query:
    
    User Query: "{user_query}"
    Solution: "{solution}"
    
    Please provide:
    1. COMPLETE or INCOMPLETE
    2. Brief explanation
    3. Any missing information
    """
    
    model_config = {
        "model": "deepseek-ai/DeepSeek-R1",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 32000,
        "temperature": 0.7
    }
    
    return query_llm(client, prompt, model_config)

def generate_final_solution_with_llm(user_query: str, command_attempts: List[Dict[str, Any]], client) -> str:
    """
    Generate a final solution from successful command attempts.
    
    Args:
        user_query (str): The original user query
        command_attempts (list): List of command attempts
        client: The LLM client to use
        
    Returns:
        str: The final solution
    """
    # Filter successful attempts
    successful_attempts = [a for a in command_attempts if a['status'] == CommandStatus.SUCCESS.value]
    
    if not successful_attempts:
        return "No successful commands to generate a solution from."
    
    # Format successful attempts
    attempts_text = ""
    for attempt in successful_attempts:
        attempts_text += f"Command: {attempt['command']}\n"
        attempts_text += f"Output: {attempt['result'][:500]}...\n"
        attempts_text += "-" * 50 + "\n"
    
    prompt = f"""
    Generate a final solution based on these successful command attempts:
    
    User Query: "{user_query}"
    
    Successful Attempts:
    {attempts_text}
    
    Please provide a comprehensive solution that:
    1. Directly answers the user query
    2. Incorporates relevant information from the command outputs
    3. Is clear and well-structured
    """
    
    model_config = {
        "model": "deepseek-ai/DeepSeek-R1",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 32000,
        "temperature": 0.7
    }
    
    return query_llm(client, prompt, model_config)

def update_commands_with_llm(user_query: str, commands: List[Dict[str, Any]], command_index: int, 
                           executed_command: str, output: str, classification: str, client) -> Tuple[bool, List[Dict[str, Any]], str]:
    """
    Determine if commands should be updated based on execution results.
    
    Args:
        user_query (str): The original user query
        commands (list): List of commands
        command_index (int): Index of the executed command
        executed_command (str): The command that was executed
        output (str): The command output
        classification (str): The classification of the output
        client: The LLM client to use
        
    Returns:
        tuple: (should_update, updated_commands, update_response)
    """
    prompt = f"""
    Determine if the remaining commands should be updated based on this execution:
    
    User Query: "{user_query}"
    Executed Command: "{executed_command}"
    Output: "{output[:500]}..."
    Classification: "{classification}"
    
    Remaining Commands:
    {json.dumps(commands[command_index+1:], indent=2)}
    
    Please provide:
    1. YES or NO for whether commands should be updated
    2. Updated commands if YES (in JSON format)
    3. Brief explanation
    """
    
    model_config = {
        "model": "deepseek-ai/DeepSeek-R1",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 32000,
        "temperature": 0.7
    }
    
    response = query_llm(client, prompt, model_config)
    
    # Parse response
    should_update = "YES" in response.upper()
    updated_commands = commands
    
    if should_update:
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                updated_commands = json.loads(json_match.group())
        except:
            pass
    
    return should_update, updated_commands, response

def fix_failed_command_with_llm(user_query: str, command: Dict[str, Any], output: str, 
                               classification: str, client) -> Tuple[Dict[str, Any], str]:
    """
    Fix a failed command using the LLM.
    
    Args:
        user_query (str): The original user query
        command (dict): The failed command
        output (str): The command output
        classification (str): The classification of the output
        client: The LLM client to use
        
    Returns:
        tuple: (fixed_command, fix_response)
    """
    prompt = f"""
    Fix this failed command:
    
    User Query: "{user_query}"
    Failed Command: "{command['action_input']}"
    Output: "{output[:500]}..."
    Classification: "{classification}"
    
    Please provide:
    1. Fixed command (in JSON format)
    2. Brief explanation of the fix
    """
    
    model_config = {
        "model_id": "deepseek-ai/DeepSeek-R1",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 32000,
        "temperature": 0.7
    }
    
    response = query_llm(client, prompt, model_config)
    
    # Parse response
    fixed_command = command
    try:
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            fixed_command = json.loads(json_match.group())
    except:
        pass
    
    return fixed_command, response

def calculate_node_value(node):
    """
    Calculate the value of a node based on its classification.
    
    Args:
        node (dict): The node to calculate value for
        
    Returns:
        float: The node's value (1.0 for success, 0.0 for failure)
    """
    classification = node.get('classification')
    return 1.0 if classification and classification.status == CommandStatus.SUCCESS else 0.0

def calculate_chain_value(node_id: str, mcts_manager) -> float:
    """
    Calculate the value of a chain of nodes from a given node to the root.
    
    Args:
        node_id (str): The ID of the node to start from
        mcts_manager: The MCTS manager containing the node map
        
    Returns:
        float: The total value of the chain
    """
    chain_value = 0.0
    current_node = mcts_manager.node_map.get(node_id)
    
    while current_node:
        # Get the node's value estimate
        value = current_node.get('valueEstimate', 0.0)
        
        # For command nodes, check if we have a classification
        if current_node.get('nodeType') == NodeType.COMMAND.value:
            value = calculate_node_value(current_node)
        
        chain_value += value
        current_node = mcts_manager.node_map.get(current_node.get('parentId'))
    
    return chain_value

def evaluate_attempt_with_llm(user_query: str, command_attempts: List[Dict[str, Any]], client) -> SolutionClassification:
    """
    Evaluate if the current attempts have reached a solution to the user query.
    
    Args:
        user_query (str): The original user query
        command_attempts (List[Dict[str, Any]]): List of command attempts so far
        client: The LLM client to use
        
    Returns:
        SolutionClassification: The classification of the solution
    """
    # Format command attempts for context
    attempts_text = ""
    for i, attempt in enumerate(command_attempts, 1):
        attempts_text += f"\nAttempt {i}:\n"
        attempts_text += f"Command: {attempt['command']}\n"
        attempts_text += f"Result: {attempt['result'][:500]}...\n"
        attempts_text += f"Status: {attempt['status']}\n"
    
    prompt = f"""
    Evaluate if the following command attempts have reached a solution to the user query.
    Your response must be in this exact format:
    
    Classification: COMPLETE or INCOMPLETE
    Brief explanation: [your explanation here]
    
    User Query: "{user_query}"
    
    Command Attempts:{attempts_text}
    """
    
    model_config = {
        "model": "deepseek-ai/DeepSeek-R1",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 32000,
        "temperature": 0.7
    }
    
    response = query_llm(client, prompt, model_config)
    
    # Extract the classification line
    classification_line = next(line for line in response.split('\n') if line.startswith('Classification:'))
    status = classification_line.split(':')[1].strip()
    
    # Extract the explanation
    explanation_line = next(line for line in response.split('\n') if line.startswith('Brief explanation:'))
    explanation = explanation_line.split(':', 1)[1].strip()
    
    # Create and return the SolutionClassification
    return SolutionClassification(
        status=SolutionStatus(status),
        reason=explanation
    )

def parse_classification_text(classification_text):
    """
    Parse a classification text into a CommandClassification object.
    
    Args:
        classification_text (str): The classification text to parse
        
    Returns:
        CommandClassification: A structured classification object
    """
    # Check for error indicators
    error_indicators = ["error", "Error", "ERROR", "undefined field", "Database Error"]
    if any(indicator in classification_text for indicator in error_indicators):
        return CommandClassification(
            status=CommandStatus.FAILURE,
            reason="Error detected in response"
        )
    
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


