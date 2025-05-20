import re
import os,sys,json
from helper_functions import query_llm, remove_think_tags
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import openai
from mcts_thinking import NodeType

# Add the sanitize_filename function at the top of the file, before main()
def sanitize_filename(filename):
    """
    Sanitizes a string to be used as a valid filename.
    
    Args:
        filename (str): The string to sanitize
    
    Returns:
        str: A sanitized string safe to use as a filename
    """
    # Replace problematic characters with underscores
    invalid_chars = r'[<>:"/\\|?*\n\r\t]'
    sanitized = re.sub(invalid_chars, '_', filename)
    sanitized = sanitized.replace(' ','_')
    
    # Truncate if too long (max 100 chars)
    if len(sanitized) > 100:
        sanitized = sanitized[:97]
    
    # Ensure it's not empty
    if not sanitized or sanitized.isspace():
        sanitized = "unnamed_query"
    
    return sanitized

def is_valid_json(json_str):
    """
    Validates if the provided string is a valid JSON.

    Args:
        json_str (str): The JSON string to validate.

    Returns:
        bool: True if valid JSON, False otherwise.
    """
    try:
        # Create a copy of the string with double curly braces replaced for validation
        validation_str = json_str.replace('{{', '{').replace('}}', '}')
        json.loads(validation_str)
        return True
    except json.JSONDecodeError:
        return False

def extract_json_elements(s):
    """
    Generator to extract JSON elements from a string.
    
    Args:
        s (str): The input string containing JSON elements.
    
    Yields:
        object: Parsed JSON objects or arrays.
    """
    decoder = json.JSONDecoder()
    idx = 0
    n = len(s)
    while idx < n:
        obj, idx_new = decoder.raw_decode(s, idx)
        yield obj
        idx = idx_new

def extract_commands_from_code_blocks(content):
    """
    Extracts commands from markdown-style code blocks.
    
    Args:
        content (str): The content to extract commands from.
        
    Returns:
        list: A list of extracted commands.
    """
    commands = []
    
    # Look for ```bash or ```shell style code blocks
    code_block_pattern = re.compile(r'```(?:bash|shell)?\s*(.*?)\s*```', re.DOTALL)
    code_blocks = code_block_pattern.findall(content)
    
    for block in code_blocks:
        # Split by lines and filter out empty ones
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        commands.extend(lines)
    
    return commands

def extract_commands(content):
    """
    Extract commands using standard JSON parsing.
    
    Args:
        content (str): The content to extract commands from.
        
    Returns:
        list: A list of extracted commands.
    """
    commands = []
    
    # Strategy 1: Try standard JSON parsing
    try:
        # Create a copy of the content with double curly braces replaced for parsing
        parsing_content = content.replace('{{', '{').replace('}}', '}')
        if is_valid_json(parsing_content):
            print("Using standard JSON parsing")
            json_data = json.loads(parsing_content)
            if isinstance(json_data, list):
                for item in json_data:
                    if isinstance(item, dict) and "action_input" in item:
                        commands.append(item["action_input"])
            elif isinstance(json_data, dict) and "action_input" in json_data:
                commands.append(json_data["action_input"])
    except Exception as e:
        print(f"Error during JSON parsing in extract_commands: {str(e)}")
    
    return commands

def extract_commands_with_llm(content, together_client):
    """
    Use an LLM to extract commands when other parsing methods fail.
    This is a last resort fallback strategy.
    
    Args:
        content (str): The content to extract commands from.
        together_client: The Together client instance to use for LLM calls
        
    Returns:
        list: A list of extracted commands as dictionaries with action and action_input keys.
    """
    print("Attempting command extraction using LLM")
    
    prompt = """Extract executable commands from the following text. 
Return only the actual commands that should be executed, one per line.
Do not include any explanations or markdown formatting.

Text to extract from:
{content}"""

    # Call LLM with the prompt
    # Create a proper x config
    model_config = {
        "model": "deepseek-ai/DeepSeek-R1",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt.format(content=content)}
        ],
        "max_tokens": 32000,
        "temperature": 0.7
    }
    
    response = query_llm(together_client, prompt.format(content=content), model_config)
    
    # Remove thinking tags and get the solution
    solution = remove_think_tags(response)
    
    # Extract action json from solution
    commands = extract_commands(solution)
    
    if commands:
        print(f"LLM extracted {len(commands)} commands")
        return commands
    else:
        print(f"Errored on Response:\n{response}")
        return []

class ActionType(Enum):
    BASH = "bash"
    PYTHON = "python"
    MCTS = "mcts"

class MCTSState:
    def __init__(self, node_type: NodeType, content: Dict[str, Any]):
        self.node_type = node_type
        self.content = content
        self.children: List[MCTSState] = []
        self.parent: Optional[MCTSState] = None
        self.value: float = 0.0
        self.visits: int = 0

    def add_child(self, child: 'MCTSState') -> None:
        child.parent = self
        self.children.append(child)

    def update_value(self, new_value: float) -> None:
        self.value = (self.value * self.visits + new_value) / (self.visits + 1)
        self.visits += 1

def parse_content(content: str, content_type: str, openai_client) -> Dict[str, Any]:
    """
    Parse content into a structured format.
    
    Args:
        content (str): The content to parse
        content_type (str): The type of content (action, think, answer)
        openai_client: The OpenAI client to use
        
    Returns:
        dict: Structured content
    """
    # Create a copy with double curly braces replaced for parsing
    parsing_content = content.replace('{{', '{').replace('}}', '}')
    if is_valid_json(parsing_content):
        return json.loads(parsing_content)
    
    # If not valid JSON, use LLM to parse
    examples = {
        "action": """
Example action node:
{
    "action": "bash",
    "action_input": "ls -la",
    "reasoning": "Checking directory contents to understand the current state",
    "confidence": 0.9
}

Example action node with multiple steps:
{
    "action": "python",
    "action_input": "import pandas as pd; df = pd.read_csv('data.csv')",
    "reasoning": "Loading data for analysis",
    "confidence": 0.85
}""",
        "think": """
Example thought node:
{
    "thought": "I need to analyze the current state and plan next steps",
    "reasoning": "The system needs to understand the context before taking action",
    "confidence": 0.95,
    "next_steps": ["check environment", "validate inputs", "plan execution"]
}

Example thought node with analysis:
{
    "thought": "The data structure suggests we need to transform it",
    "reasoning": "Current format doesn't match required schema",
    "confidence": 0.8,
    "analysis": {
        "current_format": "nested",
        "required_format": "flat",
        "transformation_needed": true
    }
}"""
    }
    
    prompt = f"""
    Parse this {content_type} content into a structured format.
    
    Here are examples of valid {content_type} node formats:
    {examples.get(content_type, "")}
    
    Content to parse:
    {content}
    
    Please provide the output in valid JSON format following the example structure above.
    """
    
    model_config = {
        "model_id": "gpt-4o",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_completion_tokens": 32000,
        "temperature": 0.7
    }
    
    response = query_llm(openai_client, prompt, model_config)
    return json.loads(response)

def parse_action_content(action_content: str, openai_client) -> List[Dict[str, Any]]:
    """
    Parse action content into a list of action dictionaries.
    
    Args:
        action_content (str): The action content to parse
        openai_client: The OpenAI client to use
        
    Returns:
        list: List of action dictionaries
    """
    return parse_content(action_content, "action", openai_client)

def parse_think_content(think_content: str, openai_client) -> Dict[str, Any]:
    """
    Parse think content into a structured format.
    
    Args:
        think_content (str): The think content to parse
        openai_client: The OpenAI client to use
        
    Returns:
        dict: Structured think content
    """
    return parse_content(think_content, "think", openai_client)

def parse_answer_content(answer_content: str, openai_client) -> Dict[str, Any]:
    """
    Parse answer content into a structured format.
    
    Args:
        answer_content (str): The answer content to parse
        openai_client: The OpenAI client to use
        
    Returns:
        dict: Structured answer content
    """
    return parse_content(answer_content, "answer", openai_client)

def create_mcts_state(node_type: NodeType, content: Dict[str, Any]) -> MCTSState:
    """
    Create a new MCTS state.
    
    Args:
        node_type (NodeType): The type of node
        content (dict): The node content
        
    Returns:
        MCTSState: The created state
    """
    return MCTSState(node_type, content)

def parse_mcts_action(action: Dict[str, Any]) -> Tuple[NodeType, str]:
    """
    Parse an MCTS action into type and input.
    
    Args:
        action (dict): The action to parse
        
    Returns:
        tuple: (node_type, action_input)
    """
    node_type = NodeType(action.get("action", "action"))
    action_input = action.get("action_input", "")
    return node_type, action_input

def validate_mcts_state(state: MCTSState) -> bool:
    """
    Validate an MCTS state.
    
    Args:
        state (MCTSState): The state to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not state.content:
        return False
    
    if state.node_type == NodeType.COMMAND:
        if "action" not in state.content or "action_input" not in state.content:
            return False
    
    if state.node_type == NodeType.SOLUTION:
        if "result" not in state.content:
            return False
    
    return True

def get_mcts_action_space(state: MCTSState) -> List[NodeType]:
    """
    Get the action space for an MCTS state.
    
    Args:
        state (MCTSState): The current state
        
    Returns:
        list: List of possible actions
    """
    if state.node_type == NodeType.COMMAND:
        return [NodeType.THOUGHT, NodeType.SOLUTION]

def get_mcts_state_value(state: MCTSState) -> float:
    """
    Get the value of an MCTS state.
    
    Args:
        state (MCTSState): The state to evaluate
        
    Returns:
        float: The state value
    """
    if state.node_type == NodeType.SOLUTION:
        # Solution nodes have direct value
        return state.content.get("value", 0.0)
    else:
        # Other nodes have value based on children
        if not state.children:
            return 0.0
        return max(child.value for child in state.children)

def format_mcts_state(state: MCTSState) -> str:
    """
    Format an MCTS state as a string.
    
    Args:
        state (MCTSState): The state to format
        
    Returns:
        str: Formatted state string
    """
    if state.node_type == NodeType.ROOT:
        return f"ROOT: {json.dumps(state.content)}"
    elif state.node_type == NodeType.COMMAND:
        formatted_state = f"Command: {state.content['action']}\n"
    else:
        formatted_state = f"SOLUTION: {state.content.get('text', '')[:100]}..."
    return formatted_state

def get_mcts_state_path(state: MCTSState) -> List[MCTSState]:
    """
    Get the path from root to the given state.
    
    Args:
        state (MCTSState): The target state
        
    Returns:
        list: List of states from root to target
    """
    path = []
    current = state
    while current:
        path.append(current)
        current = current.parent
    return list(reversed(path))

def format_mcts_path(path: List[MCTSState]) -> str:
    """
    Format an MCTS path as a string.
    
    Args:
        path (list): List of states in the path
        
    Returns:
        str: Formatted path string
    """
    return " -> ".join(format_mcts_state(state) for state in path)

def parse_commands(response_to_call, together_client=None):
    """
    Parses the commands from the given response.
    Gives preference to <action> tags and falls back to <think> tags if no commands are found in <action>.
    Uses multiple fallback strategies for more robust extraction.

    Args:
        response_to_call (str): The response containing commands.
        together_client: The Together client instance to use for LLM calls (optional)

    Returns:
        list: A list of command strings.
    """
    print(f'response_to_call = {response_to_call}')
    try:
        if not response_to_call.strip():
            print("Received an empty response for commands.")
            return []

        print(f"Raw response: {response_to_call}")

        commands = []

        # Define regex patterns for <action> tags with case-insensitivity and whitespace handling
        action_pattern = re.compile(r'<action>\s*(.*?)\s*</action>', re.DOTALL | re.IGNORECASE)

        # Process <action> tags
        action_contents = action_pattern.findall(response_to_call)
        print(f"Found {len(action_contents)} <action> tag(s).")

        for content in action_contents:
            content = content.strip()
            if not content:
                continue
            print("Processing content within <action> tags.")
            
            # First, check if the content is valid JSON
            if is_valid_json(content):
                print("Content is valid JSON, checking for required fields.")
                # Create a copy of the content with double curly braces replaced for parsing
                parsing_content = content.replace('{{', '{').replace('}}', '}')
                json_data = json.loads(parsing_content)
                
                # For JSON arrays
                if isinstance(json_data, list):
                    for item in json_data:
                        if isinstance(item, dict) and "action" in item and "action_input" in item:
                            commands.append({
                                "action": item["action"],
                                "action_input": item["action_input"]
                            })
                # For single JSON objects            
                elif isinstance(json_data, dict) and "action" in json_data and "action_input" in json_data:
                    commands.append({
                        "action": json_data["action"],
                        "action_input": json_data["action_input"]
                    })
                else:
                    print("JSON missing required fields (action and action_input), trying standard JSON parsing.")
            
            # If not valid JSON or missing required fields, try JSON extraction
            if not commands:
                print("Using standard JSON extraction for action tags.")
                action_commands = extract_commands(content)
                if action_commands:
                    commands.extend(action_commands)

        # If commands found in <action>, return them
        if commands:
            print(f"Commands found within <action> tags: {commands}")
            return commands

        # Final fallback: Try LLM-based extraction
        if not commands and together_client:
            print("Attempting LLM-based command extraction as final fallback")
            llm_commands = extract_commands_with_llm(response_to_call, together_client)
            if llm_commands:
                # Process each command from LLM
                for cmd in llm_commands:
                    commands.append({
                        "action": cmd.get("action", "command"),
                        "action_input": cmd["action_input"]
                    })

        if not commands:
            print("No valid commands found in the response.")
            print(f"Initial input received (response_to_call): {response_to_call}")
            return []  # Continue returning empty list instead of exiting
    
        print(f"Final extracted commands: {commands}")
        return commands

    except Exception as ex:
        print(f"Unexpected error while parsing commands: {ex}")
        print(f"Initial input received (response_to_call): {response_to_call}")
        return []  # Continue returning empty list instead of exiting

