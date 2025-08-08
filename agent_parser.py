import re
import os,sys,json
from helper_functions import query_llm, remove_think_tags

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
    
    # Remove trailing periods and underscores
    sanitized = sanitized.rstrip('._')
    
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
    # Create a proper model config
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


def extract_mcp_actions(content):
    """
    Extract MCP tool calls from <action> tags in the content.
    Args:
        content (str): The content to extract actions from.
    Returns:
        list: A list of dicts with tool_name and parameters.
    """
    actions = []
    action_pattern = re.compile(r'<action>\s*(.*?)\s*</action>', re.DOTALL | re.IGNORECASE)
    action_contents = action_pattern.findall(content)
    for block in action_contents:
        block = block.strip()
        if not block:
            continue
        try:
            action_json = json.loads(block)
            if isinstance(action_json, dict) and "tool_name" in action_json and "parameters" in action_json:
                actions.append(action_json)
            elif isinstance(action_json, list):
                for item in action_json:
                    if isinstance(item, dict) and "tool_name" in item and "parameters" in item:
                        actions.append(item)
        except Exception as e:
            print(f"Error parsing <action> block: {e}\nBlock: {block}")
    return actions

def parse_commands(response_to_call, together_client=None):
    """
    Parses the MCP tool calls from the given response.
    Args:
        response_to_call (str): The response containing actions.
        together_client: (ignored, for compatibility)
    Returns:
        list: A list of MCP tool call dicts.
    """
    return extract_mcp_actions(response_to_call)
