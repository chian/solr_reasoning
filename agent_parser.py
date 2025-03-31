import re
import logging
import os,sys,json
from helper_functions import query_local_llm

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
        json.loads(json_str)
        return True
    except json.JSONDecodeError:
        print(f"Invalid JSON: {json_str}")
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
        try:
            obj, idx_new = decoder.raw_decode(s, idx)
            yield obj
            idx = idx_new
        except json.JSONDecodeError:
            # Move to the next character if no valid JSON is found
            idx += 1

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

def extract_direct_commands(content):
    """
    Attempts to extract commands directly when no JSON structure is found.
    Looks for patterns like 'curl', 'wget', etc. at the start of lines.
    
    Args:
        content (str): The content to extract commands from.
        
    Returns:
        list: A list of extracted commands.
    """
    commands = []
    
    # Common command line tools to look for
    command_prefixes = [
        'curl', 'wget', 'git', 'python', 'pip', 'npm', 'node', 
        'cat', 'ls', 'cd', 'mkdir', 'rm', 'cp', 'mv',
        'grep', 'find', 'sed', 'awk', 'bash', 'ssh',
        'docker', 'kubectl', 'apt', 'yum', 'brew'
    ]
    
    # Split by lines and look for lines starting with common commands
    lines = content.split('\n')
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
            
        # Check if line starts with a common command
        for prefix in command_prefixes:
            if stripped.startswith(prefix + ' ') or stripped == prefix:
                commands.append(stripped)
                break
                
    return commands

def extract_commands_with_fallbacks(content, logging):
    """
    Extract commands using multiple strategies with fallbacks.
    
    Args:
        content (str): The content to extract commands from.
        logging: The logging object.
        
    Returns:
        list: A list of extracted commands.
    """
    commands = []
    
    # Strategy 1: Try standard JSON parsing
    if is_valid_json(content):
        logging.debug("Using standard JSON parsing")
        json_data = json.loads(content)
        if isinstance(json_data, list):
            for item in json_data:
                if isinstance(item, dict) and "action_input" in item:
                    commands.append(item["action_input"])
        elif isinstance(json_data, dict) and "action_input" in json_data:
            commands.append(json_data["action_input"])
            
    # Strategy 2: Try parsing JSON line by line
    if not commands:
        logging.debug("Trying JSON line-by-line parsing")
        for json_element in extract_json_elements(content):
            if isinstance(json_element, list):
                for item in json_element:
                    if isinstance(item, dict) and "action_input" in item:
                        commands.append(item["action_input"])
            elif isinstance(json_element, dict) and "action_input" in json_element:
                commands.append(json_element["action_input"])
                
    # Strategy 3: Look for markdown code blocks
    if not commands:
        logging.debug("Looking for code blocks")
        code_block_commands = extract_commands_from_code_blocks(content)
        if code_block_commands:
            commands.extend(code_block_commands)
            
    # Strategy 4: Direct command extraction
    if not commands:
        logging.debug("Trying direct command extraction")
        direct_commands = extract_direct_commands(content)
        if direct_commands:
            commands.extend(direct_commands)
            
    return commands

def extract_commands_with_llm(content, logging):
    """
    Use an LLM to extract commands when other parsing methods fail.
    This is a last resort fallback strategy.
    
    Args:
        content (str): The content to extract commands from.
        logging: The logging object.
        
    Returns:
        list: A list of extracted commands.
    """
    logging.debug("Attempting command extraction using LLM")
    
    # Import here to avoid circular dependency
    from solr_together import together_client
    
    prompt = """Extract executable commands from the following text. 
Return only the actual commands that should be executed, one per line.
Do not include any explanations or markdown formatting.

Text to extract from:
{content}"""

    try:
        # Call LLM with the prompt
        response = query_local_llm(together_client, prompt.format(content=content))
        
        # Remove thinking tags and get the solution
        solution = remove_think_tags(response)
        
        # Split into commands
        commands = [cmd.strip() for cmd in solution.split('\n') if cmd.strip()]
        
        if commands:
            logging.debug(f"LLM extracted {len(commands)} commands")
            return commands
            
    except Exception as e:
        logging.error(f"Error during LLM command extraction: {str(e)}")
        
    return []


def parse_commands(response_to_call, logging):
    """
    Parses the commands from the given response.
    Gives preference to <action> tags and falls back to <think> tags if no commands are found in <action>.
    Uses multiple fallback strategies for more robust extraction.

    Args:
        response_to_call (str): The response containing commands.

    Returns:
        list: A list of command strings.
    """
    print(f'response_to_call = {response_to_call}')
    try:
        if not response_to_call.strip():
            logging.warning("Received an empty response for commands.")
            return []

        logging.debug(f"Raw response: {response_to_call}")

        commands = []

        # Define regex patterns for <action> and <think> tags with case-insensitivity and whitespace handling
        tag_patterns = {
            'action': re.compile(r'<action>\s*(.*?)\s*</action>', re.DOTALL | re.IGNORECASE),
            'think': re.compile(r'<think>\s*(.*?)\s*</think>', re.DOTALL | re.IGNORECASE)
        }

        # Process <action> tags first
        action_contents = tag_patterns['action'].findall(response_to_call)
        logging.debug(f"Found {len(action_contents)} <action> tag(s).")

        for content in action_contents:
            content = content.strip()
            if not content:
                continue
            logging.debug("Processing content within <action> tags.")
            
            # Use the enhanced extraction with fallbacks
            action_commands = extract_commands_with_fallbacks(content, logging)
            if action_commands:
                commands.extend(action_commands)

        # If commands found in <action>, return them without processing <think>
        if commands:
            logging.debug(f"Commands found within <action> tags: {commands}")
            return commands

        # If no commands found in <action>, process <think> tags
        think_contents = tag_patterns['think'].findall(response_to_call)
        logging.debug(f"Found {len(think_contents)} <think> tag(s).")

        for content in think_contents:
            content = content.strip()
            if not content:
                continue
            logging.debug("Processing content within <think> tags.")
            
            # Use the enhanced extraction with fallbacks
            think_commands = extract_commands_with_fallbacks(content, logging)
            if think_commands:
                commands.extend(think_commands)

        # If still no commands found but we have code blocks directly in the response (outside tags)
        if not commands:
            logging.debug("Looking for code blocks outside of tags as last resort.")
            outside_commands = extract_commands_from_code_blocks(response_to_call)
            if outside_commands:
                commands.extend(outside_commands)
                
            # Direct command extraction as very last resort
            if not commands:
                outside_direct_commands = extract_direct_commands(response_to_call)
                if outside_direct_commands:
                    commands.extend(outside_direct_commands)
                    
                # Final fallback: Try LLM-based extraction
                if not commands:
                    logging.debug("Attempting LLM-based command extraction as final fallback")
                    llm_commands = extract_commands_with_llm(response_to_call, logging)
                    if llm_commands:
                        commands.extend(llm_commands)

        if not commands:
            logging.error("No valid commands found in the response.")
            logging.error(f"Initial input received (response_to_call): {response_to_call}")
            return []  # Continue returning empty list instead of exiting
    
        logging.debug(f"Final extracted commands: {commands}")
        return commands

    except Exception as ex:
        logging.exception(f"Unexpected error while parsing commands: {ex}")
        logging.error(f"Initial input received (response_to_call): {response_to_call}")
        return []  # Continue returning empty list instead of exiting

def test_parse_commands(logging):
    """Unit tests for parse_commands function."""

    # Test Case 1: JSON Array within <action>
    response1 = """
    <action>
    [
        {
            "action": "p3-tools",
            "action_input": "command1"
        },
        {
            "action": "p3-tools",
            "action_input": "command2"
        }
    ]
    </action>
    <think>
    [
        {
            "action": "p3-tools",
            "action_input": "command3"
        }
    ]
    </think>
    """
    expected1 = ["command1", "command2"]
    result1 = parse_commands(response1, logging)
    assert result1 == expected1, f"Test Case 1 Failed: Expected {expected1}, Got {result1}"
    print("Test Case 1 Passed: JSON Array within <action> prioritized over <think>")

    # Test Case 2: JSONL within <action>
    response2 = """
    <action>
    {
        "action": "p3-tools",
        "action_input": "command4"
    }
    {
        "action": "p3-tools",
        "action_input": "command5"
    }
    </action>
    <think>
    {
        "action": "p3-tools",
        "action_input": "command6"
    }
    </think>
    """
    expected2 = ["command4", "command5"]
    result2 = parse_commands(response2, logging)
    assert result2 == expected2, f"Test Case 2 Failed: Expected {expected2}, Got {result2}"
    print("Test Case 2 Passed: JSONL within <action> prioritized over <think>")

    # Test Case 3: Single JSON Object within <action>
    response3 = """
    <action>
    {
        "action": "p3-tools",
        "action_input": "command7"
    }
    </action>
    <think>
    {
        "action": "p3-tools",
        "action_input": "command8"
    }
    </think>
    """
    expected3 = ["command7"]
    result3 = parse_commands(response3, logging)
    assert result3 == expected3, f"Test Case 3 Failed: Expected {expected3}, Got {result3}"
    print("Test Case 3 Passed: Single JSON Object within <action> prioritized over <think>")

    # Test Case 4: No Commands in <action>, Commands in <think>
    response4 = """
    <action>
    This is some explanatory text without commands.
    </action>
    <think>
    {
        "action": "p3-tools",
        "action_input": "command9"
    }
    </think>
    """
    expected4 = ["command9"]
    result4 = parse_commands(response4, logging)
    assert result4 == expected4, f"Test Case 4 Failed: Expected {expected4}, Got {result4}"
    print("Test Case 4 Passed: No Commands in <action>, Fallback to <think>")

    # Test Case 5: Both <action> and <think> tags have no valid commands
    response5 = """
    <action>
    No commands here, just text.
    </action>
    <think>
    Still no commands in think either.
    </think>
    """
    expected5 = []
    result5 = parse_commands(response5, logging)
    assert result5 == expected5, f"Test Case 5 Failed: Expected {expected5}, Got {result5}"
    print("Test Case 5 Passed: Both <action> and <think> have no commands")

    # Test Case 6: Only <action> commands are processed when present
    response6 = """
    <action>
    {
        "action": "p3-tools",
        "action_input": "command10"
    }
    </action>
    <think>
    {
        "action": "p3-tools",
        "action_input": "command11"
    }
    {
        "action": "p3-tools",
        "action_input": "command12"
    }
    </think>
    """
    expected6 = ["command10"]
    result6 = parse_commands(response6, logging)
    assert result6 == expected6, f"Test Case 6 Failed: Expected {expected6}, Got {result6}"
    print("Test Case 6 Passed: Only <action> commands are processed when present")

    # Test Case 7: Multiple <action> tags with commands
    response7 = """
    <action>
    {
        "action": "p3-tools",
        "action_input": "command13"
    }
    </action>
    <action>
    [
        {
            "action": "p3-tools",
            "action_input": "command14"
        },
        {
            "action": "p3-tools",
            "action_input": "command15"
        }
    ]
    </action>
    <think>
    {
        "action": "p3-tools",
        "action_input": "command16"
    }
    </think>
    """
    expected7 = ["command13", "command14", "command15"]
    result7 = parse_commands(response7, logging)
    assert result7 == expected7, f"Test Case 7 Failed: Expected {expected7}, Got {result7}"
    print("Test Case 7 Passed: Multiple <action> tags with commands processed correctly")

    # Test Case 8: Invalid JSON within <action>, valid commands within <think>
    response8 = """
    <action>
    {
        "action": "p3-tools",
        "action_input": "command17",
    }  # Invalid JSON due to trailing comma
    </action>
    <think>
    {
        "action": "p3-tools",
        "action_input": "command18"
    }
    </think>
    """
    expected8 = ["command18"]
    result8 = parse_commands(response8, logging)
    assert result8 == expected8, f"Test Case 8 Failed: Expected {expected8}, Got {result8}"
    print("Test Case 8 Passed: Invalid JSON in <action>, Fallback to <think>")

    # Test Case 10: Original Use Case with only <think>
    response10 = """
    <think>
    {
        "action": "p3-tools",
        "action_input": "p3-all-genomes --eq genus,Salmonella --eq species,enterica | p3-get-genome-features --in feature_type,CDS,rna --attr genome_id --attr patric_id --attr feature_type --attr start --attr end --attr strand --attr product | p3-sort genome_id feature.sequence_id feature.start/n feature.strand"
    }
    </think>
    """
    expected10 = [
        "p3-all-genomes --eq genus,Salmonella --eq species,enterica | p3-get-genome-features --in feature_type,CDS,rna --attr genome_id --attr patric_id --attr feature_type --attr start --attr end --attr strand --attr product | p3-sort genome_id feature.sequence_id feature.start/n feature.strand"
    ]
    result10 = parse_commands(response10, logging)
    assert result10 == expected10, f"Test Case 10 Failed: Expected {expected10}, Got {result10}"
    print("Test Case 10 Passed: Original Use Case with only <think>")

    print("All tests completed successfully.")


