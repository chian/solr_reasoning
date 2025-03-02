import re
import logging
import os,sys,json

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

def parse_commands(response_to_call, logging):
    """
    Parses the commands from the given response.
    Gives preference to <action> tags and falls back to <think> tags if no commands are found in <action>.

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
            logging.debug("Processing JSON content within <action> tags.")
            
            # Try to parse as JSON
            try:
                # First try as a complete JSON object or array
                json_data = json.loads(content)
                if isinstance(json_data, list):
                    for item in json_data:
                        if isinstance(item, dict) and "action_input" in item:
                            commands.append(item["action_input"])
                            logging.debug(f"Extracted command from JSON array: {item['action_input']}")
                elif isinstance(json_data, dict) and "action_input" in json_data:
                    commands.append(json_data["action_input"])
                    logging.debug(f"Extracted command from JSON object: {json_data['action_input']}")
            except json.JSONDecodeError:
                # If not a complete JSON, try to extract JSON objects line by line
                for json_element in extract_json_elements(content):
                    if isinstance(json_element, list):
                        for item in json_element:
                            if isinstance(item, dict) and "action_input" in item:
                                commands.append(item["action_input"])
                                logging.debug(f"Extracted command from JSON array element: {item['action_input']}")
                    elif isinstance(json_element, dict) and "action_input" in json_element:
                        commands.append(json_element["action_input"])
                        logging.debug(f"Extracted command from JSON object element: {json_element['action_input']}")

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
            logging.debug("Processing JSON content within <think> tags.")
            
            # Try to parse as JSON
            try:
                # First try as a complete JSON object or array
                json_data = json.loads(content)
                if isinstance(json_data, list):
                    for item in json_data:
                        if isinstance(item, dict) and "action_input" in item:
                            commands.append(item["action_input"])
                            logging.debug(f"Extracted command from JSON array: {item['action_input']}")
                elif isinstance(json_data, dict) and "action_input" in json_data:
                    commands.append(json_data["action_input"])
                    logging.debug(f"Extracted command from JSON object: {json_data['action_input']}")
            except json.JSONDecodeError:
                # If not a complete JSON, try to extract JSON objects line by line
                for json_element in extract_json_elements(content):
                    if isinstance(json_element, list):
                        for item in json_element:
                            if isinstance(item, dict) and "action_input" in item:
                                commands.append(item["action_input"])
                                logging.debug(f"Extracted command from JSON array element: {item['action_input']}")
                    elif isinstance(json_element, dict) and "action_input" in json_element:
                        commands.append(json_element["action_input"])
                        logging.debug(f"Extracted command from JSON object element: {json_element['action_input']}")

        if not commands:
            logging.error("No valid commands found in the JSON content.")
            logging.error(f"Initial input received (response_to_call): {response_to_call}")
            return []  # Continue returning empty list instead of exiting

        logging.debug(f"Final extracted commands: {commands}")
        return commands

    except Exception as ex:
        logging.exception(f"Unexpected error while parsing commands: {ex}")
        logging.error(f"Initial input received (response_to_call): {response_to_call}")
        return []  # Continue returning empty list instead of exiting

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


