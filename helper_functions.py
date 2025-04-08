from together import Together
import re
import os

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
    Queries an LLM with the given user query using the provided configuration.
    
    Args:
        client: The client instance (Together or OpenAI).
        user_query (str): The query to send to the LLM.
        model_config (dict): Configuration for the model.
    
    Returns:
        str: The response from the LLM.
    """
    # Get the model ID from the config
    model_id = model_config.get('model_id', model_config.get('model'))
    
    # Get the messages from the config
    messages = model_config['messages']
    
    # Create the parameters dictionary
    params = {
        "model": model_id,
        "messages": messages
    }
    
    # Add any additional parameters from the config
    api_params = model_config.get('api_params', {})
    params.update(api_params)
    
    # Call the API
    response = client.chat.completions.create(**params)
    return response.choices[0].message.content

def parse_last_command_output_with_llm(terminal_output_lines, client):
    """
    Takes in the command output and uses an LLM to extract relevant information.

    Parameters:
        terminal_output_lines (list of str): The lines captured from the command output.

    Returns:
        str: The LLM's best guess at the parsed output.
    """
    output_str = "\n".join(terminal_output_lines)

    prompt = f"""You are an expert in bioinformatics. Below is the raw 
        output from a solr query execution:

        {output_str}

        Your task is to extract and return the relevant information from the output.

        Output:
        """
    print("Parsing Command Output with LLM...")
    
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
    
    return query_llm(client, prompt, model_config)

def classify_last_command_output_with_llm(user_query, cmd, observation_text, client):
    """
    Uses the LLM to classify the output of the last command.
    
    Args:
        user_query (str): The original user query
        cmd (str): The command that was executed
        observation_text (str): Output from the command execution
        client: The LLM client to use
    
    Returns:
        str: Classification of the output
    """
    # Limit the observation to a reasonable size
    observation_lines = observation_text.splitlines()[:100]
    observation_text = "\n".join(observation_lines)
    
    prompt = f"""
    You are evaluating if a command execution was TECHNICALLY SUCCESSFUL.

    User Query: "{user_query}"

    Here is the output of the command:
    ```
    {observation_text}
    ```

    Classify ONLY the TECHNICAL execution of this command:

    1. SUCCESS: Command executed correctly and returned valid, non-error data
       - No error messages, API errors, or undefined field errors
       - Returned properly formatted data
       - Output can be used for further analysis (even if incomplete)

    2. FAILURE: Command failed technically
       - Contains syntax errors
       - Shows API errors like "undefined field" or HTTP error codes
       - Returned error messages instead of data
       - Output cannot be used for further processing

    Important: This is ONLY about technical execution, NOT about answering the user's query.
    A command can succeed technically but not provide a complete answer.

    Classification: [SUCCESS/FAILURE]
    Brief explanation: 
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
    
    # Get the classification from the LLM
    response = query_llm(client, prompt, model_config)
    
    # Extract the classification and reason from the response
    try:
        # Try to find the classification and reason in the response
        status_match = re.search(r'Classification:\s*(SUCCESS|FAILURE)', response, re.IGNORECASE)
        reason_match = re.search(r'Brief explanation:\s*(.*?)(?:\n\n|\Z)', response, re.DOTALL)
        
        if status_match and reason_match:
            status = status_match.group(1).upper()
            reason = reason_match.group(1).strip()
            
            # Create a CommandClassification object
            from solr_together import CommandClassification, CommandStatus
            classification = CommandClassification(
                status=CommandStatus(status),
                reason=reason
            )
            
            # Return the original response format for backward compatibility
            return f"Classification: {status}\nBrief explanation: {reason}"
        else:
            # If we can't extract the structured data, return the original response
            return response
    except Exception as e:
        # If there's an error, return the original response
        print(f"Error parsing classification: {str(e)}")
        return response

def derive_solution_with_llm(classification, client):
    """
    Uses the LLM to derive a solution based on the classification.
    
    Args:
        classification (str): Classification of the command output
    
    Returns:
        str: Derived solution
    """
    prompt = f"""
    Based on the following classification of a command output:
    
    Classification:
    {classification}
    
    Please provide a concise solution or answer based on this information.
    If the command was not successful, indicate what would be needed to answer the query.
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
    
    return query_llm(client, prompt, model_config)

def generate_training_output(user_query, think_content, action_content, answer_content, output_dir="query_results"):
    """
    Generates a training output file for successful runs.
    
    Args:
        user_query (str): The original user query
        think_content (str): The reasoning process from <think> tags
        action_content (str): The actions taken and their outputs
        answer_content (str): The final answer
        output_dir (str): Directory to save the training output
    
    Returns:
        str: Path to the generated training file
    """
    from datetime import datetime
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with just timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"training_{timestamp}.txt"
    filepath = os.path.join(output_dir, filename)
    
    # Format the training output with template
    training_content = f"""USER_QUERY: A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer, sometimes using actions to find the answer. The reasoning process, actions, and answer are enclosed within <think></think>, <action> </action> and
<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <action> [ {{"action": type of action here, "action_input": specific command to launch here, "action_output": put observation here}}, {{ add more actions as needed}} ]
<answer> answer here </answer>. 

User: {user_query}

Assistant:
<think>
{think_content}
</think>

<action>
{action_content}
</action>

<answer>
{answer_content}
</answer>
"""
    
    # Write to file
    with open(filepath, 'w') as f:
        f.write(training_content)
    
    return filepath

def evaluate_solution_with_llm(user_query, solution, client):
    """
    Evaluates if a solution actually answers the user's query.
    
    Args:
        user_query (str): The original user query
        solution (str): The derived solution
        client: The LLM client to use
    
    Returns:
        tuple: (success_bool, evaluation_text)
    """
    prompt = f"""
    You are evaluating if this solution ANSWERS THE USER'S QUESTION.

    User Query: "{user_query}"
    
    Proposed Solution:
    ```
    {solution}
    ```

    Determine if this solution satisfactorily answers what the user asked:

    1. COMPLETE: Solution fully addresses the user's question
       - Provides all information requested
       - Answer is directly relevant to the query
       - A reasonable user would be satisfied

    2. PARTIAL: Solution partially addresses the user's question
       - Some relevant information, but incomplete
       - Missing important details requested in the query
       - A user would need to ask follow-up questions

    3. FAILURE: Solution fails to address the user's question
       - Information is irrelevant or completely wrong
       - Does not answer what was asked
       - A user would need to restart their inquiry

    Answer with COMPLETE, PARTIAL, or FAILURE and explain your reasoning.
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
    
    evaluation = query_llm(client, prompt, model_config)
    success = "complete" in evaluation.lower()
    
    return success, evaluation

def generate_final_solution_with_llm(user_query, successful_commands, client):
    """
    Generates a comprehensive final solution based on all command outputs.
    
    Args:
        user_query (str): The original user query
        successful_commands (list of dicts): List of all commands and outputs
        client: The LLM client to use
    
    Returns:
        str: The final solution that answers the user query
    """
    # Combine all observations for a comprehensive solution
    command_trace = "\n\n".join([
        f"Command: {cmd['action_input']}\nOutput: {cmd['action_output']}"
        for cmd in successful_commands
    ])
    
    # Create a prompt to generate the final solution
    solution_prompt = f"""
    You will be given a successful trace of commands and outputs and your job is to
    summarize the success in plain words so that the novice computer user can 
    understand what worked and why it worked.

    User Query: {user_query}
    
    Command Trace: 
    {command_trace}
    
    Remember, I am not asking for a solution because I already solved this problem. 
    I am asking for a solution summary only.
    """
    
    # Create a proper model config
    model_config = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": solution_prompt}
        ],
        "max_tokens": 8000,
        "temperature": 0.7
    }
    
    # Generate the solution
    solution_raw = query_llm(client, solution_prompt, model_config)
    
    # Clean up the solution
    final_solution = remove_think_tags(solution_raw)
    
    return final_solution

def update_commands_with_llm(user_query, commands, command_index, cmd, observation_text, classification, client):
    """
    Uses the LLM to update the command list based on the current command's output.
    
    Args:
        user_query (str): The original user query
        commands (list): List of commands to update
        command_index (int): Index of the current command
        cmd (str): The command that was executed
        observation_text (str): Output from the command execution
        classification (str): The classification of the command output
        client: The LLM client to use
    
    Returns:
        tuple: (should_update, updated_commands, update_response)
            - should_update (bool): Whether the commands should be updated
            - updated_commands (list): Updated list of commands
            - update_response (str): Response from the LLM about the update
    """
    # Limit the observation to a reasonable size
    observation_lines = observation_text.splitlines()[:100]
    observation_text = "\n".join(observation_lines)
    
    prompt = f"""
    Based on the following command execution:
    
    User Query: "{user_query}"
    
    Command executed:
    {cmd}
    
    Command output:
    ```
    {observation_text}
    ```
    
    Classification:
    {classification}
    
    Please provide the next command to execute to answer the query.
    Return ONLY the command, nothing else.
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
    
    next_cmd = query_llm(client, prompt, model_config)
    # Create a new command with the same action as the previous command
    new_cmd = {"action": commands[command_index]["action"], "action_input": next_cmd.strip()}
    commands[command_index + 1] = new_cmd
    return True, commands, "Commands updated based on execution result"

def fix_failed_command_with_llm(user_query, cmd, observation_text, classification, client):
    """
    When a command fails, prompts the LLM to fix just that one command.
    
    Args:
        user_query (str): The original user query
        cmd (dict): The failed command dictionary with action and action_input keys
        observation_text (str): The output of the failed command
        classification (str): The classification of the failed command
        client: The LLM client to use
    
    Returns:
        tuple: (fixed_command, llm_response)
    """
    import re
    import os
    
    # Limit the observation to a reasonable size
    observation_lines = observation_text.splitlines()[:100]
    observation_text = "\n".join(observation_lines)
    
    # Get relevant API documentation based on the command and error
    api_context, doc_filename = get_relevant_api_docs(cmd['action_input'], observation_text)
    
    # Create the prompt for fixing the failed command
    fix_prompt = f"""
    User Query: {user_query}
    
    The following command failed:
    ```
    {cmd['action_input']}
    ```
    
    Output:
    {observation_text}
    
    Classification:
    {classification}
    
    {api_context}
    
    IMPORTANT: ONLY fix this specific command syntax. DO NOT:
    - Suggest multiple commands or a completely different approach
    - Recommend tools or utilities not mentioned in the original command
    - Propose alternative APIs or endpoints
    - Change the fundamental approach
    - DO NOT TRY TO USE ALTERNATIVES TO THE CORRECT BV-BRC SOLR CALL
    - USE BV-BRC ONLY - USE BV-BRC ONLY - USE BV-BRC ONLY
        --This means the "action" should be "bash" since that's how you make a SOLR query
        --This also means the "action_input" should start with "curl" since that's how you make a SOLR query
    - Consider if you should think about using a different endpoint.
        --For example: consider genome to find genome ids before using genome_features
    
    Simply correct the syntax/parameters of THIS EXACT command to make it work.
    Focus on addressing the specific issue that caused this command to fail.
    
    Your response should be in this format:
    <fixed_command>
    [the corrected command]
    </fixed_command>
    <explanation>
    [brief explanation of what was fixed]
    </explanation>
    """
    
    # Create a proper model config
    model_config = {
        "model": "deepseek-ai/DeepSeek-R1",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": fix_prompt}
        ],
        "max_tokens": 32000,
        "temperature": 0.7
    }
    
    # Ask the LLM to fix the command
    fix_response = query_llm(client, fix_prompt, model_config)
    
    # Extract the fixed command from the LLM response
    fixed_match = re.search(r'<fixed_command>(.*?)</fixed_command>', fix_response, re.DOTALL)
    
    if fixed_match:
        # Create a new command dictionary with the fixed command
        fixed_cmd = {
            "action": cmd["action"],
            "action_input": fixed_match.group(1).strip()
        }
    else:
        # If no fixed command found, return the original command
        fixed_cmd = cmd
    
    # Add documentation filename to response if it was accessed
    if doc_filename:
        fix_response = fix_response + f"\n<accessed_documentation>{doc_filename}</accessed_documentation>"
    
    # Return the updated cmd dictionary
    return fixed_cmd, fix_response

def get_relevant_api_docs(cmd, error_text, api_docs_dir="api_docs"):
    """
    Extracts relevant API documentation based on the command and error message.
    
    Args:
        cmd (str): The command that failed
        error_text (str): The error output
        api_docs_dir (str): Directory containing API documentation files
    
    Returns:
        tuple: (api_context, doc_filename or None)
    """
    # Default empty context
    api_context = "RELEVANT API INFORMATION:\nNo specific API documentation could be found for this error."
    doc_filename = None
    
    # Check for common API errors
    field_error_match = re.search(r"undefined field (\w+)", error_text)
    endpoint_match = re.search(r"https?://[^/]+/api/(\w+)/", cmd)
    
    if field_error_match and endpoint_match:
        # Get problematic field and endpoint
        problem_field = field_error_match.group(1)
        endpoint = endpoint_match.group(1)
        
        # Try to read relevant API docs
        docs_file = os.path.join(api_docs_dir, f"{endpoint}.txt")
        if os.path.exists(docs_file):
            doc_filename = f"{endpoint}.txt"
            with open(docs_file, 'r') as f:
                api_docs = f.read()
            
            # Include the complete API documentation
            api_context = f"""
            RELEVANT API DOCUMENTATION:
            
            Error involves undefined field '{problem_field}' in the '{endpoint}' API.
            
            COMPLETE API DOCUMENTATION for {endpoint}:
            {api_docs}
            
            Important: ONLY use fields that are explicitly listed in the documentation above.
            Do not use '{problem_field}' as it is undefined and not available.
            
            If you need to filter by genus/species, you should:
            1. First query the genome API to get genome_ids
            2. Then use those IDs with the genome_feature API
            
            Example pattern:
            ```
            # Get genome_id for Salmonella enterica
            curl -s "https://www.bv-brc.org/api/genome/?eq(genus,Salmonella)&eq(species,enterica)&select(genome_id)&limit(1)"
            
            # Use genome_id to query features
            curl -s "https://www.bv-brc.org/api/genome_feature/?eq(genome_id,GENOME_ID)&select(feature_id,product)&limit(10)"
            ```
            """
    
    return api_context, doc_filename


