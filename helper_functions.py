from together import Together
import re
import os
import json
from pydantic import BaseModel, Field
from enum import Enum

# Define the status enum
class CommandStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    PARTIAL = "PARTIAL"

# Define the Pydantic model for command classification
class CommandClassification(BaseModel):
    status: CommandStatus = Field(..., description="The status of the command execution")
    reason: str = Field(..., description="Brief explanation of the classification")
    
    def is_successful(self) -> bool:
        """Helper method to check if the command was successful"""
        return self.status == CommandStatus.SUCCESS

def remove_think_tags(text: str) -> str:
    """
    Removes <think> tags and their contents from text.
    
    Args:
        text (str): The text to process
    
    Returns:
        str: Text with <think> tags and their contents removed
    """
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

def query_llm(config, user_query, model_config=None):
    """
    Queries an LLM with the given user query using the provided configuration.
    
    Args:
        config (dict): Configuration containing the client and model info.
        user_query (str): The query to send to the LLM.
        model_config (dict, optional): Override configuration for the model.
    
    Returns:
        str: The response from the LLM.
    """
    # Use the provided config or fall back to model_config
    if model_config is None:
        model_config = config
    
    # Get the client from the config
    client = config['client']
    
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
    
    # Add standard parameters if they exist in the config
    for param in ['max_tokens', 'temperature']:
        if param in model_config:
            params[param] = model_config[param]
    
    # Call the API using the client from the config
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

def evaluate_solution_with_llm(user_query, solution, config):
    """
    Evaluates if a solution actually answers the user's query.
    
    Args:
        user_query (str): The original user query
        solution (str): The derived solution
        config (dict): Configuration containing the client and model info
    
    Returns:
        tuple: (success_bool, evaluation_text)
    """
    prompt = f"""
    You are evaluating if this solution ANSWERS THE USER'S QUESTION IN EXACT TERMS. HYPOTHETICAL PLANS ARE NOT ACCEPTABLE IF IT IS NOT EXECUTED.

    User Query: "{user_query}"
    
    Proposed Solution:
    ```
    {solution}
    ```

    HINT: Check if any actions failed during execution. Look for:
    - Error messages like "Invalid request parameters", "error", "failed"
    - Actions that returned error status
    - Actions that couldn't be completed
    - Placeholder text like "INSERT_FROM_PREVIOUS_RESULT" that wasn't replaced
    These often indicate that the solution is not executable and therefore not a valid solution. 
    It is okay to have an error and either fix it or try another approach, but if the critical information for answering the query is never obtained, then it is not a valid solution.

        Determine if this solution satisfactorily answers what the user asked:

    1. COMPLETE: Solution fully addresses the user's question in exact terms
       - Provides all information requested
       - Answer is directly relevant to the query
       - Necessary information is obtained
       - A reasonable user would be satisfied

    2. PARTIAL: Solution partially addresses the user's question. e.g. some critical actions failed or some information is missing.
       - Some relevant information, but incomplete
       - Missing important details requested in the query
       - A user would need to ask follow-up questions

    3. FAILURE: Solution fails to address the user's question
       - Information is irrelevant or completely wrong
       - Does not answer what was asked
       - A user would need to restart their inquiry

    In your answer, you must start with COMPLETE, PARTIAL, or FAILURE as the first word in all caps followed by a : and then the reason. Keep it brief and make sure to follow this format. Make sure to only begin with COMPLETE, PARTIAL, or FAILURE.
    """
    # Create a proper model config for o3-mini
    model_config = {
        "model": "o3-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    }
    
    evaluation = query_llm(config, prompt, model_config)
    
    # More strict success criteria - must start with COMPLETE: in all caps
    evaluation_upper = evaluation.strip()
    success = evaluation_upper.startswith("COMPLETE:")
    
    return success, evaluation

def generate_final_solution_with_llm(user_query, successful_commands, client):
    """
    Generates a final solution based on successful command outputs.
    
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

def fix_failed_command_with_llm(user_query, action_dict, observation_text, classification, config):
    """
    When a command fails, prompts the LLM to fix just that one command.
    
    Args:
        user_query (str): The original user query
        action_dict (dict): The failed action dictionary with tool_name and parameters keys
        observation_text (str): The output of the failed command
        classification (str): The classification of the failed command
        config (dict): Configuration containing the client and model info
    
    Returns:
        tuple: (fixed_action, llm_response)
    """
    import re
    # Limit the observation to a reasonable size
    observation_lines = observation_text.splitlines()[:100]
    observation_text = "\n".join(observation_lines)
    
    # Create a more detailed prompt for fixing the command
    fix_prompt = f"""A bioinformatics command failed with an error. Please fix the command based on the error message.

Original Action:
{json.dumps(action_dict, indent=2)}

Error Message:
{observation_text}

Please provide a corrected version of the action. The action should use the same tool_name but with corrected parameters.

Return ONLY a JSON object with the corrected action in this exact format:
{{
    "tool_name": "{action_dict['tool_name']}",
    "parameters": {{
        "command": "corrected command here"
    }}
}}

Common fixes:
- Fix syntax errors in p3-tools commands
- Correct parameter names or values
- Fix typos in field names
- Ensure proper quoting

If you cannot fix the command, return: {{"unfixable": true}}"""
    
    # Use the config-based query_llm
    model_config = {
        "model": "deepseek-ai/DeepSeek-R1",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that fixes bioinformatics commands."},
            {"role": "user", "content": fix_prompt}
        ],
        "max_tokens": 1000,
        "temperature": 0.1
    }
    
    fix_response = query_llm(config, fix_prompt, model_config)
    
    # Try to parse the JSON response
    try:
        fixed_action = json.loads(fix_response.strip())
        if fixed_action.get("unfixable"):
            return None, fix_response
        return fixed_action, fix_response
    except json.JSONDecodeError:
        # If response isn't valid JSON, try to extract JSON from it
        json_match = re.search(r'\{.*\}', fix_response, re.DOTALL)
        if json_match:
            try:
                fixed_action = json.loads(json_match.group())
                if fixed_action.get("unfixable"):
                    return None, fix_response
                return fixed_action, fix_response
            except json.JSONDecodeError:
                pass
        return None, fix_response


