from together import Together

def query_local_llm(client, user_query, model="deepseek-ai/DeepSeek-R1"):
    """
    Queries the local LLM with the given user query.
    
    Args:
        client: The Together client instance.
        user_query (str): The query to send to the LLM.
        model (str): The model to use for the LLM.
    
    Returns:
        str: The response from the LLM.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_query}
        ],
        #stop=["<|endoftext|>"],
        max_tokens=32000,
        temperature=0.7
    )

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
    return query_local_llm(client, prompt)

def classify_last_command_output_with_llm(observation_lines, user_query, client):
    """
    Uses the LLM to classify the output of the last command.
    
    Args:
        observation_lines (list): Lines of output from the last command
        user_query (str): The original user query
    
    Returns:
        str: Classification of the output
    """
    # Limit the observation to a reasonable size
    observation_text = "\n".join(observation_lines[:100])
    
    prompt = f"""
    You are evaluating the output of a command that was run to answer this query:
    "{user_query}"
    
    Here is the output of the command:
    ```
    {observation_text}
    ```
    
    First, analyze the user's query to understand:
    1. What type of data they're asking for
    2. Whether they specified any quantity requirements
    3. Whether they specified any additional criteria beyond the basic request
    
    Then classify the output based on these rules:
    1. SUCCESS: The command returned data that matches the basic type and criteria requested in the query
    2. PARTIAL SUCCESS: The command returned data but it doesn't match the type or criteria from the query
    3. FAILURE: The command failed to execute or returned no valid data
    
    Important: Do not add requirements that weren't in the original query. If the user asks for "features" without specifying quantity, returning one feature is a complete success.
    
    Provide your classification and a brief explanation that references the specific requirements from the user's query.
    """
    
    return query_local_llm(client, prompt)

def justify_solution_with_llm(observation_lines, user_query, llm_response, client):
    """
    Uses the LLM to justify the solution based on the command output.
    
    Args:
        observation_lines (list): Lines of output from the last command
        user_query (str): The original user query
        llm_response (str): The LLM's response that generated the command
    
    Returns:
        str: Justification for the solution
    """
    # Limit the observation to a reasonable size
    observation_text = "\n".join(observation_lines[:100])
    
    prompt = f"""
    You are analyzing the output of a command that was run to answer this query:
    "{user_query}"
    
    Here is the output of the command:
    ```
    {observation_text}
    ```
    
    Please explain whether this output successfully answers the query, and why or why not.
    Be specific about what information was found or not found.
    """
    
    return query_local_llm(client, prompt)

def derive_solution_with_llm(classification, justification, client):
    """
    Uses the LLM to derive a solution based on the classification and justification.
    
    Args:
        classification (str): Classification of the command output
        justification (str): Justification for the classification
    
    Returns:
        str: Derived solution
    """
    prompt = f"""
    Based on the following classification and justification of a command output:
    
    Classification:
    {classification}
    
    Justification:
    {justification}
    
    Please provide a concise solution or answer based on this information.
    If the command was not successful, indicate what would be needed to answer the query.
    """
    
    return query_local_llm(client, prompt)


