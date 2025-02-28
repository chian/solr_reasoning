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
import tempfile

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

#----------------------------------
# Removed BV-BRC client portion as Terminal interaction is no longer needed

#----------------------------------
#LLM-based command execution

queries = [
    "Retrieve the annotated features of a Salmonella enterica genome.",
    "Get all coding sequences (CDS) for a Pseudomonas aeruginosa genome.",
    "Find all pseudogenes in a Mycobacterium tuberculosis strain.",
    "Extract all ribosomal RNA (rRNA) genes from a Vibrio cholerae genome.",
    "List all tRNA genes in a Klebsiella pneumoniae genome.",
    "Retrieve all virulence factors in a clinical Escherichia coli isolate.",
    "Find all phage-associated genes in a Listeria monocytogenes genome.",
    "Retrieve all hypothetical proteins in a recently sequenced Bacillus subtilis genome.",
    "Extract all protein sequences from a Clostridium difficile strain.",
    "Get all gene ontology (GO) terms associated with proteins in a Yersinia pestis genome.",
    
    # Comparative Genomics
    "Identify orthologous genes between Escherichia coli and Shigella flexneri.",
    "Retrieve paralogous genes in a Streptococcus pneumoniae genome.",
    "Compare metabolic pathways between two Helicobacter pylori strains.",
    "Find genes that are unique to a pathogenic Acinetobacter baumannii strain compared to a commensal strain.",
    "List all genes in a Vibrio parahaemolyticus genome that have no homologs in other Vibrio species.",
    "Retrieve all single nucleotide polymorphisms (SNPs) between two Neisseria gonorrhoeae isolates.",
    "Find gene fusions in a recently sequenced Burkholderia pseudomallei strain.",
    "Identify antibiotic resistance genes present in a hospital-acquired Enterobacter cloacae isolate but absent in an environmental isolate.",
    "Retrieve all plasmid-encoded genes from a Salmonella Typhimurium genome.",
    "Identify genomic islands in a newly sequenced Campylobacter jejuni strain.",
    
    # Phylogenetic & Evolutionary Analysis
    "Retrieve all publicly available genomes for Salmonella enterica for phylogenetic analysis.",
    "Construct a phylogenetic tree for 20 Klebsiella pneumoniae strains.",
    "Find SNPs unique to multidrug-resistant Staphylococcus aureus strains.",
    "Identify gene duplications in a Mycobacterium tuberculosis genome.",
    "Retrieve all core genes of Escherichia coli strains.",
    "Find strain-specific genes in a Listeria monocytogenes outbreak strain.",
    "Extract allelic variants of the gyrA gene across multiple Pseudomonas aeruginosa genomes.",
    "Retrieve historical isolates of Yersinia pestis for a time-scaled phylogeny.",
    "Identify transposon-mediated gene disruptions in a Clostridioides difficile genome.",
    "Retrieve all genomes from environmental Vibrio strains for evolutionary analysis.",
    
    # Metagenomics & Microbiome Studies
    "Identify bacterial species present in a gut microbiome sample.",
    "Find antimicrobial resistance genes in a wastewater microbiome sample.",
    "Determine the relative abundance of Bacteroides species in a human fecal sample.",
    "Identify viruses present in a marine microbiome dataset.",
    "Compare the taxonomic composition of two soil microbiome samples.",
    "Detect horizontal gene transfer events in a metagenome assembly.",
    "Identify functional pathways enriched in a gut microbiome sample.",
    "Retrieve all antibiotic resistance genes detected in a hospital ICU microbiome.",
    "Compare microbial diversity between healthy and diseased lung microbiomes.",
    "Detect plasmid-associated genes in a sewage metagenomic dataset.",
    
    # Antimicrobial Resistance (AMR) Detection
    "Identify all beta-lactamase genes in a carbapenem-resistant Enterobacter cloacae isolate.",
    "Retrieve known aminoglycoside resistance genes in a clinical Pseudomonas aeruginosa strain.",
    "Find mutations in the gyrA and parC genes associated with fluoroquinolone resistance in Neisseria gonorrhoeae.",
    "Determine if a Salmonella strain carries mobile colistin resistance (mcr) genes.",
    "Identify multidrug efflux pump genes in a hospital-acquired Klebsiella pneumoniae isolate.",
    "Compare aminoglycoside resistance determinants between two Acinetobacter baumannii strains.",
    "Retrieve all carbapenemase genes found in Escherichia coli clinical isolates.",
    "Identify point mutations in the rpoB gene associated with rifampin resistance in Mycobacterium tuberculosis.",
    "Find plasmid-mediated quinolone resistance genes in a Shigella flexneri isolate.",
    "Detect novel antimicrobial resistance determinants in a newly sequenced Staphylococcus epidermidis genome.",
    
    # Functional Genomics & Metabolic Pathway Analysis
    "Identify all genes involved in the TCA cycle in a Pseudomonas putida genome.",
    "Retrieve all genes encoding nitrogen fixation proteins in a Bradyrhizobium japonicum strain.",
    "Find all genes related to butanol biosynthesis in Clostridium acetobutylicum.",
    "Identify genes involved in quorum sensing in a Vibrio harveyi isolate.",
    "Retrieve all genes involved in sulfur metabolism in a Desulfovibrio vulgaris genome.",
    "Find biosynthetic gene clusters responsible for polyketide production in an Actinomyces strain.",
    "Identify key enzymes in the glycolysis pathway of a Lactobacillus plantarum strain.",
    "Compare amino acid biosynthesis pathways between Corynebacterium glutamicum and Escherichia coli.",
    "Retrieve all transporters involved in iron acquisition in a Yersinia pestis genome.",
    "Identify genes associated with ethanol tolerance in Saccharomyces cerevisiae.",
    
    # Viral Genomics
    "Retrieve all annotated proteins from an Influenza A virus genome.",
    "Identify mutations in the hemagglutinin gene of H1N1 isolates.",
    "Find all known antiviral resistance mutations in a Hepatitis C virus genome.",
    "Retrieve all structural proteins of a SARS-CoV-2 isolate.",
    "Compare spike protein variants between multiple SARS-CoV-2 genomes.",
    "Identify recombination events in Dengue virus genomes.",
    "Retrieve all non-structural proteins of Zika virus.",
    "Identify conserved epitopes in Ebola virus glycoproteins.",
    "Detect potential zoonotic transmission events in Rabies virus genomes.",
    "Find all accessory proteins in MERS-CoV genomes.",
    
    # Host-Pathogen Interactions
    "Identify bacterial secretion system genes in a Legionella pneumophila genome.",
    "Retrieve all effector proteins secreted by a Salmonella enterica strain.",
    "Find host receptors targeted by Listeria monocytogenes invasion proteins.",
    "Identify bacterial adhesins in a Neisseria meningitidis genome.",
    "Retrieve all toxin-antitoxin system genes in a Mycobacterium tuberculosis genome.",
    "Find all genes involved in intracellular survival of Brucella abortus.",
    "Identify host immune evasion genes in a Yersinia pestis genome.",
    "Retrieve all genes involved in iron uptake in a pathogenic Vibrio vulnificus strain.",
    "Find phage-encoded virulence factors in a Staphylococcus aureus genome.",
    "Identify bacterial genes that mimic host proteins in a Chlamydia trachomatis genome.",
    
    # Mobile Genetic Elements & Horizontal Gene Transfer
    "Identify all integrons present in a Pseudomonas aeruginosa genome.",
    "Retrieve all transposase genes in a multidrug-resistant Acinetobacter baumannii isolate.",
    "Find prophage regions in a Vibrio cholerae genome.",
    "Identify conjugative plasmid genes in a hospital-acquired Enterobacter cloacae strain.",
    "Retrieve insertion sequence elements in a newly sequenced Shigella flexneri genome.",
    "Find all genomic islands in a Klebsiella pneumoniae isolate.",
    "Identify CRISPR arrays in a Lactococcus lactis genome.",
    "Retrieve phage defense-associated genes in a Streptococcus pyogenes isolate."
]

CoT_template = PromptTemplate.from_template(textwrap.dedent(
    """
    You are an expert in bioinformatics and highly proficient with the PATRIC SOLR database.

    Genome data can be retrieved from the genome collection.
    Antibiotic resistance data can be retrieved from the genome_amr collection.
    Gene feature data can be retrieved from the genome_feature collection.
    Protein family reference data can be retrieved from the protein_family_ref collection.
    Protein feature data can be retrieved from the protein_feature collection.
    Protein structure data can be retrieved from the protein_structure collection.
    Epitope data can be retrieved from the epitope collection.
    Pathway data can be retrieved from the pathway collection.
    Subsystem data can be retrieved from the subsystem collection.
    Taxonomy data can be retrieved from the taxonomy collection.
    Experiment metadata can be retrieved from the experiment collection.
    Gene ontology reference data can be retrieved from the gene_ontology_ref collection.
    Sequence feature data can be retrieved from the sequence_feature collection.
    Spike variant data can be retrieved from the spike_variant collection.
    Surveillance data can be retrieved from the surveillance collection.
    Bioset and bioset result data can be retrieved from the bioset and bioset_result collections.
    Structured assertion data can be retrieved from the structured_assertion collection.
    Serology data can be retrieved from the serology collection.
    Protein-protein interaction (PPI) data can be retrieved from the ppi collection.

    **Goal**: Generate fast-running SOLR commands that demonstrate the solution approach but
    limit the output
    to keep runtime short. For example:
    - Use `limit(1)` (or fewer) to cap the result size
    - Use a small set of columns (e.g., `select(feature_id,product))` to avoid retrieving 
    unneeded data.
    - Avoid overly complex filters or joins unless absolutely necessary.

    When generating your chain-of-thought (<think>), keep it brief. Then produce the final commands in <action>.

    Examples:

    User Query: List all genomes associated with *Escherichia coli*. 
    Response:<think>
    First, identify the relevant command for retrieving genome data. The `https://bv-brc.org/api/genome/?eq(genome_id,<genome_id>)` endpoint allows filtering genomes based on various fields.
  
    To target *Escherichia coli*, we use filters for the genus and species fields. Genomes can be retrieved from the genome collection.
  
    The resulting command is:
    ```
    bash
    curl -H "Accept: application/json" "https://www.bv-brc.org/api/genome/?and(eq(genus,Escherichia),eq(species,coli)&limit(1)"
    ```
</think>

<action>
   [
   {{
       "action": "bash",
       "action_input": "curl -H \"Accept: application/json\" \"https://www.bv-brc.org/api/genome/?and(eq(genus,Escherichia),eq(species,coli)&limit(1)\""
   }}
   ]
</action>
                                                       
    User Query: {user_query}
    """))

def llm_for_actionable_commands(client, user_query, previous_attempts=None):
    """
    Generates commands using the LLM based on the user query and previous attempts.
    
    Args:
        client: The Together client instance
        user_query (str): The user's query
        previous_attempts (list, optional): List of previous attempts with their results
    
    Returns:
        str: The LLM's response containing commands
    """
    if previous_attempts and len(previous_attempts) > 0:
        previous_attempts_str = "\n".join([
            f"Previous attempt {i+1}:\n"
            f"Command: {attempt['command']}\n"
            f"Result: {attempt['result']}\n"
            f"Status: {attempt['status']}\n"
            for i, attempt in enumerate(previous_attempts)
        ])
        
        prompt = f"""Previous attempts to answer this query have failed. 
Here are the previous attempts and their results:

{previous_attempts_str}

Please provide a new approach to answer the original query:
{user_query}

Remember to:
1. Consider why the previous attempts failed
2. Adjust the approach based on the error messages or results
3. Try a different strategy if the previous ones weren't successful
"""
    else:
        prompt = CoT_template.format(user_query=user_query)
    
    return query_local_llm(client, prompt)

def execute_command(command, llm_response_file=None):
    """
    Executes a command and returns the output.

    Args:
        command (str): The command to execute.
        llm_response_file (str, optional): Path to the LLM response file to append observations.

    Returns:
        str: The output of the command.
    """
    try:
        logging.info(f"Executing command: {command}")
        
        # Create a temporary file to store the output
        with tempfile.NamedTemporaryFile(delete=False, mode='w+t') as temp_file:
            temp_file_path = temp_file.name
        
        # Execute the command and redirect output to the temporary file
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = process.communicate()
        output = stdout
        
        # If there's an error, include it in the output
        if stderr:
            error_message = f"Error executing command: {stderr}"
            logging.error(error_message)
            output = f"{output}\n{error_message}" if output else error_message
        
        # Write the output to the temporary file
        with open(temp_file_path, 'w') as temp_file:
            temp_file.write(output)
        
        # Append the observation to the LLM response file if provided
        if llm_response_file:
            with open(llm_response_file, 'a') as f:
                f.write(f"\n\n<observation>\n{output}\n</observation>\n")
        
        return output
    
    except Exception as e:
        error_message = f"Error executing command: {str(e)}"
        logging.exception(error_message)
        
        # Append the error observation to the LLM response file if provided
        if llm_response_file:
            with open(llm_response_file, 'a') as f:
                f.write(f"\n\n<observation>\n{error_message}\n</observation>\n")
        
        return error_message

def execute_commands(commands, query_dir):
    """
    Executes a list of commands sequentially in the same shell session and captures their outputs.

    Args:
        commands (list): A list of commands to execute.
        query_dir (str): The directory where intermediate files are saved.

    Returns:
        list: A list of dictionaries containing command details, outputs, and observations.
    """
    results = []
    
    for command in commands:
        print(f"Executing command: {command}")
        
        # Initialize observation data
        observation = {
            "stdout": "",
            "stderr": "",
            "return_code": 0,
            "output_files": []
        }

        # Detect output redirection using '>'
        redirect_match = re.search(r'>(\s*)(\S+)', command)
        if redirect_match:
            output_file = redirect_match.group(2)
            # Ensure the output file path is within the query directory
            output_file_path = os.path.join(query_dir, sanitize_filename(output_file))
            observation["output_files"].append(output_file_path)
            # Modify the command to redirect output to the correct directory
            command = re.sub(r'>(\s*)(\S+)', f'> "{output_file_path}"', command)
        
        try:
            # **Prepend the source command to each command**
            full_command = f"source /Applications/BV-BRC.app/user-env.sh; {command}"

            # Execute the command
            process = subprocess.Popen(
                full_command,
                shell=True,
                cwd=query_dir,  # Set the working directory to query_dir
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                executable='/bin/bash'  # **Ensure using bash shell**
            )
            stdout, stderr = process.communicate()
            return_code = process.returncode

            # Update observation with command outputs
            observation["stdout"] = stdout.strip()
            observation["stderr"] = stderr.strip()
            observation["return_code"] = return_code

            if return_code == 0: 
                print(f"Command '{command}' executed successfully.")
                print(f"Output:\n{stdout}")
            else:
                print(f"Command '{command}' failed with error:\n{stderr}")

            # If there is a redirected output file, read its contents
            if redirect_match:
                if os.path.exists(output_file_path):
                    with open(output_file_path, "r") as f:
                        file_content = f.read()
                    observation["file_contents"] = {
                        output_file_path: file_content
                    }
                    print(f"Output file '{output_file_path}' created with contents:\n{file_content}")
                else:
                    print(f"Expected output file '{output_file_path}' was not found.")

        except Exception as e:
            print(f"Error executing command '{command}': {e}")
            observation["stderr"] = str(e)
            observation["return_code"] = 1

        result = {
            "action": "bash",
            "action_input": command,
            "observation": observation
        }
        results.append(result)

    return results

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

def parse_commands(response_to_call):
    """
    Parses the commands from the given response.
    Gives preference to <action> tags and falls back to <think> tags if no commands are found in <action>.

    Args:
        response_to_call (str): The response containing commands.

    Returns:
        list: A list of command strings.
    """
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

#----------------------------------
# Main Execution

def log_complete_trace(query_dir, user_query, llm_response, commands, execution_results, classifications, justifications, solutions):
    """
    Logs the complete trace of events for a query into a single file.
    
    Args:
        query_dir (str): Directory where the trace file will be saved
        user_query (str): The original user query
        llm_response (str): The LLM's response to the user query
        commands (list): List of commands that were executed
        execution_results (list): Results of command executions
        classifications (list): Classifications of command outputs
        justifications (list): Justifications for the classifications
        solutions (list): Solutions derived from the classifications
    """
    trace_file_path = os.path.join(query_dir, "complete_trace.txt")
    
    with open(trace_file_path, "w") as f:
        # Write the user query
        f.write(f"# User Query\n{user_query}\n\n")
        
        # Write the LLM's initial response with reasoning
        f.write(f"# Initial LLM Response\n{llm_response}\n\n")
        
        # Write each command execution and its results
        for i, (command, result, classification, justification, solution) in enumerate(
            zip(commands, execution_results, classifications, justifications, solutions)
        ):
            f.write(f"## Command Execution {i+1}\n\n")
            
            # Extract think tag content from LLM response if available
            think_match = re.search(r'<think>\s*(.*?)\s*</think>', llm_response, re.DOTALL | re.IGNORECASE)
            if think_match:
                f.write(f"<think>\n{think_match.group(1)}\n</think>\n\n")
            
            # Write the action
            f.write(f"<action>\n{command}\n</action>\n\n")
            
            # Write the observation
            stdout = result["observation"]["stdout"]
            stderr = result["observation"]["stderr"]
            observation = stdout
            if stderr:
                observation += f"\n\nErrors:\n{stderr}"
            f.write(f"<observation>\n{observation}\n</observation>\n\n")
            
            # Write the classification
            f.write(f"<classification>\n{classification}\n</classification>\n\n")
            
            # Write the justification
            f.write(f"<justification>\n{justification}\n</justification>\n\n")
            
            # Write the solution
            f.write(f"<solution>\n{solution}\n</solution>\n\n")
            
            f.write("---\n\n")
    
    logging.info(f"Complete trace saved to {trace_file_path}")
    return trace_file_path

def main():
    for user_query in queries:
        try:
            # Create directory for query results
            query_dir = os.path.join("query_results", sanitize_filename(user_query))
            os.makedirs(query_dir, exist_ok=True)
            
            # Initialize tracking for attempts
            previous_attempts = []
            max_attempts = 3
            success = False
            
            # Try multiple attempts if needed
            for attempt_num in range(max_attempts):
                logging.info(f"Attempt {attempt_num + 1} for query: {user_query}")
                
                # Generate commands using LLM with knowledge of previous attempts
                response = llm_for_actionable_commands(client, user_query, previous_attempts)
                
                # Parse and execute commands
                commands = parse_commands(response)
                if not commands:
                    logging.warning("No valid commands generated")
                    continue
                
                # Execute each command
                for cmd in commands:
                    result = execute_command(cmd)
                    
                    # Record this attempt
                    attempt = {
                        "command": cmd,
                        "result": result,
                        "status": "success" if "error" not in result.lower() else "failure"
                    }
                    previous_attempts.append(attempt)
                    
                    # Log the attempt
                    with open(os.path.join(query_dir, f"attempt_{attempt_num + 1}.txt"), "a") as f:
                        f.write(f"Command: {cmd}\n")
                        f.write(f"Result: {result}\n")
                        f.write(f"Status: {attempt['status']}\n")
                        f.write("-" * 50 + "\n")
                    
                    # Check if successful
                    if attempt["status"] == "success":
                        success = True
                        break
                
                # Break if successful
                if success:
                    logging.info(f"Query succeeded on attempt {attempt_num + 1}")
                    break
                elif attempt_num < max_attempts - 1:
                    logging.info("Trying another attempt with adjusted approach")
                else:
                    logging.warning(f"All {max_attempts} attempts failed")
            
            # Write final summary
            with open(os.path.join(query_dir, "summary.txt"), "w") as f:
                f.write(f"Query: {user_query}\n")
                f.write(f"Number of attempts: {len(previous_attempts)}\n")
                f.write(f"Final status: {'Success' if success else 'Failure'}\n\n")
                f.write("Attempt History:\n")
                for i, attempt in enumerate(previous_attempts):
                    f.write(f"\nAttempt {i + 1}:\n")
                    f.write(f"Command: {attempt['command']}\n")
                    f.write(f"Status: {attempt['status']}\n")
                    f.write("-" * 50 + "\n")
        
        except Exception as e:
            logging.exception(f"Error processing query: {user_query}")
            continue

#----------------------------------
# Helper Functions

def log_results(user_query, response, p3_call, observation, justification, solution, classification):
    """
    Logs the results of processing a user query.

    Args:
        user_query (str): The original user query.
        response (str): The LLM's response to the user query.
        p3_call (str): The JSON-formatted p3-tools commands.
        observation (str): The parsed output from the commands.
        justification (str): The justification from the LLM.
        solution (str): The solution derived based on classification.
        classification (str): The classification of the command outputs.
    """
    log_entry = (
        "\n--------------------------------\n\n"
        f"User: {user_query}\n"
        f"Response: {response}\n"
        f"Action Format: {p3_call}\n"
        f"Observation:\n{observation}\n"
        f"Justification:\n{justification}\n"
        f"Solution:\n{solution}\n"
        f"Classification:\n{classification}\n"
    )
    # Determine log file based on classification
    if "failure" in classification.lower():
        log_filename = "classification_failure_together.log"
    else:
        log_filename = "classification_success_together.log"

    with open(log_filename, "a") as f:
        f.write(log_entry)

def parse_last_command_output_with_llm(terminal_output_lines):
    """
    Takes in the command output and uses an LLM to extract relevant information.

    Parameters:
        terminal_output_lines (list of str): The lines captured from the command output.

    Returns:
        str: The LLM's best guess at the parsed output.
    """
    output_str = "\n".join(terminal_output_lines)

    prompt = f"""You are an expert in bioinformatics. Below is the raw 
output from a p3-tools command execution:

{output_str}

Your task is to extract and return the relevant information from the output.

Output:
"""
    print("Parsing Command Output with LLM...")
    return query_local_llm(client, prompt)

def classify_last_command_output_with_llm(terminal_output_lines, user_query):
    """
    Uses LLM to classify the command output as success or failure.

    Args:
        terminal_output_lines (list of str): The lines captured from the command output.
        user_query (str): The original user query.

    Returns:
        str: "success" or "failure"
    """
    output_str = "\n".join(terminal_output_lines)

    prompt = f"""You are an expert at analyzing command outputs to determine if a command was successful
and if it successfully answered the user's query.

The user's query is: {user_query}

Below is the raw output from a p3-tools command execution:

{output_str}

Your task is to classify the output as either "success" or "failure".

Output:
"""
    print("Classifying Command Output with LLM...")
    return query_local_llm(client, prompt)

def justify_solution_with_llm(terminal_output_lines, user_query, response_to_user_query):
    """
    Uses LLM to justify the solution based on command output.

    Args:
        terminal_output_lines (list of str): The lines captured from the command output.
        user_query (str): The original user query.
        response_to_user_query (str): The LLM's response to the user query.

    Returns:
        str: The justification from the LLM.
    """
    output_str = "\n".join(terminal_output_lines)

    prompt = f"""You are an expert at analyzing command outputs to determine if a command was successful
and if it successfully answered the user's query.

The user's query is: {user_query}

The LLM's response to the user's query is: {response_to_user_query}

Below is the raw output from a p3-tools command execution:

{output_str}

Your task is to justify the LLM's response to the user's query based on the command output.
Return only the justification and a summary of the solution, not the user's query or the LLM's response.

Output:
"""
    print("Justifying LLM's Response with LLM...")
    return query_local_llm(client, prompt)

def derive_solution_with_llm(classification, justification):
    """
    Uses LLM to derive a solution based on the classification and justification.

    Args:
        classification (str): The classification result ("success" or "failure").
        justification (str): The justification for the classification.

    Returns:
        str: The derived solution or recommended action.
    """
    prompt = f"""Based on the following classification and justification, provide a concise solution or recommended action.

Classification: {classification}

Justification: {justification}

Solution:
"""
    print("Deriving Solution with LLM...")
    return query_local_llm(client, prompt)

def sanitize_filename(filename):
    """
    Sanitizes a string to be used as a valid filename by removing or replacing invalid characters.

    Args:
        filename (str): The original filename string.

    Returns:
        str: A sanitized version of the filename.
    """
    return re.sub(r'[\\/*?:"<>|]', "_", filename)

def test_parse_commands():
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
    result1 = parse_commands(response1)
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
    result2 = parse_commands(response2)
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
    result3 = parse_commands(response3)
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
    result4 = parse_commands(response4)
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
    result5 = parse_commands(response5)
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
    result6 = parse_commands(response6)
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
    result7 = parse_commands(response7)
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
    result8 = parse_commands(response8)
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
    result10 = parse_commands(response10)
    assert result10 == expected10, f"Test Case 10 Failed: Expected {expected10}, Got {result10}"
    print("Test Case 10 Passed: Original Use Case with only <think>")

    print("All tests completed successfully.")

#----------------------------------
# Entry Point

if __name__ == "__main__":
    test_parse_commands()
    main()

