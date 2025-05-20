import json
from solr_together import together_client, generate_enhanced_prompt, llm_for_actionable_commands, extract_think_content
from mcts_thinking import MCTSThinkingManager

# Step 1: Load the first query from queries.json
with open('queries.json', 'r') as f:
    queries = json.load(f)
user_query = queries[0]
print(f"Loaded query: {user_query}\n")

# Step 2: Generate a reasoning response using the LLM
print("Generating reasoning response from LLM...")
enhanced_prompt = generate_enhanced_prompt(user_query)
llm_response, _ = llm_for_actionable_commands(together_client, user_query, enhanced_prompt=enhanced_prompt)
print("\n=== LLM Response ===\n")
print(llm_response)
print("\n=== END LLM Response ===\n")

# Step 3: Extract <think> content
think_content = extract_think_content(llm_response)
print("\n=== Extracted <think> Content ===\n")
print(think_content)
print("\n=== END <think> Content ===\n")

# Step 4: Split into individual thoughts
mcts_manager = MCTSThinkingManager()
thoughts = mcts_manager.split_thoughts_from_text(think_content)
print(f"\nSplit into {len(thoughts)} thought(s):")
for i, t in enumerate(thoughts, 1):
    print(f"--- Thought {i} ---\n{t}\n")

# Step 5: Organize into a chain of thought nodes
nodes = mcts_manager.organize_thoughts_into_chain(thoughts)
print("\n=== Thought Node Chain ===\n")
for node in nodes:
    print(f"Node {node['thoughtNumber']}/{node['totalThoughts']} (ID: {node['nodeId']}, Parent: {node.get('parentId')})")
    print(f"Thought: {node['thought']}")
    print(f"Next Thought Needed: {node['nextThoughtNeeded']}")
    print(f"Depth: {node['depth']}")
    print("-") 