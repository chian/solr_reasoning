import os
import json
from typing import Any, Dict

from rl.p3_mt_grpo_instrumentation import EpisodeBuffer, train_with_buffer


def create_mock_episode(buffer: EpisodeBuffer, prompt: str, answer: str) -> None:
    """Create a mock episode for RL training without calling external APIs"""
    
    # Simulate the model discovering tools and solving the problem step by step
    # This is what the RL agent will learn to do during training
    
    # Step 1: Initial reasoning about the task
    buffer.append_episode_step("assistant", 
        '<think>I need to understand what tools are available for this bioinformatics task. Let me start by discovering what tools exist.</think>')
    
    # Step 2: Tool discovery
    buffer.append_episode_step("assistant", 
        '<action>{"tool_name": "list_available_tools", "parameters": {}}</action>')
    
    # Step 3: Tool discovery response (simulated)
    buffer.append_episode_step("user", 
        '<observation>{"success": true, "tools": [{"name": "p3_all_genomes", "description": "Get genome records"}, {"name": "p3_get_genome_features", "description": "Get genome features"}]}</observation>')
    
    # Step 4: Getting tool help
    buffer.append_episode_step("assistant", 
        '<think>I can see tools for getting genomes and features. Let me get more details about p3_all_genomes to understand how to use it.</think>')
    buffer.append_episode_step("assistant", 
        '<action>{"tool_name": "get_tool_help", "parameters": {"tool_name": "p3_all_genomes"}}</action>')
    
    # Step 5: Tool help response (simulated)
    buffer.append_episode_step("user", 
        '<observation>{"success": true, "help": {"description": "Get genome records", "parameters": {"species": {"type": "str", "optional": true}, "limit": {"type": "int", "default": 1000}}}}}</observation>')
    
    # Step 6: Using the tool
    buffer.append_episode_step("assistant", 
        '<think>Now I understand how to use p3_all_genomes. I can search for E. coli genomes and then get features from those genomes.</think>')
    buffer.append_episode_step("assistant", 
        '<action>{"tool_name": "tools/call", "parameters": {"name": "p3_all_genomes", "arguments": {"species": "Escherichia coli", "limit": 5}}}</action>')
    
    # Step 7: Tool response (simulated)
    buffer.append_episode_step("user", 
        '<observation>{"success": true, "data": [{"genome_id": "83333.111", "genome_name": "E. coli K-12"}, {"genome_id": "83333.112", "genome_name": "E. coli O157:H7"}]}</observation>')
    
    # Step 8: Final synthesis
    buffer.append_episode_step("assistant", 
        '<think>I found E. coli genomes. Now I need to get features from these genomes to identify virulence factors. Let me use p3_get_genome_features to search for virulence-related genes.</think>')
    buffer.append_episode_step("assistant", 
        '<solution>Based on my exploration, I can use p3_all_genomes to find E. coli genomes and then p3_get_genome_features to search for virulence factors within those genomes. I discovered 2 E. coli genomes that I can analyze further.</solution>')


def main():
    # Minimal demo: user sets DATASET_JSONL and MODEL_NAME, we create mock episodes for RL training
    dataset_jsonl = os.environ.get("DATASET_JSONL")
    model_name = os.environ.get("MODEL_NAME")
    num_gpus = int(os.environ.get("NUM_GPUS", "1"))

    if not dataset_jsonl or not model_name:
        raise SystemExit("Set DATASET_JSONL and MODEL_NAME env vars for the demo.")

    buffer = EpisodeBuffer()

    # Create mock episodes for RL training (no external API calls)
    with open(dataset_jsonl, "r") as f:
        for line in f:
            obj = json.loads(line)
            prompt = obj["prompt"]
            answer = obj["answer"]
            buffer.start(prompt, answer)
            # Add the initial user prompt to start the episode
            buffer.append_episode_step("user", prompt)
            # Create a mock episode instead of calling the actual P3 workflow
            create_mock_episode(buffer, prompt, answer)
            buffer.finalize()

    print(f"Created {len(buffer.prompts)} mock episodes for RL training")
    
    # Train MT-GRPO on collected episodes
    print("Starting MT-GRPO training...")
    train_with_buffer(
        buffer=buffer,
        model_name=model_name,
        num_gpus=num_gpus,
    )


if __name__ == "__main__":
    main()


