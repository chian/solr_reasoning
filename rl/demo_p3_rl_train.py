import os
import json
from typing import Any, Dict

from rl.p3_mt_grpo_instrumentation import EpisodeBuffer, train_with_buffer


def create_real_episode(buffer: EpisodeBuffer, prompt: str, answer: str) -> None:
    """Create a real episode by simulating the actual P3 workflow structure"""
    
    print(f"Creating real P3 workflow episode for: {prompt}")
    
    # Simulate the real P3 workflow structure without external API dependencies
    # This demonstrates the actual tool calling pattern that would happen in production
    
    # Step 1: Initial reasoning about the task
    buffer.append_episode_step("assistant", 
        '<think>I need to find complete genomes for Escherichia coli. Let me start by searching for E. coli genomes using the p3_all_genomes tool.</think>')
    
    # Step 2: Call p3_all_genomes tool (real tool structure)
    buffer.append_episode_step("assistant", 
        '<action>{"tool_name": "p3_all_genomes", "parameters": {"species": "Escherichia coli", "limit": 1000, "attributes": ["genome_id", "genome_name", "genome_status"]}}</action>')
    
    # Step 3: Tool response (simulated but realistic)
    buffer.append_episode_step("user", 
        '<observation>{"success": true, "data": [{"genome_id": "83333.111", "genome_name": "Escherichia coli K-12 MG1655", "genome_status": "Complete"}, {"genome_id": "83333.112", "genome_name": "Escherichia coli O157:H7 EDL933", "genome_status": "Complete"}], "total_count": 71388}</observation>')
    
    # Step 4: Process the results
    buffer.append_episode_step("assistant", 
        '<think>Great! I found E. coli genomes. The response shows there are 71,388 total E. coli genomes available. Let me get more details about the complete genomes specifically.</think>')
    
    # Step 5: Filter for complete genomes only
    buffer.append_episode_step("assistant", 
        '<action>{"tool_name": "p3_all_genomes", "parameters": {"species": "Escherichia coli", "genome_status": "Complete", "limit": 1000, "attributes": ["genome_id", "genome_name"]}}</action>')
    
    # Step 6: Complete genomes response
    buffer.append_episode_step("user", 
        '<observation>{"success": true, "data": [{"genome_id": "83333.111", "genome_name": "Escherichia coli K-12 MG1655"}, {"genome_id": "83333.112", "genome_name": "Escherichia coli O157:H7 EDL933"}], "total_count": 15432}</observation>')
    
    # Step 7: Final solution
    buffer.append_episode_step("assistant", 
        '<solution>I found 15,432 complete genomes for Escherichia coli. The search returned genome records including the well-known K-12 MG1655 strain and the pathogenic O157:H7 EDL933 strain. The total count of complete E. coli genomes is 15,432.</solution>')



# Model configuration - choose based on available memory
SMALL_MODEL = "microsoft/DialoGPT-small"  # 117M parameters, ~500MB memory
MEDIUM_MODEL = "microsoft/DialoGPT-medium"  # 345M parameters, ~1.4GB memory
LARGE_MODEL = "Qwen/Qwen2.5-7B"  # 7B parameters, ~28GB memory

def main():
    # Minimal demo: user sets DATASET_JSONL and MODEL_NAME, we create mock episodes for RL training
    dataset_jsonl = os.environ.get("DATASET_JSONL")
    model_name = os.environ.get("MODEL_NAME", SMALL_MODEL)  # Default to small model for testing
    num_gpus = int(os.environ.get("NUM_GPUS", "1"))

    if not dataset_jsonl or not model_name:
        raise SystemExit("Set DATASET_JSONL and MODEL_NAME env vars for the demo.")

    buffer = EpisodeBuffer()

    # Create real episodes for RL training - let the model generate new text
    with open(dataset_jsonl, "r") as f:
        for line in f:
            obj = json.loads(line)
            prompt = obj["prompt"]
            answer = obj["answer"]
            buffer.start(prompt, answer)
            # Add the initial user prompt to start the episode
            buffer.append_episode_step("user", prompt)
            # Don't add pre-written steps - let the model generate them during training
            buffer.finalize()

    print(f"Created {len(buffer.prompts)} real episodes for RL training")
    
    # Train MT-GRPO on collected episodes
    print("Starting MT-GRPO training...")
    train_with_buffer(
        buffer=buffer,
        model_name=model_name,
        num_gpus=num_gpus,
    )


if __name__ == "__main__":
    main()


