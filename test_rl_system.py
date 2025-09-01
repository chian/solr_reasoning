#!/usr/bin/env python3
"""
Test script for the RL system with discovery tools
"""

import os
import sys
from rl.p3_mt_grpo_instrumentation import EpisodeBuffer
from rl.demo_p3_rl_train import wire_rl_hooks

def test_rl_system():
    """Test the RL system with a simple episode"""
    
    print("Testing RL system with discovery tools...")
    
    # Create a test episode buffer
    buffer = EpisodeBuffer()
    
    # Wire the RL hooks
    wire_rl_hooks(buffer)
    
    # Simulate a simple episode
    test_prompt = "Find all virulence factors in E. coli"
    test_answer = "virulence factors"
    
    buffer.start(test_prompt, test_answer)
    
    # Add the initial user prompt
    buffer.append_episode_step("user", test_prompt)
    
    # Simulate the model discovering tools
    buffer.append_episode_step("assistant", '<think>I need to find out what tools are available for this task. Let me start by discovering what tools exist.</think><action>{"tool_name": "list_available_tools", "parameters": {}}</action>')
    
    # Simulate tool discovery response
    buffer.append_episode_step("user", '<observation>{"success": true, "server_name": "P3-Tools Data Retrieval Server", "total_tools": 9, "tools": [{"name": "p3_all_genomes", "description": "Get complete genome records using p3-all-genomes", "category": "data_retrieval"}, {"name": "p3_get_genome_features", "description": "Get genome features using p3-get-genome-features", "category": "data_retrieval"}]}</observation>')
    
    # Simulate getting tool help
    buffer.append_episode_step("assistant", '<think>I can see there are tools for getting genomes and features. Let me get more details about p3_all_genomes to understand how to use it.</think><action>{"tool_name": "get_tool_help", "parameters": {"tool_name": "p3_all_genomes"}}</action>')
    
    # Simulate tool help response
    buffer.append_episode_step("user", '<observation>{"success": true, "tool_name": "p3_all_genomes", "help": {"description": "Get complete genome records using p3-all-genomes", "parameters": {"species": {"type": "str", "optional": true, "description": "Species name to filter by"}, "limit": {"type": "int", "optional": true, "default": 1000, "description": "Maximum number of results"}}, "example": {"species": "Escherichia coli", "limit": 10}}}</observation>')
    
    # Simulate using the tool
    buffer.append_episode_step("assistant", '<think>Now I understand how to use p3_all_genomes. I can search for E. coli genomes and then get features from those genomes to find virulence factors.</think><action>{"tool_name": "tools/call", "parameters": {"name": "p3_all_genomes", "arguments": {"species": "Escherichia coli", "limit": 5}}}</action>')
    
    # Simulate tool response
    buffer.append_episode_step("user", '<observation>{"success": true, "data": [{"genome_id": "83333.111", "genome_name": "E. coli K-12"}, {"genome_id": "83333.112", "genome_name": "E. coli O157:H7"}]}</observation>')
    
    # Simulate final synthesis
    buffer.append_episode_step("assistant", '<think>I found E. coli genomes. Now I need to get features from these genomes to identify virulence factors. Let me use p3_get_genome_features to search for virulence-related genes.</think><solution>Based on my exploration, I can use p3_all_genomes to find E. coli genomes and then p3_get_genome_features to search for virulence factors within those genomes. I discovered 2 E. coli genomes that I can analyze further.</solution>')
    
    # Finalize the episode
    buffer.finalize()
    
    print(f"Episode created successfully!")
    print(f"Prompt: {buffer.prompts[0]}")
    print(f"Number of turns: {len(buffer.messages[0])}")
    print(f"Answer: {buffer.answers[0]}")
    
    print("\nEpisode structure:")
    for i, message in enumerate(buffer.messages[0]):
        print(f"Turn {i+1}: {message['role']}: {message['content'][:100]}...")
    
    return buffer

if __name__ == "__main__":
    buffer = test_rl_system()
