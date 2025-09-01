from typing import List, Dict, Any
from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig


class P3Rubric:
    """Simple rubric for P3 tasks that rewards correct answers"""
    
    def __init__(self):
        pass
    
    def get_reward_funcs(self):
        return [self.answer_reward_func]
    
    def answer_reward_func(self, completions: List, answers: List, **kwargs) -> List[float]:
        """Reward function that checks if the answer appears in the completion"""
        rewards = []
        for completion, answer in zip(completions, answers):
            # Extract the final answer from the completion
            if isinstance(completion, list) and completion:
                final_content = completion[-1].get("content", "")
            else:
                final_content = str(completion)
            
            # Check if any answer alias appears
            if isinstance(answer, (list, tuple)):
                answer_list = answer
            else:
                answer_list = [answer]
            
            found = any(str(ans).lower() in final_content.lower() for ans in answer_list)
            rewards.append(1.0 if found else 0.0)
        
        return rewards


class P3Env:
    """Environment for P3 tool execution tasks"""
    
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.rubric = P3Rubric()
    
    def get_dataset(self, **kwargs: Any) -> Dataset:
        return self.dataset
    
    def get_eval_dataset(self, **kwargs: Any) -> Dataset | None:
        return None
    
    def get_rubric(self, **kwargs: Any):
        return self.rubric.get_reward_funcs()
    
    def format_prompt(self, prompt: str, fewshot_prob: float = 1.0) -> List[Dict[str, str]]:
        """Format prompt for P3 tasks with system instructions"""
        messages = []
        
        # Load the RL prompt template that encourages tool discovery
        try:
            with open("rl/rl_prompt_template.txt", "r") as f:
                system_prompt = f.read()
        except FileNotFoundError:
            # Fallback if template not found
            system_prompt = """You are an AI assistant that helps with bioinformatics tasks using BV-BRC MCP servers. 
You have access to various tools, but you need to discover what's available and learn how to use them effectively.

Start by exploring what tools are available for this task, then build your approach step by step based on what you discover.

Format your response as:
<think>Your reasoning here</think>
<action>{"tool_name": "tool_name", "parameters": {...}}</action>
<solution>Your final answer here</solution>"""
        
        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages
