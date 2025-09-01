from typing import List, Dict, Any, Tuple
from datasets import Dataset
from trl.trainer.grpo_trainer import RewardFunc

from verifiers.envs.simple_env import SimpleEnv
from verifiers.parsers import XMLParser
from verifiers.rubrics import Rubric
from verifiers.prompts import SIMPLE_PROMPT


class P3Rubric(Rubric):
    """Simple rubric for P3 tasks that rewards correct answers"""
    
    def __init__(self):
        super().__init__()
    
    def get_reward_funcs(self) -> List[RewardFunc]:
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


class P3Env(SimpleEnv):
    """Environment for P3 tool execution tasks"""
    
    def __init__(self, dataset: Dataset, **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset
        self.parser = XMLParser(fields=["reasoning", "action", "solution"])
        self.rubric = P3Rubric()
    
    def get_dataset(self, **kwargs: Any) -> Dataset:
        return self.dataset
    
    def get_eval_dataset(self, **kwargs: Any) -> Dataset | None:
        return None
    
    def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
        return self.rubric.get_reward_funcs()
    
    def format_prompt(self, prompt: str, fewshot_prob: float = 1.0) -> List[Dict[str, str]]:
        """Format prompt for P3 tasks with system instructions"""
        messages = []
        
        # Add system prompt for P3 tasks
        system_prompt = """You are an AI assistant that helps with bioinformatics tasks using BV-BRC tools. 
You should:
1. Think through the problem step by step
2. Use appropriate tools to gather information
3. Provide a clear, structured response with your reasoning and final answer

Format your response as:
<think>Your reasoning here</think>
<action>[{"tool_name": "tool_name", "parameters": {...}}]</action>
<solution>Your final answer here</solution>"""
        
        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages
