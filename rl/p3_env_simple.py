from typing import List, Dict, Any
from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig


class P3Rubric:
    """Simple rubric for P3 tasks that rewards correct answers"""
    
    def __init__(self):
        pass
    
    def get_reward_funcs(self):
        return [self.answer_reward_func]
    
    def answer_reward_func(self, completions: List, **kwargs) -> List[float]:
        """Reward function that checks if the answer appears in the completion"""
        rewards = []
        for completion in completions:
            # Extract the final answer from the completion
            if isinstance(completion, list) and completion:
                final_content = completion[-1].get("content", "")
            else:
                final_content = str(completion)
            
            # Simple reward: give 1.0 for any completion that contains a solution tag
            # This is a basic reward for following the expected format
            if "<solution>" in final_content.lower():
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        
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
    
    def generate(self, prompts: List[str], llm=None, sampling_params=None, **kwargs) -> Dict[str, List]:
        """Generate completions for prompts - required by MT-GRPO trainer"""
        if llm is None or sampling_params is None:
            # Fallback: return empty completions if no LLM provided
            return {
                'ids': [[] for _ in prompts],
                'messages': [[] for _ in prompts],  
                'mask': [[] for _ in prompts]
            }
        
        # Format prompts as messages for the LLM
        formatted_prompts = []
        for prompt in prompts:
            messages = self.format_prompt(prompt)
            formatted_prompts.append(messages)
        
        # Generate completions using vLLM
        try:
            completions = llm.chat(formatted_prompts, sampling_params=sampling_params, use_tqdm=False)
            
            # Process completions into the expected format
            ids = []
            messages = []
            mask = []
            
            for i, completion in enumerate(completions):
                # Extract token IDs and text
                prompt_token_ids = list(completion.prompt_token_ids)
                completion_token_ids = list(completion.outputs[0].token_ids)
                completion_text = completion.outputs[0].text
                
                # DEBUG: Print what the model actually generated
                print(f"=== MODEL GENERATED TEXT ===")
                print(f"Full prompt messages: {formatted_prompts[i]}")
                print(f"Generated: '{completion_text}'")
                print(f"Token count: {len(completion_token_ids)}")
                print(f"Completion token IDs: {completion_token_ids[:100]}...")  # First 10 tokens
                print("=" * 50)
                
                # Combine prompt and completion IDs
                all_token_ids = prompt_token_ids + completion_token_ids
                ids.append(all_token_ids)
                
                # Create message sequence
                msg_sequence = formatted_prompts[i].copy()
                msg_sequence.append({"role": "assistant", "content": completion_text})
                messages.append(msg_sequence)
                
                # Create attention mask (1 for all tokens)
                attention_mask = [1] * len(all_token_ids)
                mask.append(attention_mask)
            
            return {
                'ids': ids,
                'messages': messages,
                'mask': mask
            }
            
        except Exception as e:
            print(f"Error generating completions: {e}")
            # Return empty completions on error
            return {
                'ids': [[] for _ in prompts],
                'messages': [[] for _ in prompts],  
                'mask': [[] for _ in prompts]
            }