import json
from typing import Any, Dict, List

import sys
from pathlib import Path

# Import the simplified P3 environment instead of external code
from rl.p3_env_simple import P3Env


class EpisodeBuffer:
    def __init__(self) -> None:
        self.prompts: List[str] = []
        self.messages: List[List[Dict[str, str]]] = []
        self.answers: List[Any] = []

        self.current_messages: List[Dict[str, str]] = []
        self.current_prompt: str = ""
        self.current_answer: Any = None

    def start(self, prompt: str, answer: Any) -> None:
        self.current_messages = []
        self.current_prompt = prompt
        self.current_answer = answer

    def append_assistant(self, content: str) -> None:
        self.current_messages.append({"role": "assistant", "content": content})

    def append_episode_step(self, role: str, content: str) -> None:
        """Add a step to the current episode trajectory"""
        self.current_messages.append({"role": role, "content": content})

    def append_user(self, content: str) -> None:
        self.current_messages.append({"role": "user", "content": content})

    def finalize(self) -> None:
        self.prompts.append(self.current_prompt)
        self.messages.append(list(self.current_messages))
        self.answers.append(self.current_answer)


def train_with_buffer(
    buffer: EpisodeBuffer,
    model_name: str,
    num_gpus: int,
    learning_rate: float = 1e-6,
    num_generations: int = 8,
    per_device_train_batch_size: int = 2,
    grad_accum_steps: int = 2,
    num_iterations: int = 1,
    max_steps: int = 100,
    beta: float = 0.0,
    turn_advantage_coef: float = 1.0,
):
    # Import transformers and TRL components
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOTrainer, GRPOConfig
    from datasets import Dataset
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Ensure tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build dataset in the format expected by the trainer
    dataset_data = []
    for prompt, messages, answer in zip(buffer.prompts, buffer.messages, buffer.answers):
        # The trainer expects prompt to be the initial user query
        # and messages to be the conversation history
        dataset_data.append({
            "prompt": prompt,
            "messages": messages,
            "answer": answer
        })
    
    dataset = Dataset.from_list(dataset_data)

    # Create the P3 environment with the dataset
    env = P3Env(dataset=dataset)

    # Create GRPO config
    config = GRPOConfig(
        learning_rate=learning_rate,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        num_generations=num_generations,
        beta=beta,
    )

    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        config=config,
        train_dataset=dataset,
        tokenizer=tokenizer,
        reward_function=env.get_rubric()[0],  # Use the first reward function
    )
    
    trainer.train()


