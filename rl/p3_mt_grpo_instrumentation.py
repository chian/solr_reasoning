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
    # Memory-efficient options for small models
    use_8bit: bool = False,
    use_4bit: bool = False,
    low_cpu_mem_usage: bool = True,
):
    # Import transformers and MT-GRPO components
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig  # GRPOConfig is in trl
    from datasets import Dataset
    import sys
    import os
    
    # Import the actual MT-GRPO trainer
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'external', 'Multi-Turn-RL-Agent'))
    from verifiers.trainers.mt_grpo_env_trainer import MTGRPOEnvTrainer
    
    # Load model and tokenizer with memory-efficient options
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=low_cpu_mem_usage,
        torch_dtype="auto",
        device_map="auto" if num_gpus > 0 else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Ensure tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build dataset in the format expected by TRL GRPO trainer
    dataset_data = []
    for prompt, messages, answer in zip(buffer.prompts, buffer.messages, buffer.answers):
        # TRL GRPO expects just the prompt field
        dataset_data.append({
            "prompt": prompt
        })
    
    print(f"Dataset data created with {len(dataset_data)} samples")
    print(f"Sample dataset entry: {dataset_data[0] if len(dataset_data) > 0 else 'Empty dataset'}")
    
    dataset = Dataset.from_list(dataset_data)
    print(f"Dataset created with {len(dataset)} samples")

    # Create the P3 environment with the dataset
    env = P3Env(dataset=dataset)

    # Create GRPO config with proper settings for small dataset
    # Ensure generation_batch_size is divisible by num_generations
    per_device_train_batch_size = 20  # Must be at least 20 for num_generations=20
    gradient_accumulation_steps = 1
    num_generations = 20  # Generate many completions for better multi-turn reasoning
    
    config = GRPOConfig(
        learning_rate=learning_rate,
        max_steps=min(max_steps, len(dataset)),  # Don't exceed dataset size
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_generations=num_generations,
        beta=beta,
        save_steps=1,  # Save frequently
        logging_steps=1,  # Log frequently
        eval_steps=1,  # Eval frequently
        report_to="none",  # Disable wandb
        use_vllm=True,  # Required for MT-GRPO
        vllm_mode="colocate",  # Use colocated vLLM instead of server mode
        max_completion_length=200,  # Allow longer completions for multi-turn reasoning
        temperature=0.7,  # Add some randomness for better generation
        top_p=0.9,  # Nucleus sampling for better quality
    )

    # Create MT-GRPO trainer (the actual Multi-Turn GRPO implementation)
    reward_funcs = env.get_rubric()  # Get list of reward functions
    trainer = MTGRPOEnvTrainer(
        model=model,
        env=env,
        turn_reward_funcs=reward_funcs,  # Turn-level rewards (list)
        outcome_reward_funcs=reward_funcs,  # Outcome-level rewards (list)
        turn_advantage_coef=turn_advantage_coef,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    
    print("Starting training...")
    print(f"Dataset sample keys: {list(dataset[0].keys()) if len(dataset) > 0 else 'Empty'}")
    
    trainer.train()