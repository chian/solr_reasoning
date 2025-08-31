import os
import subprocess
from typing import List, Optional


def run_mt_grpo(
    model_name: str,
    num_gpus: int,
    learning_rate: float = 1e-6,
    num_generations: int = 8,
    per_device_train_batch_size: int = 2,
    grad_accum_steps: int = 2,
    num_iterations: int = 1,
    max_steps: int = 100,
    beta: float = 0.0,
    trainer: str = "mt_grpo",
    turn_advantage_coef: float = 1.0,
    accelerate_config: str = "external/Multi-Turn-RL-Agent/configs/zero3.yaml",
    extra_args: Optional[List[str]] = None,
) -> int:
    """
    Thin wrapper to launch the upstream trainer via accelerate.
    All parameters are forwarded as CLI flags; no hardcoding of credentials.
    Returns the process return code.
    """
    script = "external/Multi-Turn-RL-Agent/verifiers/examples/triviaqa_search.py"

    cmd = [
        "accelerate",
        "launch",
        "--config-file",
        accelerate_config,
        "--num-processes",
        str(max(1, num_gpus - 1)),
        script,
        "--model_name",
        model_name,
        "--num_gpus",
        str(num_gpus),
        "--learning_rate",
        str(learning_rate),
        "--num_generations",
        str(num_generations),
        "--per_device_train_batch_size",
        str(per_device_train_batch_size),
        "--grad_accum_steps",
        str(grad_accum_steps),
        "--num_iterations",
        str(num_iterations),
        "--max_steps",
        str(max_steps),
        "--beta",
        str(beta),
        "--trainer",
        trainer,
        "--turn_advantage_coef",
        str(turn_advantage_coef),
    ]

    if extra_args:
        cmd.extend(extra_args)

    env = os.environ.copy()
    proc = subprocess.run(cmd, env=env)
    return proc.returncode


