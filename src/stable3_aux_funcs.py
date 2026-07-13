import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.monitor import Monitor
from typing import Callable, Optional

def set_seed(seed: int) -> None:
    r"""
    Sets the global seed for pseudo-random number generators in NumPy, PyTorch (CPU), 
    and CUDA to ensure reproducibility of the reinforcement learning experiments.

    :param seed: Integer value to seed the random number generators.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_env(render_mode: Optional[str] = None) -> Monitor:
    r"""
    Instantiates the RobertaEnv and wraps it in a Monitor class.
    The Monitor wrapper is essential for tracking performance metrics (episode length 
    and cumulative reward) necessary for TensorBoard logging.

    :param render_mode: Mode of rendering ("human" or "rgb_array").
    :return: A gym.Env wrapped in a Monitor.
    """
    env = gym.make("RobertaEnv-v0", render_mode=render_mode)
    env = Monitor(env)
    return env

def cossine_schedule(initial_value: float) -> Callable[[float], float]:
    r"""
    Implements a cosine annealing learning rate schedule, which allows the optimizer 
    to traverse the loss landscape smoothly.

    The learning rate \alpha(p) at progress p \in [0, 1] is defined as:
    \alpha(p) = \alpha_{initial} \cdot \left( 0.5 \cdot (1 + \cos(\pi \cdot (1 - p))) \right)

    This schedule prevents premature convergence by starting with a high learning rate 
    for exploration and decaying it as the policy nears optimality.

    :param initial_value: The starting learning rate at the beginning of the training (p=0).
    :return: A function that returns the annealed learning rate given the training progress.
    """
    def func(progress_remaining: float) -> float:
        # progress_remaining: 1.0 at start, 0.0 at end of training
        progress = 1.0 - progress_remaining 
        return initial_value * (0.5 * (1 + np.cos(np.pi * progress)))
    
    return func