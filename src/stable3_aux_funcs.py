import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.monitor import Monitor

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_env(render_mode=None):
    env = gym.make("RobertaEnv-v0", render_mode=render_mode)
    env = Monitor(env)
    return env

def cossine_schedule(initial_value: float):
    """
    Implements a cosine annealing learning rate schedule.
    """
    def func(progress_remaning):
        return initial_value * (0.5 * (1 + np.cos(np.pi * (1 - progress_remaning))))
    
    return func