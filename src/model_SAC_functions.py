from robertaEnv import RobertaEnv

import argparse
import os
import time
import gymnasium as gym
import numpy as np
import torch

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_env(render_mode=None):
    env = gym.make("RobertaEnv-v0", render_mode=render_mode)
    env = Monitor(env)
    return env

def linear_schedule(initial_value: float):
    def func(progress_remaning):
        return initial_value * (0.5 * (1 + np.cos(np.pi * (1 - progress_remaning))))
    
    return func

def create_sac(env, lr, log_dir):

    
    policy_kwards =dict(
        net_arch = dict(
            pi=[32],
            qf=[32]
        )
    )
    

    model = SAC(
        "MlpPolicy",
        env=env,
        policy_kwargs=policy_kwards,
        device="cpu",
        gamma=0.99,
        ent_coef="auto",
        buffer_size=10000000,
        batch_size=256,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        learning_rate=linear_schedule(lr),
        verbose=1,
        tensorboard_log=log_dir
    )

    return model
