import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO

from robertaEnv import RobertaEnv

def create_ppo(env, lr, log_dir):
    
    policy_keywards = dict(
        net_arch = dict(
            pi=[32],
            qi=[32]
        ),
        activation_fn=nn.ReLU
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=policy_keywards,
        learning_rate=lr,
        device="cpu",
        gamma=0.99,
        batch_size=256,
        n_epochs=10,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log=log_dir
    )

    return model