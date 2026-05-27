from robertaEnv import RobertaEnv
from robertaPolicy import RobertaPolicy

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

    custom_policy = dict(
        features_extractor_class=RobertaPolicy,
        features_extractor_kwargs=dict(features_dim=64),
        net_arch=dict(
            pi=[32],
            qf=[32]
        )
    )

    model = SAC(
        "MlpPolicy",
        env=env,
        policy_kwargs=custom_policy,
        device="cpu",
        gamma=0.99,
        ent_coef="auto",
        buffer_size=1000000,
        batch_size=256,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        learning_rate=linear_schedule(lr),
        verbose=1,
        tensorboard_log=log_dir
    )

    return model

def train_sac(lr, timesteps, seed):

    set_seed(seed)

    timestamp = time.strftime("%d/%m/%Y-%H %M %S")
    log_dir = f"logs/SAC_Roberta_lr{lr}_seed{seed}_episodes{timesteps}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    env = make_vec_env("RobertaEnv-v0", n_envs=24, seed=seed)
    eval_env = gym.make("RobertaEnv-v0")

    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=log_dir, 
        log_path=log_dir,
        eval_freq=100000,
        deterministic=True, 
        render=False
    )

    model = create_sac(env, lr, log_dir)


    print(f"\n Training SAC on RobertaEnv-v0")
    print(f"→ Learning Rate: {lr}")
    print(f"→ Seed: {seed}")
    print(f"→ Timesteps: {timesteps}")
    print(f"→ Logs: {log_dir}\n")

    model = create_sac(env, lr, log_dir)
    model.learn(total_timesteps=timesteps, callback=eval_callback, progress_bar=True)

    model_path = os.path.join(log_dir, f"SAC_Roberta_lr{lr}_seed{seed}.zip")
    model.save(model_path)

    print(f"\n Model saved to: {model_path}\n")
    env.close()

def retrain_sac(model_path: str, timesteps: int, seed: int):
    set_seed(seed)
    env = make_vec_env("RobertaEnv-v0", n_envs=16, seed=seed)

    model = SAC.load(model_path, env=env)

    model.learn(
        total_timesteps=timesteps,
        progress_bar=True,
        reset_num_timesteps=False,
        tb_log_name="SAC_FineTuning"
    )

    new_path = model_path.replace(".zip", "_v2.zip")
    model.save(new_path)
    print(f"New model saved to: {model_path}")

    env.close()