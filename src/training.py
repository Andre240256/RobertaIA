"""
Training script for Reinforcement Learning agents using Stable Baselines3.
It registers and utilizes a custom Gymnasium environment ('RobertaEnv-v0') 
and supports training with PPO (Proximal Policy Optimization) or SAC (Soft Actor-Critic).
"""

from robertaEnv import RobertaEnv
from model_SAC_functions import set_seed, create_sac
from model_PPO_functions import create_ppo

import argparse
import time
import os
import gymnasium as gym
import json

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecFrameStack

def train(lr: float, timesteps: int, seed: int, algorithm: str) -> None:
    """
    Initializes and trains a reinforcement learning model.

    This function sets up the logging directory, initializes the vectorized 
    environments (with frame stacking), configures an evaluation callback, 
    instantiates the chosen RL algorithm, and executes the training loop.
    Finally, it saves the trained model to the disk.

    Args:
        lr (float): The learning rate for the optimizer.
        timesteps (int): The total number of environment steps to train the agent.
        seed (int): The random seed for reproducibility across the environment and algorithm.
        algorithm (str): The RL algorithm to use. Must be either "SAC" or "PPO".

    Raises:
        ValueError: If an unsupported algorithm string is provided.
    """
    set_seed(seed)

    # Setup logging directory based on hyperparameter configuration
    timestamp = time.strftime("%d-%m-%Y-%H:%M:%S")
    log_dir = f"logs/{algorithm}/lr{lr}_seed{seed}_episodes{timesteps}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    # Initialize a temporary environment to extract and log base info
    env = gym.make("RobertaEnv-v0")
    _, info = env.reset()
    info['algorithm'] = algorithm
    json_path = os.path.join(log_dir, "info.json")
    with open(json_path, "w") as js:
        json.dump(info, js, indent=4, sort_keys=False)
    env.close()

    # Create vectorized environments for training and evaluation
    # Uses VecFrameStack to stack the last 4 frames as input observations
    env = make_vec_env("RobertaEnv-v0", n_envs=16, seed=seed)
    env = VecFrameStack(env, n_stack=4)
    
    eval_env = make_vec_env("RobertaEnv-v0", n_envs=1, seed=seed)
    eval_env = VecFrameStack(eval_env, n_stack=4)

    # Configure the callback to evaluate the agent periodically and save the best model
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=log_dir, 
        log_path=log_dir,
        eval_freq=100000,
        deterministic=True, 
        render=False
    )

    # Instantiate the selected RL model
    if algorithm == "SAC": 
        model = create_sac(env, lr, log_dir)
    elif algorithm == "PPO":
        model = create_ppo(env, lr, log_dir)
    else:
        raise ValueError("No supported algorithm provided.")
    
    print(f"\n Training {algorithm} on RobertaEnv-v0")
    print(f"→ Learning Rate: {lr}")
    print(f"→ Seed: {seed}")
    print(f"→ Timesteps: {timesteps}")
    print(f"→ Logs: {log_dir}\n")

    # Execute the training process
    model.learn(total_timesteps=timesteps, callback=eval_callback, progress_bar=True)

    # Save the final trained model
    model_path = os.path.join(log_dir, f"{algorithm}_Roberta_lr{lr}_seed{seed}.zip")
    model.save(model_path)

    print(f"\n Model saved to: {model_path}\n")
    env.close()

if __name__ == "__main__":
    # Register the custom environment with Gymnasium
    gym.register(
        id='RobertaEnv-v0',
        entry_point=RobertaEnv,
        max_episode_steps=1000,
    )

    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(description="Train an RL agent on RobertaEnv-v0.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--algorithm", type=str, default=None, help="Algorithm to train: 'PPO' or 'SAC'.")
    parser.add_argument("--timesteps", type=int, default=None, help="Total timesteps for training.")

    args = parser.parse_args()

    # Validate arguments and initiate training
    if args.algorithm is None or args.algorithm not in ["PPO", "SAC"]:
        print("ERROR: not defined the --algorithm for training")
        print("please use 'PPO' or 'SAC' for training")
    elif args.timesteps is None:
        print("ERROR: not defined the --timesteps for training")
    else:
        print("Initing training!")
        train(args.lr, args.timesteps, args.seed, args.algorithm)
