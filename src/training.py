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

def train(lr, timesteps, seed, algorithm):

    set_seed(seed)

    timestamp = time.strftime("%d-%m-%Y-%H:%M:%S")
    log_dir = f"logs/{algorithm}/lr{lr}_seed{seed}_episodes{timesteps}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    env = gym.make("RobertaEnv-v0")
    _, info = env.reset()
    info['algorithm'] = algorithm
    json_path = os.path.join(log_dir, "info.json")
    with open(json_path, "w") as js:
        json.dump(info, js, indent=4, sort_keys=False)
    env.close()

    env = make_vec_env("RobertaEnv-v0", n_envs=16, seed=seed)
    env = VecFrameStack(env, n_stack=4)
    eval_env = make_vec_env("RobertaEnv-v0", n_envs=1, seed=seed)
    eval_env = VecFrameStack(eval_env, n_stack=4)

    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=log_dir, 
        log_path=log_dir,
        eval_freq=100000,
        deterministic=True, 
        render=False
    )

    if algorithm == "SAC": 
        model = create_sac(env, lr, log_dir)
    elif algorithm == "PPO":
        model = create_ppo(env, lr, log_dir)
    else:
        ValueError(f"No suported algorithm.")
    
    print(f"\n Training {algorithm} on RobertaEnv-v0")
    print(f"→ Learning Rate: {lr}")
    print(f"→ Seed: {seed}")
    print(f"→ Timesteps: {timesteps}")
    print(f"→ Logs: {log_dir}\n")

    model.learn(total_timesteps=timesteps, callback=eval_callback, progress_bar=True)

    model_path = os.path.join(log_dir, f"{algorithm}_Roberta_lr{lr}_seed{seed}.zip")
    model.save(model_path)

    print(f"\n Model saved to: {model_path}\n")
    env.close()

if __name__ == "__main__":
    gym.register(
        id='RobertaEnv-v0',
        entry_point=RobertaEnv,
        max_episode_steps=1000,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--algorithm", type=str, default=None)
    parser.add_argument("--timesteps", type=int, default=None)

    args = parser.parse_args()

    if args.algorithm == None or (args.algorithm != "PPO" and args.algorithm != "SAC") :
        print("ERROR: not defined the --algorithm for training")
        print("please use 'PPO' or 'SAC' for training")

        

    if args.timesteps == None:
        print("ERROR: not defined the --timesteps for training")
    else:
        print("Initing training!")
        train(args.lr, args.timesteps, args.seed, args.algorithm)

