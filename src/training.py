from model_SAC_functions import set_seed, create_sac

import time
import os
import gymnasium as gym

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

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