from robertaEnv import RobertaEnv
from model_SAC_functions import make_env

import argparse
import math
import numpy as np
import os
import gymnasium as gym
import matplotlib.pyplot as plt

from stable_baselines3 import SAC


def run_demo(model_path, episodes):

    if not os.path.exists(model_path):
        print(f"\n Model not found: {model_path}\n")
        return
    
    print(f"\n Running SAC Demo: {model_path}\n")

    img_dir = model_path.replace(".zip", "") + "/images_test"
    os.makedirs(img_dir, exist_ok=True)

    env = make_env(render_mode="human")
    model = SAC.load(model_path)

    for ep in range(episodes):
        obs, info = env.reset()
        setpoint = info["setpoint"]
        print(f"Setpoint: {setpoint * 180 / math.pi}")
        done = False
        ep_reward = 0
        tic = 0
        phi = []
        step = []


        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward

            env.render()

            tic += 1
            step.append(tic)
            phi.append(obs[0])
            
            done = terminated or truncated
            if terminated:
                print("Killed")
            if truncated:
                print("Max steps, truncating episode")


        print(f"Episode {ep + 1} Reward = {ep_reward}")

        setpoint_vec = np.ones_like(phi) * setpoint

        plt.plot(step, phi, color='b')
        plt.plot(step, setpoint_vec, color='r')
        plt.xlabel('Steps')
        plt.ylabel('Angle (phi)')
        plt.grid()
        img_pth = os.path.join(img_dir, f"episode_{ep+1}_reward_{ep_reward}.png")
        plt.savefig(img_pth)
        plt.close()
    
    env.close()

if __name__ == "__main__":

    gym.register(
        id='RobertaEnv-v0',
        entry_point=RobertaEnv,
        max_episode_steps=1000,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=10)
    # parser.add_argument("--help")

    args = parser.parse_args()

    if args.model is None:
        print("\n ERROR: missing --model path\n")
    else:
        run_demo(args.model, args.episodes)
