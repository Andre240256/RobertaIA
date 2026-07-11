from robertaEnv import RobertaEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack

import argparse
import math
import numpy as np
import os
import gymnasium as gym
import matplotlib.pyplot as plt

from stable_baselines3 import SAC
from stable_baselines3 import PPO


def run_demo(model_path, episodes, device = "cpu"):
    r"""
    Loads a pre-trained Reinforcement Learning model and evaluates its performance 
    in a simulated environment. Generates performance plots for each episode to 
    visualize the convergence of the angle \phi toward the setpoint \phi^*.

    :param model_path: Path to the .zip file containing the trained model (SAC or PPO).
    :param episodes: Number of evaluation episodes to perform.
    :param device: Hardware device for inference ("cpu" or "cuda").
    """

    if not os.path.exists(model_path):
        print(f"\n Model not found: {model_path}\n")
        return
    

    img_dir = model_path.replace(".zip", "") + "/images_test"
    os.makedirs(img_dir, exist_ok=True)

    env = make_vec_env("RobertaEnv-v0", n_envs=1,
                       env_kwargs={'render_mode':'human'})
    env = VecFrameStack(env, n_stack=4)

    algorithm = model_path.replace("logs/", "").split('/')[0]
    print(f"\n Running {algorithm} Demo: {model_path}\n")

    if algorithm == "SAC":
        model = SAC.load(model_path, device=device)
    elif algorithm == "PPO":
        model = PPO.load(model_path, device=device)
    else:
        raise ValueError('Name does not contain a valid algorithm.')

    for ep in range(episodes):
        obs = env.reset()
        setpoint = obs[0][-2]
        print(f"Setpoint: {setpoint * 180 / math.pi}")
        done = False
        ep_reward = 0
        tic = 0
        phi = []
        step = []


        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, info = env.step(action)
            ep_reward += reward[0]

            env.render()

            tic += 1
            step.append(tic)
            phi.append(obs[0][-4])
            
            done = dones[0]
            if done:
                if info[0].get("TimeLimit.truncated", False):
                    print("Max steps, truncating episode")
                else:
                    print("Killed")


        print(f"Episode {ep + 1} Reward = {ep_reward}")

        setpoint_vec = np.ones_like(phi) * setpoint

        plt.plot(step[:-3], phi[:-3], color='b')
        plt.plot(step[:-3], setpoint_vec[:-3], color='r')
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

    parser = argparse.ArgumentParser(description="Evaluate RL mechanical arm controller.")
    parser.add_argument("--model", type=str, default=None, help="Path to the model .zip file.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run.")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda).")

    args = parser.parse_args()

    if args.model is None:
        print("\n ERROR: missing --model path\n")
    else:
        run_demo(args.model, args.episodes, device=args.device)
