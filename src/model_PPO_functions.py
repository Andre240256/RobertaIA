import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO

from robertaEnv import RobertaEnv


class ActorCritic(nn.Module):

    def __init__(self, nb_observations, nb_actions):
        super().__init__()

        self.nb_observations = nb_actions
        self.nb_actions = nb_actions

        self.actor_fc1 = nn.Linear(nb_observations, 64)
        self.actor_fc2 = nn.Linear(64, 64)
        self.actor_fc3 = nn.Linear(64, nb_actions)

        self.actor_logstd = nn.Parameter(torch.zeros(1, nb_actions))

        self.critic_fc1 = nn.Linear(nb_observations, 64)
        self.critic_fc2 = nn.Linear(64, 64)
        self.critic_fc3 = nn.Linear(64, nb_actions)
        

    def forward(self, x: torch.Tensor):
        a = F.relu(self.actor_fc1(x))
        a = F.relu(self.actor_fc2(a))
        action_mean = torch.tanh(self.actor_fc3(a)) 
        
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd) 

        v = F.relu(self.critic_fc1(x))
        v = F.relu(self.critic_fc2(v))
        value = self.critic_fc3(v)
        return action_mean, action_std, value
    
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