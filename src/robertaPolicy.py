import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

class RobertaPolicy(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64):

        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]

        self.fc1 = nn.Linear(n_input_channels, 64)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(64, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations = F.leaky_relu(self.fc1(observations))
        observations = self.dropout(observations)
        observations = F.leaky_relu(self.fc2(observations))

        return observations
