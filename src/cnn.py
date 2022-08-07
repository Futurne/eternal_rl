import gym

import torch
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CNNFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(
            self,
            observation_space: gym.spaces.Box,
            hidden_dim: int,
            n_channels: int,
            n_layers: int,
        ):
        super(CNNFeaturesExtractor, self).__init__(observation_space, hidden_dim)

        in_channels = observation_space.shape[0]
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, n_channels, 3, 1, padding='same'),
            nn.ReLU(),
        )
        self.cnn = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(n_channels, n_channels, 3, 1, padding='same'),
                nn.BatchNorm2d(n_channels),
                nn.ReLU(),
            )
            for _ in range(n_layers)
        ])

        with torch.no_grad():
            obs = torch.as_tensor(observation_space.sample()[None]).float()
            emb = self.project(obs)
            for layer in self.cnn:
                emb = layer(emb)
            flat_dim = emb.flatten().shape[-1]

        self.flatten = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.FloatTensor) -> torch.FloatTensor:
        embeddings = self.project(observations)
        for layer in self.cnn:
            embeddings = layer(embeddings) + embeddings

        return self.flatten(embeddings)


if __name__ == '__main__':
    import numpy as np
    from torchinfo import summary

    observation_space = gym.spaces.Box(
        low=0,
        high=1,
        shape=[4 * 13, 4, 4],
        dtype=np.uint8
    )
    model = CNNFeaturesExtractor(
        observation_space,
        hidden_dim=128,
        n_channels=16,
        n_layers=3
    )

    summary(
        model,
        input_size=[(64, 4 * 13, 4, 4),],
        device='cpu',
    )

