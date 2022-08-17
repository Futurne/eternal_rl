import gym
import einops
from einops.layers.torch import Rearrange

import torch
import torch.nn as nn


class CNNFeaturesExtractor(nn.Module):
    def __init__(
            self,
            observation_space: gym.spaces.Box,
            n_channels: int,
            n_layers: int,
        ):
        super().__init__()
        n_sides, n_class, map_size, _ = observation_space.shape

        self.embed_classes = nn.Sequential(
            Rearrange('b s c s1 s2 -> (b s1 s2) (s c)'),  # Do batch linear inference on each token

            nn.Linear(n_sides * n_class, n_channels),
            nn.LayerNorm(n_channels),
            nn.LeakyReLU(),
        )

        self.cnn = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(n_channels, n_channels, 3, 1, padding='same'),
                nn.BatchNorm2d(n_channels),
                nn.ReLU(),
            )
            for _ in range(n_layers)
        ])

    def forward(self, observations: torch.FloatTensor) -> torch.FloatTensor:
        """Embed each tokens in a more meaningful way.

        Input
        -----
            observations: One-hot rendering of the map.
                Shape of [batch_size, 4, n_class, map_size, map_size].

        Output
        ------
            features: Extracted features of each token.
                Shape of [batch_size, n_channels, map_size, map_size].
        """
        b_size, m_size = observations.shape[0], observations.shape[-1]

        x = self.embed_classes(observations)
        x = einops.rearrange(x, '(b s1 s2) c -> b c s1 s2', b=b_size, s2=m_size)  # Go back to CNN-like shape
        for layer in self.cnn:
            x = layer(x) + x

        return x

