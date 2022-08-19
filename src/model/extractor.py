#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gym
import einops
from einops.layers.torch import Rearrange
from positional_encodings.torch_encodings import PositionalEncoding2D

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
        n_sides, n_class, *_ = observation_space.shape

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


class TransformerFeaturesExtractor(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            n_heads: int,
            ff_size: int,
            dropout: float,
            n_layers: int,
    ):
        super().__init__()

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_size,
            n_heads,
            ff_size,
            dropout,
            batch_first = True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.pos_encoding = PositionalEncoding2D(hidden_size)

        # Pointer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            hidden_size,
            n_heads,
            ff_size,
            dropout,
            batch_first = True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, n_layers)
        self.mha = nn.MultiheadAttention(
            hidden_size,
            n_heads,
            dropout = 0,  
            batch_first = True,
        )
        self.rotate_token = nn.Sequential(
            nn.Linear(hidden_size, 4),
            nn.Softmax(dim=2),
        )
        self.decoder_toks = nn.Parameter(torch.randn(2, hidden_size))

    def forward(
        self,
        tiles: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Embed the tiles based on themselves.
        Compute two token embeddings based on the tiles embeddings and on themselves.

        Input
        -----
            tiles: List of tokens embeddings.
                Shape of [batch_size, hidden_size, map_size, map_size].

        Output
        ------
            x: Tiles embeddings after the transformer encoder module.
                Shape of [batch_size, map_size * map_size, hidden_size].
            y: The two decoder token embeddings after the transformer decoder module.
                Shape of [batch_size, 2, hidden_size].
        """
        batch_size = tiles.shape[0]

        # Encoder
        x = einops.rearrange(tiles, 'b h m1 m2 -> b m1 m2 h')
        x = x + self.pos_encoding(x)  # Add 2D encoding
        x = einops.rearrange(x, 'b m1 m2 h -> b (m1 m2) h')
        x = self.encoder(x)

        # Decoder
        y = einops.repeat(self.decoder_toks, 's h -> b s h', b=batch_size)
        y = self.decoder(y, x)

        return x, y



class FeaturesExtractorModel(nn.Module):
    def __init__(
            self,
            observation_space: gym.spaces.Box,
            hidden_size: int,
            cnn_layers: int,
            n_heads: int,
            ff_size: int,
            dropout: float,
            n_layers: int,
    ):
        super().__init__()

        self.cnn = CNNFeaturesExtractor(
            observation_space,
            hidden_size,
            cnn_layers,
        )

        self.encoder = TransformerFeaturesExtractor(
            hidden_size,
            n_heads,
            ff_size,
            dropout,
            n_layers,
        )

    def forward(
        self,
        tiles: torch.FloatTensor,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        """Embed each tokens in a more meaningful way.

        Input
        -----
            observations: One-hot rendering of the map.
                Shape of [batch_size, 4, n_class, map_size, map_size].

        Output
        ------
            x: Tiles embeddings after the transformer encoder module.
                Shape of [batch_size, map_size * map_size, hidden_size].
            y: The two decoder token embeddings after the transformer decoder module.
                Shape of [batch_size, 2, hidden_size].

        The output is doubled because it returns both the latent representation for the
        actor and the critic network.
        """
        x = self.cnn(tiles)
        features = self.encoder(x)
        return features, features

    def forward_actor(self, tiles: torch.FloatTensor) -> torch.FloatTensor:
        return self.forward(tiles)[0]

    def forward_critic(self, tiles: torch.FloatTensor) -> torch.FloatTensor:
        return self.forward(tiles)[1]

