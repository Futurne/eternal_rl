#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math

import gym
import einops
from einops.layers.torch import Rearrange
from positional_encodings.torch_encodings import PositionalEncoding2D

import torch
import torch.nn as nn


class TileIdEncoding(nn.Module):
    """From https://pytorch.org/tutorials/beginner/transformer_tutorial.html.
    """
    def __init__(
            self,
            hidden_size: int,
            dropout: float = 0.1,
            max_classes: int = 24,
            const_term: float = 10000.0,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_classes).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2) * (-math.log(const_term) / hidden_size))
        pe = torch.zeros(max_classes, hidden_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        """
        Input
        -----
            x: Tiles with their value as a long ID.
                Shape of [batch_size, 4, map_size, map_size].

        Output
        ------
            x: Tiles encoding depending on their ID.
                Shape of [batch_size, 4, map_size, map_size, hidden_size].
        """
        x = self.pe[x]
        return self.dropout(x)


class CNNFeaturesExtractor(nn.Module):
    def __init__(
            self,
            observation_space: gym.spaces.Box,
            n_channels: int,
            n_layers: int,
            max_classes: int = 23,
            dropout: float = 0.1,
            const_term_encoding: float = 10000.0,
        ):
        super().__init__()
        n_sides = observation_space.shape[0]

        self.embed_classes = nn.Sequential(
            TileIdEncoding(n_channels, dropout, max_classes, const_term_encoding),  # Encode tiles based on their class ID
            Rearrange('b r s1 s2 c -> b s1 s2 (c r)'),  # Concat rolls and class embeddings
            nn.Linear(n_sides * n_channels, n_channels),  # Reduce dims
            nn.LayerNorm(n_channels),
            nn.LeakyReLU(),
            Rearrange('b s1 s2 c -> b c s1 s2'),  # CNN-like shape
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
        x = observations.argmax(dim=2)  # Get class ids
        x = self.embed_classes(x)
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
            const_term_encoding: float = 10000.0,
    ):
        super().__init__()

        self.cnn = CNNFeaturesExtractor(
            observation_space,
            hidden_size,
            cnn_layers,
            dropout=dropout,
            const_term_encoding=const_term_encoding,
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

