#!/usr/bin/env python
# -*- coding: utf-8 -*-


import einops
from positional_encodings.torch_encodings import PositionalEncoding2D

import torch
import torch.nn as nn


class PointerModel(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            n_heads: int,
            ff_size: int,
            dropout: float,
            n_layers: int,
            n_rolls: int = 4,
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
            1,
            dropout,
            batch_first = True,
        )
        self.rotate_token = nn.Sequential(
            nn.Linear(hidden_size, 4),
            nn.Softmax(dim=2),
        )
        self.first_tok = nn.Parameter(torch.randn(1, hidden_size))
        self.second_tok = nn.Parameter(torch.randn(1, hidden_size))

    def forward(
        self,
        tiles: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Select two tiles in the `tiles` tensor to swap and gives their rotations.

        Input
        -----
            tiles: List of tokens embeddings.
                Shape of [batch_size, hidden_size, map_size, map_size].

        Output
        ------
            selected: Selected tiles (one-hot).
                Shape of [batch_size, 2, map_size * map_size].
            rolls: Rolls for the selected tiles (one-hot).
                Shape of [batch_size, 2, n_rolls].
        """
        batch_size = tiles.shape[0]

        # Encoder
        x = einops.rearrange(tiles, 'b h m1 m2 -> b m1 m2 h')
        x = x + self.pos_encoding(x)  # Add 2D encoding
        x = einops.rearrange(x, 'b m1 m2 h -> b (m1 m2) h')
        x = self.encoder(x)

        # Pointer
        ## First tile selection & rotations
        first_token = einops.repeat(self.first_tok, 's h -> b s h', b=batch_size)
        query = self.decoder(first_token, x)
        tile_emb, first_selected_tile = self.mha(query, x, x, need_weights=True)
        first_rotation = self.rotate_token(tile_emb)  # [batch_size, 1, 4]

        ## Second tile selection
        second_token = einops.repeat(self.second_tok, 's h -> b s h', b=batch_size)
        tokens = torch.concat(
            (first_token, second_token),
            dim=1,
        )
        query = self.decoder(tokens, x)[:, 1:]
        tile_emb, second_selected_tile = self.mha(query, x, x, need_weights=True)
        second_rotation = self.rotate_token(tile_emb)

        return (
            torch.concat((first_selected_tile, second_selected_tile), dim=1),
            torch.concat((first_rotation, second_rotation), dim=1)
        )

