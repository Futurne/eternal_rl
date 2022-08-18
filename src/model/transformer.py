import gym

import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
from positional_encodings import PositionalEncoding2D, PositionalEncoding1D

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TransformerFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(
            self,
            observation_space: gym.spaces.Box,
            hidden_dim: int,
            n_heads: int,
            n_layers: int,
            n_class: int,
        ):
        super(TransformerFeaturesExtractor, self).__init__(observation_space, hidden_dim)
        self.hidden_dim = hidden_dim

        # Embedding layers
        self.class_emb = nn.Embedding(n_class, hidden_dim)
        self.loc_emb = PositionalEncoding1D(hidden_dim)
        self.glob_emb = PositionalEncoding2D(hidden_dim)

        # Aggregator tokens
        self.loc_tok = nn.Parameter(torch.randn(1, hidden_dim))
        self.glob_tok = nn.Parameter(torch.randn(1, hidden_dim))

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=2 * hidden_dim,
            batch_first=True,
        )
        self.local_encoder = nn.TransformerEncoder(encoder_layer, n_layers)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=2 * hidden_dim,
            batch_first=True,
        )
        self.global_encoder = nn.TransformerEncoder(encoder_layer, n_layers)

        self.flatten_2D = Rearrange('b s1 s2 h -> b (s1 s2) h')
        self.flat_1D_tile = Rearrange('b s1 s2 t h -> (b s1 s2) t h')

    def forward(self, observations: torch.LongTensor) -> torch.FloatTensor:
        """
        Input
        -----
            observations: Raw boards.
                Shape of [batch_size, size, size, 4].

        Output
        ------
            embeddings: Embeddings of each board.
                Shape of [batch_size, hidden_dim].
        """
        observations = observations.long()
        b_size = observations.shape[0]
        size = observations.shape[1]

        # Embeddings
        h = self.class_emb(observations)                # [batch_size, size, size, 4, hidden_dim]
        h = self.flat_1D_tile(h)                        # To get a 1D tile vector along the local scope
        h = h + self.loc_emb(h)                         # Add local positional embedding
        # h is of shape [batch_size * size * size, 4 , hidden_dim]

        # Local transformer
        """
        loc_tok = einops.repeat(
            self.loc_tok,
            't h -> b t h',
            b=h.shape[0]
        )
        h = torch.cat((h, loc_tok), dim=1)              # Append the aggregator token
        """
        h = self.local_encoder(h)
        h = h.sum(dim=1)
        # h = h[:, -1]                                    # Get the info from the aggregator token
        h = h.view(b_size, size, -1, self.hidden_dim)
        # h is of shape [batch_size, size, size, hidden_dim]

        # Global transformer
        h = h + self.glob_emb(h)                        # Add global positional embedding
        h = self.flatten_2D(h)                          # [batch_size, size * size, hidden_dim]
        """
        glob_tok = einops.repeat(
                observation = np.expand_dims(observation, axis=-1)  # Shape is [size, size, 4, 1]
            self.glob_tok,
            't h -> b t h',
            b=b_size
        )
        h = torch.cat((h, glob_tok), dim=1)
        """
        h = self.global_encoder(h)
        # return h[:, -1]                                 # [batch_size, hidden_dim]
        return h.sum(dim=1)


if __name__ == '__main__':
    import numpy as np
    from torchinfo import summary

    observation_space = gym.spaces.Box(
        low=0,
        high=1,
        shape=[8, 8, 4, 1],  # [size, size, 4, 1]
        dtype=np.uint8
    )
    model = TransformerFeaturesExtractor(
        observation_space,
        hidden_dim=128,
        n_heads=4,
        n_layers=3,
        n_class=13,
    )

    summary(
        model,
        input_data=[torch.zeros((64, 8, 8, 4), dtype=torch.long),],
        device='cpu',
    )
