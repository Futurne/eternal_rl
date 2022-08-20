#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gym
import einops
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.distributions import Distribution

import torch
import torch.nn as nn
from torch.distributions import Categorical

from src.model.extractor import FeaturesExtractorModel


class PointerActorCritic(ActorCriticPolicy):
    """Custom network for policy and value function.

    You can find the original ActorCriticPolicy implementation here:
    https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/policies.html#ActorCriticPolicy
    """
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: callable,
        hidden_size: int = 50,
        cnn_layers: int = 1,
        n_heads: int = 5,
        ff_size: int = 100,
        dropout: float = 0.1,
        n_layers: int = 3,
        *args,
        **kwargs,
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.hidden_size = hidden_size
        self.cnn_layers = cnn_layers
        self.n_heads = n_heads
        self.ff_size = ff_size
        self.dropout = dropout
        self.n_layers = n_layers
        self.n_rolls = observation_space.shape[0]

        super(PointerActorCritic, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self):
        self.mlp_extractor = FeaturesExtractorModel(
            self.observation_space,
            self.hidden_size,
            self.cnn_layers,
            self.n_heads,
            self.ff_size,
            self.dropout,
            self.n_layers,
        )

    def _build(self, lr_schedule: Schedule):
        """Create the networks and the optimizer.
        """
        self._build_mlp_extractor()

        self.mha = nn.MultiheadAttention(
            self.hidden_size,
            self.n_heads,
            dropout = 0,  # 0% otherwise the attention will not sum up to 1
            batch_first = True,
        )
        self.roll = nn.Sequential(
            nn.Linear(self.hidden_size, self.n_rolls),
            nn.Softmax(dim=2),
        )
        self.value = nn.Linear(2 * self.hidden_size, 1)


        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def value_net(
            self,
            features: tuple[torch.FloatTensor, torch.FloatTensor],
    ) -> torch.FloatTensor:
        """Value network predicting the value of an action.

        Input
        -----
            features: Tuple of (tiles_embeddings, decoder_tokens_embeddings).
                tiles_embeddings: Features extracted from the tiles.
                    Shape of [batch_size, n_tiles, hidden_size].
                decoder_tokens_embeddings: Features extracted from the decoder based on the tiles.
                    Shape of [batch_size, 2, hidden_size].

        Output
        ------
            value: Predicted values of the actions.
                Shape of [batch_size, 1].
        """
        tiles, tokens = features
        tokens = einops.rearrange(tokens, 'b s h -> b (s h)')
        return self.value(tokens)

    def action_net(
            self,
            features: tuple[torch.FloatTensor, torch.FloatTensor],
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        """Action network predicting the future action.

        Input
        -----
            features: Tuple of (tiles_embeddings, decoder_tokens_embeddings).
                tiles_embeddings: Features extracted from the tiles.
                    Shape of [batch_size, n_tiles, hidden_size].
                decoder_tokens_embeddings: Features extracted from the decoder based on the tiles.
                    Shape of [batch_size, 2, hidden_size].

        Output
        ------
            selected: The selected tiles to swap.
                Shape of [batch_size, 2, n_tiles].
            rolls: The rolls to apply to the selected tiles.
                Shape of [batch_size, 2, n_rolls].
        """
        tiles, tokens = features
        _, selected = self.mha(tokens, tiles, tiles, need_weights=True)
        rolls = self.roll(tokens)
        return selected, rolls

    def _get_action_dist_from_latent(
            self,
            features: tuple[torch.FloatTensor, torch.FloatTensor],
    ) -> Distribution:
        """Action network predicting the future action.

        Input
        -----
            features: Tuple of (tiles_embeddings, decoder_tokens_embeddings).
                tiles_embeddings: Features extracted from the tiles.
                    Shape of [batch_size, n_tiles, hidden_size].
                decoder_tokens_embeddings: Features extracted from the decoder based on the tiles.
                    Shape of [batch_size, 2, hidden_size].

        Output
        ------
            action_dist: List of batched distribution of the actions.
                The first two are the distributions of the selected tiles to swap.
                The last two are the distributions of the rolls to apply to selected tiles.
        """
        selected, rolls = self.action_net(features)
        self.action_dist.distribution = [
            Categorical(probs=selected[:, i])
            for i in range(selected.shape[1])
        ] + [
            Categorical(probs=rolls[:, i])
            for i in range(rolls.shape[1])
        ]
        return self.action_dist

    def extract_features(self, observations: torch.ByteTensor) -> torch.FloatTensor:
        """Overwrite the `extract_features` method.
        Cast the observations to a FloatTensor.
        """
        return observations.float()

