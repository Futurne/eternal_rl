#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from typing import Optional

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
        manual_orient: bool,
        net_arch: Optional[dict[str, any]] = None,
        *args,
        **kwargs,
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.manual_orient = manual_orient

        # Default args
        self.net_arch = {
            'hidden_size': 10,
            'cnn_layers': 1,
            'n_heads': 2,
            'ff_size': 20,
            'dropout': 0.1,
            'n_layers': 3,
            'const_term_encoding': 10000.0,
        }
        if net_arch:
            self.net_arch |= net_arch

        # Makes sure that all parameters are here
        assert all(
            param in self.net_arch
            for param in [
                'hidden_size',
                'cnn_layers',
                'n_heads',
                'ff_size',
                'dropout',
                'n_layers',
                'const_term_encoding',
            ]
        )

        super(PointerActorCritic, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch = self.net_arch,  # Overwrite default net_arch
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self):
        self.mlp_extractor = FeaturesExtractorModel(
            self.observation_space,
            self.net_arch['hidden_size'],
            self.net_arch['cnn_layers'],
            self.net_arch['n_heads'],
            self.net_arch['ff_size'],
            self.net_arch['dropout'],
            self.net_arch['n_layers'],
            self.net_arch['const_term_encoding'],
        )

    def _build(self, lr_schedule: Schedule):
        """Create the networks and the optimizer.
        """
        self._build_mlp_extractor()

        self.mha = nn.MultiheadAttention(
            self.net_arch['hidden_size'],
            self.net_arch['n_heads'],
            dropout = 0,  # 0% otherwise the attention will not sum up to 1
            batch_first = True,
        )

        if self.manual_orient:
            self.roll = nn.Sequential(
                nn.Linear(self.net_arch['hidden_size'], self.action_space.nvec[-1]),
                nn.Softmax(dim=2),
            )

        self.value = nn.Linear(2 * self.net_arch['hidden_size'], 1)

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
                Only outputted if the `manual_orient` parameter is `True`.
                Shape of [batch_size, 2, n_rolls].
        """
        tiles, tokens = features
        _, selected = self.mha(tokens, tiles, tiles, need_weights=True)
        if not self.manual_orient:
            return selected

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
        actions = self.action_net(features)

        if self.manual_orient:
            selected, rolls = actions
            rolls = [
                Categorical(probs=rolls[:, i])
                for i in range(rolls.shape[1])
            ]
        else:
            selected = actions
            rolls = []

        self.action_dist.distribution = [
            Categorical(probs=selected[:, i])
            for i in range(selected.shape[1])
        ] + rolls
        return self.action_dist

    def extract_features(self, observations: torch.ByteTensor) -> torch.FloatTensor:
        """Overwrite the `extract_features` method.
        Cast the observations to a FloatTensor.
        """
        return observations.float()

    def save(self, path: str):
        """Save the parameters to be loaded later.

        Input
        -----
            path: Directory where to put the models.
        """
        if not os.path.isdir(path):
            os.makedirs(path)

        torch.save(
            self.mlp_extractor.state_dict(),
            os.path.join(path, 'mlp_extractor.pt')
        )
        torch.save(
            self.mha.state_dict(),
            os.path.join(path, 'mha.pt')
        )
        torch.save(
            self.value.state_dict(),
            os.path.join(path, 'value.pt')
        )
        if self.manual_orient:
            torch.save(
                self.roll.state_dict(),
                os.path.join(path, 'roll.pt')
            )

    def load_from_state_dict(self, path: str):
        """Load from previously saved parameters.

        Input
        -----
            path: Directory where to load the models.
        """
        self.mlp_extractor.load_state_dict(
            torch.load(os.path.join(path, 'mlp_extractor.pt'))
        )
        self.mha.load_state_dict(
            torch.load(os.path.join(path, 'mha.pt'))
        )
        self.value.load_state_dict(
            torch.load(os.path.join(path, 'value.pt'))
        )
        if self.manual_orient:
            self.roll.load_state_dict(
                torch.load(os.path.join(path, 'roll.pt'))
            )

