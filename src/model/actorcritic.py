#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gym
from stable_baselines3.common.policies import ActorCriticPolicy

from src.model.extractor import CNNFeaturesExtractor, TransformerFeaturesExtractor


class PointerModel(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        n_rolls: int,
    ):
        super().__init__()



class PointerActorCritic(ActorCriticPolicy):
    """Custom network for policy and value function.

    You can find the original ActorCriticPolicy implementation here:
    https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/policies.html#ActorCriticPolicy
    """
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        hidden_size: int,
        cnn_layers: int,
        n_heads: int,
        ff_size: int,
        dropout: float,
        n_layers: int,
        *args,
        **kwargs,
    ):
        super(PointerActorCritic, self).__init__(
            observation_space,
            action_space,
            *args,
            **kwargs,
        )

        self.ortho_init = False

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
        self.mha = nn.MultiheadAttention(
            hidden_size,
            n_heads,
            dropout = 0,  # 0% otherwise the attention will not sum up to 1
            batch_first = True,
        )
        self.value = nn.Linear(2 * hidden_size, 1)

    def _build_mlp_extractor(self):
        self.mlp_extractor = nn.Sequential(
            self.cnn,
            self.encoder,
        )

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
        # TODO: do action prediction
        pass

    def _get_action_dist_from_latent(
            self,
            features: tuple[torch.FloatTensor, torch.FloatTensor],
    ) -> torch.FloatTensor:
        # TODO: return distribution based on the prediction of actions
        pass

