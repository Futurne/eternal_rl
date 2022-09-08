#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gym
import torch
import numpy as np

from src.environment.environment import EternityEnv
from src.model.extractor import CNNFeaturesExtractor, TransformerFeaturesExtractor, FeaturesExtractorModel
from src.model.actorcritic import PointerActorCritic


def test_cnn_extractor():
    observation_space = gym.spaces.Box(
        low = 0,
        high = 1,
        shape = [4, 13, 4, 4],
        dtype = np.uint8
    )
    model = CNNFeaturesExtractor(
        observation_space,
        n_channels = 10,
        n_layers = 2,
    )
    env = EternityEnv('instances/eternity_A.txt', manual_orient=True)

    obs = torch.FloatTensor(env.render())
    obs = torch.unsqueeze(obs, dim=0)
    assert model(obs).shape == torch.Size([1, 10, 4, 4])


def test_transformer_extractor():
    hidden_size = 10
    batch_size = 64
    map_size = 4

    model = TransformerFeaturesExtractor(
        hidden_size = hidden_size,
        n_heads = 2,
        ff_size = 20,
        dropout = 0.1,
        n_layers = 3,
    )

    tiles = torch.randn(batch_size, hidden_size, map_size, map_size)
    x, y = model(tiles)
    assert x.shape == torch.Size([batch_size, map_size * map_size, hidden_size])
    assert y.shape == torch.Size([batch_size, 2, hidden_size])


def test_features_extractor_model():
    env = EternityEnv('instances/eternity_A.txt', manual_orient=False)
    hidden_size = 10
    batch_size = 64
    n_tiles = env.action_space.nvec[0]

    model = FeaturesExtractorModel(
        env.observation_space,
        hidden_size,
        cnn_layers = 1,
        n_heads = 2,
        ff_size = 2 * hidden_size,
        dropout = 0.1,
        n_layers = 3
    )

    obs = torch.randn(
        batch_size,
        env.observation_space.shape[0],
        env.observation_space.shape[1],
        env.observation_space.shape[2],
        env.observation_space.shape[3],
    )

    for x, y in model(obs):
        assert x.shape == torch.Size([batch_size, n_tiles, hidden_size])
        assert y.shape == torch.Size([batch_size, 2, hidden_size])


def test_pointer_actor_critic():
    env = EternityEnv('instances/eternity_A.txt', manual_orient=True)
    hidden_size = 10
    batch_size = 64
    n_tiles = env.action_space.nvec[0]

    model = PointerActorCritic(
        env.observation_space,
        env.action_space,
        lambda _: 1e-4,
        True,
        dict(
            hidden_size = hidden_size,
            cnn_layers = 1,
            n_heads = 2,
            ff_size = 2 * hidden_size,
            dropout = 0.1,
            n_layers = 3,
        ),
    )
    tiles = torch.randn(batch_size, n_tiles, hidden_size)
    decoder_tokens = torch.randn(batch_size, 2, hidden_size)

    values = model.value_net((tiles, decoder_tokens))
    assert values.shape == torch.Size([batch_size, 1])

    selected, rolls = model.action_net((tiles, decoder_tokens))
    assert selected.shape == torch.Size([batch_size, 2, n_tiles])
    assert rolls.shape == torch.Size([batch_size, 2, env.observation_space.shape[0]])
    assert torch.all(
        torch.abs(
            selected.sum(dim=2) - torch.ones(64, 2)
        ) < 1e-3
    )
    assert torch.all(
        torch.abs(
            rolls.sum(dim=2) - torch.ones(64, 2)
        ) < 1e-3
    )

    obs = torch.randn(
        batch_size,
        env.observation_space.shape[0],
        env.observation_space.shape[1],
        env.observation_space.shape[2],
        env.observation_space.shape[3],
    )
    actions, values, log_prob = model(obs)
    assert actions.shape == torch.Size([batch_size, 4])
    assert values.shape == torch.Size([batch_size, 1])
    assert log_prob.shape == torch.Size([batch_size])

    env = EternityEnv('instances/eternity_A.txt', manual_orient=False)

    model = PointerActorCritic(
        env.observation_space,
        env.action_space,
        lambda _: 1e-4,
        False,
        dict(
            hidden_size = hidden_size,
            cnn_layers = 1,
            n_heads = 2,
            ff_size = 2 * hidden_size,
            dropout = 0.1,
            n_layers = 3,
        ),
    )
    obs = torch.randn(
        batch_size,
        env.observation_space.shape[0],
        env.observation_space.shape[1],
        env.observation_space.shape[2],
        env.observation_space.shape[3],
    )
    actions, values, log_prob = model(obs)
    assert actions.shape == torch.Size([batch_size, 2])
    assert values.shape == torch.Size([batch_size, 1])
    assert log_prob.shape == torch.Size([batch_size])

