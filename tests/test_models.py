#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gym
import torch
import numpy as np
from torchinfo import summary

from src.environment import EternityEnv
from src.cnn import CNNFeaturesExtractor
from src.pointer import PointerModel


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
    env = EternityEnv('instances/eternity_A.txt', 0)

    obs = torch.FloatTensor(env.render())
    obs = torch.unsqueeze(obs, dim=0)
    assert model(obs).shape == torch.Size([1, 10, 4, 4])


def test_pointer_model():
    model = PointerModel(
        hidden_size = 10,
        n_heads = 2,
        ff_size = 20,
        dropout = 0.1,
        n_layers = 3,
    )

    tiles = torch.randn(64, 10, 4, 4)
    selected, rotations = model(tiles)
    assert selected.shape == torch.Size([64, 2, 16])
    assert rotations.shape == torch.Size([64, 2, 4])
    assert torch.all(
        torch.abs(
            selected.sum(dim=2) - torch.ones(64, 2)
        ) < 1e-3
    )
    assert torch.all(
        torch.abs(
            rotations.sum(dim=2) - torch.ones(64, 2)
        ) < 1e-3
    )

