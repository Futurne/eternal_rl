#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gym
import torch
import numpy as np
from torchinfo import summary

from src.environment import EternityEnv
from src.cnn import CNNFeaturesExtractor


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

