#!/usr/bin/env python
# -*- coding: utf-8 -*-

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from src.train import TrainEternal


def load_config_file(config_path: str) -> dict[str, any]:
    config = {
        'seed': 0,
        'num_cpu': 4,
        'net_arch': {},
        'manual_orient': False,
    }  # Default config
    with open(config_path, 'r') as config_file:
        config |= yaml.safe_load(config_file)

    # Preprocess values
    config['total_timesteps'] = int(float(config['total_timesteps']))

    return config


if __name__ == '__main__':
    config = load_config_file('config.yaml')
    train = TrainEternal(config)
    train.train()

