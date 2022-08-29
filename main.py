#!/usr/bin/env python
# -*- coding: utf-8 -*-

import yaml
import typer
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from src.train import TrainEternal


def load_config_file(config_path: str) -> dict[str, any]:
    config = {
        'seed': 0,
        'num_cpu': 4,
        'net_arch': {},
        'manual_orient': False,
        'gae_lambda': 0.99,
        'clip_range': 0.2,
        'normalize_advangate': True,
        'ent_coef': 0,
        'vf_coef': 0.5,
    }  # Default config
    with open(config_path, 'r') as config_file:
        config |= yaml.safe_load(config_file)

    # Preprocess values
    config['total_timesteps'] = int(float(config['total_timesteps']))
    config['lr'] = float(config['lr'])

    return config


def main(config_path: str):
    config = load_config_file(config_path)
    train = TrainEternal(config)
    train.train()


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 2:  # For sweeps
        main(sys.argv[1])
    else:
        typer.run(main)

