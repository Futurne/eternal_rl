#!/usr/bin/env python
# -*- coding: utf-8 -*-

import yaml
import typer
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from src.train.train import TrainEternal


def load_config_file(config_path: str) -> dict[str, any]:
    config = {
        'gae_lambda': 0.99,
        'clip_range': 0.2,
        'normalize_advantage': True,
        'ent_coef': 0,
        'vf_coef': 0.5,
    }  # Default config
    with open(config_path, 'r') as config_file:
        config |= yaml.safe_load(config_file)

    # Load default configuration is any
    if 'default_conf' in config:
        with open(config['default_conf'], 'r') as config_file:
            config |= {
                key: value for key, value in yaml.safe_load(config_file).items()
                if key not in config
            }

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

    main(sys.argv[1])
    sys.exit(0)
    if len(sys.argv) > 2:  # For sweeps
        main(sys.argv[1])
    else:
        typer.run(main)

