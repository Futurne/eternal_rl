#!/usr/bin/env python
# -*- coding: utf-8 -*-

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from src.environment import EternityEnv
from src.model.actorcritic import PointerActorCritic


def load_config_file(config_path: str) -> dict[str, any]:
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    return config


def train_agent(config: dict[str, any]):
    env = EternityEnv(config['instance_path'], config['max_steps'], config['seed'])
    # env = make_vec_env(env, n_envs=config['num_cpu'], seed=config['seed'],)
    model = PPO(PointerActorCritic, env, verbose=1)
    model.learn(int(float(config['total_timesteps'])))


if __name__ == '__main__':
    config = load_config_file('config.yaml')
    train_agent(config)
