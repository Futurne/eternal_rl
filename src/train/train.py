#!/usr/bin/env python
# -*- coding: utf-8 -*-

import wandb
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv 
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from src.train.callback import EternalCallback
from src.environment.environment import EternityEnv, next_instance
from src.model.actorcritic import PointerActorCritic


class TrainEternal:
    def __init__(self, config: dict[str, any]):
        self.__dict__ |= config
        set_random_seed(self.seed)
        self.tensor_log_dir = None

    def make_env(self) -> SubprocVecEnv:
        """Build the vectorized environments.

        See: https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#multiprocessing-unleashing-the-power-of-vectorized-environments.
        """
        _init = lambda seed: Monitor(
            EternityEnv(
                self.instance_path,
                self.manual_orient,
                reward_type = self.reward_type,
                reward_penalty = self.reward_penalty,
                seed = seed,
            ),
            info_keywords = ('matchs', 'ratio'),
        )
        env = SubprocVecEnv([
            lambda: _init(cpu_id + self.seed) for cpu_id in range(self.num_cpu)
        ])
        return env

    def train_on_env(self, pretrain_rollouts: int = 0) -> tuple[int, EternalCallback]:
        run_id = 0
        with wandb.init(
            project = 'Eternal RL',
            entity = 'pierrotlc',
            config = self.__dict__,
            group = self.group,
            # sync_tensorboard = True,  # Auto-upload the tensorboard metrics
        ) as run:
            # Get wandb config => sweeps can change this config
            config = {
                k: v
                for k, v in wandb.config.items()
                if '.' not in k
            }
            for k, v in wandb.config.items():
                if '.' not in k:
                    continue

                k1, k2 = k.split('.')
                if k1 in config:
                    config[k1][k2] = v
                else:
                    config[k1] = {k2: v}

            self.__dict__ |= config

            run_id = run.id

            if not self.tensor_log_dir:
                self.tensor_log_dir = f'runs/{run_id}'
                wandb.tensorboard.patch(root_logdir=self.tensor_log_dir)

            # Create env
            env = self.make_env()

            # Create agent
            model = PPO(
                PointerActorCritic,
                env,
                n_steps = self.n_steps,
                learning_rate = self.lr,
                batch_size = self.batch_size,
                gamma = self.gamma,
                gae_lambda = self.gae_lambda,
                clip_range = self.clip_range,
                normalize_advantage = self.normalize_advantage,
                ent_coef = self.ent_coef,
                vf_coef = self.vf_coef,
                verbose = 1,
                tensorboard_log = self.tensor_log_dir,
                policy_kwargs = {
                    'net_arch': self.net_arch,
                    'manual_orient': self.manual_orient,
                },
                seed = self.seed,
            )
            if self.load_pretrain:
                model.policy.load_from_state_dict(self.load_pretrain)

            # Callback
            callback = EternalCallback(
                # gif_path = f'gifs/{run_id}',
                # gif_length = 25,
                model_path = f'models/{run_id}',
                pretrain_rollouts = pretrain_rollouts,
            )

            # Train the agent
            model.learn(
                self.total_timesteps,
                callback = callback,
            )

            env.close()

        return run_id, callback

    def train(self):
        run_id, callback = self.train_on_env()
        while self.curriculum_learning and callback.has_won:
            self.instance_path = next_instance(self.instance_path)
            print(f'Upgrading instance to: {self.instance_path}.')
            self.load_pretrain = f'models/{run_id}'
            pretrain_rollouts = callback.pretrain_rollouts + callback.n_rollouts
            run_id, callback = self.train_on_env(pretrain_rollouts)

