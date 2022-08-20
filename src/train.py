#!/usr/bin/env python
# -*- coding: utf-8 -*-

import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecVideoRecorder
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from src.environment import EternityEnv
from src.model.actorcritic import PointerActorCritic


class TrainEternal:
    def __init__(self, config: dict[str, any]):
        self.__dict__ |= config
        set_random_seed(self.seed)

    def make_env(self) -> SubprocVecEnv:
        """Build the vectorized environments.

        See: https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#multiprocessing-unleashing-the-power-of-vectorized-environments.
        """
        _init = lambda seed: Monitor(EternityEnv(self.instance_path, self.max_steps, seed))
        env = SubprocVecEnv([lambda: _init(cpu_id + self.seed) for cpu_id in range(self.num_cpu)])
        return env

    def train(self):
        with wandb.init(
            project = 'Eternal RL',
            entity = 'pierrotlc',
            config = self.__dict__,
            group = self.group,
            sync_tensorboard = True,  # Auto-upload the tensorboard metrics
            # monitor_gym = True,  # Auto-upload the videos of the agent playing
        ) as run:
            # Create env
            env = self.make_env()
            """
            env = VecVideoRecorder(
                env,
                f'videos/{run.id}',
                record_video_trigger = lambda x: x % 2000,
            )
            """

            # Create agent
            model = PPO(
                PointerActorCritic,
                env,
                verbose = 1,
                tensorboard_log = f'runs/{run.id}',
                policy_kwargs = {'net_arch': self.net_arch},
            )

            # Train the agent
            model.learn(
                self.total_timesteps,
                callback = WandbCallback(
                    verbose = 2,
                    # model_save_path = f'models/{run.id}',
                    # model_save_freq = 100,
                ),
            )

