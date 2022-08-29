#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import torch
import wandb
import imageio
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecVideoRecorder
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from src.environment import EternityEnv
from src.model.actorcritic import PointerActorCritic


class EternalCallback(WandbCallback):
    def __init__(
            self,
            model_path: str = None,
            gif_path: str = None,
            gif_length: int = None,
            verbose: int = 0,
            *args,
            **kwargs
    ):
        super(EternalCallback, self).__init__(verbose, *args, **kwargs)

        self.model_path = model_path
        self.gif_path = gif_path
        self.gif_length = gif_length

        if gif_path:
            assert gif_length  # gif_length must be setted when using gifs!

        self.n_rollouts = 0

    def _on_rollout_end(self):
        super()._on_rollout_end()

        if self.model_path:
            self.model.policy.save(self.model_path)

        if self.gif_path:
            filepath = self.create_gif()
            wandb.log(
                {'gif': wandb.Video(filepath, fps=3, format='gif')}
            )

        self.n_rollouts += 1

    def create_gif(self) -> str:
        images = []
        model = self.model
        env = model.env
        obs = env.reset()
        img = env.render(mode='rgb_array')

        with torch.no_grad():
            for _ in range(self.gif_length):
                images.append(img)
                action, _ = model.predict(obs)
                obs, _, _, _ = env.step(action)
                img = model.env.render(mode='rgb_array')

        filepath = os.path.join(self.gif_path, f'{self.n_rollouts}.gif')
        if not os.path.isdir(self.gif_path):
            os.makedirs(self.gif_path)
        imageio.mimsave(filepath, images, fps=3)

        del model
        del env
        return filepath


class TrainEternal:
    def __init__(self, config: dict[str, any]):
        self.__dict__ |= config
        set_random_seed(self.seed)

    def make_env(self) -> SubprocVecEnv:
        """Build the vectorized environments.

        See: https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#multiprocessing-unleashing-the-power-of-vectorized-environments.
        """
        _init = lambda seed: Monitor(
            EternityEnv(
                self.instance_path,
                self.max_steps,
                self.manual_orient,
                reward_type = self.reward_type,
                reward_penalty = self.reward_penalty,
                seed = seed,
            ),
            info_keywords = ('matchs', 'ratio'),
        )
        env = SubprocVecEnv([lambda: _init(cpu_id + self.seed) for cpu_id in range(self.num_cpu)])
        return env

    def train(self):
        with wandb.init(
            project = 'Eternal RL',
            entity = 'pierrotlc',
            config = self.__dict__,
            group = self.group,
            sync_tensorboard = True,  # Auto-upload the tensorboard metrics
        ) as run:
            # Get wandb config => sweeps can change this config
            self.__dict__ |= wandb.config

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
                verbose = 1,
                tensorboard_log = f'runs/{run.id}',
                policy_kwargs = {
                    'net_arch': self.net_arch,
                    'manual_orient': self.manual_orient,
                },
                seed = self.seed,
            )
            if self.load_pretrain:
                model.policy.load_from_state_dict(self.load_pretrain)

            # Train the agent
            model.learn(
                self.total_timesteps,
                callback = EternalCallback(
                    # gif_path = f'gifs/{run.id}',
                    # gif_length = 25,
                    model_path = f'models/{run.id}',
                ),
            )

            env.close()

