#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import torch
import wandb
import imageio
import numpy as np
from wandb.integration.sb3 import WandbCallback


ROLLOUTS_MEAN = 10
REWARDS_THRESHOLD = 0.90


class EternalCallback(WandbCallback):
    def __init__(
            self,
            model_path: str = None,
            gif_path: str = None,
            gif_length: int = None,
            pretrain_rollouts: int = 0,
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

        self.pretrain_rollouts = pretrain_rollouts
        self.n_rollouts = 0
        self.has_won = False

    def _on_step(self) -> bool:
        """Compute the mean episode length and stop the training
        if the mean episode length is lower than a specific threshold.
        """
        rewards = self.model.env.env_method('get_episode_rewards')
        rewards = [
            np.mean(episodes[-ROLLOUTS_MEAN:]) for episodes in rewards
            if len(episodes) > ROLLOUTS_MEAN
        ]
        rewards = np.mean(rewards) if rewards else float('+inf')
        if rewards >= REWARDS_THRESHOLD and self.n_rollouts > ROLLOUTS_MEAN:
            self.has_won = True
            return False  # Stop the training!

        return super()._on_step()

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

        return filepath

