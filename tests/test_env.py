#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from src.environment import EternityEnv, read_instance_file, to_one_hot


def test_read_instance():
    instance = read_instance_file('instances/eternity_A.txt')
    real = np.array(
        [
            [
                [0,  0,  0,  0],
                [2,  6,  4,  4],
                [12, 8,  8,  5],
                [6, 11,  4,  1],
            ],
            [
                [1,  8,  3,  0],
                [9,  4, 10,  0],
                [11, 7,  8,  0],
                [0,  3,  7,  0],
            ],
            [
                [2,  6,  4,  4],
                [12, 8,  8,  5],
                [3, 11,  4,  1],
                [0,  0,  0,  0],
            ],
            [
                [0,  1,  8,  3],
                [0,  9,  4, 10],
                [0, 11,  7,  8],
                [3,  6,  3,  7],
            ]
        ]
    )
    assert np.all(instance[:, 0, 0] == np.array([0, 1, 2, 0]))
    assert np.all(instance[:, 3, 3] == np.array([1, 0, 0, 7]))
    assert np.all(instance[:, 3, 2] == np.array([4, 7, 0, 3]))
    assert np.all(instance == real)


def test_one_hot():
    instance = read_instance_file('instances/eternity_trivial_A.txt')
    instance = to_one_hot(instance)
    real = np.array(
        [
            [
                [[0, 0], [0, 0]],
                [[1, 0], [0, 0]],
                [[0, 0], [0, 1]],
                [[0, 1], [0, 0]],
                [[0, 0], [1, 0]],
            ],
            [
                [[0, 0], [0, 0]],
                [[0, 1], [0, 0]],
                [[1, 0], [0, 0]],
                [[0, 0], [1, 0]],
                [[0, 0], [0, 1]],
            ],
            [
                [[1, 1], [1, 1]],
                [[0, 0], [0, 0]],
                [[0, 0], [0, 0]],
                [[0, 0], [0, 0]],
                [[0, 0], [0, 0]],
            ],
            [
                [[1, 1], [1, 1]],
                [[0, 0], [0, 0]],
                [[0, 0], [0, 0]],
                [[0, 0], [0, 0]],
                [[0, 0], [0, 0]],
            ]
        ]
    )
    assert np.all(instance == real)

    instance = read_instance_file('instances/eternity_A.txt')
    instance = to_one_hot(instance)
    assert np.all(
        np.argmax(instance[:, :, 0, 0], axis=1) == np.array([0, 1, 2, 0])
    )
    assert np.all(
        np.argmax(instance[:, :, 3, 3], axis=1) == np.array([1, 0, 0, 7])
    )
    assert np.all(
        np.argmax(instance[:, :, 3, 2], axis=1) == np.array([4, 7, 0, 3])
    )


def test_matchs_tiles():
    env = EternityEnv('instances/eternity_A.txt', 0, 'mlp')
    env.render('human')
    assert env.count_matchs() == 12
    assert env.count_tile_matchs((0, 0)) == 1
    assert env.count_tile_matchs((1, 2)) == 3

