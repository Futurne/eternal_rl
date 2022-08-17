#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
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
    env = EternityEnv('instances/eternity_A.txt', 0)
    assert env.count_matchs() == 12
    assert env.count_tile_matchs((0, 0)) == 1
    assert env.count_tile_matchs((1, 2)) == 3


def test_swap_tiles():
    env = EternityEnv('instances/eternity_A.txt', 0)

    coords_1, coords_2 = (0, 0), (3, 2)
    tile_1 = env.instance[:, :, coords_1[0], coords_1[1]].copy()
    tile_2 = env.instance[:, :, coords_2[0], coords_2[1]].copy()
    env.swap_tiles(coords_1, coords_2)

    assert np.all(tile_1 == env.instance[:, :, coords_2[0], coords_2[1]])
    assert np.all(tile_2 == env.instance[:, :, coords_1[0], coords_1[1]])


@pytest.mark.parametrize(
    'coords, roll_value',
    [
        (
            (0, 0),
            1
        ),
        (
            (3, 2),
            -1
        ),
        (
            (1, 1),
            10
        )
    ]
)
def test_roll_tiles(coords: tuple[int, int], roll_value: int):
    env = EternityEnv('instances/eternity_A.txt', 0)
    tile = env.instance[:, :, coords[0], coords[1]].argmax(axis=1)

    env.roll_tile(coords, roll_value)
    assert np.all(
        env.instance[:, :, coords[0], coords[1]].argmax(axis=1) == np.roll(tile, roll_value)
    )
