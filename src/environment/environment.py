import os
from itertools import product

import gym
import numpy as np
from gym import spaces
from numpy.random import default_rng

from src.environment.draw import display_solution


NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3

ORIGIN_NORTH = 0
ORIGIN_SOUTH = 1
ORIGIN_WEST = 2
ORIGIN_EAST = 3

WALL_ID = 0

ENV_DIR = 'instances'
ENV_ORDERED = [
    'eternity_trivial_A.txt',
    'eternity_trivial_B.txt',
    'eternity_A.txt',
    'eternity_B.txt',
    'eternity_C.txt',
    'eternity_D.txt',
    'eternity_E.txt',
    'eternity_complet.txt',
]


class EternityEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array', 'computer']}

    def __init__(
            self,
            instance_path: str,
            manual_orient: bool,
            reward_type: str = 'win_ratio',
            reward_penalty: float = 0.0,
            seed: int = 0,
    ):
        super().__init__()

        self.rng = default_rng(seed)
        self.instance_path = instance_path
        instance = read_instance_file(instance_path)
        self.instance = to_one_hot(instance)  # Shape is [4, n_class, size, size]

        self.size = self.instance.shape[-1]
        self.n_class = self.instance.shape[1]
        self.n_pieces = self.size * self.size
        self.max_steps = self.n_pieces * 2
        self.best_matchs = 2 * self.size * (self.size - 1)
        self.matchs = 0

        self.manual_orient = manual_orient
        if manual_orient:
            self.action_space = spaces.MultiDiscrete([
                self.n_pieces,  # Tile id to swap
                self.n_pieces,  # Tile id to swap
                4,  # How much rolls for the first tile
                4,  # How much rolls for the second tile
            ])
        else:
            self.action_space = spaces.MultiDiscrete([
                self.n_pieces,  # Tile id to swap
                self.n_pieces,  # Tile id to swap
            ])

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=self.render().shape,
            dtype=np.uint8
        )

        self.reward_type = reward_type
        self.reward_penalty = reward_penalty

    def step(self, action: np.ndarray) -> tuple[np.ndarray, int, bool, dict]:
        """Swap the two choosen tiles and orient them in the best possible way.

        Input
        -----
            action: Id of the tiles to swap and their rolling shift values.
                In the form of [tile_1_id, tile_2_id, roll_1, roll_2].

        Output
        ------
            observation: Array of observation, output of self.render.
            reward: Number of matching sides.
            done: Is there still sides that are not matched?
            info: -
        """
        coords = [
            (a // self.size, a % self.size)
            for a in action[:2]
        ]

        if self.manual_orient:
            rolls = [a for a in action[2:]]

        previous_matchs = self.matchs

        # Swap tiles
        self.swap_tiles(coords[0], coords[1])

        # Reorient the two tiles
        if self.manual_orient:
            self.roll_tile(coords[0], rolls[0])
            self.roll_tile(coords[1], rolls[1])
        else:
            self.best_orientation(coords[0])
            self.best_orientation(coords[1])

        self.matchs = self.count_matchs()
        self.tot_steps += 1

        observation = self.render()
        win = self.matchs == self.best_matchs
        timeout = self.tot_steps == self.max_steps
        done = win or timeout

        match self.reward_type:
            case 'win_ratio':
                reward = self.matchs / self.best_matchs if done else 0
            case 'win':
                reward = int(win)
            case 'delta':
                reward = (self.matchs - previous_matchs) / self.best_matchs
            case 'penalty':
                reward = 0
            case _:
                raise RuntimeError(f'Unknown reward type: {self.reward_type}.')

        reward -= self.reward_penalty  # Small penalty at each step

        info = {
            'matchs': self.matchs,
            'ratio': self.matchs / self.best_matchs,
            'win': win,
        }

        if win and self.instance_path == os.path.join(ENV_DIR, 'eternity_complet.txt'):
            self.render(mode='rgb_array', output_file='solution.png')

        return observation, reward, done, info

    def reset(self) -> np.ndarray:
        """Scramble the tiles and randomly orient them.
        """
        instance = self.instance
        instance = instance.reshape(4 * self.n_class, -1)  # Shape is [4 * n_class, size * size]
        instance = np.transpose(
            instance,
            axes=(1, 0)
        )  # Shape is [size * size, 4 * n_class]

        # Scramble the tiles
        instance = self.rng.permutation(instance)

        # Randomly orient the tiles
        for tile_id, tile in enumerate(instance):
            shift_value = self.rng.integers(low=0, high=4) * self.n_class
            instance[tile_id] = np.roll(tile, shift_value)

        instance = instance.reshape(self.size, self.size, 4, self.n_class)
        instance = np.transpose(
            instance,
            axes=(2, 3, 0, 1)
        )  # Shape is [4, n_class, size, size]
        self.instance = instance

        self.matchs = self.count_matchs()
        self.tot_steps = 0

        return self.render()

    def render(self, mode: str='computer', output_file: str=None) -> np.ndarray:
        """Transform the instance into an observation.

        The observation is a one-hot map of shape [4, n_class, size, size].
        """
        if mode == 'computer':
            return self.instance
        if mode == 'rgb_array':
            solution = self.instance.argmax(axis=1)
            return display_solution(solution, self.matchs, output_file)

    def count_tile_matchs(self, coords: tuple[int, int]) -> int:
        """Count the matchs a tile has with its neighbours.
        """
        matchs = 0
        tile = self.instance[:, :, coords[0], coords[1]]

        tile_sides = [NORTH, EAST, SOUTH, WEST]
        other_sides = [SOUTH, WEST, NORTH, EAST]
        other_coords = [
            (coords[0] + 1, coords[1]),  # (y, x)
            (coords[0], coords[1] + 1),
            (coords[0] - 1, coords[1]),
            (coords[0], coords[1] - 1)
        ]

        for side_t, side_o, coords_o in zip(tile_sides, other_sides, other_coords):
            if coords_o[0] < 0 or coords_o[0] >= self.size or\
                    coords_o[1] < 0 or coords_o[1] >= self.size:
                continue  # Those coords are outside the square

            tile_class = tile[side_t]
            other_class = self.instance[side_o, :, coords_o[0], coords_o[1]]

            if tile_class[WALL_ID] != 1:  # Ignore the walls
                matchs += int(np.all(tile_class == other_class))  # Add one if the one-hots are the same

        return matchs

    def count_pairs_matchs(self, coords_1: tuple[int, int], coords_2: tuple[int, int]) -> int:
        matchs = self.count_tile_matchs(coords_1) + self.count_tile_matchs(coords_2)

        # Seek for duplicates matchs
        tile_1_sides = [NORTH, EAST, SOUTH, WEST]
        tile_2_sides = [SOUTH, WEST, NORTH, EAST]
        tile_1_neighbours = [
            (coords_1[0] + 1, coords_1[1]),  # (y, x)
            (coords_1[0], coords_1[1] + 1),
            (coords_1[0] - 1, coords_1[1]),
            (coords_1[0], coords_1[1] - 1)
        ]

        for side_1, side_2, coords in zip(tile_1_sides, tile_2_sides, tile_1_neighbours):
            if coords == coords_2:  # Potential duplicate !
                color_1 = self.instance[side_1, :, coords_1[0], coords_1[1]].argmax()
                color_2 = self.instance[side_2, :, coords_2[0], coords_2[1]].argmax()
                if color_1 == color_2 and color_1 != WALL_ID:
                    matchs -= 1  # Remove duplicate

                break  # There can only be one duplicate

        return matchs

    def count_matchs(self) -> int:
        """Count all matchs for the current state.
        """
        matchs = sum(
            self.count_tile_matchs((y, x))
            for x, y in product(range(self.size), range(self.size))
        )
        return matchs // 2  # Sides have all been checked twice

    def swap_tiles(self, tile_1_coords: tuple[int, int], tile_2_coords: tuple[int, int]):
        """Swap the two given tiles.
        """
        tile_1 = self.instance[:, :, tile_1_coords[0], tile_1_coords[1]].copy()
        tile_2 = self.instance[:, :, tile_2_coords[0], tile_2_coords[1]]

        self.instance[:, :, tile_1_coords[0], tile_1_coords[1]] = tile_2
        self.instance[:, :, tile_2_coords[0], tile_2_coords[1]] = tile_1

    def best_orientation(self, coords: tuple[int, int]) -> int:
        """Reorient the given tile to maximise the matchs.

        Return the number of matchs of this tile.
        """
        tile = self.instance[:, :, coords[0], coords[1]]
        max_matchs = 0
        best_tile = tile.copy()

        for shift_value in range(4):
            shift_value *= self.n_class
            shifted_tile = np.roll(tile.flatten(), shift_value)

            shifted_tile = shifted_tile.reshape(4, self.n_class)
            self.instance[:, :, coords[0], coords[1]] = shifted_tile

            matchs = self.count_tile_matchs(coords)
            if matchs > max_matchs:
                max_matchs = matchs
                best_tile = shifted_tile.copy()

        self.instance[:, :, coords[0], coords[1]] = best_tile
        return max_matchs

    def roll_tile(self, coords: tuple[int, int], shift_value: int):
        """Reorient the tile by doing a circular shift.
        """
        shift_value *= self.n_class
        tile = self.instance[:, :, coords[0], coords[1]]
        tile = np.roll(tile.flatten(), shift_value)
        tile = tile.reshape(4, self.n_class)
        self.instance[:, :, coords[0], coords[1]] = tile

    def seed(self, seed: int):
        """Modify the seed.
        """
        self.rng = default_rng(seed)


def read_instance_file(instance_path: str) -> np.ndarray:
    """Read the instance file and return a matrix containing the ordered elements.

    Output
    ------
        data: Matrix containing each tile.
            Shape of [4, y-axis, x-axis].
            Origin is in the bottom left corner.
    """
    with open(instance_path, 'r') as instance_file:
        instance_file = iter(instance_file)

        n = int(next(instance_file))

        data = np.zeros((4, n * n), dtype=np.uint8)
        for element_id, element in enumerate(instance_file):
            class_ids = element.split(' ')
            class_ids = [int(c) for c in class_ids]
            data[:, element_id] = class_ids

    for tile_id in range(data.shape[-1]):
        tile = data[:, tile_id]
        tile[NORTH], tile[EAST], tile[SOUTH], tile[WEST] =\
                tile[ORIGIN_NORTH], tile[ORIGIN_EAST], tile[ORIGIN_SOUTH], tile[ORIGIN_WEST]

    data = data.reshape((4, n, n))
    return data


def to_one_hot(data: np.ndarray) -> np.ndarray:
    """Change each tile to a one-hot encoding version of its elements.

    Input
    -----
        data: Matrix containing each class ids of its tiles.
            Shape of [4, size, size].

    Output
    ------
        data: Matrix containing one-hot encoding of each class ids of its tiles.
            Shape of [4, n_class, size, size].
    """
    n_class = np.max(data) + 1
    size = data.shape[-1]

    one_hot_data = np.zeros((4, n_class, size, size), dtype=np.uint8)
    for x, y, side_id in product(range(size), range(size), range(4)):
        class_id = data[side_id, y, x]
        one_hot_data[side_id, class_id, y, x] = 1

    return one_hot_data


def next_instance(instance_path: str) -> str:
    instance_id = [
        i + 1 for i, path in enumerate(ENV_ORDERED)
        if os.path.join(ENV_DIR, path) == instance_path
    ][0]
    instance_id = min(instance_id, len(ENV_ORDERED) - 1)
    return os.path.join(ENV_DIR, ENV_ORDERED[instance_id])

