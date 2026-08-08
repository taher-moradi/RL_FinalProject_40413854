"""
tests/test_env.py
Unit tests for Environment functionality and Generator path validity.
"""

import pytest
import numpy as np
from environments.generator import MazeGenerator
from environments.maze import DynamicMazeEnv


def test_maze_generator():
    """Tests grid size, obstacle percentage, and component placement for Student ID."""
    generator = MazeGenerator(student_id="40413854")
    grid, positions = generator.generate_valid_maze()

    # Grid dimensions must be 16x16
    assert grid.shape == (16, 16)

    # Key position dictionary integrity
    assert "start" in positions
    assert "key" in positions
    assert "door" in positions
    assert "goal" in positions

    # At least 5 penalty tiles
    penalties = np.sum(grid == MazeGenerator.PENALTY)
    assert penalties >= 5

    # At least 15% walls/obstacles
    walls = np.sum(grid == MazeGenerator.WALL)
    assert (walls / 256.0) >= 0.15


def test_environment_initialization():
    """Tests reset method and initial state values."""
    generator = MazeGenerator(student_id="40413854")
    grid, positions = generator.generate_valid_maze()
    env = DynamicMazeEnv(grid, positions, max_energy=50)

    state = env.reset()
    r, c, key, energy = state

    assert (r, c) == positions["start"]
    assert key == 0
    assert energy == 50


def test_door_blocking_without_key():
    """Tests locked door behavior when key is not collected."""
    generator = MazeGenerator(student_id="40413854")
    grid, positions = generator.generate_valid_maze()
    env = DynamicMazeEnv(grid, positions, max_energy=50)

    # Position agent right next to locked door
    door_r, door_c = positions["door"]
    env.agent_pos = (door_r, max(0, door_c - 1))
    env.has_key = 0

    # Execute move towards door tile
    next_state, reward, done, info = env.step(DynamicMazeEnv.RIGHT)

    # Agent should remain in position if key is missing
    if (door_r, door_c - 1) != positions["door"]:
        assert env.agent_pos != positions["door"] or env.has_key == 1


def test_energy_depletion():
    """Tests episode termination upon energy depletion."""
    generator = MazeGenerator(student_id="40413854")
    grid, positions = generator.generate_valid_maze()
    env = DynamicMazeEnv(grid, positions, max_energy=3)

    env.reset()
    done = False

    for _ in range(3):
        _, _, done, _ = env.step(DynamicMazeEnv.UP)

    assert done is True
    assert env.energy == 0


if __name__ == "__main__":
    pytest.main(["-v", "tests/test_env.py"])