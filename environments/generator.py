"""
environments/generator.py
Dynamic Maze Map Generator with BFS Path Validation.
"""

import os
import json
import numpy as np
from collections import deque
from typing import Tuple, List, Dict, Optional


class MazeGenerator:
    """
    Maze generator based on Student ID and BFS validation.
    
    Map Element Encoding:
    0: Empty (Normal tile)
    1: Wall (Obstacle)
    2: Penalty (Penalty tile)
    3: Start (Start point)
    4: Key (Key point)
    5: Door (Locked Door)
    6: Goal (Goal point)
    """

    EMPTY = 0
    WALL = 1
    PENALTY = 2
    START = 3
    KEY = 4
    DOOR = 5
    GOAL = 6

    def __init__(self, student_id: str = "40123454"):
        self.student_id = student_id
        # Calculate Base Seed using second-to-last digit
        self.base_seed = int(student_id[-2])
        # Calculate Grid Size: 15 + (base_seed % 4) -> 15 + (5 % 4) = 16
        self.grid_size = 15 + (self.base_seed % 4)

    def generate_valid_maze(self, save_path: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]]]:
        """
        Generates a valid maze layout ensuring a feasible path exists via BFS.
        """
        attempt_seed = self.base_seed

        while True:
            np.random.seed(attempt_seed)
            grid = np.zeros((self.grid_size, self.grid_size), dtype=int)

            # Assign unique positions for key elements
            coords = self._get_unique_coordinates(count=4, seed=attempt_seed)
            start_pos, key_pos, door_pos, goal_pos = coords[0], coords[1], coords[2], coords[3]

            grid[start_pos] = self.START
            grid[key_pos] = self.KEY
            grid[door_pos] = self.DOOR
            grid[goal_pos] = self.GOAL

            # Minimum 15% walls/obstacles (At least 39 cells for 16x16 grid)
            num_walls = int(0.18 * (self.grid_size ** 2))
            
            # Minimum 5 penalty tiles
            num_penalties = 6

            # Place obstacles and penalty tiles
            empty_positions = [
                (r, c) for r in range(self.grid_size) for c in range(self.grid_size)
                if grid[r, c] == self.EMPTY
            ]

            np.random.shuffle(empty_positions)

            for _ in range(num_walls):
                if empty_positions:
                    pos = empty_positions.pop()
                    grid[pos] = self.WALL

            for _ in range(num_penalties):
                if empty_positions:
                    pos = empty_positions.pop()
                    grid[pos] = self.PENALTY

            # BFS Validation:
            # Step 1: Path from Start to Key (Door locked)
            path_to_key = self._bfs_search(grid, start_pos, key_pos, allow_door=False)
            # Step 2: Path from Key to Goal (Door unlocked/passable)
            path_to_goal = self._bfs_search(grid, key_pos, goal_pos, allow_door=True)

            if path_to_key is not None and path_to_goal is not None:
                positions = {
                    "start": start_pos,
                    "key": key_pos,
                    "door": door_pos,
                    "goal": goal_pos
                }

                if save_path:
                    self.save_map(grid, positions, save_path)

                return grid, positions

            attempt_seed += 1000  # Shift seed for next attempt if invalid

    def _get_unique_coordinates(self, count: int, seed: int) -> List[Tuple[int, int]]:
        """Generates unique random coordinates for key elements."""
        rng = np.random.default_rng(seed)
        indices = rng.choice(self.grid_size * self.grid_size, size=count, replace=False)
        return [(int(idx // self.grid_size), int(idx % self.grid_size)) for idx in indices]

    def _bfs_search(
        self,
        grid: np.ndarray,
        start: Tuple[int, int],
        target: Tuple[int, int],
        allow_door: bool
    ) -> Optional[List[Tuple[int, int]]]:
        """
        BFS algorithm to validate path existence between start and target.
        """
        queue = deque([(start, [start])])
        visited = {start}
        rows, cols = grid.shape

        while queue:
            (curr_r, curr_c), path = queue.popleft()

            if (curr_r, curr_c) == target:
                return path

            # 4 Movement Directions: UP, DOWN, LEFT, RIGHT
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for dr, dc in directions:
                nr, nc = curr_r + dr, curr_c + dc

                if 0 <= nr < rows and 0 <= nc < cols:
                    cell_type = grid[nr, nc]

                    if cell_type == self.WALL:
                        continue
                    if cell_type == self.DOOR and not allow_door:
                        continue

                    if (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append(((nr, nc), path + [(nr, nc)]))

        return None

    def save_map(self, grid: np.ndarray, positions: Dict[str, Tuple[int, int]], filepath: str) -> None:
        """Saves grid map and component positions to a JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "student_id": self.student_id,
            "grid_size": self.grid_size,
            "grid": grid.tolist(),
            "positions": {k: list(v) for k, v in positions.items()}
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def load_map(filepath: str) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]]]:
        """Loads grid map and positions from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        grid = np.array(data["grid"], dtype=int)
        positions = {k: tuple(v) for k, v in data["positions"].items()}
        return grid, positions


if __name__ == "__main__":
    generator = MazeGenerator(student_id="40123454")
    maze_grid, pos_dict = generator.generate_valid_maze(
        save_path="environments/maps/maze_40123454.json"
    )
    print(f"Map successfully generated. Shape: {maze_grid.shape}")
    print(f"Key positions: {pos_dict}")