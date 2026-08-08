"""
main.py
Master Project Launcher. Provides CLI arguments to run GUI app or execute full experiment benchmarks.
Updated: Adds --random_map flag to generate new maps dynamically.
"""

import argparse
from gui.app import MazeGUIApp
from experiments.run_experiments import main as run_benchmarks


def main():
    parser = argparse.ArgumentParser(description="Dynamic Maze Reinforcement Learning Project Launcher")
    parser.add_argument("--mode", type=str, default="gui", choices=["gui", "experiment"], help="Launch GUI mode or run CLI experiment benchmarks.")
    parser.add_argument("--student_id", type=str, default="40413854", help="Student ID for seed calculation.")
    parser.add_argument("--random_map", action="store_true", default=True, help="Generate a fresh new random map layout on launch.")

    args = parser.parse_args()

    if args.mode == "gui":
        print(f"Starting Pygame GUI Mode (Random Map={args.random_map}) for Student ID: {args.student_id}...")
        app = MazeGUIApp(student_id=args.student_id, random_map=args.random_map)
        app.run()
    elif args.mode == "experiment":
        print("Starting Benchmark Experiments...")
        run_benchmarks()


if __name__ == "__main__":
    main()