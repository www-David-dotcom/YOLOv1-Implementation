from __future__ import annotations

import argparse

from yolo.utils.visualization import plot_training_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", default="outputs/logs/train_history.csv")
    parser.add_argument("--output", default="outputs/figures/training_curve.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_training_history(args.history, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()