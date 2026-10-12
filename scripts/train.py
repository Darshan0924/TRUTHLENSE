#!/usr/bin/env python3
"""Train the TruthLens baseline from a labeled local dataset."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from truthlens.ml import DatasetError, train_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path, help="Root containing train/, val/, test/ class folders")
    parser.add_argument("--output", type=Path, default=Path("models/truthlens.pt"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2, help="CPU threads")
    args = parser.parse_args()
    try:
        result = train_model(
            args.dataset, args.output,
            epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.learning_rate, seed=args.seed, threads=args.threads,
            progress=lambda row: print(json.dumps(row), flush=True),
        )
    except (DatasetError, ValueError, ImportError, OSError) as exc:
        parser.exit(1, f"Training failed: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
