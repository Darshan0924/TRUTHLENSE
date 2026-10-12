#!/usr/bin/env python3
"""Evaluate a trained TruthLens checkpoint on its held-out test split."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from truthlens.ml import DatasetError, ModelLoadError, evaluate_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path, help="Optionally save test metrics as JSON")
    args = parser.parse_args()
    try:
        result = evaluate_model(args.checkpoint, args.dataset, batch_size=args.batch_size, threads=args.threads)
        rendered = json.dumps(result, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
    except (DatasetError, ModelLoadError, ValueError, ImportError, OSError) as exc:
        parser.exit(1, f"Evaluation failed: {exc}\n")
    print(rendered)


if __name__ == "__main__":
    main()
