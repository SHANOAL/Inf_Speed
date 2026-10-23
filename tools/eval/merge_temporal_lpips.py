#!/usr/bin/env python3
"""Merge frame-wise LPIPS statistics from independent video shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pooled_method(paths: list[Path], method: str) -> tuple[int, dict]:
    shards = [load(path) for path in paths]
    counts = np.asarray([item["num_videos"] for item in shards], dtype=np.float64)
    means = np.stack([item["methods"][method]["mean"] for item in shards])
    stds = np.stack([item["methods"][method]["std"] for item in shards])
    total = int(counts.sum())
    mean = np.sum(counts[:, None] * means, axis=0) / total
    m2 = np.sum((counts[:, None] - 1.0) * stds**2 + counts[:, None] * (means - mean) ** 2, axis=0)
    std = np.sqrt(m2 / (total - 1))
    ci95 = 1.96 * std / np.sqrt(total)
    overall = sum(
        item["num_videos"] * item["methods"][method]["overall_mean"] for item in shards
    ) / total
    return total, {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "ci95": ci95.tolist(),
        "overall_mean": float(overall),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stepvar", required=True, type=Path)
    parser.add_argument("--faststar", nargs="+", required=True, type=Path)
    parser.add_argument("--attnstar", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    stepvar = load(args.stepvar)
    step_count = int(stepvar["num_videos"])
    fast_count, fast = pooled_method(args.faststar, "FastSTAR")
    attn_count, attn = pooled_method(args.attnstar, "AttnSTAR")
    if len({step_count, fast_count, attn_count}) != 1:
        raise ValueError(f"video-count mismatch: {step_count}, {fast_count}, {attn_count}")
    output = {
        "metric": "LPIPS-AlexNet",
        "num_videos": step_count,
        "frame_count": stepvar["frame_count"],
        "methods": {
            "StepVAR": stepvar["methods"]["StepVAR"],
            "FastSTAR": fast,
            "AttnSTAR": attn,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    for method, values in output["methods"].items():
        print(f"{method}: overall LPIPS={values['overall_mean']:.9f}")


if __name__ == "__main__":
    main()
