#!/usr/bin/env python3
"""Build a deterministic, dimension-balanced subset of rewritten VBench prompts."""

import argparse
import json
import random
from collections import Counter


FULL_16_DIMS = [
    "subject_consistency",
    "background_consistency",
    "aesthetic_quality",
    "imaging_quality",
    "object_class",
    "multiple_objects",
    "color",
    "spatial_relationship",
    "scene",
    "temporal_style",
    "overall_consistency",
    "human_action",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
    "appearance_style",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per_dimension", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260710)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.input, "r", encoding="utf-8") as f:
        rows = json.load(f)

    prompt_frequency = Counter(row["prompt_en"] for row in rows)
    rng = random.Random(args.seed)
    used = set()
    selected = []

    for dim in FULL_16_DIMS:
        candidates = [
            row for row in rows
            if dim in row.get("dimension", [])
            and row.get("refined_prompt")
            and prompt_frequency[row["prompt_en"]] == 1
        ]
        rng.shuffle(candidates)
        choices = [row for row in candidates if row["prompt_en"] not in used][:args.per_dimension]
        if len(choices) != args.per_dimension:
            raise RuntimeError(
                f"dimension {dim!r} has only {len(choices)} usable prompts; "
                f"requested {args.per_dimension}"
            )
        for row in choices:
            item = dict(row)
            item["dimension"] = [dim]
            selected.append(item)
            used.add(item["prompt_en"])

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2, ensure_ascii=False)
    print(f"wrote {len(selected)} prompts to {args.output}")


if __name__ == "__main__":
    main()
