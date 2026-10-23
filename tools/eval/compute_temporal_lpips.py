#!/usr/bin/env python3
"""Aggregate frame-wise LPIPS to a dense reference over matched videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_method(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("method must be NAME=VIDEO_DIR")
    name, directory = value.split("=", 1)
    return name, Path(directory).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--method", action="append", required=True, type=parse_method)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=0)
    return parser.parse_args()


def decode_pair(reference_path: Path, candidate_path: Path):
    import cv2

    reference = cv2.VideoCapture(str(reference_path))
    candidate = cv2.VideoCapture(str(candidate_path))
    if not reference.isOpened() or not candidate.isOpened():
        raise RuntimeError(f"failed to open {reference_path.name}")
    while True:
        ok_ref, frame_ref = reference.read()
        ok_candidate, frame_candidate = candidate.read()
        if not ok_ref or not ok_candidate:
            if ok_ref != ok_candidate:
                raise RuntimeError(f"frame-count mismatch for {reference_path.name}")
            break
        if frame_ref.shape != frame_candidate.shape:
            raise RuntimeError(f"shape mismatch for {reference_path.name}")
        yield frame_ref, frame_candidate
    reference.release()
    candidate.release()


def score_video(reference_path, candidate_path, model, device, batch_size):
    import numpy as np
    import torch

    values = []
    ref_batch = []
    candidate_batch = []

    def flush():
        if not ref_batch:
            return
        ref_np = np.stack(ref_batch)[..., ::-1].copy()
        candidate_np = np.stack(candidate_batch)[..., ::-1].copy()
        ref = torch.from_numpy(ref_np).permute(0, 3, 1, 2).float().to(device)
        candidate = torch.from_numpy(candidate_np).permute(0, 3, 1, 2).float().to(device)
        ref = ref.div(127.5).sub(1.0)
        candidate = candidate.div(127.5).sub(1.0)
        with torch.no_grad():
            batch_values = model(ref, candidate).reshape(-1).detach().cpu().numpy()
        values.extend(float(v) for v in batch_values)
        ref_batch.clear()
        candidate_batch.clear()

    for frame_ref, frame_candidate in decode_pair(reference_path, candidate_path):
        ref_batch.append(frame_ref)
        candidate_batch.append(frame_candidate)
        if len(ref_batch) >= batch_size:
            flush()
    flush()
    return values


def main() -> None:
    import lpips
    import numpy as np
    import torch

    args = parse_args()
    reference_dir = args.reference_dir.resolve()
    reference_names = sorted(path.name for path in reference_dir.glob("*.mp4"))
    end_index = args.end_index if args.end_index > 0 else len(reference_names)
    reference_names = reference_names[args.start_index:end_index]
    if args.limit > 0:
        reference_names = reference_names[: args.limit]
    if not reference_names:
        raise SystemExit("no reference videos found")

    device = torch.device(args.device)
    model = lpips.LPIPS(net="alex", verbose=False).eval().to(device)
    result = {
        "metric": "LPIPS-AlexNet",
        "reference_dir": str(reference_dir),
        "num_videos": len(reference_names),
        "start_index": args.start_index,
        "end_index": end_index,
        "frame_count": None,
        "methods": {},
    }

    for method_name, method_dir in args.method:
        candidate_names = {path.name for path in method_dir.glob("*.mp4")}
        missing = [name for name in reference_names if name not in candidate_names]
        if missing:
            raise SystemExit(f"{method_name}: missing {len(missing)} videos")
        rows = []
        for index, filename in enumerate(reference_names, 1):
            values = score_video(
                reference_dir / filename,
                method_dir / filename,
                model,
                device,
                args.batch_size,
            )
            if rows and len(values) != len(rows[0]):
                raise RuntimeError(f"unexpected frame count for {filename}")
            rows.append(values)
            if index % 25 == 0 or index == len(reference_names):
                print(f"[{method_name}] {index}/{len(reference_names)}", flush=True)

        values = np.asarray(rows, dtype=np.float64)
        mean = values.mean(axis=0)
        std = values.std(axis=0, ddof=1)
        ci95 = 1.96 * std / np.sqrt(values.shape[0])
        result["frame_count"] = int(values.shape[1])
        result["methods"][method_name] = {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "ci95": ci95.tolist(),
            "overall_mean": float(values.mean()),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
