#!/usr/bin/env python3
"""Evaluate prompt-aligned videos against a dense reference with resumable metrics."""

import argparse
import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


CPU_METRICS = {"psnr", "ssim"}


def parse_method(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("method must be NAME=VIDEO_DIR")
    name, path = value.split("=", 1)
    return name, Path(path).resolve()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_dir", required=True, type=Path)
    parser.add_argument("--method", action="append", required=True, type=parse_method)
    parser.add_argument("--output_root", required=True, type=Path)
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["psnr", "ssim", "lpips"],
        default=["psnr", "ssim", "lpips"],
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--lpips_device", default="cuda:0")
    parser.add_argument("--lpips_batch_size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def pair_id(filename):
    return hashlib.sha1(filename.encode("utf-8")).hexdigest()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(tmp, path)


def result_path(output_dir, filename):
    return output_dir / "per_video" / f"{pair_id(filename)}.json"


def cpu_pair(task):
    filename, reference_path, candidate_path, metrics, frame_stride = task
    import cv2
    import numpy as np
    from skimage.metrics import structural_similarity

    cv2.setNumThreads(1)
    reference = cv2.VideoCapture(reference_path)
    candidate = cv2.VideoCapture(candidate_path)
    if not reference.isOpened() or not candidate.isOpened():
        return {"video": filename, "error": "failed to open one or both videos"}

    sums = {metric: 0.0 for metric in metrics}
    sampled_frames = 0
    decoded_frames = 0
    identical_frames = 0
    error = None
    while True:
        ok_ref, frame_ref = reference.read()
        ok_candidate, frame_candidate = candidate.read()
        if not ok_ref or not ok_candidate:
            if ok_ref != ok_candidate:
                error = "frame-count mismatch"
            break
        if frame_ref.shape != frame_candidate.shape:
            error = f"shape mismatch: {frame_ref.shape} vs {frame_candidate.shape}"
            break
        if decoded_frames % frame_stride == 0:
            if "psnr" in metrics:
                delta = frame_ref.astype(np.float64) - frame_candidate.astype(np.float64)
                mse = float(np.mean(delta * delta))
                if mse == 0:
                    psnr = 100.0
                    identical_frames += 1
                else:
                    psnr = 20.0 * math.log10(255.0 / math.sqrt(mse))
                sums["psnr"] += psnr
            if "ssim" in metrics:
                sums["ssim"] += float(
                    structural_similarity(
                        frame_ref,
                        frame_candidate,
                        channel_axis=-1,
                        data_range=255,
                    )
                )
            sampled_frames += 1
        decoded_frames += 1

    reference.release()
    candidate.release()
    if error is not None:
        return {"video": filename, "error": error}
    if sampled_frames == 0:
        return {"video": filename, "error": "no frames decoded"}

    row = {
        "video": filename,
        "decoded_frames": decoded_frames,
        "sampled_frames": sampled_frames,
        "frame_stride": frame_stride,
        "identical_frames": identical_frames,
    }
    if "psnr" in metrics:
        row["PSNR"] = sums["psnr"] / sampled_frames
    if "ssim" in metrics:
        row["SSIM"] = sums["ssim"] / sampled_frames
    return row


def lpips_pair(reference_path, candidate_path, model, device, batch_size, frame_stride):
    import cv2
    import numpy as np
    import torch

    reference = cv2.VideoCapture(str(reference_path))
    candidate = cv2.VideoCapture(str(candidate_path))
    if not reference.isOpened() or not candidate.isOpened():
        raise RuntimeError("failed to open one or both videos")

    ref_batch = []
    candidate_batch = []
    total = 0.0
    sampled_frames = 0
    decoded_frames = 0

    def flush():
        nonlocal total, sampled_frames
        if not ref_batch:
            return
        ref_np = np.stack(ref_batch)[..., ::-1].copy()
        candidate_np = np.stack(candidate_batch)[..., ::-1].copy()
        ref_tensor = torch.from_numpy(ref_np).permute(0, 3, 1, 2).float().to(device)
        candidate_tensor = torch.from_numpy(candidate_np).permute(0, 3, 1, 2).float().to(device)
        ref_tensor = ref_tensor.div(127.5).sub(1.0)
        candidate_tensor = candidate_tensor.div(127.5).sub(1.0)
        with torch.no_grad():
            values = model(ref_tensor, candidate_tensor).reshape(-1)
        total += float(values.sum().item())
        sampled_frames += int(values.numel())
        ref_batch.clear()
        candidate_batch.clear()

    while True:
        ok_ref, frame_ref = reference.read()
        ok_candidate, frame_candidate = candidate.read()
        if not ok_ref or not ok_candidate:
            if ok_ref != ok_candidate:
                raise RuntimeError("frame-count mismatch")
            break
        if frame_ref.shape != frame_candidate.shape:
            raise RuntimeError(f"shape mismatch: {frame_ref.shape} vs {frame_candidate.shape}")
        if decoded_frames % frame_stride == 0:
            ref_batch.append(frame_ref)
            candidate_batch.append(frame_candidate)
            if len(ref_batch) >= batch_size:
                flush()
        decoded_frames += 1
    flush()
    reference.release()
    candidate.release()
    if sampled_frames == 0:
        raise RuntimeError("no frames decoded")
    return total / sampled_frames, sampled_frames


def cached_row(path, filename):
    if not path.exists():
        return {"video": filename}
    row = load_json(path)
    if row.get("video") != filename:
        raise RuntimeError(f"cache collision for {filename}")
    return row


def cpu_metrics_for_method(reference_dir, candidate_dir, output_dir, names, metrics, args):
    required_keys = {metric.upper() for metric in metrics}
    tasks = []
    for filename in names:
        path = result_path(output_dir, filename)
        row = cached_row(path, filename) if args.resume else {"video": filename}
        if required_keys.issubset(row) and row.get("frame_stride") == args.frame_stride:
            continue
        tasks.append(
            (
                filename,
                str(reference_dir / filename),
                str(candidate_dir / filename),
                tuple(metrics),
                args.frame_stride,
            )
        )

    if not tasks:
        return
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(cpu_pair, task): task[0] for task in tasks}
        for future in as_completed(futures):
            filename = futures[future]
            result = future.result()
            path = result_path(output_dir, filename)
            row = cached_row(path, filename) if args.resume else {"video": filename}
            if "error" not in result:
                row.pop("error", None)
            row.update(result)
            save_json(row, path)
            completed += 1
            if completed % 25 == 0 or completed == len(tasks):
                print(f"[cpu] {output_dir.name}: {completed}/{len(tasks)}", flush=True)


def lpips_for_method(reference_dir, candidate_dir, output_dir, names, args):
    import lpips
    import torch

    device = torch.device(args.lpips_device)
    model = lpips.LPIPS(net="alex", verbose=False).eval().to(device)
    pending = []
    for filename in names:
        path = result_path(output_dir, filename)
        row = cached_row(path, filename)
        if args.resume and "LPIPS" in row and row.get("lpips_frame_stride") == args.frame_stride:
            continue
        pending.append(filename)

    for index, filename in enumerate(pending, start=1):
        path = result_path(output_dir, filename)
        row = cached_row(path, filename)
        try:
            value, frames = lpips_pair(
                reference_dir / filename,
                candidate_dir / filename,
                model,
                device,
                args.lpips_batch_size,
                args.frame_stride,
            )
            row["LPIPS"] = value
            row["lpips_sampled_frames"] = frames
            row["lpips_frame_stride"] = args.frame_stride
            row.pop("error", None)
        except Exception as exc:
            row["lpips_error"] = str(exc)
        save_json(row, path)
        if index % 10 == 0 or index == len(pending):
            print(f"[lpips] {output_dir.name}: {index}/{len(pending)}", flush=True)


def write_summary(method, reference_dir, candidate_dir, output_dir, names, metrics, args):
    rows = []
    errors = []
    required = {metric.upper() for metric in metrics}
    for filename in names:
        path = result_path(output_dir, filename)
        if not path.exists():
            errors.append({"video": filename, "error": "missing result"})
            continue
        row = load_json(path)
        missing = sorted(required - set(row))
        if row.get("error") or row.get("lpips_error") or missing:
            errors.append(
                {
                    "video": filename,
                    "error": row.get("error") or row.get("lpips_error") or f"missing {missing}",
                }
            )
            continue
        rows.append(row)

    summary = {
        "method": method,
        "reference_dir": str(reference_dir),
        "candidate_dir": str(candidate_dir),
        "metrics": list(metrics),
        "frame_stride": args.frame_stride,
        "num_videos": len(rows),
        "expected_videos": len(names),
        "num_errors": len(errors),
        "total_sampled_frames": sum(row.get("sampled_frames", 0) for row in rows),
        "errors": errors,
        "per_video": rows,
    }
    for metric in metrics:
        key = metric.upper()
        values = [float(row[key]) for row in rows]
        summary[f"avg_{key}"] = sum(values) / len(values) if values else None
        frame_key = "lpips_sampled_frames" if metric == "lpips" else "sampled_frames"
        weights = [int(row.get(frame_key, 0)) for row in rows]
        weight_sum = sum(weights)
        summary[f"frame_weighted_{key}"] = (
            sum(value * weight for value, weight in zip(values, weights)) / weight_sum
            if weight_sum
            else None
        )
    save_json(summary, output_dir / "results.json")
    print(
        f"[summary] {method}: videos={len(rows)}/{len(names)} errors={len(errors)} "
        + " ".join(
            f"{metric.upper()}={summary.get(f'avg_{metric.upper()}')}" for metric in metrics
        ),
        flush=True,
    )


def main():
    args = parse_args()
    reference_dir = args.reference_dir.resolve()
    if args.frame_stride < 1:
        raise SystemExit("frame_stride must be positive")
    reference_names = {path.name for path in reference_dir.glob("*.mp4")}
    if not reference_names:
        raise SystemExit(f"no reference videos in {reference_dir}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    metrics = tuple(dict.fromkeys(args.metrics))
    cpu_metrics = tuple(metric for metric in metrics if metric in CPU_METRICS)
    for method, candidate_dir in args.method:
        candidate_names = {path.name for path in candidate_dir.glob("*.mp4")}
        if candidate_names != reference_names:
            raise SystemExit(
                f"{method}: filename mismatch reference_only={len(reference_names - candidate_names)} "
                f"candidate_only={len(candidate_names - reference_names)}"
            )
        names = sorted(reference_names)
        if args.limit > 0:
            names = names[: args.limit]
        output_dir = args.output_root / method
        (output_dir / "per_video").mkdir(parents=True, exist_ok=True)
        print(f"[{method}] videos={len(names)} metrics={metrics}", flush=True)
        if cpu_metrics:
            cpu_metrics_for_method(reference_dir, candidate_dir, output_dir, names, cpu_metrics, args)
        if "lpips" in metrics:
            lpips_for_method(reference_dir, candidate_dir, output_dir, names, args)
        write_summary(method, reference_dir, candidate_dir, output_dir, names, metrics, args)


if __name__ == "__main__":
    main()
