#!/usr/bin/env python3
"""Run VBench-I2V dimensions independently and compute official aggregate scores."""

import argparse
import csv
import glob
import importlib.util
import json
import os
import os.path as osp
import subprocess
import sys
import time


REPO_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
VBENCH_DIR = osp.join(REPO_ROOT, "VBench")
EVALUATE_I2V = osp.join(VBENCH_DIR, "evaluate_i2v.py")
FULL_INFO = osp.join(VBENCH_DIR, "vbench2_beta_i2v", "vbench2_i2v_full_info.json")
SCORE_CONSTANTS = osp.join(VBENCH_DIR, "scripts", "constant.py")

I2V_DIMS = [
    "camera_motion",
    "i2v_subject",
    "i2v_background",
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", action="append", required=True, help="NAME=VIDEOS_PATH")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--full_info", default=FULL_INFO)
    parser.add_argument("--ratio", default="16-9")
    parser.add_argument("--dimensions", nargs="+", default=I2V_DIMS)
    parser.add_argument("--cuda_visible_devices")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true", default=True)
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def latest_result(output_dir):
    paths = glob.glob(osp.join(output_dir, "*_eval_results.json"))
    return max(paths, key=osp.getmtime) if paths else None


def parse_score(output_dir, dimension):
    path = latest_result(output_dir)
    if not path:
        return None, None
    data = load_json(path)
    value = data.get(dimension)
    if isinstance(value, list) and value:
        value = value[0]
    if isinstance(value, dict):
        value = value.get("score")
    return (float(value), path) if isinstance(value, (int, float)) else (None, path)


def load_constants():
    spec = importlib.util.spec_from_file_location("vbench_i2v_score_constants", SCORE_CONSTANTS)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load VBench constants from {SCORE_CONSTANTS}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def official_scores(summary):
    constants = load_constants()
    upload = {}
    for raw_key, display_key in constants.I2VKEY.items():
        value = summary.get(raw_key)
        if not isinstance(value, (int, float)):
            return None
        upload[display_key] = float(value)

    normalized = {}
    for key in constants.TASK_INFO_I2V:
        bounds = constants.NORMALIZE_DIC_I2V[key]
        normalized[key] = (upload[key] - bounds["Min"]) / (bounds["Max"] - bounds["Min"])

    quality = sum(
        normalized[key] * constants.DIM_WEIGHT_I2V[key]
        for key in constants.I2V_QUALITY_LIST
    ) / sum(constants.DIM_WEIGHT_I2V[key] for key in constants.I2V_QUALITY_LIST)
    i2v = sum(
        normalized[key] * constants.DIM_WEIGHT_I2V[key]
        for key in constants.I2V_LIST
    ) / sum(constants.DIM_WEIGHT_I2V[key] for key in constants.I2V_LIST)
    total = (
        quality * constants.I2V_QUALITY_WEIGHT + i2v * constants.I2V_WEIGHT
    ) / (constants.I2V_QUALITY_WEIGHT + constants.I2V_WEIGHT)
    return {"quality": quality, "i2v": i2v, "total": total}


def run_dimension(videos_dir, dimension, output_dir, args):
    cmd = [
        sys.executable,
        EVALUATE_I2V,
        "--output_path", output_dir,
        "--full_json_dir", args.full_info,
        "--videos_path", videos_dir,
        "--dimension", dimension,
        "--load_ckpt_from_local", "True",
        "--ratio", args.ratio,
        "--mode", "vbench_standard",
        "--imaging_quality_preprocessing_mode", "longer",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{VBENCH_DIR}:{REPO_ROOT}:{env.get('PYTHONPATH', '')}"
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.makedirs(output_dir, exist_ok=True)
    log_path = osp.join(output_dir, "eval.log")
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("[cmd] " + " ".join(cmd) + "\n")
        log.flush()
        start = time.time()
        process = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        elapsed = time.time() - start
        log.write(f"\n[exit] code={process.returncode} elapsed_s={elapsed:.1f}\n")
    return process.returncode, elapsed, log_path


def evaluate_method(name, videos_dir, args):
    method_root = osp.join(args.output_root, name)
    os.makedirs(method_root, exist_ok=True)
    summary_path = osp.join(method_root, "vbench_i2v_scores.json")
    summary = load_json(summary_path) if args.resume and osp.exists(summary_path) else {}
    summary.update({"_method": name, "_videos_dir": osp.abspath(videos_dir), "_dimensions": args.dimensions})
    summary.setdefault("_status", {})

    for dimension in args.dimensions:
        output_dir = osp.join(method_root, f"dim_{dimension}")
        score, result_path = parse_score(output_dir, dimension)
        if args.resume and score is not None:
            summary[dimension] = score
            summary["_status"][dimension] = {"ok": True, "cached": True, "result_file": result_path}
            print(f"[{name}] {dimension}: cached {score:.6f}", flush=True)
            continue
        code, elapsed, log_path = run_dimension(videos_dir, dimension, output_dir, args)
        score, result_path = parse_score(output_dir, dimension)
        ok = code == 0 and score is not None
        summary[dimension] = score if ok else None
        summary["_status"][dimension] = {
            "ok": ok,
            "exit_code": code,
            "elapsed_s": round(elapsed, 1),
            "result_file": result_path,
            "log": log_path,
        }
        save_json(summary, summary_path)
        if not ok and not args.continue_on_error:
            break

    failed = [dim for dim in I2V_DIMS if not isinstance(summary.get(dim), (int, float))]
    summary["_num_scored_dimensions"] = len(I2V_DIMS) - len(failed)
    official = official_scores(summary)
    summary["_quality_score"] = official["quality"] if official else None
    summary["_i2v_score"] = official["i2v"] if official else None
    summary["_total_score"] = official["total"] if official else None
    summary["_failed_dimensions"] = failed
    summary["_score_protocol"] = "VBench-I2V official normalized Quality/I2V weighting"
    save_json(summary, summary_path)
    return summary


def write_summary(summaries, output_root, dimensions):
    path = osp.join(output_root, "summary.csv")
    header = ["method", "total_score", "quality_score", "i2v_score", *dimensions, "failed"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for name, summary in summaries.items():
            writer.writerow([
                name,
                summary.get("_total_score"),
                summary.get("_quality_score"),
                summary.get("_i2v_score"),
                *[summary.get(dim) for dim in dimensions],
                ",".join(summary.get("_failed_dimensions", [])),
            ])
    return path


def main():
    args = parse_args()
    methods = []
    for entry in args.method:
        if "=" not in entry:
            raise SystemExit(f"--method must be NAME=VIDEOS_PATH, got {entry!r}")
        name, path = entry.split("=", 1)
        methods.append((name, osp.abspath(path)))
    os.makedirs(args.output_root, exist_ok=True)
    summaries = {name: evaluate_method(name, path, args) for name, path in methods}
    save_json(summaries, osp.join(args.output_root, "summary.json"))
    print(write_summary(summaries, args.output_root, args.dimensions))


if __name__ == "__main__":
    main()
