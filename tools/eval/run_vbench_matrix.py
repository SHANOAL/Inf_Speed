#!/usr/bin/env python3
"""
Robust VBench runner for one-seed InfinityStar evaluations.

The script runs VBench one dimension at a time, stores every dimension in its
own directory, and writes an aggregate summary even when some dimensions fail.
It supports both official prompt-named videos and sampled vXXX.mp4 videos.
"""

import argparse
import glob
import importlib.util
import json
import math
import os
import os.path as osp
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path


REPO_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
VBENCH_DIR = osp.join(REPO_ROOT, "VBench")
EVALUATE_PY = osp.join(VBENCH_DIR, "evaluate.py")
VBENCH_FULL_INFO = osp.join(VBENCH_DIR, "vbench", "VBench_full_info.json")
VBENCH_SCORE_CONSTANTS = osp.join(VBENCH_DIR, "scripts", "constant.py")

PAPER_10_DIMS = [
    "subject_consistency",
    "background_consistency",
    "aesthetic_quality",
    "imaging_quality",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
    "overall_consistency",
    "temporal_style",
    "appearance_style",
]

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


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    os.makedirs(osp.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if osp.exists(tmp_path):
            os.remove(tmp_path)


def official_vbench_scores(summary):
    """Return the official normalized Quality, Semantic, and Total scores."""
    spec = importlib.util.spec_from_file_location("vbench_score_constants", VBENCH_SCORE_CONSTANTS)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load VBench score constants from {VBENCH_SCORE_CONSTANTS}")
    constants = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(constants)

    raw = {}
    for task in constants.TASK_INFO:
        value = summary.get(task.replace(" ", "_"))
        if not isinstance(value, (int, float)):
            return None
        raw[task] = float(value)

    normalized = {}
    for task, value in raw.items():
        bounds = constants.NORMALIZE_DIC[task]
        normalized[task] = (value - bounds["Min"]) / (bounds["Max"] - bounds["Min"])

    quality_weight = sum(constants.DIM_WEIGHT[task] for task in constants.QUALITY_LIST)
    semantic_weight = sum(constants.DIM_WEIGHT[task] for task in constants.SEMANTIC_LIST)
    quality = sum(
        normalized[task] * constants.DIM_WEIGHT[task] for task in constants.QUALITY_LIST
    ) / quality_weight
    semantic = sum(
        normalized[task] * constants.DIM_WEIGHT[task] for task in constants.SEMANTIC_LIST
    ) / semantic_weight
    total = (
        quality * constants.QUALITY_WEIGHT + semantic * constants.SEMANTIC_WEIGHT
    ) / (constants.QUALITY_WEIGHT + constants.SEMANTIC_WEIGHT)
    return {"quality": quality, "semantic": semantic, "total": total}


def parse_methods(entries, method_name, videos_path):
    methods = []
    for entry in entries or []:
        if "=" not in entry:
            raise SystemExit(f"--method must be NAME=VIDEOS_PATH, got {entry!r}")
        name, path = entry.split("=", 1)
        methods.append((name, osp.abspath(path)))
    if videos_path:
        methods.append((method_name, osp.abspath(videos_path)))
    if not methods:
        raise SystemExit("Provide --method NAME=VIDEOS_PATH or --videos_path.")
    return methods


def dimension_list(args):
    if args.dimensions:
        return args.dimensions
    if args.dimension_set == "full16":
        return FULL_16_DIMS
    return PAPER_10_DIMS


def latest_eval_json(output_dir):
    files = glob.glob(osp.join(output_dir, "*_eval_results.json"))
    if not files:
        return None
    return max(files, key=osp.getmtime)


def parse_score(output_dir, dim):
    result_file = latest_eval_json(output_dir)
    if not result_file:
        return None, None
    data = load_json(result_file)
    value = data.get(dim)
    score = None
    if isinstance(value, list) and value:
        head = value[0]
        if isinstance(head, (int, float)):
            score = float(head)
        elif isinstance(head, dict) and isinstance(head.get("score"), (int, float)):
            score = float(head["score"])
    elif isinstance(value, dict) and isinstance(value.get("score"), (int, float)):
        score = float(value["score"])
    elif isinstance(value, (int, float)):
        score = float(value)
    if score is not None and not math.isfinite(score):
        score = None
    return score, result_file


def read_tail(path, limit=4000):
    if not osp.exists(path):
        return ""
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - limit), os.SEEK_SET)
        return f.read().decode("utf-8", errors="replace")


def reserve_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def full_info_by_prompt(full_info_path):
    return {item["prompt_en"]: item for item in load_json(full_info_path)}


def standard_video_name(prompt, index=0):
    return f"{prompt}-{index}.mp4"


def make_sampled_workspace(videos_dir, sampled_prompts, dim, full_info_path):
    full_by_prompt = full_info_by_prompt(full_info_path)
    tmp = tempfile.TemporaryDirectory(prefix=f"vbench_{dim}_")
    tmp_videos = osp.join(tmp.name, "videos")
    os.makedirs(tmp_videos, exist_ok=True)
    tmp_full_info = []
    for i, prompt in enumerate(sampled_prompts):
        if prompt.get("dimension") != dim:
            continue
        src = osp.join(videos_dir, f"v{i:03d}.mp4")
        prompt_en = prompt["prompt_en"]
        if not osp.exists(src) or prompt_en not in full_by_prompt:
            continue
        dst = osp.join(tmp_videos, standard_video_name(prompt_en, 0))
        os.symlink(src, dst)
        tmp_full_info.append(full_by_prompt[prompt_en])
    if not tmp_full_info:
        tmp.cleanup()
        return None
    tmp_info_path = osp.join(tmp.name, "full_info.json")
    save_json(tmp_full_info, tmp_info_path)
    return tmp, tmp_videos, tmp_info_path


def run_vbench_dim(videos_dir, full_info_path, dim, output_dir, log_path, args):
    cmd = [
        sys.executable,
        EVALUATE_PY,
        "--output_path",
        output_dir,
        "--full_json_dir",
        full_info_path,
        "--videos_path",
        videos_dir,
        "--dimension",
        dim,
        "--load_ckpt_from_local",
        "True",
        "--mode",
        "vbench_standard",
        "--imaging_quality_preprocessing_mode",
        args.imaging_quality_preprocessing_mode,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{VBENCH_DIR}:{REPO_ROOT}:{env.get('PYTHONPATH', '')}"
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = str(reserve_free_port())
    env["RANK"] = "0"
    env["LOCAL_RANK"] = "0"
    env["WORLD_SIZE"] = "1"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.makedirs(output_dir, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("[cmd] " + " ".join(cmd) + "\n")
        log.flush()
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        elapsed = time.time() - t0
        log.write(f"\n[exit] code={proc.returncode} elapsed_s={elapsed:.1f}\n")
    return proc.returncode, elapsed


def evaluate_method(name, videos_dir, dims, output_root, args, sampled_prompts=None):
    method_out = osp.join(output_root, name)
    os.makedirs(method_out, exist_ok=True)
    summary_path = osp.join(method_out, "vbench_scores.json")
    summary = load_json(summary_path) if args.resume and osp.exists(summary_path) else {}
    previous_dims = summary.get("_dimensions", []) if isinstance(summary.get("_dimensions"), list) else []
    summary_dims = list(dict.fromkeys(previous_dims + dims))
    summary["_method"] = name
    summary["_videos_dir"] = videos_dir
    summary["_dimensions"] = summary_dims
    summary.setdefault("_status", {})

    for dim in dims:
        dim_out = osp.join(method_out, f"dim_{dim}")
        score, result_file = parse_score(dim_out, dim)
        if args.resume and score is not None:
            summary[dim] = score
            summary["_status"][dim] = {
                "ok": True,
                "cached": True,
                "result_file": result_file,
            }
            print(f"[{name}] {dim}: cached {score:.6f}", flush=True)
            continue

        workspace = None
        run_videos_dir = videos_dir
        run_full_info = args.full_json_dir
        if sampled_prompts is not None:
            workspace = make_sampled_workspace(videos_dir, sampled_prompts, dim, args.full_json_dir)
            if workspace is None:
                summary[dim] = None
                summary["_status"][dim] = {"ok": False, "error": "no valid sampled videos"}
                save_json(summary, summary_path)
                print(f"[{name}] {dim}: no valid sampled videos", flush=True)
                continue
            tmp, run_videos_dir, run_full_info = workspace

        log_path = osp.join(dim_out, "eval.log")
        print(f"[{name}] {dim}: running", flush=True)
        try:
            code, elapsed = run_vbench_dim(run_videos_dir, run_full_info, dim, dim_out, log_path, args)
        finally:
            if workspace is not None:
                workspace[0].cleanup()

        score, result_file = parse_score(dim_out, dim)
        ok = code == 0 and score is not None
        summary[dim] = score if ok else None
        summary["_status"][dim] = {
            "ok": ok,
            "exit_code": code,
            "elapsed_s": round(elapsed, 1),
            "result_file": result_file,
            "log": log_path,
        }
        if not ok:
            summary["_status"][dim]["log_tail"] = read_tail(log_path)
        save_json(summary, summary_path)

        if ok:
            print(f"[{name}] {dim}: score={score:.6f} dt={elapsed:.1f}s", flush=True)
        else:
            print(f"[{name}] {dim}: FAILED code={code} dt={elapsed:.1f}s", flush=True)
            if not args.continue_on_error:
                break

    numeric = [summary.get(dim) for dim in summary_dims if isinstance(summary.get(dim), (int, float))]
    failed = [dim for dim in summary_dims if not isinstance(summary.get(dim), (int, float))]
    official = official_vbench_scores(summary)
    summary["_raw_mean"] = sum(numeric) / len(numeric) if numeric else None
    summary["_quality_score"] = official["quality"] if official else None
    summary["_semantic_score"] = official["semantic"] if official else None
    summary["_total_score"] = official["total"] if official else None
    summary["_aggregate"] = summary["_total_score"]
    summary["_score_protocol"] = "VBench official normalized 4:1 Quality/Semantic weighting"
    summary["_num_scored_dimensions"] = len(numeric)
    summary["_failed_dimensions"] = failed
    save_json(summary, summary_path)
    return summary


def write_table(summaries, dims, output_root):
    table_dims = list(dims)
    for summary in summaries.values():
        for dim in summary.get("_dimensions", []):
            if dim not in table_dims:
                table_dims.append(dim)
    rows = []
    header = [
        "method", "total_score", "quality_score", "semantic_score", "raw_mean", "num_dims"
    ] + table_dims + ["failed"]
    rows.append(header)
    for method, summary in summaries.items():
        row = [
            method,
            "" if summary.get("_total_score") is None else f"{summary['_total_score']:.6f}",
            "" if summary.get("_quality_score") is None else f"{summary['_quality_score']:.6f}",
            "" if summary.get("_semantic_score") is None else f"{summary['_semantic_score']:.6f}",
            "" if summary.get("_raw_mean") is None else f"{summary['_raw_mean']:.6f}",
            str(summary.get("_num_scored_dimensions", 0)),
        ]
        for dim in table_dims:
            value = summary.get(dim)
            row.append("" if value is None else f"{value:.6f}")
        row.append(",".join(summary.get("_failed_dimensions", [])))
        rows.append(row)

    csv_path = osp.join(output_root, "summary.csv")
    md_path = osp.join(output_root, "summary.md")
    with open(csv_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(",".join('"' + cell.replace('"', '""') + '"' for cell in row) + "\n")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(header) + " |\n")
        f.write("| " + " | ".join(["---"] * len(header)) + " |\n")
        for row in rows[1:]:
            f.write("| " + " | ".join(row) + " |\n")
    return csv_path, md_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", action="append", help="NAME=VIDEOS_PATH. Can be repeated.")
    parser.add_argument("--method_name", default="method")
    parser.add_argument("--videos_path")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--full_json_dir", default=VBENCH_FULL_INFO)
    parser.add_argument("--dimension_set", choices=["paper10", "full16"], default="paper10")
    parser.add_argument("--dimensions", nargs="+")
    parser.add_argument("--sampled_prompts", help="JSON list for v000.mp4 style sampled videos.")
    parser.add_argument("--cuda_visible_devices")
    parser.add_argument("--imaging_quality_preprocessing_mode", default="longer")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true", default=True)
    args = parser.parse_args()

    dims = dimension_list(args)
    methods = parse_methods(args.method, args.method_name, args.videos_path)
    sampled_prompts = load_json(args.sampled_prompts) if args.sampled_prompts else None
    os.makedirs(args.output_root, exist_ok=True)

    summaries = {}
    for name, videos_dir in methods:
        if not osp.isdir(videos_dir):
            raise SystemExit(f"Videos dir does not exist: {videos_dir}")
        summaries[name] = evaluate_method(name, videos_dir, dims, args.output_root, args, sampled_prompts)

    save_json(summaries, osp.join(args.output_root, "summary.json"))
    csv_path, md_path = write_table(summaries, dims, args.output_root)
    print(f"summary: {osp.join(args.output_root, 'summary.json')}", flush=True)
    print(f"csv: {csv_path}", flush=True)
    print(f"markdown: {md_path}", flush=True)


if __name__ == "__main__":
    main()
