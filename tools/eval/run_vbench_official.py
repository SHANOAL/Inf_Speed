"""
Run official VBench evaluation on saved videos.

For each method dir (containing .mp4 generated from rewritten VBench prompts),
this:
  1. Reads sampled_prompts.json to learn which video belongs to which dimension.
  2. Loads VBench's official full-info JSON to recover auxiliary labels.
  3. For each dimension:
       a. Builds a temp dir with prompt-named symlinks, e.g. "<prompt>-0.mp4"
       b. Builds a temp full-info JSON containing only sampled prompts.
       c. Calls VBench/evaluate.py in vbench_standard mode.
  3. Aggregates per-dim scores into <output>/vbench_scores.json

Usage:
  python tools/eval/run_vbench_official.py \\
      --videos_root output/vbench_faststar_vs_v3 \\
      --methods baseline faststar v3 \\
      --sampled_prompts output/vbench_faststar_vs_v3/sampled_prompts.json
"""
import sys, os, json, shutil, subprocess, argparse, tempfile, time
import os.path as osp

REPO_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
VBENCH_DIR = osp.join(REPO_ROOT, "VBench")
EVALUATE_PY = osp.join(VBENCH_DIR, "evaluate.py")
VBENCH_FULL_INFO = osp.join(VBENCH_DIR, "vbench", "VBench_full_info.json")

ALL_DIMS = [
    "subject_consistency", "background_consistency", "temporal_flickering",
    "motion_smoothness", "dynamic_degree", "aesthetic_quality", "imaging_quality",
    "object_class", "multiple_objects", "human_action", "color",
    "spatial_relationship", "scene", "appearance_style", "temporal_style",
    "overall_consistency",
]


def run_dim(videos_dir, full_json_dir, dim, output_dir):
    """Invoke VBench evaluate.py for a single dimension."""
    cmd = [
        sys.executable, EVALUATE_PY,
        "--videos_path", videos_dir,
        "--dimension", dim,
        "--output_path", output_dir,
        "--full_json_dir", full_json_dir,
        "--mode", "vbench_standard",
        "--load_ckpt_from_local", "True",
    ]
    print(f"  [{dim}] running ...", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=VBENCH_DIR, capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=VBENCH_DIR))
    dt = time.time() - t0
    if r.returncode != 0:
        print(f"  [{dim}] FAILED ec={r.returncode} dt={dt:.0f}s", flush=True)
        print(f"    STDERR tail: {r.stderr[-2000:]}", flush=True)
        return None
    print(f"  [{dim}] done dt={dt:.0f}s", flush=True)
    return True


def parse_dim_score(output_dir, dim):
    """Find the latest results_*_eval_results.json and pull the dim score."""
    files = sorted([f for f in os.listdir(output_dir) if f.endswith("_eval_results.json")], reverse=True)
    if not files:
        return None
    p = osp.join(output_dir, files[0])
    d = json.load(open(p))
    if dim in d:
        v = d[dim]
        if isinstance(v, list) and len(v) > 0:
            return v[0] if isinstance(v[0], (int, float)) else (v[0].get("score") if isinstance(v[0], dict) else None)
        if isinstance(v, dict):
            return v.get("score")
        if isinstance(v, (int, float)):
            return v
    return None


def load_full_info_by_prompt():
    with open(VBENCH_FULL_INFO, "r") as f:
        items = json.load(f)
    return {it["prompt_en"]: it for it in items}


def standard_video_name(prompt_en, sample_index=0):
    if os.sep in prompt_en or (os.altsep and os.altsep in prompt_en):
        raise ValueError(f"Prompt cannot be used as a standard VBench filename: {prompt_en!r}")
    return f"{prompt_en}-{sample_index}.mp4"


def evaluate_method(videos_root, method, prompts, output_root, dims, resume):
    method_videos = osp.join(videos_root, method, "videos")
    out = osp.join(output_root, method)
    os.makedirs(out, exist_ok=True)
    full_info_by_prompt = load_full_info_by_prompt()
    scores = {}
    results_file = osp.join(out, "vbench_scores.json")
    if osp.exists(results_file):
        try:
            scores = json.load(open(results_file))
        except Exception:
            scores = {}

    for dim in dims:
        if resume and dim in scores and scores[dim] is not None:
            print(f"  [{dim}] cached score={scores[dim]:.4f}", flush=True)
            continue
        # Find the sampled videos for this dim.
        dim_videos = [(i, p) for i, p in enumerate(prompts) if p["dimension"] == dim]
        if not dim_videos:
            print(f"  [{dim}] no videos found, skip", flush=True)
            continue
        # Build a VBench-standard temp dir. We cannot use custom_input for all
        # dimensions, because object/color/scene/style dimensions require the
        # auxiliary_info fields from VBench_full_info.json.
        with tempfile.TemporaryDirectory(prefix=f"vb_{method}_{dim}_") as tmp:
            videos_link_dir = osp.join(tmp, "videos")
            os.makedirs(videos_link_dir, exist_ok=True)
            dim_full_info = []
            for vi, p in dim_videos:
                src = osp.join(method_videos, f"v{vi:03d}.mp4")
                if not osp.exists(src):
                    print(f"  [{dim}] missing {src}", flush=True)
                    continue
                prompt_en = p["prompt_en"]
                if prompt_en not in full_info_by_prompt:
                    print(f"  [{dim}] prompt missing from VBench full info: {prompt_en}", flush=True)
                    continue
                dst = osp.join(videos_link_dir, standard_video_name(prompt_en, 0))
                os.symlink(src, dst)
                dim_full_info.append(full_info_by_prompt[prompt_en])
            if not dim_full_info:
                print(f"  [{dim}] no valid videos after linking, skip", flush=True)
                scores[dim] = None
                continue
            full_info_path = osp.join(tmp, "full_info.json")
            with open(full_info_path, "w") as f:
                json.dump(dim_full_info, f, indent=2, ensure_ascii=False)
            dim_out = osp.join(out, f"dim_{dim}")
            os.makedirs(dim_out, exist_ok=True)
            ok = run_dim(videos_link_dir, full_info_path, dim, dim_out)
            if ok:
                s = parse_dim_score(dim_out, dim)
                scores[dim] = s
                print(f"    -> score = {s}", flush=True)
            else:
                scores[dim] = None
        # Save incrementally
        with open(results_file, "w") as f:
            json.dump(scores, f, indent=2)
    # Final aggregate
    vals = [v for v in scores.values() if isinstance(v, (int, float))]
    agg = sum(vals) / len(vals) if vals else None
    scores["_aggregate"] = agg
    with open(results_file, "w") as f:
        json.dump(scores, f, indent=2)
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos_root", required=True, help="dir containing per-method subdirs, each with videos/")
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--sampled_prompts", required=True, help="JSON list with {prompt_en, dimension, ...}")
    parser.add_argument("--output_root", default=None, help="where to save per-method vbench_scores.json (default: videos_root/vbench_eval)")
    parser.add_argument("--dimensions", nargs="+", default=ALL_DIMS)
    parser.add_argument("--resume", action="store_true")
    cli = parser.parse_args()

    prompts = json.load(open(cli.sampled_prompts))
    out_root = cli.output_root or osp.join(cli.videos_root, "vbench_eval")
    os.makedirs(out_root, exist_ok=True)

    summary = {}
    for method in cli.methods:
        print(f"\n=== {method} ===", flush=True)
        scores = evaluate_method(cli.videos_root, method, prompts, out_root, cli.dimensions, cli.resume)
        summary[method] = scores
        if scores.get("_aggregate"):
            print(f"\n  >>> {method} VBench aggregate = {scores['_aggregate']:.4f}", flush=True)

    with open(osp.join(out_root, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {out_root}/summary.json", flush=True)


if __name__ == "__main__":
    main()
