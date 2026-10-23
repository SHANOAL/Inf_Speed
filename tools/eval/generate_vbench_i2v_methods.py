#!/usr/bin/env python3
"""Generate VBench-I2V videos for dense and token-reduction methods."""

import argparse
import json
import os
import os.path as osp
import sys
import time

import numpy as np


REPO_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from infinity.schedules.dynamic_resolution import get_dynamic_resolution_meta  # noqa: E402
from tools.eval.generate_vbench_methods import (  # noqa: E402
    METHODS,
    build_args,
    configure_method,
)
from tools.infer_video_720p import InferencePipe, perform_inference  # noqa: E402
from tools.run_infinity import save_video  # noqa: E402


OURS_CONFIG_KEYS = {
    "sparstar", "sparstar_density", "sparstar_block_size",

    "prune_ctx_pool",

    "prune_attn_temporal_mode",
    "prune_attn_temporal_anchors",
    "prune_attn_random_select",
    "prune_skip_code",
    "prune_skip_code_beta",

    "scoring", "variant",
    "prune_w_spatial", "prune_w_temporal", "prune_w_text", "prune_fusion_p",
    "prune_attn_query_stride", "prune_attn_max_queries", "prune_per_frame",
    "prune_attn_mask_text_padding", "prune_attn_text_pool", "prune_attn_text_topk",
    "prune_attn_temporal_smooth", "prune_attn_shared_fraction",
    "prune_attn_text_query_gate", "prune_recover_mode", "prune_recover_code_mask",
    "prune_recover_neighbors", "prune_recover_alpha",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods",
        nargs="+",
        required=True,
        choices=["dense", *sorted(METHODS)],
    )
    parser.add_argument(
        "--full_info",
        default=osp.join(REPO_ROOT, "VBench", "vbench2_beta_i2v", "vbench2_i2v_full_info.json"),
    )
    parser.add_argument(
        "--image_dir",
        default=osp.join(REPO_ROOT, "VBench", "vbench2_beta_i2v", "data", "crop", "16-9"),
    )
    parser.add_argument("--output_root", default=osp.join(REPO_ROOT, "output"))
    parser.add_argument("--checkpoints_dir", default=osp.join(REPO_ROOT, "wieghts"))
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--samples_per_prompt", type=int, default=1)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--dimensions", nargs="+")
    parser.add_argument("--ours_config_json")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_rows(path, dimensions, offset, max_samples):
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if dimensions:
        selected = set(dimensions)
        rows = [row for row in rows if selected.intersection(row.get("dimension", []))]
    if offset:
        rows = rows[offset:]
    if max_samples:
        rows = rows[:max_samples]
    return rows


def load_ours_config(path):
    if not path:
        return
    with open(path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    override = loaded.get("generator_config", loaded)
    unknown = set(override) - OURS_CONFIG_KEYS
    if unknown:
        raise ValueError(f"unsupported Ours config keys: {sorted(unknown)}")
    METHODS["ours"].update(override)


def configure_dense(args):
    args.prune_scoring = "faststar"
    args.prune_ratios_per_scale = {}
    args.prune_keep_ratio = 1.0
    args.prune_token_threshold = 10**9


def generation_meta(cli, method, output_dir, row_count):
    return {
        "task": "VBench-I2V",
        "method": method,
        "config": {} if method == "dense" else METHODS[method],
        "full_info": osp.abspath(cli.full_info),
        "image_dir": osp.abspath(cli.image_dir),
        "image_ratio": "16-9",
        "num_prompt_image_pairs": row_count,
        "samples_per_prompt": cli.samples_per_prompt,
        "seed_base": cli.seed,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "output_dir": osp.abspath(output_dir),
    }


def check_resume_protocol(meta_path, expected, overwrite):
    if not osp.exists(meta_path) or overwrite:
        return
    with open(meta_path, "r", encoding="utf-8") as f:
        previous = json.load(f)
    keys = ("task", "method", "full_info", "image_dir", "samples_per_prompt", "seed_base")
    mismatches = [key for key in keys if previous.get(key) != expected.get(key)]
    if mismatches:
        raise RuntimeError(
            f"refusing to mix I2V protocols in {osp.dirname(meta_path)}; mismatched {mismatches}"
        )


def generate_method(pipe, args, cli, rows, method, prune_ratios):
    output_dir = osp.join(cli.output_root, f"vbench_i2v_{method}")
    os.makedirs(output_dir, exist_ok=True)
    meta = generation_meta(cli, method, output_dir, len(rows))
    meta_path = osp.join(output_dir, "_generation_meta.json")
    check_resume_protocol(meta_path, meta, cli.overwrite)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    if method == "dense":
        configure_dense(args)
    else:
        configure_method(args, method)
        args.prune_ratios_per_scale = dict(prune_ratios)

    generated = 0
    skipped = 0
    total_time = 0.0
    for item_index, item in enumerate(rows):
        prompt = item["prompt_en"]
        image_path = osp.join(cli.image_dir, item["image_name"])
        if not osp.exists(image_path):
            raise FileNotFoundError(image_path)
        for sample_index in range(cli.samples_per_prompt):
            video_path = osp.join(output_dir, f"{prompt}-{sample_index}.mp4")
            if osp.exists(video_path) and not cli.overwrite:
                skipped += 1
                continue
            global_item_index = cli.offset + item_index
            seed = cli.seed + global_item_index * cli.samples_per_prompt + sample_index
            result = perform_inference(
                pipe,
                {"prompt": prompt, "image_path": image_path, "seed": seed},
                args,
            )
            save_video(result["output"], fps=args.fps, save_filepath=video_path)
            generated += 1
            total_time += float(result["elapsed_time"])
            if generated % 25 == 0:
                print(
                    f"[{method}] generated={generated} skipped={skipped} "
                    f"avg_infer_s={total_time / generated:.2f}",
                    flush=True,
                )

    done = {
        **meta,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "generated": generated,
        "skipped": skipped,
        "avg_infer_s": total_time / generated if generated else None,
    }
    with open(osp.join(output_dir, "_generation_done.json"), "w", encoding="utf-8") as f:
        json.dump(done, f, indent=2, ensure_ascii=False)
    print(f"[{method}] done generated={generated} skipped={skipped}", flush=True)


def main():
    cli = parse_args()
    if cli.samples_per_prompt < 1:
        raise ValueError("samples_per_prompt must be positive")
    load_ours_config(cli.ours_config_json)
    rows = load_rows(cli.full_info, cli.dimensions, cli.offset, cli.max_samples)
    if not rows:
        raise ValueError("no I2V prompt-image pairs selected")

    args = build_args(cli.checkpoints_dir)
    pipe = InferencePipe(args)
    dynamic_meta, templates = get_dynamic_resolution_meta(args.dynamic_scale_schedule, args.video_frames)
    template = templates[np.argmin(np.abs(templates - 0.571))]
    schedule = dynamic_meta[template][args.pn]["pt2scale_schedule"][(args.video_frames - 1) // 4 + 1]
    num_scales = len(schedule)
    prune_ratios = {
        num_scales - 4: 0.80,
        num_scales - 3: 0.70,
        num_scales - 2: 0.60,
        num_scales - 1: 0.30,
    }

    for method in cli.methods:
        generate_method(pipe, args, cli, rows, method, prune_ratios)


if __name__ == "__main__":
    main()
