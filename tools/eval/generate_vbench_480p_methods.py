#!/usr/bin/env python3
"""Generate the 480p VBench T2V benchmark with dense and pruned methods."""

import argparse
import json
import os.path as osp

import numpy as np

from infinity.schedules.dynamic_resolution import (
    get_dynamic_resolution_meta,
    get_first_full_spatial_size_scale_index,
)
from tools.eval import generate_vbench_methods as base


OURS_CONFIG_KEYS = {
    "sparstar", "sparstar_density", "sparstar_block_size",

    "prune_ctx_pool",

    "prune_attn_temporal_mode",
    "prune_attn_temporal_anchors",
    "prune_attn_random_select",
    "prune_skip_code",
    "prune_skip_code_beta",

    "scoring",
    "variant",
    "prune_w_spatial",
    "prune_w_temporal",
    "prune_w_text",
    "prune_fusion_p",
    "prune_attn_query_stride",
    "prune_attn_max_queries",
    "prune_per_frame",
    "prune_attn_mask_text_padding",
    "prune_attn_text_pool",
    "prune_attn_text_topk",
    "prune_attn_temporal_smooth",
    "prune_attn_shared_fraction",
    "prune_attn_text_query_gate",
    "prune_recover_mode",
    "prune_recover_code_mask",
    "prune_recover_neighbors",
    "prune_recover_alpha",
}


def build_480p_args(checkpoints_dir):
    args = base.build_args(checkpoints_dir)
    args.pn = "0.40M"
    args.model_path = osp.join(checkpoints_dir, "infinitystar_8b_480p_weights")
    args.image_scale_repetition = "[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]"
    args.video_scale_repetition = "[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1]"
    args.detail_scale_min_tokens = 350
    args.semantic_scales = 11
    return args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods",
        nargs="+",
        required=True,
        choices=["dense", *sorted(base.METHODS)],
    )
    parser.add_argument(
        "--prompt_json",
        default=osp.join(base.REPO_ROOT, "evaluation", "VBench_rewrited_prompt.json"),
    )
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--checkpoints_dir", default=osp.join(base.REPO_ROOT, "wieghts"))
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--min_free_gb", type=float, default=5.0)
    parser.add_argument("--bypass_last_n", type=int, default=0)
    parser.add_argument("--ours_config_json")
    parser.add_argument(
        "--prompt_field",
        choices=("refined_prompt", "prompt_en"),
        default="refined_prompt",
    )
    parser.add_argument(
        "--suffix",
        default="Close-up on big objects, emphasize scale and detail",
    )
    return parser.parse_args()


def load_ours_config(path):
    if not path:
        return
    with open(path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    override = loaded.get("generator_config", loaded)
    unknown = set(override) - OURS_CONFIG_KEYS
    if unknown:
        raise ValueError(f"unsupported Ours config keys: {sorted(unknown)}")
    base.METHODS["ours"].update(override)


def main():
    cli = parse_args()
    load_ours_config(cli.ours_config_json)
    rows = base.load_prompts(cli.prompt_json, cli.offset, cli.max_samples)
    if not rows:
        raise ValueError("no VBench prompts selected")

    args = build_480p_args(cli.checkpoints_dir)
    print("[480p VBench] Loading model ...", flush=True)
    pipe = base.InferencePipe(args)

    meta, templates = get_dynamic_resolution_meta(args.dynamic_scale_schedule, args.video_frames)
    template = templates[np.argmin(np.abs(templates - 0.571))]
    scale_schedule = meta[template][args.pn]["pt2scale_schedule"][(args.video_frames - 1) // 4 + 1]
    args.first_full_spatial_size_scale_index = get_first_full_spatial_size_scale_index(scale_schedule)
    args.tower_split_index = args.first_full_spatial_size_scale_index + 1
    context_info = pipe.get_scale_pack_info(
        scale_schedule,
        args.first_full_spatial_size_scale_index,
        args,
    )

    num_scales = len(scale_schedule)
    prune_ratios = {
        num_scales - 4: 0.80,
        num_scales - 3: 0.70,
        num_scales - 2: 0.60,
        num_scales - 1: 0.30,
    }
    tau_list = [args.tau_image] * args.tower_split_index + [args.tau_video] * (
        num_scales - args.tower_split_index
    )
    cfg_list = [args.cfg] * (num_scales - cli.bypass_last_n) + [1] * cli.bypass_last_n
    print(
        f"[480p VBench] scales={num_scales} prune_ratios={prune_ratios} prompts={len(rows)}",
        flush=True,
    )

    base.METHODS.setdefault("dense", {"scoring": "faststar", "variant": "dense_480p"})
    for method in cli.methods:
        if method == "dense":
            args.prune_ratios_per_scale = {}
            args.prune_keep_ratio = 1.0
            args.prune_token_threshold = 10**9
        else:
            args.prune_ratios_per_scale = dict(prune_ratios)
            args.prune_keep_ratio = 1.0
            args.prune_token_threshold = 200
        output_dir = osp.join(cli.output_root, f"vbench_{method}")
        base.generate_method(
            pipe,
            args,
            rows,
            method,
            output_dir,
            cfg_list,
            tau_list,
            scale_schedule,
            context_info,
            cli,
        )


if __name__ == "__main__":
    main()
