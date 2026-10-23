#!/usr/bin/env python3
"""Generate full VBench videos for InfinityStar pruning methods.

Outputs VBench-standard prompt-named files, e.g. "<prompt_en>-0.mp4", so the
videos can be evaluated directly by tools/eval/run_vbench_matrix.py.
"""

import argparse
import json
import os
import os.path as osp
import shutil
import sys
import time
from collections import OrderedDict

import numpy as np
import torch
from tqdm import tqdm


REPO_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from infinity.schedules.dynamic_resolution import (  # noqa: E402
    get_dynamic_resolution_meta,
    get_first_full_spatial_size_scale_index,
)
from infinity.utils.arg_util import Args  # noqa: E402
from tools.infer_video_720p import InferencePipe  # noqa: E402
from tools.run_infinity import gen_one_example, save_video  # noqa: E402


METHODS = {
    "faststar": {"scoring": "faststar"},
    "stepvar": {"scoring": "stepvar"},
    "fastvar": {"scoring": "fastvar"},
    "sparsevar": {"scoring": "sparsevar"},
    "tome": {"scoring": "tome"},
    "ours": {
        "scoring": "attn",
        "variant": "frame_adaptive_text2",
        "prune_w_spatial": 1.0,
        "prune_w_temporal": 1.0,
        "prune_w_text": 2.0,
        "prune_attn_max_queries": 2048,
        "prune_per_frame": True,
    },
}


def save_json_atomic(obj, path):
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


def save_video_atomic(frames, fps, path):
    stem, extension = osp.splitext(path)
    tmp_path = f"{stem}.tmp.{os.getpid()}{extension}"
    try:
        save_video(frames, fps=fps, save_filepath=tmp_path)
        os.replace(tmp_path, path)
    finally:
        if osp.exists(tmp_path):
            os.remove(tmp_path)


def wait_for_output_space(path, min_free_gb):
    min_free_bytes = int(min_free_gb * 1024**3)
    while shutil.disk_usage(path).free < min_free_bytes:
        free_gb = shutil.disk_usage(path).free / 1024**3
        print(
            f"[disk] waiting for {min_free_gb:.1f} GiB free in {path}; "
            f"currently {free_gb:.1f} GiB",
            flush=True,
        )
        time.sleep(60)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", required=True, choices=sorted(METHODS))
    parser.add_argument(
        "--prompt_json",
        default=osp.join(REPO_ROOT, "evaluation", "VBench_rewrited_prompt.json"),
    )
    parser.add_argument("--output_root", default=osp.join(REPO_ROOT, "output"))
    parser.add_argument("--checkpoints_dir", default=osp.join(REPO_ROOT, "wieghts"))
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
        help="Prompt field used for generation. refined_prompt matches InfinityStar's paper protocol.",
    )
    parser.add_argument("--suffix", default="Close-up on big objects, emphasize scale and detail")
    return parser.parse_args()


def build_args(checkpoints_dir):
    args = Args()
    args.pn = "0.90M"
    args.fps = 16
    args.video_frames = 81
    args.model_path = osp.join(checkpoints_dir, "infinitystar_8b_720p_weights")
    args.checkpoint_type = "torch_shard"
    args.vae_path = osp.join(checkpoints_dir, "infinitystar_videovae.pth")
    args.text_encoder_ckpt = osp.join(checkpoints_dir, "text_encoder/flan-t5-xl-official/")
    args.model_type = "infinity_qwen8b"
    args.text_channels = 2048
    args.dynamic_scale_schedule = "infinity_elegant_clip20frames_v2"
    args.bf16 = 1
    args.use_apg = 1
    args.use_cfg = 0
    args.cfg = 34
    args.tau_image = 1
    args.tau_video = 0.4
    args.apg_norm_threshold = 0.05
    args.image_scale_repetition = "[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]"
    args.video_scale_repetition = "[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1, 1]"
    args.append_duration2caption = 1
    args.use_two_stage_lfq = 1
    args.detail_scale_min_tokens = 750
    args.semantic_scales = 12
    args.max_repeat_times = 10000
    args.enable_rewriter = 0

    args.prune_keep_ratio = 1.0
    args.prune_token_threshold = 200
    args.prune_scoring = "faststar"
    args.prune_last_iter_full = True
    args.prune_per_frame = False
    args.prune_fusion_p = 2.0
    args.prune_recover_mode = "pu"
    args.prune_recover_code_mask = None
    args.prune_recover_neighbors = 4
    args.prune_recover_alpha = 1.0
    args.prune_zero_fill = False
    args.prune_full_chunks_per_scale = None
    args.prune_full_tail_chunks = 0
    args.adaptive_cfg_bypass = False
    args.prune_w_str = 0.5
    args.prune_power_iter = 3
    args.prune_w_temp = 1.0
    args.prune_w_spatial = 1.0
    args.prune_w_temporal = 1.0
    args.prune_w_text = 1.0
    args.prune_attn_query_stride = None
    args.prune_attn_max_queries = None
    args.prune_attn_mask_text_padding = False
    args.prune_attn_text_pool = "max"
    args.prune_attn_text_topk = 4
    args.prune_attn_temporal_smooth = 0.0
    args.prune_attn_shared_fraction = 0.0
    args.prune_attn_text_query_gate = 0.0
    args.sparsevar_anchor_stride = 4
    args.tome_chunk_size = 1024
    args.prune_error_aware = False
    args.prune_mode = "prune"
    return args


def load_prompts(path, offset, max_samples):
    with open(path, "r", encoding="utf-8") as f:
        all_items = json.load(f)
    prompt_items = OrderedDict()
    for item in all_items:
        prompt_en = item.get("prompt_en")
        if prompt_en and prompt_en not in prompt_items:
            prompt_items[prompt_en] = item
    rows = list(prompt_items.values())
    if offset > 0:
        rows = rows[offset:]
    if max_samples > 0:
        rows = rows[:max_samples]
    return rows


def configure_method(args, method):
    cfg = METHODS[method]
    args.prune_scoring = cfg["scoring"]
    args.prune_w_spatial = cfg.get("prune_w_spatial", 1.0)
    args.prune_w_temporal = cfg.get("prune_w_temporal", 1.0)
    args.prune_w_text = cfg.get("prune_w_text", 1.0)
    args.prune_fusion_p = cfg.get("prune_fusion_p", 2.0)
    args.prune_attn_query_stride = cfg.get("prune_attn_query_stride")
    args.prune_attn_max_queries = cfg.get("prune_attn_max_queries")
    args.prune_per_frame = cfg.get("prune_per_frame", False)
    args.prune_attn_mask_text_padding = cfg.get("prune_attn_mask_text_padding", False)
    args.prune_attn_text_pool = cfg.get("prune_attn_text_pool", "max")
    args.prune_attn_text_topk = cfg.get("prune_attn_text_topk", 4)
    args.prune_attn_temporal_smooth = cfg.get("prune_attn_temporal_smooth", 0.0)
    args.prune_attn_shared_fraction = cfg.get("prune_attn_shared_fraction", 0.0)
    args.prune_attn_text_query_gate = cfg.get("prune_attn_text_query_gate", 0.0)
    args.prune_zero_fill = False
    args.prune_recover_mode = cfg.get("prune_recover_mode", "pu")
    args.prune_recover_code_mask = cfg.get("prune_recover_code_mask")
    args.prune_recover_neighbors = cfg.get("prune_recover_neighbors", 4)
    args.prune_recover_alpha = cfg.get("prune_recover_alpha", 1.0)
    args.prune_attn_temporal_mode = cfg.get("prune_attn_temporal_mode", "column")
    args.prune_attn_temporal_anchors = cfg.get("prune_attn_temporal_anchors", 1024)
    args.prune_attn_random_select = cfg.get("prune_attn_random_select", False)
    args.prune_ctx_pool = cfg.get("prune_ctx_pool", 0)
    args.sparstar = cfg.get("sparstar", 0)
    args.sparstar_density = cfg.get("sparstar_density", "")
    args.sparstar_block_size = cfg.get("sparstar_block_size", 128)
    args.prune_mode = "prune"


def generate_method(pipe, args, rows, method, output_dir, cfg_list, tau_list, scale_schedule, context_info, cli):
    configure_method(args, method)
    os.makedirs(output_dir, exist_ok=True)
    meta_path = osp.join(output_dir, "_generation_meta.json")
    if osp.exists(meta_path) and not cli.overwrite:
        with open(meta_path, "r", encoding="utf-8") as f:
            previous_meta = json.load(f)
        previous_prompt_field = previous_meta.get("prompt_field", "prompt_en")
        if previous_prompt_field != cli.prompt_field:
            raise RuntimeError(
                f"refusing to mix prompt protocols in {output_dir}: existing videos use "
                f"{previous_prompt_field!r}, requested {cli.prompt_field!r}. Use a new output "
                "directory or pass --overwrite."
            )
    meta = {
        "method": method,
        "config": METHODS[method],
        "num_prompts": len(rows),
        "seed_base": cli.seed,
        "prompt_json": osp.abspath(cli.prompt_json),
        "prompt_field": cli.prompt_field,
        "suffix": cli.suffix,
        "output_dir": osp.abspath(output_dir),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_json_atomic(meta, meta_path)

    generated = 0
    skipped = 0
    total_time = 0.0
    print(f"\n[{method}] output={output_dir}", flush=True)
    print(f"[{method}] total prompts={len(rows)} scoring={args.prune_scoring}", flush=True)

    for idx, item in enumerate(tqdm(rows, desc=f"{method}")):
        prompt_en = item["prompt_en"]
        video_path = osp.join(output_dir, f"{prompt_en}-0.mp4")
        if osp.exists(video_path) and not cli.overwrite:
            skipped += 1
            continue

        wait_for_output_space(output_dir, cli.min_free_gb)
        prompt_text = item.get(cli.prompt_field) or prompt_en
        gen_prompt = f"<<<t=5s>>>{prompt_text}, {cli.suffix}" if cli.suffix else f"<<<t=5s>>>{prompt_text}"
        # subsets may pin the seed used for this prompt in the full run
        seed = int(item["_seed"]) if "_seed" in item else cli.seed + idx

        torch.cuda.synchronize()
        t0 = time.time()
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True), torch.no_grad():
            img, _ = gen_one_example(
                pipe.infinity,
                pipe.vae,
                pipe.text_tokenizer,
                pipe.text_encoder,
                gen_prompt,
                cfg_list=cfg_list,
                tau_list=tau_list,
                scale_schedule=scale_schedule,
                g_seed=seed,
                cfg_insertion_layer=[0],
                vae_type=args.vae_type,
                sampling_per_bits=1,
                enable_positive_prompt=0,
                low_vram_mode=True,
                args=args,
                get_visual_rope_embeds=pipe.get_visual_rope_embeds,
                context_info=context_info,
            )
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        total_time += elapsed

        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        save_video_atomic(img.cpu().numpy(), fps=args.fps, path=video_path)
        generated += 1

        if generated % 25 == 0:
            avg = total_time / generated
            print(f"  [{method}] generated={generated} skipped={skipped} avg={avg:.1f}s", flush=True)

    done = {
        **meta,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "generated": generated,
        "skipped": skipped,
        "avg_time_s": total_time / generated if generated else None,
    }
    save_json_atomic(done, osp.join(output_dir, "_generation_done.json"))
    print(f"[{method}] done generated={generated} skipped={skipped}", flush=True)
    if generated:
        print(f"[{method}] avg_time={total_time / generated:.1f}s/video", flush=True)


def main():
    cli = parse_args()
    if cli.ours_config_json:
        with open(cli.ours_config_json, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        override = loaded.get("generator_config", loaded)
        allowed = {
            "scoring", "variant",
            "prune_w_spatial", "prune_w_temporal", "prune_w_text", "prune_fusion_p",
            "prune_attn_query_stride", "prune_attn_max_queries", "prune_per_frame",
            "prune_attn_mask_text_padding", "prune_attn_text_pool", "prune_attn_text_topk",
            "prune_attn_temporal_smooth", "prune_attn_shared_fraction",
            "prune_attn_text_query_gate",
            "prune_recover_mode", "prune_recover_code_mask",
            "prune_recover_neighbors", "prune_recover_alpha",
        }
        unknown = set(override) - allowed
        if unknown:
            raise ValueError(f"unsupported Ours config keys: {sorted(unknown)}")
        METHODS["ours"].update(override)
    rows = load_prompts(cli.prompt_json, cli.offset, cli.max_samples)

    args = build_args(cli.checkpoints_dir)
    print("[VBench generation] Loading model ...", flush=True)
    pipe = InferencePipe(args)

    meta, h_div_w_templates = get_dynamic_resolution_meta(args.dynamic_scale_schedule, args.video_frames)
    hwt = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - 0.571))]
    scale_schedule = meta[hwt][args.pn]["pt2scale_schedule"][(args.video_frames - 1) // 4 + 1]
    num_scales = len(scale_schedule)
    args.first_full_spatial_size_scale_index = get_first_full_spatial_size_scale_index(scale_schedule)
    args.tower_split_index = args.first_full_spatial_size_scale_index + 1
    context_info = pipe.get_scale_pack_info(scale_schedule, args.first_full_spatial_size_scale_index, args)

    args.prune_ratios_per_scale = {
        num_scales - 4: 1.0,
        num_scales - 3: 1.0,
        num_scales - 2: 1.0,
        num_scales - 1: 1.0,
    }
    tower_split = args.tower_split_index
    tau_list = [args.tau_image] * tower_split + [args.tau_video] * (num_scales - tower_split)
    cfg_list = [args.cfg] * (num_scales - cli.bypass_last_n) + [1] * cli.bypass_last_n

    print(f"[VBench generation] methods={cli.methods}", flush=True)
    print(f"[VBench generation] prompts={len(rows)} offset={cli.offset} max_samples={cli.max_samples}", flush=True)
    print(f"[VBench generation] prune_ratios={args.prune_ratios_per_scale}", flush=True)
    print(f"[VBench generation] cfg tail={cfg_list[-5:]}", flush=True)

    for method in cli.methods:
        out_dir = osp.join(cli.output_root, f"vbench_{method}")
        generate_method(pipe, args, rows, method, out_dir, cfg_list, tau_list, scale_schedule, context_info, cli)


if __name__ == "__main__":
    main()
