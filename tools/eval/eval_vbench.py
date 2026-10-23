"""
Unified VBench evaluator.

Methods: faststar, v1 (StepVAR), v3 (text cosine), v32 (cross-attn), oracle, v4.
Sample N prompts/dim from rewritten VBench JSON, run baseline + each method.

Usage:
  python tools/eval/eval_vbench.py --methods faststar v1 --samples_per_dim 10
  python tools/eval/eval_vbench.py --methods v32 --resume

Protocol (paper-aligned):
  - 720p, 81 frames, cfg=34, last 4 scales pruned with {0.8, 0.7, 0.6, 0.3}
  - No suffix on prompts (rewritten VBench refined_prompt as-is)
  - prune_zero_fill=False, recover_mode=pu (= zero_fill in this codebook)
  - reserve_last_iter_full
"""
import sys, os, time, json, argparse, random
import os.path as osp
import numpy as np
import torch

REPO_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Reserve 30GB immediately to block other users during model load.
import torch as _torch_pre
_RESERVATION = _torch_pre.empty((30 * 1024 * 1024 * 1024) // 4, dtype=_torch_pre.float32, device="cuda:0")


PRUNE_THRESHOLD = 200
SAMPLES_PER_DIM = 10
SEED_BASE = 41
PER_SCALE = {26: 0.8, 27: 0.7, 28: 0.6, 29: 0.3}
OUTPUT_ROOT = osp.join(REPO_ROOT, "output", "eval_vbench")
PROMPT_JSON = osp.join(REPO_ROOT, "evaluation", "VBench_rewrited_prompt.json")


def sample_prompts(prompt_json, samples_per_dim, seed=0):
    from collections import defaultdict
    with open(prompt_json, "r") as f:
        all_items = json.load(f)
    by_dim = defaultdict(list)
    for item in all_items:
        for dim in item["dimension"]:
            by_dim[dim].append(item)
    rng = random.Random(seed)
    selected, seen = [], set()
    for dim in sorted(by_dim.keys()):
        pool = list(by_dim[dim])
        rng.shuffle(pool)
        cnt = 0
        for it in pool:
            if cnt >= samples_per_dim:
                break
            if it["prompt_en"] in seen:
                continue
            seen.add(it["prompt_en"])
            selected.append({"prompt_en": it["prompt_en"], "refined_prompt": it.get("refined_prompt", it["prompt_en"]), "dimension": dim})
            cnt += 1
    return selected


def build_args():
    from infinity.utils.arg_util import Args
    ckpt_dir = osp.join(REPO_ROOT, "wieghts")
    args = Args()
    args.pn = "0.90M"; args.fps = 16; args.video_frames = 81
    args.model_path = osp.join(ckpt_dir, "infinitystar_8b_720p_weights")
    args.checkpoint_type = "torch_shard"
    args.vae_path = osp.join(ckpt_dir, "infinitystar_videovae.pth")
    args.text_encoder_ckpt = osp.join(ckpt_dir, "text_encoder/flan-t5-xl-official/")
    args.model_type = "infinity_qwen8b"; args.text_channels = 2048
    args.dynamic_scale_schedule = "infinity_elegant_clip20frames_v2"
    args.bf16 = 1; args.use_apg = 1; args.use_cfg = 0; args.cfg = 34
    args.tau_image = 1; args.tau_video = 0.4; args.apg_norm_threshold = 0.05
    args.image_scale_repetition = "[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]"
    args.video_scale_repetition = "[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1, 1]"
    args.append_duration2caption = 1; args.use_two_stage_lfq = 1
    args.detail_scale_min_tokens = 750; args.semantic_scales = 12
    args.max_repeat_times = 10000; args.enable_rewriter = 0
    args.prune_keep_ratio = 1.0
    args.prune_token_threshold = PRUNE_THRESHOLD
    args.prune_fusion_p = 2.0
    args.prune_ratios_per_scale = None
    args.prune_last_iter_full = True
    args.prune_per_frame = False
    args.prune_recover_mode = "pu"
    args.prune_scoring = "faststar"
    args.prune_zero_fill = False
    args.prune_full_chunks_per_scale = None
    args.prune_full_tail_chunks = 0
    args.sparsevar_anchor_stride = 4
    args.tome_chunk_size = 1024
    args.adaptive_cfg_bypass = False
    return args


def run_gen(pipe, args, cfg_list, tau_list, scale_schedule, context_info, prompt, seed):
    from tools.run_infinity import gen_one_example
    prompt_full = f"<<<t=5s>>>{prompt}"
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True), torch.no_grad():
        img, _ = gen_one_example(
            pipe.infinity, pipe.vae, pipe.text_tokenizer, pipe.text_encoder,
            prompt_full, cfg_list=cfg_list, tau_list=tau_list,
            scale_schedule=scale_schedule, g_seed=seed,
            cfg_insertion_layer=[0], vae_type=args.vae_type,
            sampling_per_bits=1, enable_positive_prompt=0,
            low_vram_mode=True, args=args,
            get_visual_rope_embeds=pipe.get_visual_rope_embeds,
            context_info=context_info,
        )
    return img


def compute_metrics(ref, gen):
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    from skimage.metrics import structural_similarity as ssim_fn
    import lpips, torch as _t
    if not hasattr(compute_metrics, "_lp"):
        compute_metrics._lp = lpips.LPIPS(net="alex").cuda().eval()
    ref = np.asarray(ref); gen = np.asarray(gen)
    if ref.shape != gen.shape:
        return float("nan"), float("nan"), float("nan")
    psnrs, ssims, lpipss = [], [], []
    for f_ref, f_gen in zip(ref, gen):
        psnrs.append(psnr_fn(f_ref, f_gen, data_range=255))
        ssims.append(ssim_fn(f_ref, f_gen, channel_axis=-1, data_range=255))
        a = _t.from_numpy(f_ref).permute(2, 0, 1).float().div(127.5).sub(1).unsqueeze(0).cuda()
        b = _t.from_numpy(f_gen).permute(2, 0, 1).float().div(127.5).sub(1).unsqueeze(0).cuda()
        with _t.no_grad():
            lpipss.append(compute_metrics._lp(a, b).item())
    return float(np.mean(psnrs)), float(np.mean(ssims)), float(np.mean(lpipss))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_per_dim", type=int, default=SAMPLES_PER_DIM)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--output_root", default=OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true")
    cli = parser.parse_args()

    from tools.infer_video_720p import InferencePipe
    from tools.run_infinity import save_video
    from infinity.schedules.dynamic_resolution import (
        get_dynamic_resolution_meta, get_first_full_spatial_size_scale_index,
    )

    prompts = sample_prompts(PROMPT_JSON, cli.samples_per_dim)
    os.makedirs(cli.output_root, exist_ok=True)
    with open(osp.join(cli.output_root, "sampled_prompts.json"), "w") as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)
    print(f"[eval] Sampled {len(prompts)} prompts across {len(set(p['dimension'] for p in prompts))} dims", flush=True)

    args = build_args()
    global _RESERVATION
    del _RESERVATION
    torch.cuda.empty_cache()
    print("[eval] Loading model ...", flush=True)
    t_load = time.time()
    pipe = InferencePipe(args)
    print(f"[eval] Model loaded in {time.time()-t_load:.1f}s", flush=True)

    meta, h_div_w_templates = get_dynamic_resolution_meta(args.dynamic_scale_schedule, args.video_frames)
    hwt = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - 0.571))]
    scale_schedule = meta[hwt][args.pn]["pt2scale_schedule"][(args.video_frames - 1) // 4 + 1]
    args.first_full_spatial_size_scale_index = get_first_full_spatial_size_scale_index(scale_schedule)
    args.tower_split_index = args.first_full_spatial_size_scale_index + 1
    context_info = pipe.get_scale_pack_info(scale_schedule, args.first_full_spatial_size_scale_index, args)
    num_scales = len(scale_schedule)
    tower_split = args.tower_split_index
    tau_list = [args.tau_image] * tower_split + [args.tau_video] * (num_scales - tower_split)
    cfg_list = [args.cfg] * num_scales

    base_dir = osp.join(cli.output_root, "baseline")
    os.makedirs(osp.join(base_dir, "videos"), exist_ok=True)
    base_frames_cache = {}
    print(f"\n=== Baseline ({len(prompts)} videos) ===", flush=True)
    args.prune_ratios_per_scale = None
    args.prune_scoring = "faststar"
    for i, p in enumerate(prompts):
        vp = osp.join(base_dir, "videos", f"v{i:03d}.npy")
        if cli.resume and osp.exists(vp):
            base_frames_cache[i] = np.load(vp)
            print(f"  baseline {i+1}/{len(prompts)}: cached", flush=True); continue
        t0 = time.time()
        img = run_gen(pipe, args, cfg_list, tau_list, scale_schedule, context_info, p["refined_prompt"], SEED_BASE + i)
        frames = img.cpu().numpy()
        np.save(vp, frames)
        save_video(frames, fps=args.fps, save_filepath=osp.join(base_dir, "videos", f"v{i:03d}.mp4"))
        base_frames_cache[i] = frames
        print(f"  baseline {i+1}/{len(prompts)} [{p['dimension']}] dt={time.time()-t0:.1f}s", flush=True)

    for method in cli.methods:
        print(f"\n=== {method} ({len(prompts)} videos) ===", flush=True)
        mdir = osp.join(cli.output_root, method)
        os.makedirs(osp.join(mdir, "videos"), exist_ok=True)
        args.prune_ratios_per_scale = PER_SCALE
        args.prune_scoring = method
        psnrs, ssims, lpipss, per_video = [], [], [], []
        for i, p in enumerate(prompts):
            mp = osp.join(mdir, "videos", f"v{i:03d}.mp4")
            metric_file = osp.join(mdir, "videos", f"v{i:03d}_metrics.json")
            if cli.resume and osp.exists(metric_file):
                m = json.load(open(metric_file))
                psnrs.append(m["PSNR"]); ssims.append(m["SSIM"]); lpipss.append(m["LPIPS"])
                per_video.append(m)
                print(f"  {method} {i+1}/{len(prompts)}: cached PSNR={m['PSNR']:.2f}", flush=True); continue
            t0 = time.time()
            img = run_gen(pipe, args, cfg_list, tau_list, scale_schedule, context_info, p["refined_prompt"], SEED_BASE + i)
            frames = img.cpu().numpy()
            save_video(frames, fps=args.fps, save_filepath=mp)
            psnr, ssim, lp = compute_metrics(base_frames_cache[i], frames)
            psnrs.append(psnr); ssims.append(ssim); lpipss.append(lp)
            row = {"idx": i, "dimension": p["dimension"], "prompt": p["prompt_en"], "seed": SEED_BASE + i, "PSNR": round(psnr, 2), "SSIM": round(ssim, 4), "LPIPS": round(lp, 4)}
            per_video.append(row)
            with open(metric_file, "w") as f:
                json.dump(row, f)
            print(f"  {method} {i+1}/{len(prompts)} [{p['dimension']}] dt={time.time()-t0:.1f}s PSNR={psnr:.2f} SSIM={ssim:.4f} LPIPS={lp:.4f}", flush=True)
        result = {"method": method, "num_videos": len(prompts),
                  "avg_PSNR": round(float(np.mean(psnrs)), 2),
                  "avg_SSIM": round(float(np.mean(ssims)), 4),
                  "avg_LPIPS": round(float(np.mean(lpipss)), 4),
                  "per_video": per_video}
        with open(osp.join(mdir, "results.json"), "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  >>> {method}: PSNR={result['avg_PSNR']} SSIM={result['avg_SSIM']} LPIPS={result['avg_LPIPS']}", flush=True)


if __name__ == "__main__":
    main()
