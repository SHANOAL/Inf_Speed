"""
StageVAR on InfinityStar: standalone benchmark.

StageVAR (arXiv:2512.16483) adapted for video:
  - CFG=0 for late scales (only unconditional branch)
  - L2 norm importance token selection
  - Cache-fill restoration (RTR)
  - Per-scale keep ratios on last 4 scales

Runs baseline + StageVAR, 10 videos, computes PSNR/SSIM/LPIPS.

Usage:
  CUDA_VISIBLE_DEVICES=3 python baselines/bench_stagevar.py
"""
import sys, os, time, json
import os.path as osp
import numpy as np
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

SAVE_DIR = 'baselines/output/stagevar'
os.makedirs(SAVE_DIR, exist_ok=True)

NUM_VIDEOS = 10
SEED_BASE = 42
PRUNE_THRESHOLD = 200

PROMPTS = [
    "A golden retriever running on a sandy beach with ocean waves in the background, cinematic lighting, ultra-realistic",
    "A red sports car drives fast along a coastal mountain road at sunset, waves crashing on rocks below, cinematic wide shot",
    "A cat sitting on a windowsill watching rain fall outside, cozy indoor lighting, close-up",
    "An astronaut floating in space with Earth in the background, cinematic, high detail",
    "A busy Tokyo street at night with neon signs reflecting on wet pavement, slow motion",
    "A butterfly landing on a colorful flower in a garden, macro shot, natural lighting",
    "A steam train crossing a stone bridge over a misty valley, aerial cinematic shot",
    "A chef preparing sushi in a traditional Japanese restaurant, close-up on hands",
    "A surfer riding a massive wave at sunset, slow motion, golden hour lighting",
    "A snowy mountain landscape with northern lights in the sky, time lapse, 4K",
]


def build_args():
    from infinity.utils.arg_util import Args
    ckpt_dir = os.environ.get('INFSTAR_CKPT', './wieghts')
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
    args.prune_w_str = 0.5; args.prune_power_iter = 3
    args.prune_w_temp = 1.0; args.prune_fusion_p = 2.0
    args.prune_ratios_per_scale = None
    args.prune_last_iter_full = False
    args.stagevar_config = None
    return args


def run_gen(pipe, args, cfg_list, tau_list, scale_schedule, context_info, prompt, seed):
    from tools.run_infinity import gen_one_example
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True), \
         torch.no_grad():
        img, _ = gen_one_example(
            pipe.infinity, pipe.vae, pipe.text_tokenizer, pipe.text_encoder,
            prompt, cfg_list=cfg_list, tau_list=tau_list,
            scale_schedule=scale_schedule, g_seed=seed,
            cfg_insertion_layer=[0], vae_type=args.vae_type,
            sampling_per_bits=1, enable_positive_prompt=0,
            low_vram_mode=True, args=args,
            get_visual_rope_embeds=pipe.get_visual_rope_embeds,
            context_info=context_info,
        )
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
    return img


def compute_metrics(frames_ref, frames_test):
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    import lpips

    n = min(len(frames_ref), len(frames_test))
    frames_ref, frames_test = frames_ref[:n], frames_test[:n]

    psnr_vals, ssim_vals = [], []
    for i in range(n):
        psnr_vals.append(peak_signal_noise_ratio(frames_ref[i], frames_test[i], data_range=255))
        ssim_vals.append(structural_similarity(
            frames_ref[i], frames_test[i], channel_axis=2, data_range=255))

    lpips_fn = lpips.LPIPS(net="alex").eval().cuda()
    lpips_vals = []
    for i in range(n):
        ref_t = torch.from_numpy(frames_ref[i]).permute(2, 0, 1).float().unsqueeze(0).cuda() / 127.5 - 1.0
        tst_t = torch.from_numpy(frames_test[i]).permute(2, 0, 1).float().unsqueeze(0).cuda() / 127.5 - 1.0
        with torch.no_grad():
            lpips_vals.append(lpips_fn(ref_t, tst_t).item())
    del lpips_fn
    torch.cuda.empty_cache()

    return {
        "PSNR": float(np.mean(psnr_vals)),
        "SSIM": float(np.mean(ssim_vals)),
        "LPIPS": float(np.mean(lpips_vals)),
    }


def main():
    from tools.infer_video_720p import InferencePipe
    from tools.run_infinity import save_video
    from infinity.schedules.dynamic_resolution import (
        get_dynamic_resolution_meta, get_first_full_spatial_size_scale_index,
    )

    args = build_args()
    print("[stagevar] Loading model ...")
    t_load = time.time()
    pipe = InferencePipe(args)
    print(f"[stagevar] Model loaded in {time.time() - t_load:.1f}s")

    meta, h_div_w_templates = get_dynamic_resolution_meta(
        args.dynamic_scale_schedule, args.video_frames)
    hwt = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - 0.571))]
    scale_schedule = meta[hwt][args.pn]["pt2scale_schedule"][
        (args.video_frames - 1) // 4 + 1]
    args.first_full_spatial_size_scale_index = get_first_full_spatial_size_scale_index(scale_schedule)
    args.tower_split_index = args.first_full_spatial_size_scale_index + 1
    context_info = pipe.get_scale_pack_info(
        scale_schedule, args.first_full_spatial_size_scale_index, args)
    num_scales = len(scale_schedule)
    tower_split = args.tower_split_index
    tau_list = ([args.tau_image] * tower_split +
                [args.tau_video] * (num_scales - tower_split))

    print(f"[stagevar] num_scales={num_scales}, GPU={torch.cuda.get_device_name(0)}")
    print(f"[stagevar] Last 4 scales:")
    for si in range(num_scales - 4, num_scales):
        pn = scale_schedule[si]
        print(f"  scale {si}: ({pn[0]},{pn[1]},{pn[2]}) = {pn[0]*pn[1]*pn[2]} tokens")

    # StageVAR config (matching paper): α={0.96, 0, 0} for last 3 scales
    # α=0.96 → keep ~17.6% rank → keep_ratio=0.176
    # α=0   → skip scale entirely
    sv_config = {
        num_scales - 3: 0.176,   # α=0.96 → 17.6% rank, block-level RP+RTR
        num_scales - 2: 0.0,     # α=0 → skip entirely
        num_scales - 1: 0.0,     # α=0 → skip entirely
    }
    cfg_list_full = [args.cfg] * num_scales

    print(f"[stagevar] StageVAR config (α-mapped): {sv_config}")
    print(f"[stagevar] Last iter full-token: True")

    # ================================================================
    # Phase 1: Baseline (10 videos)
    # ================================================================
    print(f"\n{'='*70}")
    print(f"PHASE 1: Generating {NUM_VIDEOS} baseline videos")
    print(f"{'='*70}")

    args.stagevar_config = None
    args.prune_ratios_per_scale = None
    args.prune_last_iter_full = False

    baseline_dir = osp.join(SAVE_DIR, "baseline")
    os.makedirs(baseline_dir, exist_ok=True)

    prompt0 = f"<<<t=5s>>>{PROMPTS[0]}, Close-up on big objects, emphasize scale and detail"

    print("  warmup ...")
    _ = run_gen(pipe, args, cfg_list_full, tau_list, scale_schedule,
                context_info, prompt0, SEED_BASE)
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    t0 = time.time()
    _ = run_gen(pipe, args, cfg_list_full, tau_list, scale_schedule,
                context_info, prompt0, SEED_BASE)
    torch.cuda.synchronize()
    baseline_time = time.time() - t0
    print(f"  Baseline latency: {baseline_time:.2f}s")

    baseline_frames_list = []
    for vi in range(NUM_VIDEOS):
        vpath = osp.join(baseline_dir, f"video_{vi:02d}.mp4")
        prompt_i = f"<<<t=5s>>>{PROMPTS[vi]}, Close-up on big objects, emphasize scale and detail"
        seed_i = SEED_BASE + vi
        print(f"  baseline {vi+1}/{NUM_VIDEOS}: {PROMPTS[vi][:50]}...")
        img = run_gen(pipe, args, cfg_list_full, tau_list, scale_schedule,
                      context_info, prompt_i, seed_i)
        frames = img.cpu().numpy()
        save_video(frames, fps=args.fps, save_filepath=vpath)
        baseline_frames_list.append(frames)

    # ================================================================
    # Phase 2: StageVAR (10 videos)
    # ================================================================
    print(f"\n{'='*70}")
    print(f"PHASE 2: Generating {NUM_VIDEOS} StageVAR videos")
    print(f"  CFG=0 (uncond only) + L2 norm selection on last 4 scales")
    print(f"  per-scale keep = {sv_config}")
    print(f"{'='*70}")

    args.stagevar_config = sv_config
    args.prune_last_iter_full = True
    args.prune_token_threshold = PRUNE_THRESHOLD

    accel_dir = osp.join(SAVE_DIR, "stagevar")
    os.makedirs(accel_dir, exist_ok=True)

    print("  warmup ...")
    _ = run_gen(pipe, args, cfg_list_full, tau_list, scale_schedule,
                context_info, prompt0, SEED_BASE)
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    t0 = time.time()
    _ = run_gen(pipe, args, cfg_list_full, tau_list, scale_schedule,
                context_info, prompt0, SEED_BASE)
    torch.cuda.synchronize()
    accel_time = time.time() - t0
    speedup = baseline_time / accel_time
    print(f"  StageVAR latency: {accel_time:.2f}s  (speedup={speedup:.2f}x)")

    accel_frames_list = []
    for vi in range(NUM_VIDEOS):
        vpath = osp.join(accel_dir, f"video_{vi:02d}.mp4")
        prompt_i = f"<<<t=5s>>>{PROMPTS[vi]}, Close-up on big objects, emphasize scale and detail"
        seed_i = SEED_BASE + vi
        print(f"  stagevar {vi+1}/{NUM_VIDEOS}: {PROMPTS[vi][:50]}...")
        img = run_gen(pipe, args, cfg_list_full, tau_list, scale_schedule,
                      context_info, prompt_i, seed_i)
        frames = img.cpu().numpy()
        save_video(frames, fps=args.fps, save_filepath=vpath)
        accel_frames_list.append(frames)

    # ================================================================
    # Phase 3: Compute metrics
    # ================================================================
    print(f"\n{'='*70}")
    print(f"PHASE 3: Computing PSNR / SSIM / LPIPS")
    print(f"{'='*70}")

    all_psnr, all_ssim, all_lpips = [], [], []
    for vi in range(NUM_VIDEOS):
        print(f"  metrics {vi+1}/{NUM_VIDEOS} ...")
        m = compute_metrics(baseline_frames_list[vi], accel_frames_list[vi])
        all_psnr.append(m["PSNR"])
        all_ssim.append(m["SSIM"])
        all_lpips.append(m["LPIPS"])
        print(f"    PSNR={m['PSNR']:.2f}  SSIM={m['SSIM']:.4f}  LPIPS={m['LPIPS']:.4f}")

    avg_psnr = float(np.mean(all_psnr))
    avg_ssim = float(np.mean(all_ssim))
    avg_lpips = float(np.mean(all_lpips))

    # ================================================================
    # Results
    # ================================================================
    results = {
        "method": "StageVAR (adapted for video)",
        "gpu": torch.cuda.get_device_name(0),
        "resolution": "720p", "duration": "5s", "frames": 81,
        "num_videos": NUM_VIDEOS,
        "baseline_latency_s": round(baseline_time, 2),
        "stagevar_latency_s": round(accel_time, 2),
        "speedup": round(speedup, 2),
        "stagevar_config": {str(k): v for k, v in sv_config.items()},
        "avg_PSNR": round(avg_psnr, 2),
        "avg_SSIM": round(avg_ssim, 4),
        "avg_LPIPS": round(avg_lpips, 4),
        "per_video": [
            {"idx": vi, "prompt": PROMPTS[vi], "seed": SEED_BASE + vi,
             "PSNR": round(all_psnr[vi], 2),
             "SSIM": round(all_ssim[vi], 4),
             "LPIPS": round(all_lpips[vi], 4)}
            for vi in range(NUM_VIDEOS)
        ],
    }
    json_path = osp.join(SAVE_DIR, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print("RESULTS: StageVAR on InfinityStar (Video)")
    print(f"{'='*80}")
    print(f"GPU: {results['gpu']}")
    print(f"Task: T2V 720p 5s (81 frames), {NUM_VIDEOS} videos averaged")
    print(f"StageVAR: CFG=0 + block-level L2 norm RP + cache-fill RTR + scale skip")
    print(f"Per-scale config: {sv_config} (0=skip)")
    print(f"{'-'*80}")
    print(f"{'Method':<30} {'Latency':>8} {'Speedup':>8} {'PSNR↑':>7} {'SSIM↑':>7} {'LPIPS↓':>7}")
    print(f"{'-'*80}")
    print(f"{'Baseline (InfinityStar)':<30} {baseline_time:>7.2f}s {'1.00x':>8} {'ref':>7} {'ref':>7} {'ref':>7}")
    print(f"{'StageVAR (video)':<30} {accel_time:>7.2f}s {speedup:>7.2f}x "
          f"{avg_psnr:>7.2f} {avg_ssim:>7.4f} {avg_lpips:>7.4f}")
    print(f"{'='*80}")
    print(f"\nResults saved: {json_path}")
    print(f"Videos: {baseline_dir}/  &  {accel_dir}/")


if __name__ == "__main__":
    main()
