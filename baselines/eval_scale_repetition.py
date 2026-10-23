# Benchmark: compare original vs reduced scale_repetition on a single 720p video
import sys, os, time
import os.path as osp
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from infinity.utils.arg_util import Args
from tools.infer_video_720p import InferencePipe, perform_inference
from tools.run_infinity import save_video


def build_args(checkpoints_dir, video_scale_repetition_str):
    args = Args()
    args.pn = '0.90M'
    args.fps = 16
    args.video_frames = 81
    args.model_path = os.path.join(checkpoints_dir, 'infinitystar_8b_720p_weights')
    args.checkpoint_type = 'torch_shard'
    args.vae_path = os.path.join(checkpoints_dir, 'infinitystar_videovae.pth')
    args.text_encoder_ckpt = os.path.join(checkpoints_dir, 'text_encoder/flan-t5-xl-official/')
    args.model_type = 'infinity_qwen8b'
    args.text_channels = 2048
    args.dynamic_scale_schedule = 'infinity_elegant_clip20frames_v2'
    args.bf16 = 1
    args.use_apg = 1
    args.use_cfg = 0
    args.cfg = 34
    args.tau_image = 1
    args.tau_video = 0.4
    args.apg_norm_threshold = 0.05
    args.image_scale_repetition = '[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]'
    args.video_scale_repetition = video_scale_repetition_str
    args.append_duration2caption = 1
    args.use_two_stage_lfq = 1
    args.detail_scale_min_tokens = 750
    args.semantic_scales = 12
    args.max_repeat_times = 10000
    args.enable_rewriter = 0
    return args


def run_once(pipe, args, prompt, seed, tag, save_dir):
    """Run one inference, return elapsed seconds."""
    data = {'seed': seed, 'prompt': prompt}
    # warm up
    result = perform_inference(pipe, data, args)
    warmup_time = result['elapsed_time']
    print(f"[{tag}] warmup run: {warmup_time:.2f}s")

    # timed run
    result = perform_inference(pipe, data, args)
    elapsed = result['elapsed_time']
    print(f"[{tag}] timed run: {elapsed:.2f}s")

    out_path = osp.join(save_dir, f'{tag}.mp4')
    save_video(result['output'], fps=args.fps, save_filepath=out_path)
    print(f"[{tag}] saved to {out_path}")
    return elapsed


if __name__ == '__main__':
    checkpoints_dir = os.environ.get('INFSTAR_CKPT', './wieghts')
    save_dir = 'baselines/output/scale_repetition'
    os.makedirs(save_dir, exist_ok=True)

    prompt = "A golden retriever running on a sandy beach with ocean waves in the background, cinematic lighting, ultra-realistic"
    seed = 42

    # --- Config A: Original repetition ---
    rep_original = '[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1, 1]'
    # --- Config B: Reduced repetition for small scales ---
    rep_reduced  = '[1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 2, 1, 1]'

    import json
    orig_steps = sum(json.loads(rep_original))
    reduced_steps = sum(json.loads(rep_reduced))
    print(f"Original video_scale_repetition total steps:  {orig_steps}")
    print(f"Reduced  video_scale_repetition total steps:  {reduced_steps}")
    print(f"Theoretical step reduction: {orig_steps - reduced_steps} steps ({(1 - reduced_steps/orig_steps)*100:.1f}%)\n")

    # Load model once
    print("=" * 60)
    print("Loading model...")
    print("=" * 60)
    args_orig = build_args(checkpoints_dir, rep_original)
    pipe = InferencePipe(args_orig)

    # Run original
    print("\n" + "=" * 60)
    print(f"Running ORIGINAL: video_scale_repetition = {rep_original}")
    print("=" * 60)
    t_orig = run_once(pipe, args_orig, prompt, seed, 'original', save_dir)

    # Run reduced (reuse same pipe, just change args)
    print("\n" + "=" * 60)
    print(f"Running REDUCED: video_scale_repetition = {rep_reduced}")
    print("=" * 60)
    args_reduced = build_args(checkpoints_dir, rep_reduced)
    t_reduced = run_once(pipe, args_reduced, prompt, seed, 'reduced', save_dir)

    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Original  repetition: {rep_original}")
    print(f"Reduced   repetition: {rep_reduced}")
    print(f"Original  time:  {t_orig:.2f}s")
    print(f"Reduced   time:  {t_reduced:.2f}s")
    speedup = t_orig / t_reduced
    saving_pct = (1 - t_reduced / t_orig) * 100
    print(f"Speedup:  {speedup:.2f}x")
    print(f"Time saved: {saving_pct:.1f}%")
    print(f"\nVideos saved to: {save_dir}/")
    print(f"  original.mp4  -- full repetition")
    print(f"  reduced.mp4   -- reduced repetition")
    print(f"Compare them visually to check quality difference.")
