# Benchmark: 3D Spatio-Temporal Token Merging for InfinityStar 720p
import sys, os, time
import os.path as osp
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from infinity.utils.arg_util import Args
# NOTE: compute_merged_dims was part of the old ToMe baseline code.
# Re-implement locally if this benchmark needs to be re-run.
def compute_merged_dims(t, h, w, ratio):
    """Estimate merged dimensions for 3D token merging."""
    import math
    total = t * h * w
    target = int(total * ratio)
    scale = (target / total) ** (1/3)
    return max(1, round(t * scale)), max(1, round(h * scale)), max(1, round(w * scale))
from tools.infer_video_720p import InferencePipe, perform_inference
from tools.run_infinity import save_video


def build_args(ckpt_dir, tome_target_ratio=1.0, tome_token_threshold=20000, tome_alpha=0.7):
    args = Args()
    args.pn = '0.90M'
    args.fps = 16
    args.video_frames = 81
    args.model_path = os.path.join(ckpt_dir, 'infinitystar_8b_720p_weights')
    args.checkpoint_type = 'torch_shard'
    args.vae_path = os.path.join(ckpt_dir, 'infinitystar_videovae.pth')
    args.text_encoder_ckpt = os.path.join(ckpt_dir, 'text_encoder/flan-t5-xl-official/')
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
    args.video_scale_repetition = '[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1, 1]'
    args.append_duration2caption = 1
    args.use_two_stage_lfq = 1
    args.detail_scale_min_tokens = 750
    args.semantic_scales = 12
    args.max_repeat_times = 10000
    args.enable_rewriter = 0
    args.tome_target_ratio = tome_target_ratio
    args.tome_token_threshold = tome_token_threshold
    args.tome_alpha = tome_alpha
    return args


def run_once(pipe, args, prompt, seed, tag, save_dir):
    data = {'seed': seed, 'prompt': prompt}
    # warmup
    result = perform_inference(pipe, data, args)
    print(f"[{tag}] warmup: {result['elapsed_time']:.2f}s")
    # timed
    result = perform_inference(pipe, data, args)
    elapsed = result['elapsed_time']
    print(f"[{tag}] timed:  {elapsed:.2f}s")
    out_path = osp.join(save_dir, f'{tag}.mp4')
    save_video(result['output'], fps=args.fps, save_filepath=out_path)
    print(f"[{tag}] saved:  {out_path}")
    return elapsed


if __name__ == '__main__':
    ckpt_dir = os.environ.get('INFSTAR_CKPT', './wieghts')
    save_dir = 'baselines/output/token_merging'
    os.makedirs(save_dir, exist_ok=True)

    prompt = "A golden retriever running on a sandy beach with ocean waves in the background, cinematic lighting, ultra-realistic"
    seed = 42

    # Preview merge dims for each affected scale
    print("=" * 60)
    print("3D MERGE DIMENSION PREVIEW")
    print("=" * 60)
    scales_to_check = [(20, 24, 43), (20, 30, 53), (20, 45, 80)]
    for t, h, w in scales_to_check:
        orig = t * h * w
        for ratio in [0.7, 0.5, 0.3]:
            t2, h2, w2 = compute_merged_dims(t, h, w, ratio)
            actual = t2 * h2 * w2
            print(f"  ({t},{h},{w})={orig:6d} -> ratio={ratio:.1f} -> ({t2},{h2},{w2})={actual:6d}  actual={actual/orig:.3f}")
    print()

    print("Loading model...")
    args_base = build_args(ckpt_dir, tome_target_ratio=1.0)
    pipe = InferencePipe(args_base)

    configs = [
        ('baseline',     1.0,  20000, 0.7),  # no merge
        ('r07_a07',      0.7,  20000, 0.7),  # keep 70%, blend 30% original
        ('r05_a07',      0.5,  20000, 0.7),  # keep 50%, blend 30% original
        ('r05_a10',      0.5,  20000, 1.0),  # keep 50%, no blend
        ('r07_t30k_a07', 0.7,  30000, 0.7),  # keep 70%, only scale 28-29
    ]

    results = []
    for tag, ratio, threshold, alpha in configs:
        print(f"\n{'='*60}")
        print(f"Config: {tag} (ratio={ratio}, threshold={threshold}, alpha={alpha})")
        print(f"{'='*60}")
        a = build_args(ckpt_dir, tome_target_ratio=ratio, tome_token_threshold=threshold, tome_alpha=alpha)
        t = run_once(pipe, a, prompt, seed, tag, save_dir)
        results.append((tag, ratio, threshold, alpha, t))

    # Summary
    t_base = results[0][4]
    print(f"\n{'='*60}")
    print("3D SPATIO-TEMPORAL TOKEN MERGING RESULTS")
    print(f"{'='*60}")
    print(f"{'Config':20s}  {'Time':>7s}  {'Speedup':>7s}  {'Saved':>6s}")
    print("-" * 50)
    for tag, ratio, threshold, alpha, t in results:
        speedup = t_base / t
        saving = (1 - t / t_base) * 100
        print(f"  {tag:18s}  {t:6.1f}s  {speedup:5.2f}x  {saving:+5.1f}%")
    print(f"\nVideos in: {save_dir}/")
