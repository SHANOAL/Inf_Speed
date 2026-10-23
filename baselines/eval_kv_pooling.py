"""
KV Pooling benchmark - single config runner.
Usage: CUDA_VISIBLE_DEVICES=X python baselines/bench_kvpool.py --tag TAG --ratio R --threshold T
"""
import sys, os, time, argparse
import os.path as osp
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from infinity.utils.arg_util import Args
from tools.infer_video_720p import InferencePipe, perform_inference
from tools.run_infinity import save_video


def build_args(ckpt_dir, kv_pool_ratio=1.0, kv_pool_threshold=20000):
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
    args.kv_pool_ratio = kv_pool_ratio
    args.kv_pool_threshold = kv_pool_threshold
    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', type=str, required=True)
    parser.add_argument('--ratio', type=float, default=1.0)
    parser.add_argument('--threshold', type=int, default=20000)
    cli = parser.parse_args()

    ckpt_dir = os.environ.get('INFSTAR_CKPT', './wieghts')
    save_dir = 'baselines/output/kv_pooling'
    os.makedirs(save_dir, exist_ok=True)

    prompt = "A golden retriever running on a sandy beach with ocean waves in the background, cinematic lighting, ultra-realistic"
    seed = 42

    print(f"[{cli.tag}] Loading model... (ratio={cli.ratio}, threshold={cli.threshold})")
    args = build_args(ckpt_dir, kv_pool_ratio=cli.ratio, kv_pool_threshold=cli.threshold)
    pipe = InferencePipe(args)

    data = {'seed': seed, 'prompt': prompt}

    # warmup
    result = perform_inference(pipe, data, args)
    print(f"[{cli.tag}] warmup: {result['elapsed_time']:.2f}s")

    # timed run
    result = perform_inference(pipe, data, args)
    elapsed = result['elapsed_time']
    print(f"[{cli.tag}] timed:  {elapsed:.2f}s")

    out_path = osp.join(save_dir, f'{cli.tag}.mp4')
    save_video(result['output'], fps=args.fps, save_filepath=out_path)
    print(f"[{cli.tag}] saved:  {out_path}")

    # Write result to file for collection
    result_path = osp.join(save_dir, f'{cli.tag}_result.txt')
    with open(result_path, 'w') as f:
        f.write(f"{cli.tag},{cli.ratio},{cli.threshold},{elapsed:.2f}\n")
    print(f"[{cli.tag}] DONE. Time={elapsed:.2f}s")
