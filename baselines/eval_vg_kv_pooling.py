# Variance-Guided Spatial KV Pooling benchmark
# Usage: KEEP_RATIO=0.5 POOL_KERNEL=2 THRESHOLD=20000 TAG=k05_p2 python bench_vg_template.py
import sys, os, time
import os.path as osp
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from infinity.utils.arg_util import Args
from tools.infer_video_720p import InferencePipe, perform_inference
from tools.run_infinity import save_video

KEEP_RATIO = float(os.environ.get('KEEP_RATIO', '1.0'))
POOL_KERNEL = int(os.environ.get('POOL_KERNEL', '2'))
THRESHOLD = int(os.environ.get('THRESHOLD', '20000'))
TAG = os.environ.get('TAG', 'test')

def build_args(ckpt_dir):
    args = Args()
    args.pn = '0.90M'; args.fps = 16; args.video_frames = 81
    args.model_path = os.path.join(ckpt_dir, 'infinitystar_8b_720p_weights')
    args.checkpoint_type = 'torch_shard'
    args.vae_path = os.path.join(ckpt_dir, 'infinitystar_videovae.pth')
    args.text_encoder_ckpt = os.path.join(ckpt_dir, 'text_encoder/flan-t5-xl-official/')
    args.model_type = 'infinity_qwen8b'; args.text_channels = 2048
    args.dynamic_scale_schedule = 'infinity_elegant_clip20frames_v2'
    args.bf16 = 1; args.use_apg = 1; args.use_cfg = 0; args.cfg = 34
    args.tau_image = 1; args.tau_video = 0.4; args.apg_norm_threshold = 0.05
    args.image_scale_repetition = '[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]'
    args.video_scale_repetition = '[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1, 1]'
    args.append_duration2caption = 1; args.use_two_stage_lfq = 1
    args.detail_scale_min_tokens = 750; args.semantic_scales = 12
    args.max_repeat_times = 10000; args.enable_rewriter = 0
    args.kv_pool_keep_ratio = KEEP_RATIO
    args.kv_pool_pool_kernel = POOL_KERNEL
    args.kv_pool_threshold = THRESHOLD
    return args

if __name__ == '__main__':
    ckpt_dir = os.environ.get('INFSTAR_CKPT', './wieghts')
    save_dir = 'baselines/output/vg_kv_pooling'

    prompt = "A golden retriever running on a sandy beach with ocean waves in the background, cinematic lighting, ultra-realistic"
    seed = 42

    print(f"Config: keep_ratio={KEEP_RATIO}, pool_kernel={POOL_KERNEL}, threshold={THRESHOLD}, tag={TAG}")
    print("Loading model...")
    args = build_args(ckpt_dir)
    pipe = InferencePipe(args)

    data = {'seed': seed, 'prompt': prompt}
    # warmup
    result = perform_inference(pipe, data, args)
    print(f"[{TAG}] warmup: {result['elapsed_time']:.2f}s")
    # timed
    result = perform_inference(pipe, data, args)
    elapsed = result['elapsed_time']
    print(f"[{TAG}] timed:  {elapsed:.2f}s")
    out_path = osp.join(save_dir, f'{TAG}.mp4')
    save_video(result['output'], fps=args.fps, save_filepath=out_path)
    print(f"[{TAG}] saved:  {out_path}")
    print(f"[{TAG}] DONE. Time={elapsed:.2f}s")
