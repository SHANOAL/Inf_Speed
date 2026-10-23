# tools/ — FastSTAR/InfinityStar evaluation suite

Cleaned 2026-06-10. Old per-method scripts moved to `tools/_legacy/`. Old launchers moved to `launchers/_legacy/`. Code-modifying backups in `/backups/`.

## Layout

```
tools/
  eval/
    eval_10v.py        # unified 10-prompt evaluator (--methods baseline faststar v1 v3 v32 oracle v4)
    eval_vbench.py     # unified VBench evaluator (--methods ... --samples_per_dim N)
  run_infinity.py      # generation helpers (gen_one_example, save_video) — used by both evaluators
  infer_video_720p.py  # InferencePipe wrapper used by eval_*
  infer_video_480p.py  # 480p inference pipe (unused by current evals)
  infer_interact_480p.py
  prompt_rewriter.py
  save_dataset_features.py
  visualize_pruning_masks.py
  analyze_token_importance.py
  ablation_*.py        # CFG-bypass / temporal / per-frame ablations (kept, called rarely)
  _legacy/             # 34 deprecated one-off scripts kept for reference

launchers/
  run.sh                # bash launchers/run.sh {10v|vbench} --methods ... [--gpu auto|N]
  watch_and_launch.sh   # used by run.sh when --gpu auto: waits for >=70GB free GPU, retries on OOM
  _legacy/              # 8 old per-method launchers
```

## Quick start

10-prompt protocol (zero_fill, baseline + methods):
```
bash launchers/run.sh 10v --methods faststar v1 v3 --output_dir output/eval_10v_$(date +%s)
```

VBench protocol (160 prompts × N methods, ~3h per method):
```
bash launchers/run.sh vbench --methods faststar v3 --samples_per_dim 10
```

Pin to a specific GPU instead of auto-waiting:
```
bash launchers/run.sh 10v --methods faststar --gpu 6
```

## Methods

| `--methods` value | Description |
|---|---|
| `baseline` | full inference, no pruning (auto-run as reference for any other method) |
| `faststar` | STTP scoring (cross-scale + cross-clip dissim, L2 fusion) — arXiv:2603.07192 |
| `v1` | Structure (PCA) + Texture (high-pass) — IS the StepVAR paper (arXiv:2603.01757) |
| `v3` | Pooled text cosine — negative result, kept for ablation |
| `v32` | Cross-attention text guidance — broken since 2026-06-02, needs `mat_qkv→q_proj` fix |
| `v4` | v1 + cross-scale convergence — shape-mismatch bug, ≡ v1 currently |
| `oracle` | Probe-forward to rank by true per-token change — upper bound under zero_fill |

## Protocol invariants (both evaluators)

- 720p, 81 frames, cfg=34
- Last 4 scales pruned: keep ratios `{26:0.8, 27:0.7, 28:0.6, 29:0.3}`
- `prune_zero_fill=False`, `prune_recover_mode=pu` (≡ zero_fill in this FSQ codebook)
- `prune_last_iter_full=True` (reserve final iteration of multi-iter scales)

## Output layout

```
output/<eval_name>/
  baseline/
    videos/v00.mp4, v00.npy, ...
  <method_name>/
    videos/v00.mp4, v00_metrics.json, ...
    results.json     # {avg_PSNR, avg_SSIM, avg_LPIPS, per_video: [...]}
  sampled_prompts.json   # (vbench only)
```

## See also

- `/backups/` — snapshots of `infinity/models/{basic,infinity,st_token_pruning,tome}.py` and `baselines/faststar.py` before edits
- `tools/_legacy/eval_new_methods_unified.py` — original v1 / v4 / error_aware / hybrid_2full comparison (10v)
- `tools/_legacy/eval_merge_sparse.py` — token-merge variants (10v)
- `tools/_legacy/test_stepvar.py` — the run that established v1 = StepVAR
