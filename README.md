# Inf_Speed: training-free acceleration for spacetime autoregressive video generation

Inf_Speed accelerates the late, token-heavy scales of next-scale (STAR) video generators such
as [InfinityStar](https://github.com/FoundationVision/InfinityStar) by computing only a subset
of tokens per frame and completing the rest. It needs no training and leaves the tokenizer,
text encoder, transformer weights and sampling schedule unchanged.

> Work in progress. Numbers below come from runs whose protocol is described in
> [Evaluation protocol](#evaluation-protocol); VBench scores for the current version are pending.

## Method

At each pruned scale (the final four scales, keeping 80/70/60/30 % of the tokens):

1. **Selection (IDS).** Every visual token is scored by how well it matches two reference sets,
   using the pretrained query/key projections of the first attention layer:
   - the prompt: max-similarity to the valid text keys;
   - the conditioning content: max-similarity to the keys of the most prompt-relevant tokens of
     the frames the clip is generated from (the first frame in T2V/I2V; the first frame and the
     last frames of the previous clip in interactive generation). Matching by content rather
     than by grid index lets the score follow moving objects.

   The two scores are min-max normalised per frame, fused with an l4 norm, and the top-k
   positions are kept independently in every frame. No L x L attention map is formed, so
   scoring costs about 1 % of generation time.
2. **Recovery (RC).** Selected positions keep the transformer output. Skipped positions keep
   their cached feature and receive the inverse-distance-weighted transformer update of the two
   nearest selected positions (`c_i + 0.5 * sum_j w_ij (z_j - c_j)`).
3. **Context tokens (experimental, `prune_ctx_pool`).** Skipped positions are average-pooled in
   `pool x pool` windows and processed alongside the selected tokens, then discarded, so that
   selected tokens still attend to a low-resolution summary of the skipped area.

Configurations live in [`configs/attnstar/`](configs/attnstar): `attnstar_v2` (current method),
`attnstar_v2_ctx4` (with context tokens), `paper_v0` (earlier version with spatial attention
scoring and query gating) and `random_rc` (random selection, same budget and recovery).

## Results so far

720p, 81 frames, 944 VBench prompts, InfinityStar-8B, pruning ratios (20,30,40,70)% on the final
four scales. Paired fidelity to the dense output generated with the same prompt and seed on the
same GPU type (H100).

| Method | PSNR | SSIM | LPIPS |
|---|---|---|---|
| **Inf_Speed (`attnstar_v2`)** | **29.257** | **0.8615** | **0.1067** |
| earlier version (`paper_v0`) | 28.874 | 0.8607 | 0.1105 |
| FastSTAR | 28.322 | 0.8489 | 0.1214 |
| StepVAR | 27.642 | 0.8277 | 0.1497 |
| FastVAR | 27.615 | 0.8351 | 0.1390 |

`attnstar_v2` minus FastSTAR: PSNR +0.94 dB (95 % CI [+0.89, +0.98], Wilcoxon p = 7e-150, better on
95 % of videos), LPIPS -0.0147.

Component study on a 63-prompt stride sample (11 VBench dimensions), PSNR to dense:

| Variant | dPSNR vs `paper_v0` |
|---|---|
| `attnstar_v2` | +0.36 [+0.17, +0.58] |
| random selection, same budget and RC | -0.71 [-0.86, -0.57] |

Keeping only the prompt term costs 0.41 dB and keeping only the conditioning term costs 0.27 dB;
per-frame selection is worth 0.14 dB; RC is worth 0.30 dB over reusing the
cache. Filling skipped positions with sampled or expected codes instead of zero updates lowers
fidelity in every setting we tried.

In progress: 480p T2V, 720p I2V (1,118 VBench-I2V examples), interactive long-video generation,
context tokens, speed/quality frontiers and VBench.

## Evaluation protocol

- Dense and accelerated videos use the same prompts, seeds, guidance and decoding settings.
  Seeds are pinned per prompt (`_seed` in the prompt json) so subsets and shards reproduce the
  full-run seeds.
- **Paired metrics must be computed on one GPU type.** Generation is bit-exact across GPUs of the
  same type, but dense videos sampled on A100 and H100 from the same seed diverge (PSNR 17-19 dB).
- Statistics: per-video differences with bootstrap 95 % confidence intervals and Wilcoxon
  signed-rank tests (`experiments/paired_stats.py`).

## Usage

Set up the InfinityStar environment (`requirements.txt`, Python 3.10, torch 2.5.1 + cu124,
flash-attn 2.7.4) and place the released weights under `./wieghts` (or point `INFSTAR_CKPT` at
them).

```bash
# accelerated and dense 720p T2V on a prompt file
python tools/eval/generate_vbench_methods.py --methods ours \
    --ours_config_json configs/attnstar/attnstar_v2.json \
    --prompt_json evaluation/VBench_rewrited_prompt.json --output_root out/attnstar_v2
python tools/eval/generate_dense_subset.py --methods ours \
    --ours_config_json configs/attnstar/paper_v0.json \
    --prompt_json evaluation/VBench_rewrited_prompt.json --output_root out/dense

# baselines: --methods faststar | stepvar | fastvar | sparsevar | tome
python tools/eval/generate_vbench_methods.py --methods faststar --output_root out/faststar

# paired fidelity and statistics
python tools/eval/eval_full_reference_metrics.py --reference_dir out/dense/vbench_ours \
    --method attnstar_v2=out/attnstar_v2/vbench_ours --output_root out/fidelity
python experiments/paired_stats.py ours=out/fidelity/attnstar_v2/results.json \
    faststar=out/fidelity/faststar/results.json
```

Other drivers: `generate_vbench_480p_methods.py`, `generate_vbench_i2v_methods.py`
(`--full_info`, `--image_dir`), `generate_interact_methods.py` (`--method`, `--config`,
`--seeds`, `--data`).

Tests (inside the project environment): `python tests/test_ids_anchor.py`,
`python tests/test_context_tokens.py`.

## Layout

```
infinity/models/st_token_pruning.py   IDS, RC, context tokens
infinity/models/infinity.py           pruning hooks in the InfinityStar inference loops
baselines/                            FastSTAR, StepVAR, FastVAR, SparseVAR, ToMe, StageVAR
tools/eval/                           generation and evaluation drivers
configs/attnstar/                     method configurations
experiments/                          statistics
tests/                                unit tests
```

## Acknowledgement

Built on [InfinityStar](https://github.com/FoundationVision/InfinityStar) (MIT License,
copyright FoundationVision); its original README is kept in
[`docs/INFINITYSTAR_README.md`](docs/INFINITYSTAR_README.md). Baselines re-implement FastVAR,
SparseVAR, StepVAR, FastSTAR and ToMe for the spacetime token grid.
