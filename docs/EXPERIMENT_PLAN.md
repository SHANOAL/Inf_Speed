# Experiment Plan

## Protocol Freeze

- Primary paper protocol: rewritten VBench prompts from `refined_prompt`.
- Secondary protocol: raw `prompt_en`, reported separately with its own dense baseline.
- T2V main setting: InfinityStar 8B, 720p, 5 seconds, 81 frames, seed base 41.
- Main benchmark: 944 unique VBench prompts and all 16 official dimensions.
- Official aggregate: normalize every dimension with VBench constants, then compute Quality, Semantic, and Total with the official 4:1 Quality-to-Semantic weighting.
- Never compare raw 16-dimension means with paper-reported VBench Total scores.

## P0: Dense Baseline Calibration

Input videos: `logs/vbench_videos_720p`

Output: `output/vbench_baseline_eval_matrix`

Acceptance criteria:

- Exactly 944 prompt-named videos are present.
- All 16 dimensions finish without failures.
- Reproduced Quality, Semantic, and Total are checked against the paper values 84.73, 79.78, and 83.74.
- Any material deviation is investigated before accelerated results enter the paper table.

## P1: 720p T2V Main Table

Methods:

- Dense InfinityStar
- SparseVAR
- FastVAR
- ToMe
- FastSTAR
- StepVAR
- Ours

Each accelerated method must use the same refined prompts, seed mapping, suffix, resolution, frame count, pruning ratios, and VBench evaluator as the dense baseline.

Required outputs per method:

- `output/vbench_<method>/_generation_meta.json`
- `output/vbench_<method>/_generation_done.json`
- 944 MP4 files
- `output/vbench_<method>_eval_matrix/<method>/vbench_scores.json`
- `output/vbench_<method>_eval_matrix/summary.csv`

## P2: Efficiency Table

Use a fixed prompt set and one A100 80GB. Run one warmup followed by measured samples.

Report:

- Transformer latency
- End-to-end latency including text encoding and VAE decoding
- Peak allocated GPU memory
- Speedup relative to dense InfinityStar
- Output resolution, duration, frame count, and pruning ratios

## P3: Paper Ablations

Screen candidates on a deterministic dimension-balanced VBench subset, then expand only stable winners.

Required ablations:

1. Hyperparameters: pruning ratios and selected scoring weights.
2. Scoring module: spatial, temporal, text, per-frame selection, and prompt-guided gating.
3. CFG bypass: disabled versus the selected bypass policy.
4. Recovery: partial update, FastSTAR recovery, and proposed residual interpolation recovery.

Stage A uses 3 prompts per dimension. Stage B uses 10 prompts per dimension. The final paper table uses the largest feasible protocol and clearly states the sample count.

## P4: Method Exploration

First-stage candidates:

- Current Ours
- Temporal smoothing at 0.25
- Shared-token fraction at 0.125
- Temporal smoothing plus shared tokens
- One-neighbor residual recovery
- Two-neighbor residual recovery with alpha 0.25

Promotion rule:

- No failed VBench dimensions.
- Improvement is not concentrated in only one prompt.
- Total score improves at both subset stages.
- Quality and Semantic trade-offs are reported, not hidden.
- A promoted method must be confirmed on the full protocol before replacing Ours.

## P5: 720p I2V

Prerequisites:

- Download the official VBench-I2V image suite.
- Implement a unified generation script for dense and accelerated methods.
- Implement official I2V Quality, I2V, and Total aggregation.
- Run a small smoke test before full generation.

The full official suite requests five samples per image-prompt pair. Any reduced-sample protocol must be labeled explicitly and applied to every method, including the dense baseline.

## P6: 480p T2V and I2V

Use the released 480p checkpoint and a separate protocol manifest. Do not mix its results with 720p outputs. Prioritize 480p I2V because the released checkpoint documentation recommends I2V over T2V.

## P7: T2I

GenEval and DPG require separate inference and evaluation entrypoints. Add them only after the video main table is stable. Record the prompt rewriter policy and report rewritten and non-rewritten results separately when applicable.

## Resource Priority

1. Dense baseline calibration
2. Refined 720p T2V main table
3. Main-table VBench evaluation
4. Targeted ablations and exploration
5. 720p I2V
6. 480p tasks
7. Raw-prompt secondary protocol
8. T2I supplements

Exploration must never preempt a running main-table job.
