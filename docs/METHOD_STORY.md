# Method Story

## Core Principle

Preserve what the prompt needs, then propagate the updates that pruning skips.

The method has two components. Extra mask stabilization mechanisms are excluded
unless they produce a clear, balanced VBench gain on the confirmation subset.

## 1. Prompt-Aware Attention Importance

Previous token reduction methods often use feature variation, texture, or motion
as a proxy for importance. Those signals can favor detailed backgrounds while
missing the object named by the prompt.

Our score reuses the generator's own attention. Visual tokens are ranked by the
attention they receive, with prompt alignment controlling which visual queries
matter. Per-frame selection lets the important region follow a moving subject.

This gives one clean claim: token importance should follow generation intent,
not visual activity alone.

## 2. Residual Update Propagation

Pruning leaves non-selected tokens with stale cached features. Direct feature
interpolation can blur local structure. We instead interpolate the transformer
update from nearby selected tokens and add that update to each cached token.

This gives the second claim: propagate the update, not the feature value.

## Complexity Rule

The automatic optimizer uses the following policy:

- An added mechanism must improve confirmed VBench total by at least 0.001 per
  mechanism, without a quality or semantic regression larger than 0.001.
- A non-core simplification may replace the base when its total is within
  0.0005 and both quality and semantic scores are within 0.001.
- Removing a core component requires a clear total gain of at least 0.001.
- Among statistically equivalent candidates, choose the lower-complexity one.

Temporal smoothing and shared-mask mixing are therefore optional candidates,
not part of the core method. The current small search gain from combining both
is too small to justify adding them to the paper story.

## Paper Evidence

- Motivation heatmaps: prior scores highlight background texture, while the
  prompt-aware score highlights the named subject.
- Scoring ablation: remove prompt alignment and per-frame selection separately.
- Recovery ablation: compare cached partial update, direct interpolation, and
  residual update propagation.
- Main benchmark: report the final automatically selected simple configuration.
