"""FastVAR-style cached token pruning for InfinityStar.

FastVAR scores tokens by their mean-squared distance to the scale-wise token
mean, forwards the high-variance tokens, and restores skipped positions from a
cache aligned from the previous scale.
"""

import torch
import torch.nn.functional as F


def _align_cache(prev_feat, prev_pn, pn, B, C, device, dtype):
    if prev_feat is None:
        return None
    t, h, w = [int(v) for v in pn]
    L = t * h * w
    if prev_pn is None:
        return prev_feat if prev_feat.shape[1] == L else None

    t0, h0, w0 = [int(v) for v in prev_pn]
    if prev_feat.shape[1] != t0 * h0 * w0:
        return prev_feat if prev_feat.shape[1] == L else None

    x = prev_feat.to(device=device, dtype=dtype).view(B, t0, h0, w0, C)
    if (h0, w0) != (h, w):
        x = x.permute(0, 1, 4, 2, 3).reshape(B * t0, C, h0, w0)
        x = F.interpolate(x, size=(h, w), mode="area")
        x = x.reshape(B, t0, C, h, w).permute(0, 1, 3, 4, 2).contiguous()
    if t0 != t:
        x = x.reshape(B, t0, h * w * C).permute(0, 2, 1)
        x = F.interpolate(x, size=t, mode="linear", align_corners=False)
        x = x.permute(0, 2, 1).reshape(B, t, h, w, C)
    return x.reshape(B, L, C)


def fastvar_compute_importance(x, pn):
    """Return shared spatial importance [B, h*w] following FastVAR MSE scoring."""
    B, L, C = x.shape
    t, h, w = [int(v) for v in pn]
    hw = h * w
    xf = x.view(B, t, hw, C).float()
    mean = xf.mean(dim=2, keepdim=True)
    score = ((xf - mean) ** 2).mean(dim=-1)
    return score.mean(dim=1)


def fastvar_prune_tokens(x, rope_cache, pn, keep_ratio):
    B, L, C = x.shape
    t, h, w = [int(v) for v in pn]
    hw = h * w
    k = max(1, int(hw * keep_ratio))

    importance = fastvar_compute_importance(x, pn)
    _, topk_hw = importance.topk(k, dim=-1)
    topk_hw, _ = topk_hw.sort(dim=-1)

    offsets = torch.arange(t, device=x.device).unsqueeze(1) * hw
    flat_idx = (topk_hw.unsqueeze(1) + offsets.unsqueeze(0)).reshape(B, t * k)
    pruned_x = x.gather(1, flat_idx.unsqueeze(-1).expand(-1, -1, C))
    pruned_rope = rope_cache[:, :, :, :, flat_idx[0], :]
    return pruned_x, pruned_rope, flat_idx, topk_hw


def fastvar_cached_recover(pruned_output, flat_idx, base, prev_scale_cache=None, prev_pn=None, pn=None):
    """Restore skipped tokens from the previous-scale cache, then scatter updates."""
    B, L, C = base.shape
    cache = _align_cache(
        prev_scale_cache,
        prev_pn,
        pn,
        B,
        C,
        base.device,
        base.dtype,
    )
    if cache is None:
        cache = base
    result = cache.clone()
    idx = flat_idx.unsqueeze(-1).expand(-1, -1, C)
    result.scatter_(1, idx, pruned_output)
    return result
