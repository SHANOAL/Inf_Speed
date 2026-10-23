"""
StageVAR baseline (arXiv:2512.16483) adapted for video.

Block-level L2 norm token selection + cache-fill RTR (Reduced Token Restoration).
Used as a comparison method.
"""

import torch


def stagevar_compute_indices(x, pn, keep_ratio):
    """
    Compute representative token indices via L2 norm importance (Frieze et al. 2004).
    P_i >= ||F^(i)||^2 / ||F||_F^2. We use deterministic top-k as approximation.

    Returns:
        topk_hw: [B, k] spatial indices (sorted)
        flat_indices: [B, t*k] full sequence indices
    """
    B, L, C = x.shape
    t, h, w = pn
    hw = h * w
    k = max(1, int(hw * keep_ratio))

    x_frames = x.view(B, t, hw, C)
    norms_sq = (x_frames ** 2).sum(dim=-1).mean(dim=1)  # [B, hw]
    _, topk_hw = norms_sq.topk(k, dim=-1)
    topk_hw, _ = topk_hw.sort(dim=-1)

    frame_offsets = torch.arange(t, device=x.device).unsqueeze(1) * hw
    flat_indices = (topk_hw.unsqueeze(1) + frame_offsets.unsqueeze(0)).reshape(B, t * k)
    return topk_hw, flat_indices


def stagevar_gather(x, flat_indices):
    """Gather tokens at flat_indices from x [B, L, C]."""
    return x.gather(1, flat_indices.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


def stagevar_scatter(selected, flat_indices, full_cached):
    """Scatter selected tokens into full_cached at flat_indices positions."""
    out = full_cached.clone()
    out.scatter_(1, flat_indices.unsqueeze(-1).expand(-1, -1, out.shape[-1]), selected)
    return out


def stagevar_gather_rope(rope_cache, flat_indices):
    """Gather RoPE entries at flat_indices. rope: [2,1,1,1,L,dim]."""
    return rope_cache[:, :, :, :, flat_indices[0], :]
