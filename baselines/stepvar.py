"""
StepVAR (arXiv:2506.02386) reproduction.

Structure-Texture Guided Pruning for Visual Autoregressive Models.
Scoring: PCA structural + 3x3 high-pass texture, spatial-only.
Recovery: Nearest Neighbor (NN) interpolation.
"""

import torch
import torch.nn.functional as F


def _norm01(s):
    mn = s.min(dim=-1, keepdim=True).values
    mx = s.max(dim=-1, keepdim=True).values
    return (s - mn) / (mx - mn + 1e-8)


def stepvar_compute_importance(x, pn, w_str=0.5, power_iter=3):
    """
    StepVAR scoring: PCA structural + high-pass texture, averaged over frames.

    Args:
        x: [B, t*h*w, C] input features
        pn: (t, h, w) scale dimensions
        w_str: weight for structural score (default 0.5)
        power_iter: PCA power iteration steps (default 3)

    Returns:
        importance: [B, h*w] higher = more important
    """
    B, L, C = x.shape
    t, h, w = pn
    hw = h * w

    x_frames = x.view(B, t, hw, C).float()

    # Structural: PCA via power iteration
    x_flat = x_frames.reshape(B, t * hw, C)
    x_centered = x_flat - x_flat.mean(dim=1, keepdim=True)
    v = torch.randn(B, C, 1, device=x.device, dtype=x_centered.dtype)
    v = v / (v.norm(dim=1, keepdim=True) + 1e-8)
    for _ in range(power_iter):
        Xv = torch.bmm(x_centered, v)
        v = torch.bmm(x_centered.transpose(1, 2), Xv)
        v = v / (v.norm(dim=1, keepdim=True) + 1e-8)
    proj = torch.bmm(x_centered, v).squeeze(-1).abs()
    S_str = _norm01(proj.view(B, t, hw).mean(dim=1))

    # Texture: 3x3 high-pass filter
    x_spatial = x_frames.reshape(B * t, hw, C).transpose(1, 2).reshape(B * t, C, h, w)
    x_padded = F.pad(x_spatial, (1, 1, 1, 1), mode='replicate')
    x_low = F.avg_pool2d(x_padded, kernel_size=3, stride=1, padding=0)
    x_high = x_spatial - x_low
    S_txt = _norm01((x_high ** 2).sum(dim=1).view(B, t, hw).mean(dim=1))

    importance = _norm01(w_str * S_str + S_txt)
    return importance


def stepvar_prune_tokens(x, rope_cache, pn, keep_ratio, w_str=0.5, power_iter=3):
    """
    StepVAR pruning: compute importance -> top-k -> prune.

    Returns:
        pruned_x: [B, t*k, C]
        pruned_rope: pruned RoPE cache
        flat_indices: [B, t*k]
        topk_hw: [B, k] selected spatial positions
    """
    B, L, C = x.shape
    t, h, w = pn
    hw = h * w
    k = max(1, int(hw * keep_ratio))

    importance = stepvar_compute_importance(x, pn, w_str=w_str, power_iter=power_iter)

    _, topk_hw = importance.topk(k, dim=-1)
    topk_hw, _ = topk_hw.sort(dim=-1)

    frame_offsets = torch.arange(t, device=x.device).unsqueeze(1) * hw
    flat_indices = (topk_hw.unsqueeze(1) + frame_offsets.unsqueeze(0)).reshape(B, t * k)

    pruned_x = x.gather(1, flat_indices.unsqueeze(-1).expand(-1, -1, C))
    pruned_rope = rope_cache[:, :, :, :, flat_indices[0], :]

    return pruned_x, pruned_rope, flat_indices, topk_hw


def stepvar_nn_recover(pruned_output, topk_hw, pn, device):
    """
    StepVAR NN recovery: fill non-selected positions from nearest selected neighbor.
    """
    B = pruned_output.shape[0]
    C = pruned_output.shape[-1]
    t, h, w = pn
    hw = h * w
    k = topk_hw.shape[1]

    kept_hw = topk_hw.to(device)
    if kept_hw.shape[0] == 1 and B > 1:
        kept_hw = kept_hw.expand(B, -1)
    kept_h = (kept_hw // w).float()
    kept_w = (kept_hw % w).float()

    all_pos = torch.arange(hw, device=device)
    all_h = (all_pos // w).float()
    all_w = (all_pos % w).float()

    dh = all_h.view(1, hw, 1) - kept_h.unsqueeze(1)
    dw = all_w.view(1, hw, 1) - kept_w.unsqueeze(1)
    nn_idx = (dh * dh + dw * dw).argmin(dim=2)

    pruned_frames = pruned_output.view(B, t, k, C)
    gather_idx = nn_idx.view(B, 1, hw, 1).expand(B, t, hw, C)
    full = pruned_frames.gather(2, gather_idx)
    return full.reshape(B, t * hw, C)
