"""
FastSTAR (arXiv:2603.07192) reproduction.

Spatiotemporal Token Pruning based on cross-scale spatial dissimilarity
and cross-clip temporal dissimilarity, with Partial Update.
Scoring uses raw cosine dissimilarity (no normalization) per paper.

Scoring:
  - Spatial: cosine dissimilarity between current input and previous scale/iteration output
  - Temporal: cosine dissimilarity between current clip and previous clip at same scale
  - Fusion: Lp-norm (p=2 by default)
"""

import torch
import torch.nn.functional as F


def _norm01(s):
    """Min-max normalize along last dimension to [0, 1]."""
    mn = s.min(dim=-1, keepdim=True).values
    mx = s.max(dim=-1, keepdim=True).values
    return (s - mn) / (mx - mn + 1e-8)


def _fs_align(prev_feat, prev_pn, B, t, h, w, hw, C, L):
    """Align previous feature [B, L_prev, C] (grid prev_pn=(t_p,h_p,w_p)) to the
    current (t,h,w) grid via spatial bilinear + temporal linear interpolation.
    Returns [B, t*hw, C] or None."""
    if prev_feat is None:
        return None
    if prev_pn is None:
        return prev_feat if prev_feat.shape[1] == L else None
    t_p, h_p, w_p = int(prev_pn[0]), int(prev_pn[1]), int(prev_pn[2])
    if t_p * h_p * w_p != prev_feat.shape[1]:
        return prev_feat if prev_feat.shape[1] == L else None
    pf = prev_feat.view(B, t_p, h_p, w_p, C).float()
    if (h_p, w_p) != (h, w):
        pf = pf.permute(0, 1, 4, 2, 3).reshape(B * t_p, C, h_p, w_p)
        pf = F.interpolate(pf, size=(h, w), mode='bilinear', align_corners=False)
        pf = pf.reshape(B, t_p, C, h, w).permute(0, 1, 3, 4, 2).contiguous()
    if t_p != t:
        pf = pf.reshape(B, t_p, hw * C).permute(0, 2, 1)
        pf = F.interpolate(pf, size=t, mode='linear', align_corners=False)
        pf = pf.permute(0, 2, 1).contiguous()
    return pf.reshape(B, t * hw, C)


def faststar_compute_importance(x, pn, prev_scale_feat=None, prev_clip_feat=None,
                                 fusion_p=2.0, prev_scale_pn=None, prev_clip_pn=None):
    """
    FastSTAR scoring: Lp fusion of spatial and temporal cosine dissimilarity.

    Args:
        x: [B, t*h*w, C] current transformer input features
        pn: (t, h, w) scale dimensions
        prev_scale_feat: [B, t*h*w, C] feature from previous scale/iteration
                         (for spatial convergence detection). None = skip.
        prev_clip_feat: [B, t*h*w, C] feature from previous clip at same scale
                        (for temporal motion detection). None = skip.
        fusion_p: Lp-norm degree (default 2)

    Returns:
        importance: [B, h*w] — higher = more important (needs update)
    """
    B, L, C = x.shape
    t, h, w = pn
    hw = h * w

    x_frames = x.view(B, t, hw, C).float()

    s_spatial = None
    s_temporal = None

    # Spatial dissimilarity: how much did this position change from previous scale?
    _ps = _fs_align(prev_scale_feat, prev_scale_pn, B, t, h, w, hw, C, L)
    if _ps is not None:
        prev_frames = _ps.view(B, t, hw, C).float()
        num = (x_frames * prev_frames).sum(dim=-1)
        den = (x_frames.norm(dim=-1) * prev_frames.norm(dim=-1)).clamp_min(1e-8)
        cos_sim = num / den                                    # [B, t, hw]
        s_spatial = (1.0 - cos_sim).mean(dim=1)                # [B, hw]

    # Temporal dissimilarity: how different is this position from previous clip?
    _pc = _fs_align(prev_clip_feat, prev_clip_pn, B, t, h, w, hw, C, L)
    if _pc is not None:
        prev_clip_frames = _pc.view(B, t, hw, C).float()
        num = (x_frames * prev_clip_frames).sum(dim=-1)
        den = (x_frames.norm(dim=-1) * prev_clip_frames.norm(dim=-1)).clamp_min(1e-8)
        cos_sim = num / den                                    # [B, t, hw]
        s_temporal = (1.0 - cos_sim).mean(dim=1)               # [B, hw]

    # Lp-norm fusion
    if s_spatial is not None and s_temporal is not None:
        importance = (s_spatial.pow(fusion_p) + s_temporal.pow(fusion_p)).pow(1.0 / fusion_p)
    elif s_spatial is not None:
        importance = s_spatial
    elif s_temporal is not None:
        importance = s_temporal
    else:
        importance = torch.ones(B, hw, device=x.device)

    return importance


def faststar_prune_tokens(x, rope_cache, pn, keep_ratio,
                           prev_scale_feat=None, prev_clip_feat=None,
                           fusion_p=2.0, prev_scale_pn=None, prev_clip_pn=None):
    """
    FastSTAR pruning: compute importance -> top-k selection -> prune.

    Args:
        x: [B, t*h*w, C]
        rope_cache: [2, 1, 1, 1, t*h*w, rope_dim]
        pn: (t, h, w)
        keep_ratio: float in (0, 1]
        prev_scale_feat: for spatial scoring
        prev_clip_feat: for temporal scoring
        fusion_p: Lp-norm degree

    Returns:
        pruned_x: [B, t*k, C]
        pruned_rope: pruned RoPE cache
        flat_indices: [B, t*k] indices in the full sequence
        topk_hw: [B, k] selected spatial positions
    """
    B, L, C = x.shape
    t, h, w = pn
    hw = h * w
    k = max(1, int(hw * keep_ratio))

    importance = faststar_compute_importance(
        x, pn,
        prev_scale_feat=prev_scale_feat,
        prev_clip_feat=prev_clip_feat,
        fusion_p=fusion_p,
        prev_scale_pn=prev_scale_pn,
        prev_clip_pn=prev_clip_pn,
    )

    # Shared mask: select top-k spatial positions shared across frames
    _, topk_hw = importance.topk(k, dim=-1)                    # [B, k]
    topk_hw, _ = topk_hw.sort(dim=-1)

    # Expand to all frames
    frame_offsets = torch.arange(t, device=x.device).unsqueeze(1) * hw
    flat_indices = (topk_hw.unsqueeze(1) + frame_offsets.unsqueeze(0)).reshape(B, t * k)

    pruned_x = x.gather(1, flat_indices.unsqueeze(-1).expand(-1, -1, C))

    idx_for_rope = flat_indices[0]
    pruned_rope = rope_cache[:, :, :, :, idx_for_rope, :]

    return pruned_x, pruned_rope, flat_indices, topk_hw


def faststar_partial_update(pruned_output, flat_indices, cache):
    """
    FastSTAR Partial Update: scatter pruned output into cache.
    Unselected positions retain cached values (equivalent to zero residual update).

    Args:
        pruned_output: [B, t*k, C]
        flat_indices: [B, t*k]
        cache: [B, t*h*w, C] previous state

    Returns:
        full_output: [B, t*h*w, C]
    """
    result = cache.clone()
    idx = flat_indices.unsqueeze(-1).expand(-1, -1, cache.shape[-1])
    result.scatter_(1, idx, pruned_output)
    return result


def faststar_create_code_mask(topk_hw, pn, target_pn, device):
    """
    Zero-fill mask for FastSTAR code accumulation.
    Selected positions = 1.0, non-selected = 0.0.
    Matches paper: f_k = Interpolate(r_hat_k) * M + f_{k-1}
    """
    B = topk_hw.shape[0]
    t = pn[0]
    h, w = pn[1], pn[2]
    if topk_hw.ndim == 3:
        if topk_hw.shape[1] != t:
            raise ValueError(
                f"per-frame mask shape {tuple(topk_hw.shape)} does not match pn={pn}"
            )
        mask = torch.zeros(B, t, h * w, device=device)
        mask.scatter_(2, topk_hw, 1.0)
        mask = mask.view(B, 1, t, h, w)
    else:
        mask = torch.zeros(B, h * w, device=device)
        mask.scatter_(1, topk_hw, 1.0)
        mask = mask.view(B, 1, 1, h, w)
        mask = mask.expand(-1, -1, t, -1, -1)

    target_t = target_pn[-3]
    target_h, target_w = target_pn[-2], target_pn[-1]
    if (t, h, w) != (target_t, target_h, target_w):
        mask = F.interpolate(mask, size=(target_t, target_h, target_w), mode='nearest')
    return mask
