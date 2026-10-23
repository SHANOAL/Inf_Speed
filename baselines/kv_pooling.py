"""
Variance-Guided Spatial KV Pooling baseline.

Only compresses KV cache (no FFN reduction), limited speedup.
Used as a comparison method.
"""

import torch
import torch.nn.functional as F


def compute_spatial_importance(key_states, pn):
    """Compute per-token spatial importance for KV pooling."""
    B, H, L, D = key_states.shape
    t, h, w = pn
    k_avg = key_states.mean(dim=1)
    k_spatial = k_avg.reshape(B, t, h, w, D)
    k_2d = k_spatial.reshape(B * t, h, w, D).permute(0, 3, 1, 2).float()
    k_padded = F.pad(k_2d, (1, 1, 1, 1), mode='replicate')
    local_mean = F.avg_pool2d(k_padded, kernel_size=3, stride=1, padding=0)
    local_var = ((k_2d - local_mean) ** 2).mean(dim=1)
    importance = local_var.reshape(B, t, h, w)
    frame_min = importance.reshape(B, t, -1).min(dim=-1, keepdim=True).values.unsqueeze(-1)
    frame_max = importance.reshape(B, t, -1).max(dim=-1, keepdim=True).values.unsqueeze(-1)
    importance = (importance - frame_min) / (frame_max - frame_min + 1e-8)
    return importance


def variance_guided_spatial_pool_kv(key_states, value_states, pn,
                                     keep_ratio=0.5, pool_kernel=2):
    """Variance-guided spatial-only KV pooling."""
    B, H, L, D = key_states.shape
    t, h, w = pn
    assert L == t * h * w
    importance = compute_spatial_importance(key_states, pn)
    n_spatial = h * w
    n_keep = max(4, int(n_spatial * keep_ratio))
    imp_flat = importance.reshape(B, t, n_spatial)
    _, top_indices = imp_flat.topk(n_keep, dim=-1, sorted=False)
    k_frames = key_states.reshape(B, H, t, h * w, D)
    v_frames = value_states.reshape(B, H, t, h * w, D)
    idx_exp = top_indices.unsqueeze(1).unsqueeze(-1).expand(B, H, t, n_keep, D)
    k_important = k_frames.gather(3, idx_exp)
    v_important = v_frames.gather(3, idx_exp)
    k_spatial = k_frames.reshape(B * H * t, h, w, D).permute(0, 3, 1, 2)
    v_spatial = v_frames.reshape(B * H * t, h, w, D).permute(0, 3, 1, 2)
    h_pool = (h + pool_kernel - 1) // pool_kernel
    w_pool = (w + pool_kernel - 1) // pool_kernel
    n_pooled = h_pool * w_pool
    k_pooled_spatial = F.adaptive_avg_pool2d(k_spatial, (h_pool, w_pool))
    v_pooled_spatial = F.adaptive_avg_pool2d(v_spatial, (h_pool, w_pool))
    k_pooled_spatial = k_pooled_spatial.permute(0, 2, 3, 1).reshape(B, H, t, n_pooled, D)
    v_pooled_spatial = v_pooled_spatial.permute(0, 2, 3, 1).reshape(B, H, t, n_pooled, D)
    k_merged = torch.cat([k_important, k_pooled_spatial], dim=3)
    v_merged = torch.cat([v_important, v_pooled_spatial], dim=3)
    L_new = t * (n_keep + n_pooled)
    k_out = k_merged.reshape(B, H, L_new, D)
    v_out = v_merged.reshape(B, H, L_new, D)
    return k_out, v_out


def should_pool_kv(pn, token_threshold=20000, min_hw=8):
    """Only pool scales with enough spatial tokens to benefit."""
    t, h, w = pn
    total = t * h * w
    return total >= token_threshold and h >= min_hw and w >= min_hw
