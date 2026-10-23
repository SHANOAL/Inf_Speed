"""
Spatio-Temporal Token Pruning for InfinityStar.

Prune spatial tokens BEFORE transformer blocks -> reduces both attention AND FFN compute.
Supports two modes:
  - Shared mask (default): same spatial positions pruned across all frames
  - Per-frame adaptive mask: each frame independently selects top-k positions
Nearest-neighbor spatial recovery AFTER blocks for full-resolution logits.
RoPE cache pruned to match token positions.

Scoring versions:
  v1 (original):  Spatial PCA + High-pass texture + Temporal cosine dissimilarity
  v2 (AGST):      v1 + Feature Activation Energy + Cross-iteration Convergence
"""

import torch
import torch.nn.functional as F


def _norm01(s):
    """Min-max normalize along last dimension to [0, 1]."""
    mn = s.min(dim=-1, keepdim=True).values
    mx = s.max(dim=-1, keepdim=True).values
    return (s - mn) / (mx - mn + 1e-8)


# ---------------------------------------------------------------------------
# v1: original scoring (kept for backward compat / ablation)
# ---------------------------------------------------------------------------

def compute_token_importance(x, pn, w_str=0.5, power_iter=3,
                              w_temp=1.0, fusion_p=2.0,
                              return_components=False,
                              per_frame=False):
    """
    Spatio-Temporal Guided Token Importance (v1).

    Args:
        x: [B, t*h*w, C] input features
        pn: (t, h, w) scale dimensions
        w_str:    weight on structural PCA score inside spatial fusion (default 0.5)
        power_iter: number of power-iteration steps for PCA (3-5 suffices)
        w_temp:   weight of temporal branch in lp-norm fusion (default 1.0)
        fusion_p: lp-norm degree for spatial/temporal fusion (default 2, per FastSTAR)
        return_components: if True, also return individual score maps
        per_frame: if True, return per-frame importance [B, t, h*w]
                   instead of frame-averaged [B, h*w]
    Returns:
        importance: [B, t, h*w] if per_frame else [B, h*w]
        (optional) dict with component score maps
    """
    B, L, C = x.shape
    t, h, w = pn
    hw = h * w

    x_frames = x.view(B, t, hw, C).float()

    # -------- Spatial 1: Structural score via PCA (power iteration) --------
    x_flat = x_frames.reshape(B, t * hw, C)
    x_centered = x_flat - x_flat.mean(dim=1, keepdim=True)
    v = torch.randn(B, C, 1, device=x.device, dtype=x_centered.dtype)
    v = v / (v.norm(dim=1, keepdim=True) + 1e-8)
    for _ in range(power_iter):
        Xv = torch.bmm(x_centered, v)
        v = torch.bmm(x_centered.transpose(1, 2), Xv)
        v = v / (v.norm(dim=1, keepdim=True) + 1e-8)
    proj = torch.bmm(x_centered, v).squeeze(-1).abs()
    S_str_full = proj.view(B, t, hw)                       # [B, t, hw]

    # -------- Spatial 2: Textural score via 3x3 high-pass filter --------
    x_spatial = x_frames.reshape(B * t, hw, C).transpose(1, 2).reshape(B * t, C, h, w)
    x_padded = F.pad(x_spatial, (1, 1, 1, 1), mode='replicate')
    x_low = F.avg_pool2d(x_padded, kernel_size=3, stride=1, padding=0)
    x_high = x_spatial - x_low
    S_txt_full = (x_high ** 2).sum(dim=1).view(B, t, hw)   # [B, t, hw]

    # -------- Temporal: cosine dissimilarity between adjacent frames --------
    cos_pair = None
    if t >= 2:
        f_prev = x_frames[:, :-1, :, :]
        f_cur  = x_frames[:,  1:, :, :]
        num = (f_prev * f_cur).sum(dim=-1)
        den = (f_prev.norm(dim=-1) * f_cur.norm(dim=-1)).clamp_min(1e-8)
        cos_pair = num / den                                # [B, t-1, hw]
        dissim = 1.0 - cos_pair                             # [B, t-1, hw]

    # -------- Branch: shared mask vs per-frame mask --------
    if per_frame:
        S_str = _norm01(S_str_full)                         # [B, t, hw]
        S_txt = _norm01(S_txt_full)                         # [B, t, hw]

        if t >= 2:
            S_tmp_pf = torch.zeros(B, t, hw, device=x.device, dtype=x_frames.dtype)
            S_tmp_pf[:, 0, :] = dissim[:, 0, :]
            if t > 2:
                S_tmp_pf[:, 1:-1, :] = 0.5 * (dissim[:, :-1, :] + dissim[:, 1:, :])
            S_tmp_pf[:, -1, :] = dissim[:, -1, :]
            S_tmp = _norm01(S_tmp_pf)                       # [B, t, hw]
        else:
            S_tmp = torch.zeros(B, t, hw, device=x.device, dtype=x_frames.dtype)

        S_spatial = _norm01(w_str * S_str + S_txt)          # [B, t, hw]

        if t >= 2 and w_temp > 0.0:
            importance = (S_spatial.pow(fusion_p) + w_temp * S_tmp.pow(fusion_p)).pow(1.0 / fusion_p)
        else:
            importance = S_spatial
    else:
        S_str = _norm01(S_str_full.mean(dim=1))             # [B, hw]
        S_txt = _norm01(S_txt_full.mean(dim=1))             # [B, hw]

        if t >= 2:
            S_tmp = _norm01(dissim.mean(dim=1))             # [B, hw]
        else:
            S_tmp = torch.zeros(B, hw, device=x.device, dtype=x_frames.dtype)

        S_spatial = _norm01(w_str * S_str + S_txt)          # [B, hw]

        if t >= 2 and w_temp > 0.0:
            importance = (S_spatial.pow(fusion_p) + w_temp * S_tmp.pow(fusion_p)).pow(1.0 / fusion_p)
        else:
            importance = S_spatial

    if return_components:
        comps = {
            'S_str': S_str,
            'S_txt': S_txt,
            'S_tmp': S_tmp,
            'S_spatial': S_spatial,
            'S_total': importance,
        }
        if t >= 2:
            comps['cos_dissim_per_frame'] = (1.0 - cos_pair)
        return importance, comps
    return importance


# ---------------------------------------------------------------------------
# v2: AGST — Attention-Guided Spatiotemporal Token Scoring
# ---------------------------------------------------------------------------

def _compute_spatial_scores(x_frames, pn, w_str, power_iter):
    """Shared spatial scoring for both v1 and v2.
    Returns S_str_full [B,t,hw], S_txt_full [B,t,hw] (unnormalized).
    """
    B, t, hw, C = x_frames.shape
    h, w = pn[1], pn[2]

    x_flat = x_frames.reshape(B, t * hw, C)
    x_centered = x_flat - x_flat.mean(dim=1, keepdim=True)
    v = torch.randn(B, C, 1, device=x_frames.device, dtype=x_centered.dtype)
    v = v / (v.norm(dim=1, keepdim=True) + 1e-8)
    for _ in range(power_iter):
        Xv = torch.bmm(x_centered, v)
        v = torch.bmm(x_centered.transpose(1, 2), Xv)
        v = v / (v.norm(dim=1, keepdim=True) + 1e-8)
    S_str_full = torch.bmm(x_centered, v).squeeze(-1).abs().view(B, t, hw)

    x_spatial = x_frames.reshape(B * t, hw, C).transpose(1, 2).reshape(B * t, C, h, w)
    x_padded = F.pad(x_spatial, (1, 1, 1, 1), mode='replicate')
    x_low = F.avg_pool2d(x_padded, kernel_size=3, stride=1, padding=0)
    x_high = x_spatial - x_low
    S_txt_full = (x_high ** 2).sum(dim=1).view(B, t, hw)

    return S_str_full, S_txt_full


def _compute_temporal_scores(x_frames):
    """Shared temporal scoring: per-pair cosine dissimilarity.
    Returns dissim [B, t-1, hw] or None if single frame.
    """
    B, t, hw, C = x_frames.shape
    if t < 2:
        return None
    f_prev = x_frames[:, :-1]
    f_cur  = x_frames[:, 1:]
    num = (f_prev * f_cur).sum(dim=-1)
    den = (f_prev.norm(dim=-1) * f_cur.norm(dim=-1)).clamp_min(1e-8)
    return 1.0 - num / den


def compute_token_importance_v2(
    x, pn,
    pu_cache=None,
    w_str=0.5, power_iter=3,
    w_temp=1.0, w_act=0.5, w_conv=1.0,
    fusion_p=2.0,
    per_frame=False,
    return_components=False,
):
    """
    AGST: Attention-Guided Spatiotemporal Token Scoring (v2).

    Four scoring axes fused via weighted Lp-norm:
      Axis 1 — Feature Activation Energy:  ||x_token||_2
      Axis 2 — Cross-Iteration Convergence: ||x_current - pu_cache||_2
      Axis 3 — Spatial Complexity: PCA structural + high-pass textural
      Axis 4 — Per-Frame Temporal Motion: cosine dissimilarity

    Args:
        x:  [B, t*h*w, C] input features (before transformer)
        pn: (t, h, w)
        pu_cache: [B, t*h*w, C] or None — previous iteration PU output
        w_str:   structural PCA weight inside spatial fusion
        power_iter: PCA power iterations
        w_temp:  temporal axis weight in Lp fusion
        w_act:   activation energy weight
        w_conv:  convergence weight (ignored when pu_cache is None)
        fusion_p: Lp-norm degree
        per_frame: per-frame adaptive mask
        return_components: return individual score maps
    Returns:
        importance: [B, t, hw] if per_frame else [B, hw]
        (optional) dict of component maps
    """
    B, L, C = x.shape
    t, h, w = pn
    hw = h * w

    x_frames = x.view(B, t, hw, C).float()

    # --- Axis 1: Feature Activation Energy ---
    E_act_full = x_frames.norm(dim=-1)                      # [B, t, hw]

    # --- Axis 2: Cross-Iteration Convergence ---
    has_conv = pu_cache is not None
    if has_conv:
        cache_frames = pu_cache.view(B, t, hw, C).float()
        E_conv_full = (x_frames - cache_frames).norm(dim=-1)  # [B, t, hw]

    # --- Axis 3: Spatial Complexity ---
    S_str_full, S_txt_full = _compute_spatial_scores(x_frames, pn, w_str, power_iter)

    # --- Axis 4: Temporal Motion ---
    dissim = _compute_temporal_scores(x_frames)               # [B, t-1, hw] or None

    # --- Fusion ---
    if per_frame:
        S_str = _norm01(S_str_full)
        S_txt = _norm01(S_txt_full)
        E_act = _norm01(E_act_full)

        if dissim is not None:
            S_tmp_pf = torch.zeros(B, t, hw, device=x.device, dtype=x_frames.dtype)
            S_tmp_pf[:, 0] = dissim[:, 0]
            if t > 2:
                S_tmp_pf[:, 1:-1] = 0.5 * (dissim[:, :-1] + dissim[:, 1:])
            S_tmp_pf[:, -1] = dissim[:, -1]
            S_tmp = _norm01(S_tmp_pf)
        else:
            S_tmp = torch.zeros(B, t, hw, device=x.device, dtype=x_frames.dtype)

        S_spatial = _norm01(w_str * S_str + S_txt)

        terms = S_spatial.pow(fusion_p)
        if w_temp > 0 and dissim is not None:
            terms = terms + w_temp * S_tmp.pow(fusion_p)
        if w_act > 0:
            terms = terms + w_act * E_act.pow(fusion_p)
        if has_conv and w_conv > 0:
            E_conv = _norm01(E_conv_full)
            terms = terms + w_conv * E_conv.pow(fusion_p)

        importance = terms.pow(1.0 / fusion_p)
    else:
        S_str = _norm01(S_str_full.mean(dim=1))
        S_txt = _norm01(S_txt_full.mean(dim=1))
        E_act = _norm01(E_act_full.mean(dim=1))

        if dissim is not None:
            S_tmp = _norm01(dissim.mean(dim=1))
        else:
            S_tmp = torch.zeros(B, hw, device=x.device, dtype=x_frames.dtype)

        S_spatial = _norm01(w_str * S_str + S_txt)

        terms = S_spatial.pow(fusion_p)
        if w_temp > 0 and dissim is not None:
            terms = terms + w_temp * S_tmp.pow(fusion_p)
        if w_act > 0:
            terms = terms + w_act * E_act.pow(fusion_p)
        if has_conv and w_conv > 0:
            E_conv = _norm01(E_conv_full.mean(dim=1))
            terms = terms + w_conv * E_conv.pow(fusion_p)

        importance = terms.pow(1.0 / fusion_p)

    if return_components:
        comps = {
            'S_str': S_str, 'S_txt': S_txt, 'S_tmp': S_tmp,
            'S_spatial': S_spatial,
            'E_act': E_act,
            'S_total': importance,
        }
        if has_conv:
            comps['E_conv'] = _norm01(E_conv_full.mean(dim=1)) if not per_frame else _norm01(E_conv_full)
        if dissim is not None:
            comps['cos_dissim_per_frame'] = dissim
        return importance, comps
    return importance


def prune_tokens(x, rope_cache, pn, keep_ratio,
                 w_str=0.5, power_iter=3, w_temp=1.0, fusion_p=2.0,
                 per_frame=False,
                 scoring='v1', pu_cache=None,
                 w_act=0.5, w_conv=1.0,
                 text_emb=None,
                 prev_scale_feat=None,
                 prune_counts=None):
    """
    Prune spatial tokens before transformer blocks.

    Args:
        x: [B, t*h*w, C]
        rope_cache: [2, 1, 1, 1, t*h*w, rope_dim]
        pn: (t, h, w)
        keep_ratio: float in (0, 1]
        per_frame: if True, each frame selects its own top-k positions
        scoring: 'v1' (original) or 'v2' (AGST)
        pu_cache: [B, t*h*w, C] for v2 convergence scoring (optional)
        w_act:   v2 activation energy weight
        w_conv:  v2 convergence weight
    Returns:
        pruned_x, pruned_rope, flat_indices, topk_hw
    """
    B, L, C = x.shape
    t, h, w = pn
    hw = h * w
    k = max(1, int(hw * keep_ratio))

    if scoring == 'v3':
        importance = compute_token_importance_v3(
            x, pn, text_emb=text_emb,
            w_temp=w_temp, fusion_p=fusion_p, per_frame=per_frame,
        )
    elif scoring == 'v2':
        importance = compute_token_importance_v2(
            x, pn, pu_cache=pu_cache,
            w_str=w_str, power_iter=power_iter,
            w_temp=w_temp, w_act=w_act, w_conv=w_conv,
            fusion_p=fusion_p, per_frame=per_frame,
        )
    elif scoring == 'v4':
        importance = compute_token_importance_v4(
            x, pn, prev_scale_feat=prev_scale_feat,
            w_str=w_str, power_iter=power_iter,
            w_temp=w_temp, w_conv=w_conv,
            fusion_p=fusion_p, per_frame=per_frame,
        )
    elif scoring == 'diff':
        importance = compute_diffusion_importance(
            x, pn, per_frame=per_frame,
        )
    else:
        importance = compute_token_importance(
            x, pn, w_str=w_str, power_iter=power_iter,
            w_temp=w_temp, fusion_p=fusion_p, per_frame=per_frame,
        )

    # v5: error-accumulation boost
    if prune_counts is not None:
        importance = apply_error_accumulation_boost(importance, prune_counts, pn)

    if per_frame:
        _, topk_hw = importance.topk(k, dim=-1)             # [B, t, k]
        topk_hw, _ = topk_hw.sort(dim=-1)
        frame_offsets = torch.arange(t, device=x.device).view(1, t, 1) * hw
        flat_indices = (topk_hw + frame_offsets).reshape(B, t * k)
    else:
        _, topk_hw = importance.topk(k, dim=-1)             # [B, k]
        topk_hw, _ = topk_hw.sort(dim=-1)
        frame_offsets = torch.arange(t, device=x.device).unsqueeze(1) * hw
        flat_indices = (topk_hw.unsqueeze(1) + frame_offsets.unsqueeze(0)).reshape(B, t * k)

    pruned_x = x.gather(1, flat_indices.unsqueeze(-1).expand(-1, -1, C))

    idx_for_rope = flat_indices[0]
    pruned_rope = rope_cache[:, :, :, :, idx_for_rope, :]

    return pruned_x, pruned_rope, flat_indices, topk_hw


def recover_tokens(pruned_output, topk_hw, pn, device, per_frame=False):
    """
    Recover full token set via nearest-neighbor spatial propagation.

    Args:
        pruned_output: [B, t*k, C]
        topk_hw: [B, k] (shared) or [B, t, k] (per_frame)
        pn: (t, h, w)
        device: torch device
        per_frame: if True, each frame has its own kept positions
    Returns:
        full_output: [B, t*h*w, C]
    """
    B, _, C = pruned_output.shape
    t, h, w = pn
    hw = h * w

    all_positions = torch.arange(hw, device=device)
    all_h_coords = all_positions // w
    all_w_coords = all_positions % w

    if per_frame:
        k = topk_hw.shape[2]
        pruned_per_frame = pruned_output.view(B, t, k, C)
        full_frames = []
        for fi in range(t):
            kept_hw = topk_hw[0, fi]                        # [k]
            kept_h = kept_hw // w
            kept_w = kept_hw % w
            dh = all_h_coords.unsqueeze(1) - kept_h.unsqueeze(0)
            dw = all_w_coords.unsqueeze(1) - kept_w.unsqueeze(0)
            dist = dh * dh + dw * dw
            nn_idx = dist.argmin(dim=1)                     # [hw]
            full_frames.append(pruned_per_frame[:, fi, nn_idx, :])
        return torch.stack(full_frames, dim=1).reshape(B, t * hw, C)
    else:
        k = topk_hw.shape[1]
        kept_hw = topk_hw[0]
        kept_h_coords = kept_hw // w
        kept_w_coords = kept_hw % w
        dh = all_h_coords.unsqueeze(1) - kept_h_coords.unsqueeze(0)
        dw = all_w_coords.unsqueeze(1) - kept_w_coords.unsqueeze(0)
        dist = dh * dh + dw * dw
        nn_idx = dist.argmin(dim=1)
        pruned_per_frame = pruned_output.view(B, t, k, C)
        full_per_frame = pruned_per_frame[:, :, nn_idx, :]
        return full_per_frame.reshape(B, t * hw, C)


def partial_update_recover(pruned_output, flat_indices, cache):
    """
    Partial Update (PU) recovery: scatter pruned transformer output into
    cached full-size tensor. Non-selected positions retain their cached
    values from the previous iteration (or the original input if first iter).

    Args:
        pruned_output: [B, t*k, C] transformer output for selected tokens
        flat_indices:  [B, t*k]    indices of selected tokens in full sequence
        cache:         [B, t*h*w, C] previous iteration output (or input)
    Returns:
        full_output:   [B, t*h*w, C] cache with selected positions updated
    """
    result = cache.clone()
    idx = flat_indices.unsqueeze(-1).expand(-1, -1, cache.shape[-1])
    result.scatter_(1, idx, pruned_output)
    return result


def _idw_weights(non_sel, sel, w, n_neighbors, device):
    """Compute inverse-distance-weighted interpolation weights.

    Args:
        non_sel: [M] indices of non-selected spatial positions
        sel:     [k] indices of selected spatial positions
        w:       grid width (for converting flat index -> 2D coords)
        n_neighbors: number of nearest neighbors for IDW
        device:  torch device
    Returns:
        dict with 'nn_idx' [M, N] and 'w' [M, N] (normalized weights)
    """
    non_h = (non_sel // w).float()
    non_w = (non_sel % w).float()
    sel_h = (sel // w).float()
    sel_w = (sel % w).float()

    dh = non_h.unsqueeze(1) - sel_h.unsqueeze(0)
    dw = non_w.unsqueeze(1) - sel_w.unsqueeze(0)
    dist = (dh ** 2 + dw ** 2).sqrt()                       # [M, k]

    nn = min(n_neighbors, dist.shape[1])
    _, nn_idx = dist.topk(nn, dim=-1, largest=False)         # [M, nn]
    nn_dist = dist.gather(1, nn_idx).clamp_min(0.5)

    weights = 1.0 / nn_dist
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return {'nn_idx': nn_idx, 'w': weights}


def partial_update_recover_interp(pruned_output, flat_indices, cache, pn,
                                   topk_hw, per_frame=False, n_neighbors=4):
    """
    IDW-interpolation recovery for non-selected positions.

    Selected positions   -> exact transformer output (same as PU).
    Non-selected positions -> inverse-distance-weighted interpolation from
    N nearest selected positions' NEW output, avoiding stale cached values
    that cause edge blur.

    Args:
        pruned_output: [B, t*k, C]
        flat_indices:  [B, t*k]
        cache:         [B, t*h*w, C]  (used only for dtype / fallback)
        pn:            (t, h, w)
        topk_hw:       [B, k] (shared) or [B, t, k] (per_frame)
        per_frame:     bool
        n_neighbors:   number of IDW neighbors (default 4)
    Returns:
        full_output: [B, t*h*w, C]
    """
    B, L, C = cache.shape
    t, h, w = pn
    hw = h * w

    # Start from cache, then overwrite everything we can
    result = cache.clone()
    idx_expand = flat_indices.unsqueeze(-1).expand(-1, -1, C)
    result.scatter_(1, idx_expand, pruned_output)

    if per_frame:
        k = topk_hw.shape[2]
        pruned_pf = pruned_output.view(B, t, k, C)
        for fi in range(t):
            kept = topk_hw[0, fi]
            is_sel = torch.zeros(hw, dtype=torch.bool, device=cache.device)
            is_sel[kept] = True
            non_sel = (~is_sel).nonzero(as_tuple=True)[0]
            if len(non_sel) == 0:
                continue
            wt = _idw_weights(non_sel, kept, w, n_neighbors, cache.device)
            fd = pruned_pf[:, fi]                            # [B, k, C]
            nd = fd[:, wt['nn_idx'].reshape(-1)].view(B, len(non_sel), -1, C)
            interp = (nd * wt['w'].unsqueeze(0).unsqueeze(-1)).sum(dim=2)
            result[:, fi * hw + non_sel] = interp.to(result.dtype)
    else:
        k = topk_hw.shape[1]
        kept = topk_hw[0]
        is_sel = torch.zeros(hw, dtype=torch.bool, device=cache.device)
        is_sel[kept] = True
        non_sel = (~is_sel).nonzero(as_tuple=True)[0]
        if len(non_sel) == 0:
            return result

        wt = _idw_weights(non_sel, kept, w, n_neighbors, cache.device)
        pruned_pf = pruned_output.view(B, t, k, C)

        for fi in range(t):
            fd = pruned_pf[:, fi]                            # [B, k, C]
            nd = fd[:, wt['nn_idx'].reshape(-1)].view(B, len(non_sel), -1, C)
            interp = (nd * wt['w'].unsqueeze(0).unsqueeze(-1)).sum(dim=2)
            result[:, fi * hw + non_sel] = interp.to(result.dtype)

    return result

def partial_update_recover_interp_residual(pruned_output, flat_indices, cache, pn,
                                            topk_hw, per_frame=False, n_neighbors=4,
                                            residual_scale=1.0):
    """
    Residual IDW-interpolation recovery.

    Selected positions   -> exact transformer output (same as PU/interp).
    Non-selected positions -> cache + IDW-interpolated RESIDUAL from neighbors.

    Instead of interpolating full features (which can smear), we interpolate
    only the delta (output - cache) at selected positions, then add it back
    to the cached value at each non-selected position.  This preserves more
    of the cached spatial detail while propagating the transformer update.
    """
    B, L, C = cache.shape
    t, h, w = pn
    hw = h * w

    result = cache.clone()
    idx_expand = flat_indices.unsqueeze(-1).expand(-1, -1, C)
    result.scatter_(1, idx_expand, pruned_output)

    cache_pf = cache.view(B, t, hw, C)

    if per_frame:
        k = topk_hw.shape[2]
        pruned_pf = pruned_output.view(B, t, k, C)
        for fi in range(t):
            kept = topk_hw[0, fi]
            is_sel = torch.zeros(hw, dtype=torch.bool, device=cache.device)
            is_sel[kept] = True
            non_sel = (~is_sel).nonzero(as_tuple=True)[0]
            if len(non_sel) == 0:
                continue
            wt = _idw_weights(non_sel, kept, w, n_neighbors, cache.device)
            fd = pruned_pf[:, fi]                            # [B, k, C]
            cache_sel = cache_pf[:, fi, kept]                # [B, k, C]
            delta = fd - cache_sel                           # residual at selected
            nd = delta[:, wt['nn_idx'].reshape(-1)].view(B, len(non_sel), -1, C)
            interp_delta = (nd * wt['w'].unsqueeze(0).unsqueeze(-1)).sum(dim=2)
            cache_non = cache_pf[:, fi, non_sel]             # [B, M, C]
            result[:, fi * hw + non_sel] = (
                cache_non + float(residual_scale) * interp_delta
            ).to(result.dtype)
    else:
        k = topk_hw.shape[1]
        kept = topk_hw[0]
        is_sel = torch.zeros(hw, dtype=torch.bool, device=cache.device)
        is_sel[kept] = True
        non_sel = (~is_sel).nonzero(as_tuple=True)[0]
        if len(non_sel) == 0:
            return result

        wt = _idw_weights(non_sel, kept, w, n_neighbors, cache.device)
        pruned_pf = pruned_output.view(B, t, k, C)

        for fi in range(t):
            fd = pruned_pf[:, fi]                            # [B, k, C]
            cache_sel = cache_pf[:, fi, kept]                # [B, k, C]
            delta = fd - cache_sel
            nd = delta[:, wt['nn_idx'].reshape(-1)].view(B, len(non_sel), -1, C)
            interp_delta = (nd * wt['w'].unsqueeze(0).unsqueeze(-1)).sum(dim=2)
            cache_non = cache_pf[:, fi, non_sel]
            result[:, fi * hw + non_sel] = (
                cache_non + float(residual_scale) * interp_delta
            ).to(result.dtype)

    return result


# ---------------------------------------------------------------------------
# v3: Text-Visual Alignment + Temporal Variance
# ---------------------------------------------------------------------------

def compute_token_importance_v3(
    x, pn,
    text_emb=None,
    w_temp=1.0,
    fusion_p=2.0,
    per_frame=False,
    return_components=False,
):
    """
    Text-guided spatiotemporal token importance (v3).

    Spatial:  cosine similarity between each visual token and pooled text embedding
    Temporal: frame-wise variance of feature vectors (self-contained, no prev clip needed)

    Args:
        x:  [B, t*h*w, C] input features
        pn: (t, h, w)
        text_emb: [B, L_text, C] text token embeddings (conditional only)
                  If None, falls back to activation energy for spatial score
        w_temp:   temporal branch weight in Lp fusion
        fusion_p: Lp-norm degree
        per_frame: per-frame adaptive mask
        return_components: return individual score maps
    Returns:
        importance: [B, t, h*w] if per_frame else [B, h*w]
    """
    B, L, C = x.shape
    t, h, w = pn
    hw = h * w

    x_frames = x.view(B, t, hw, C).float()

    # -------- Spatial: Text-Visual Alignment (v3.1: per-token top-k, not pooled) --------
    if text_emb is not None:
        te = text_emb.float()
        te = te / (te.norm(dim=-1, keepdim=True) + 1e-8)                  # [B, L_text, C]
        x_norm = x_frames / (x_frames.norm(dim=-1, keepdim=True) + 1e-8)  # [B, t, hw, C]
        sim = torch.einsum('btpc,bjc->btpj', x_norm, te)                 # [B, t, hw, L_text]
        k_txt = min(8, sim.shape[-1])
        S_spatial_full = sim.topk(k_txt, dim=-1).values.mean(dim=-1)     # [B, t, hw] best-matching words
        S_spatial_full = (S_spatial_full + 1.0) / 2.0
    else:
        S_spatial_full = x_frames.norm(dim=-1)

    # -------- Temporal: Pairwise cosine dissimilarity --------
    if t >= 2:
        f_prev = x_frames[:, :-1]
        f_cur = x_frames[:, 1:]
        num = (f_prev * f_cur).sum(dim=-1)
        den = (f_prev.norm(dim=-1) * f_cur.norm(dim=-1)).clamp_min(1e-8)
        dissim = 1.0 - num / den                            # [B, t-1, hw]
    else:
        dissim = None

    # -------- Fusion --------
    if per_frame:
        S_spatial = _norm01(S_spatial_full)
        if dissim is not None and w_temp > 0:
            S_tmp_pf = torch.zeros(B, t, hw, device=x.device, dtype=x_frames.dtype)
            S_tmp_pf[:, 0] = dissim[:, 0]
            if t > 2:
                S_tmp_pf[:, 1:-1] = 0.5 * (dissim[:, :-1] + dissim[:, 1:])
            S_tmp_pf[:, -1] = dissim[:, -1]
            S_temp = _norm01(S_tmp_pf)
            importance = (S_spatial.pow(fusion_p) + w_temp * S_temp.pow(fusion_p)).pow(1.0 / fusion_p)
        else:
            importance = S_spatial
    else:
        S_spatial = _norm01(S_spatial_full.mean(dim=1))
        if dissim is not None and w_temp > 0:
            S_temp = _norm01(dissim.mean(dim=1))
            importance = (S_spatial.pow(fusion_p) + w_temp * S_temp.pow(fusion_p)).pow(1.0 / fusion_p)
        else:
            S_temp = torch.zeros(B, hw, device=x.device, dtype=x_frames.dtype)
            importance = S_spatial

    if return_components:
        comps = {
            'S_spatial': S_spatial,
            'S_temp': S_temp if S_temp_full is not None else torch.zeros(B, hw, device=x.device),
            'S_total': importance,
        }
        return importance, comps
    return importance


# ---------------------------------------------------------------------------
# v4: Convergence-Aware Spatiotemporal Scoring
# Combines FastSTAR's cross-scale convergence with v1's spatial complexity
# ---------------------------------------------------------------------------

def compute_token_importance_v4(
    x, pn,
    prev_scale_feat=None,
    w_str=0.5, power_iter=3,
    w_temp=1.0, w_conv=1.5,
    fusion_p=2.0,
    per_frame=False,
    return_components=False,
):
    """
    Convergence-Aware Spatiotemporal Token Scoring (v4).

    Fuses three axes via Lp-norm:
      Axis 1 - Spatial Complexity: PCA structural + high-pass textural (from v1)
      Axis 2 - Temporal Motion: cosine dissimilarity between adjacent frames (from v1)
      Axis 3 - Cross-Scale Convergence: cosine dissimilarity vs previous scale output (from FastSTAR)

    Args:
        x:  [B, t*h*w, C] input features
        pn: (t, h, w)
        prev_scale_feat: [B, t*h*w, C] previous scale/iteration output (None = skip)
        w_str:   PCA weight inside spatial fusion
        power_iter: PCA iterations
        w_temp:  temporal axis weight
        w_conv:  convergence axis weight
        fusion_p: Lp-norm degree
        per_frame: per-frame adaptive mask
        return_components: return individual score maps
    Returns:
        importance: [B, t, h*w] if per_frame else [B, h*w]
    """
    B, L, C = x.shape
    t, h, w = pn
    hw = h * w

    x_frames = x.view(B, t, hw, C).float()

    # --- Axis 1: Spatial Complexity (PCA + high-pass) ---
    S_str_full, S_txt_full = _compute_spatial_scores(x_frames, pn, w_str, power_iter)

    # --- Axis 2: Temporal Motion ---
    dissim = _compute_temporal_scores(x_frames)

    # --- Axis 3: Cross-Scale Convergence ---
    S_conv_full = None
    if prev_scale_feat is not None and prev_scale_feat.shape[1] == L:
        prev_frames = prev_scale_feat.view(B, t, hw, C).float()
        num = (x_frames * prev_frames).sum(dim=-1)
        den = (x_frames.norm(dim=-1) * prev_frames.norm(dim=-1)).clamp_min(1e-8)
        S_conv_full = 1.0 - num / den  # [B, t, hw]

    # --- Fusion ---
    if per_frame:
        S_str = _norm01(S_str_full)
        S_txt = _norm01(S_txt_full)
        S_spatial = _norm01(w_str * S_str + S_txt)

        terms = S_spatial.pow(fusion_p)

        if dissim is not None and w_temp > 0:
            S_tmp_pf = torch.zeros(B, t, hw, device=x.device, dtype=x_frames.dtype)
            S_tmp_pf[:, 0] = dissim[:, 0]
            if t > 2:
                S_tmp_pf[:, 1:-1] = 0.5 * (dissim[:, :-1] + dissim[:, 1:])
            S_tmp_pf[:, -1] = dissim[:, -1]
            S_tmp = _norm01(S_tmp_pf)
            terms = terms + w_temp * S_tmp.pow(fusion_p)

        if S_conv_full is not None and w_conv > 0:
            S_conv = _norm01(S_conv_full)
            terms = terms + w_conv * S_conv.pow(fusion_p)

        importance = terms.pow(1.0 / fusion_p)
    else:
        S_str = _norm01(S_str_full.mean(dim=1))
        S_txt = _norm01(S_txt_full.mean(dim=1))
        S_spatial = _norm01(w_str * S_str + S_txt)

        terms = S_spatial.pow(fusion_p)

        if dissim is not None and w_temp > 0:
            S_tmp = _norm01(dissim.mean(dim=1))
            terms = terms + w_temp * S_tmp.pow(fusion_p)
        else:
            S_tmp = torch.zeros(B, hw, device=x.device, dtype=x_frames.dtype)

        if S_conv_full is not None and w_conv > 0:
            S_conv = _norm01(S_conv_full.mean(dim=1))
            terms = terms + w_conv * S_conv.pow(fusion_p)

        importance = terms.pow(1.0 / fusion_p)

    if return_components:
        comps = {
            'S_str': S_str, 'S_txt': S_txt,
            'S_spatial': S_spatial,
            'S_tmp': S_tmp if dissim is not None else torch.zeros(B, hw, device=x.device),
            'S_total': importance,
        }
        if S_conv_full is not None:
            comps['S_conv'] = _norm01(S_conv_full.mean(dim=1)) if not per_frame else _norm01(S_conv_full)
        return importance, comps
    return importance


# ============================================================
# v5: Error-Accumulation Aware Scoring
# ============================================================
# Idea: tokens pruned many times across scales accumulate error.
# Boost importance of frequently-pruned positions so they get
# computed more often, reducing error propagation.

def apply_error_accumulation_boost(importance, prune_counts, pn, boost_alpha=0.5):
    """
    Boost importance of positions that have been pruned frequently.
    
    Args:
        importance: [B, h*w] or [B, t, h*w] base importance scores
        prune_counts: [B, h*w] cumulative prune counts per spatial position
        pn: (t, h, w)
        boost_alpha: strength of the error-accumulation boost
    Returns:
        boosted importance (same shape as input)
    """
    if prune_counts is None:
        return importance
    
    import torch
    t, h, w = pn
    hw = h * w
    
    # Handle resolution mismatch (prev scale may differ)
    if prune_counts.shape[-1] != hw:
        import torch.nn.functional as F
        B_c = prune_counts.shape[0]
        h_prev = int(prune_counts.shape[-1] ** 0.5)
        w_prev = h_prev
        if h_prev * w_prev != prune_counts.shape[-1]:
            return importance
        counts_2d = prune_counts.view(B_c, 1, h_prev, w_prev).float()
        counts_2d = F.interpolate(counts_2d, size=(h, w), mode='nearest')
        prune_counts = counts_2d.view(B_c, hw)
    
    # Normalize to [0, 1]
    mn = prune_counts.min(dim=-1, keepdim=True).values
    mx = prune_counts.max(dim=-1, keepdim=True).values
    norm_counts = (prune_counts - mn) / (mx - mn + 1e-8)
    
    if importance.dim() == 3:
        # per-frame: [B, t, hw]
        norm_counts = norm_counts.unsqueeze(1).expand_as(importance)
    
    return importance * (1.0 + boost_alpha * norm_counts)


# ============================================================
# v6: Output-Norm Adaptive Scoring  
# ============================================================
# Idea: after the first transformer block processes all tokens,
# measure the per-token output magnitude change (residual norm).
# Tokens with large residuals are being actively refined by
# attention and should be kept for subsequent blocks.

def compute_output_norm_importance(block_input, block_output, pn, per_frame=False):
    """
    Score tokens by the magnitude of change after a transformer block.
    
    Args:
        block_input: [B, t*h*w, C] input to first transformer block
        block_output: [B, t*h*w, C] output of first transformer block
        pn: (t, h, w)
        per_frame: per-frame adaptive mask
    Returns:
        importance: [B, h*w] or [B, t, h*w]
    """
    import torch
    B, L, C = block_input.shape
    t, h, w = pn
    hw = h * w
    
    residual = (block_output - block_input).float()
    residual_norm = residual.norm(dim=-1)  # [B, t*h*w]
    residual_frames = residual_norm.view(B, t, hw)
    
    if per_frame:
        importance = residual_frames
    else:
        importance = residual_frames.mean(dim=1)  # [B, hw]
    
    # Normalize
    mn = importance.min(dim=-1, keepdim=True).values
    mx = importance.max(dim=-1, keepdim=True).values
    importance = (importance - mn) / (mx - mn + 1e-8)
    
    return importance


# ============================================================
# Token Merging: merge non-selected into nearest selected
# ============================================================

def compute_merge_assignment(topk_hw, pn):
    """
    For each spatial position, find which selected token it maps to.
    Returns assignment [hw] where assignment[i] = index in topk (0..k-1).
    Selected positions map to themselves.
    """
    import torch
    t, h, w = pn
    hw = h * w
    k = topk_hw.shape[-1]
    sel = topk_hw[0]  # [k], shared mask

    # 2D coords for all positions
    all_y = torch.arange(h, device=sel.device).view(h, 1).expand(h, w).reshape(hw).float()
    all_x = torch.arange(w, device=sel.device).view(1, w).expand(h, w).reshape(hw).float()
    sel_y = all_y[sel]
    sel_x = all_x[sel]

    # Squared distance [hw, k]
    dy = all_y.unsqueeze(1) - sel_y.unsqueeze(0)
    dx = all_x.unsqueeze(1) - sel_x.unsqueeze(0)
    dist = dy * dy + dx * dx
    assignment = dist.argmin(dim=1)  # [hw]
    return assignment


def enrich_selected_tokens(pruned_x, full_x, topk_hw, pn, alpha=0.3):
    """
    Enrich selected tokens by blending in info from assigned non-selected tokens.

    pruned_x: [B, t*k, C] - the selected tokens
    full_x:   [B, t*h*w, C] - all tokens before pruning
    topk_hw:  [B, k] - selected spatial positions
    alpha:    blend weight for non-selected contributions
    """
    import torch
    B, _, C = full_x.shape
    t, h, w = pn
    hw = h * w
    k = topk_hw.shape[-1]

    assignment = compute_merge_assignment(topk_hw, pn)  # [hw]

    full_frames = full_x.view(B, t, hw, C).float()

    # Accumulate non-selected contributions per selected bucket
    idx_exp = assignment.view(1, 1, hw, 1).expand(B, t, hw, C)
    bucket_sum = torch.zeros(B, t, k, C, device=full_x.device, dtype=torch.float32)
    bucket_sum.scatter_add_(2, idx_exp, full_frames)

    ones = torch.ones(hw, device=full_x.device)
    bucket_cnt = torch.zeros(k, device=full_x.device)
    bucket_cnt.scatter_add_(0, assignment, ones)  # [k]
    bucket_cnt = bucket_cnt.view(1, 1, k, 1).clamp_min(1)

    bucket_avg = bucket_sum / bucket_cnt  # [B, t, k, C]

    pruned_frames = pruned_x.view(B, t, k, C).float()

    # Blend: enriched = selected + alpha * (bucket_avg - selected)
    enriched = pruned_frames + alpha * (bucket_avg - pruned_frames)
    return enriched.to(pruned_x.dtype).reshape(B, t * k, C)


def unmerge_recover(merged_output, topk_hw, pn, pu_base=None, beta=0.5):
    """
    Recover full tokens from merged output.
    Each position gets its nearest selected token's output.
    Optionally blend with PU base for smoother result.

    merged_output: [B, t*k, C]
    topk_hw: [B, k]
    pn: (t, h, w)
    pu_base: [B, t*h*w, C] optional PU cache for blending
    beta: weight for PU base on non-selected positions (0=pure merge, 1=pure PU)
    """
    import torch
    B, _, C = merged_output.shape
    t, h, w = pn
    hw = h * w
    k = topk_hw.shape[-1]

    assignment = compute_merge_assignment(topk_hw, pn)  # [hw]

    merged_frames = merged_output.view(B, t, k, C)
    idx_exp = assignment.view(1, 1, hw, 1).expand(B, t, hw, C)
    full = torch.gather(merged_frames, 2, idx_exp)  # [B, t, hw, C]
    full = full.reshape(B, t * hw, C)

    if pu_base is not None:
        # For selected positions, use merged output; for non-selected, blend
        sel_mask = torch.zeros(hw, device=full.device)
        sel_mask.scatter_(0, topk_hw[0], 1.0)
        sel_mask = sel_mask.view(1, 1, hw, 1).expand(B, t, hw, C).reshape(B, t * hw, C)
        full = sel_mask * full + (1 - sel_mask) * (beta * pu_base + (1 - beta) * full)

    return full


# ---------------------------------------------------------------------------
# Attention-based scoring: uses model's own attention as importance signal
# ---------------------------------------------------------------------------

def idpruner_select(imp, feat, k, lam=0.5):
    """IDPruner greedy: argmax[ lam*Imp - (1-lam)*maxSim(to selected) ]. Returns [B,k] sorted."""
    B, hw, C = feat.shape
    fn_ = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    rng = imp.amax(-1, keepdim=True) - imp.amin(-1, keepdim=True)
    imn = (imp - imp.amin(-1, keepdim=True)) / (rng + 1e-8)
    sel = torch.zeros(B, k, dtype=torch.long, device=feat.device)
    chosen = torch.zeros(B, hw, dtype=torch.bool, device=feat.device)
    bidx = torch.arange(B, device=feat.device)
    first = imn.argmax(-1)
    sel[:, 0] = first; chosen[bidx, first] = True
    m = torch.bmm(fn_, fn_[bidx, first].unsqueeze(-1)).squeeze(-1)
    for t in range(1, k):
        score = (lam * imn - (1 - lam) * m).masked_fill(chosen, -1e9)
        nxt = score.argmax(-1)
        sel[:, t] = nxt; chosen[bidx, nxt] = True
        m = torch.maximum(m, torch.bmm(fn_, fn_[bidx, nxt].unsqueeze(-1)).squeeze(-1))
    return sel.sort(dim=-1)[0]


def compute_attention_importance(
    x, pn,
    attn_module,
    text_k=None,
    text_lens=None,
    prev_clip_hidden=None,
    w_spatial=1.0, w_temporal=1.0, w_text=1.0,
    fusion_p=2.0,
    chunk_size=256,
    query_stride=None,
    max_query_tokens=None,
    rope_cache=None,
    per_frame=False,
    text_pool='max',
    text_topk=4,
    temporal_smooth=0.0,
    text_query_gate=0.0,
    temporal_mode='column',
    prev_importance=None,
    temporal_anchors=1024,
    anchor_chunk=8192,
    random_select=False,
):
    """
    Intrinsic Dependency Scoring (IDS), Eq. (3) to Eq. (8). Scores each position
    by the spatial, cross-clip and text-conditioned dependency it receives, with
    prompt-aware query gating, and fuses the three by a weighted l_p norm.

    Returns a shared spatial score [B, hw] by default. With per_frame=True,
    returns [B, t, hw] so moving content can receive a different mask in each
    frame without changing the token budget.

    temporal_mode='column' keeps the original cross-clip column sum, which
    scores positions of the *previous* clip and reuses that index. With
    temporal_mode='anchor', the temporal score is query-side and symmetric to
    text relevance: each current token is scored by its best match against the
    keys of the previous clip's top-`temporal_anchors` important tokens
    (content matching without RoPE), so the score follows moving content.
    w_spatial=0 skips the visual self-attention pass entirely.
    """
    from infinity.models.rope import apply_rotary_emb

    B, L, C = x.shape
    t, h, w = pn
    hw = h * w

    if random_select:
        # control: same budget and recovery, uniformly random selection. A private
        # generator keeps the model's sampling RNG untouched.
        gen = torch.Generator(device=x.device)
        gen.manual_seed(1000003 * t + 1009 * h + w)
        shape = (B, t, hw) if per_frame else (B, hw)
        return torch.rand(shape, generator=gen, device=x.device)

    num_heads = attn_module.num_heads
    head_dim = attn_module.head_dim
    num_kv_heads = attn_module.num_key_value_heads
    kv_groups = num_heads // num_kv_heads
    scale = head_dim ** -0.5

    import os as _os_av
    if query_stride is None:
        query_stride = int(_os_av.environ.get("ATTN_Q_STRIDE", "1"))
    if max_query_tokens is None and _os_av.environ.get("ATTN_Q_MAX"):
        max_query_tokens = int(_os_av.environ["ATTN_Q_MAX"])
    query_stride = max(1, int(query_stride))
    max_query_tokens = None if max_query_tokens in (None, 0, "0") else max(1, int(max_query_tokens))

    with torch.no_grad():
        q = attn_module.q_norm(
            attn_module.q_proj(x).view(B, L, num_heads, head_dim)
        ).permute(0, 2, 1, 3).float()

        k_self = attn_module.k_norm(
            attn_module.k_proj(x).view(B, L, num_kv_heads, head_dim)
        ).permute(0, 2, 1, 3).float()

        use_anchor = (
            temporal_mode == 'anchor' and prev_clip_hidden is not None and w_temporal > 0
            and (prev_importance is not None or text_k is not None)
        )
        q_content = q.clone() if use_anchor else None

        if rope_cache is not None:
            q, k_self = apply_rotary_emb(q, k_self, rope_cache)

        if num_kv_heads != num_heads:
            k_self = k_self.repeat_interleave(kv_groups, dim=1)

    # Prompt-guided query gating. Text-relevant visual queries contribute more
    # when measuring which visual keys receive attention. A zero gate preserves
    # the original attention-column score exactly.
    S_text = None
    text_query_weights = None
    if text_k is not None and (w_text > 0 or text_query_gate > 0):
        tk = text_k.float()
        if tk.shape[1] != num_heads:
            tk = tk.repeat_interleave(kv_groups, dim=1)
        if tk.shape[0] != B:
            tk = tk[:B]
        text_scores = torch.matmul(q, tk.transpose(-1, -2)) * scale

        valid_lens = None
        if text_lens is not None:
            valid_lens = torch.as_tensor(text_lens, device=x.device, dtype=torch.long)
            if valid_lens.numel() != B:
                valid_lens = valid_lens[:B]
            if valid_lens.numel() != B:
                raise ValueError(f"expected {B} text lengths, got {valid_lens.numel()}")
            valid_lens = valid_lens.clamp(min=1, max=tk.shape[-2])
            valid = torch.arange(tk.shape[-2], device=x.device).view(1, 1, 1, -1)
            valid = valid < valid_lens.view(B, 1, 1, 1)
            text_scores = text_scores.masked_fill(~valid, -torch.inf)

        if text_pool == 'max':
            pooled_text = text_scores.amax(dim=-1)
        elif text_pool == 'topk_mean':
            k_text = max(1, min(int(text_topk), text_scores.shape[-1]))
            top_values = text_scores.topk(k_text, dim=-1).values
            finite = torch.isfinite(top_values)
            pooled_text = top_values.masked_fill(~finite, 0.0).sum(dim=-1)
            pooled_text = pooled_text / finite.sum(dim=-1).clamp_min(1)
        elif text_pool == 'logmeanexp':
            if valid_lens is None:
                denom = text_scores.shape[-1]
                pooled_text = torch.logsumexp(text_scores, dim=-1) - torch.log(
                    text_scores.new_tensor(float(denom))
                )
            else:
                pooled_text = torch.logsumexp(text_scores, dim=-1)
                pooled_text = pooled_text - valid_lens.float().log().view(B, 1, 1)
        else:
            raise ValueError(f"unsupported text_pool={text_pool!r}")

        gate_strength = max(0.0, float(text_query_gate))
        if gate_strength > 0:
            gate = _norm01(pooled_text.view(B, num_heads, t, hw))
            text_query_weights = 1.0 + gate_strength * gate
            text_query_weights = text_query_weights / text_query_weights.mean(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)
            text_query_weights = text_query_weights.reshape(B, num_heads, L)

        if w_text > 0:
            S_text = pooled_text.mean(dim=1).view(B, t, hw)
            if not per_frame:
                S_text = S_text.mean(dim=1)

        del text_scores

    query_idx = None
    if query_stride > 1 or (max_query_tokens is not None and L > max_query_tokens):
        query_idx = torch.arange(0, L, query_stride, device=x.device)
        if max_query_tokens is not None and query_idx.numel() > max_query_tokens:
            pick = torch.linspace(0, query_idx.numel() - 1, max_query_tokens, device=x.device)
            query_idx = query_idx[pick.round().long()].unique(sorted=True)
        if query_idx.numel() == 0:
            query_idx = torch.zeros(1, dtype=torch.long, device=x.device)
    q_cols = q.index_select(2, query_idx) if query_idx is not None else q
    query_weights = (
        text_query_weights.index_select(2, query_idx)
        if text_query_weights is not None and query_idx is not None
        else text_query_weights
    )
    if query_weights is not None:
        query_weights = query_weights / query_weights.mean(dim=-1, keepdim=True).clamp_min(1e-8)
    query_scale = float(L) / float(q_cols.shape[2])

    # -------- Eq. (5): spatial score, gated attention column sum --------
    col_sum = torch.zeros(B, num_heads, L, device=x.device, dtype=torch.float32)
    for i in range(0, q_cols.shape[2] if w_spatial > 0 else 0, chunk_size):
        qc = q_cols[:, :, i:i + chunk_size]
        scores = torch.matmul(qc, k_self.transpose(-1, -2)) * scale
        attention = scores.softmax(dim=-1)
        if query_weights is not None:
            attention = attention * query_weights[:, :, i:i + chunk_size].unsqueeze(-1)
        col_sum += attention.sum(dim=-2) * query_scale
    _vnorm_mode = _os_av.environ.get("ATTN_VNORM")
    if _vnorm_mode in {"1", "2", "3"}:
        with torch.no_grad():
            _vself = attn_module.v_proj(x).view(B, L, num_kv_heads, head_dim).permute(0, 2, 1, 3).float()
            if num_kv_heads != num_heads:
                _vself = _vself.repeat_interleave(kv_groups, dim=1)
            _vnorm = _vself.norm(dim=-1)  # [B, num_heads, L]
        if _vnorm_mode == "2":
            _vnorm = _vnorm.sqrt()
        elif _vnorm_mode == "3":
            _vmin = _vnorm.amin(dim=-1, keepdim=True); _vmax = _vnorm.amax(dim=-1, keepdim=True)
            _vnorm = (_vnorm - _vmin) / (_vmax - _vmin + 1e-8) + 0.5
        col_sum = col_sum * _vnorm
    S_spatial = col_sum.mean(dim=1).view(B, t, hw) if w_spatial > 0 else None
    if S_spatial is not None and not per_frame:
        S_spatial = S_spatial.mean(dim=1)

    # -------- Eq. (7): temporal score, cross-clip column sum --------
    S_temporal = None
    if use_anchor:
        with torch.no_grad():
            Bp = prev_clip_hidden.shape[0]
            k_prev = attn_module.k_norm(
                attn_module.k_proj(prev_clip_hidden).view(Bp, -1, num_kv_heads, head_dim)
            ).permute(0, 2, 1, 3).float()
            if k_prev.shape[0] != B:
                k_prev = k_prev[:B]
            L_prev = k_prev.shape[2]
            if prev_importance is None:
                # The preceding clip was never pruned (e.g. the single-frame first
                # clip), so it has no IDS score. Rank its tokens by prompt relevance.
                tk_prev = text_k.float()
                if tk_prev.shape[1] != num_heads:
                    tk_prev = tk_prev.repeat_interleave(kv_groups, dim=1)
                tk_prev = tk_prev[:B]
                q_prev = attn_module.q_norm(
                    attn_module.q_proj(prev_clip_hidden[:B]).view(B, -1, num_heads, head_dim)
                ).permute(0, 2, 1, 3).float()
                rel = torch.matmul(q_prev, tk_prev.transpose(-1, -2)) * scale
                if text_lens is not None:
                    vl = torch.as_tensor(text_lens, device=x.device, dtype=torch.long)[:B]
                    vl = vl.clamp(min=1, max=tk_prev.shape[-2])
                    valid = torch.arange(tk_prev.shape[-2], device=x.device).view(1, 1, 1, -1) < vl.view(B, 1, 1, 1)
                    rel = rel.masked_fill(~valid, -torch.inf)
                imp_prev = rel.amax(dim=-1).mean(dim=1)  # [B, L_prev]
                del q_prev, rel, tk_prev
            else:
                imp_prev = prev_importance.float()
            if imp_prev.shape[0] != B:
                imp_prev = imp_prev[:B]
            if imp_prev.ndim == 2:  # shared [B, hw] -> every previous frame
                imp_prev = imp_prev.unsqueeze(1).expand(-1, max(1, L_prev // hw), -1)
            imp_prev = imp_prev.reshape(B, -1)
            if imp_prev.shape[1] != L_prev:
                raise ValueError(f"prev importance {tuple(imp_prev.shape)} vs prev clip L={L_prev}")
            m = max(1, min(int(temporal_anchors), L_prev))
            anchor_idx = imp_prev.topk(m, dim=-1).indices  # [B, m]
            k_anchor = k_prev.gather(
                2, anchor_idx.view(B, 1, m, 1).expand(-1, k_prev.shape[1], -1, head_dim)
            )
            if num_kv_heads != num_heads:
                k_anchor = k_anchor.repeat_interleave(kv_groups, dim=1)
            match = torch.empty(B, num_heads, L, device=x.device, dtype=torch.float32)
            for i in range(0, L, anchor_chunk):
                match[:, :, i:i + anchor_chunk] = (
                    torch.matmul(q_content[:, :, i:i + anchor_chunk], k_anchor.transpose(-1, -2)) * scale
                ).amax(dim=-1)
            S_temporal = match.mean(dim=1).view(B, t, hw)
            if not per_frame:
                S_temporal = S_temporal.mean(dim=1)
            del k_prev, k_anchor, match, q_content
    elif prev_clip_hidden is not None and w_temporal > 0:
        with torch.no_grad():
            k_prev = attn_module.k_norm(
                attn_module.k_proj(prev_clip_hidden).view(
                    prev_clip_hidden.shape[0], -1, num_kv_heads, head_dim
                )
            ).permute(0, 2, 1, 3).float()
            if rope_cache is not None and k_prev.shape[2] == L:
                _, k_prev = apply_rotary_emb(q[:k_prev.shape[0]].clone(), k_prev, rope_cache)
            if num_kv_heads != num_heads:
                k_prev = k_prev.repeat_interleave(kv_groups, dim=1)
        if k_prev.shape[0] != B:
            k_prev = k_prev[:B]
        L_prev = k_prev.shape[2]
        cross_col = torch.zeros(B, num_heads, L_prev, device=x.device, dtype=torch.float32)
        for i in range(0, q_cols.shape[2], chunk_size):
            qc = q_cols[:, :, i:i + chunk_size]
            scores = torch.matmul(qc, k_prev.transpose(-1, -2)) * scale
            attention = scores.softmax(dim=-1)
            if query_weights is not None:
                attention = attention * query_weights[:, :, i:i + chunk_size].unsqueeze(-1)
            cross_col += attention.sum(dim=-2) * query_scale
        t_prev = max(1, L_prev // hw)
        S_temporal = cross_col.mean(dim=1).view(B, t_prev, hw)
        if per_frame:
            if t_prev != t:
                S_temporal = F.interpolate(
                    S_temporal.transpose(1, 2), size=t, mode='linear', align_corners=False
                ).transpose(1, 2)
        else:
            S_temporal = S_temporal.mean(dim=1)

    del q, k_self, query_weights, text_query_weights

    # -------- Eq. (8): weighted l_p fusion --------
    terms = 0.0
    if S_spatial is not None:
        terms = terms + w_spatial * _norm01(S_spatial).pow(fusion_p)
    if S_temporal is not None:
        terms = terms + w_temporal * _norm01(S_temporal).pow(fusion_p)
    if S_text is not None:
        terms = terms + w_text * _norm01(S_text).pow(fusion_p)

    importance = terms.pow(1.0 / fusion_p)
    if per_frame and temporal_smooth > 0 and t > 1:
        alpha = min(1.0, max(0.0, float(temporal_smooth)))
        by_position = importance.transpose(1, 2)
        smoothed = F.avg_pool1d(
            F.pad(by_position, (1, 1), mode='replicate'), kernel_size=3, stride=1
        ).transpose(1, 2)
        importance = (1.0 - alpha) * importance + alpha * smoothed

    return importance


def select_attention_tokens(
    x, rope_cache, pn, importance, keep_ratio, shared_fraction=0.0,
):
    """Select shared or frame-adaptive top-k tokens from attention importance."""
    B, _, C = x.shape
    t, h, w = pn
    hw = h * w
    k = max(1, int(hw * keep_ratio))

    if importance.ndim == 3:
        if tuple(importance.shape[1:]) != (t, hw):
            raise ValueError(
                f"per-frame importance shape {tuple(importance.shape)} does not match pn={pn}"
            )
        shared_fraction = min(1.0, max(0.0, float(shared_fraction)))
        shared_k = min(k, max(0, int(round(k * shared_fraction))))
        if shared_k > 0:
            _, shared_hw = importance.mean(dim=1).topk(shared_k, dim=-1)
            shared_hw = shared_hw.unsqueeze(1).expand(-1, t, -1)
            if shared_k < k:
                shared_mask = torch.zeros(B, hw, dtype=torch.bool, device=x.device)
                shared_mask.scatter_(1, shared_hw[:, 0], True)
                adaptive_scores = importance.masked_fill(shared_mask.unsqueeze(1), -torch.inf)
                _, adaptive_hw = adaptive_scores.topk(k - shared_k, dim=-1)
                topk_hw = torch.cat((shared_hw, adaptive_hw), dim=-1)
            else:
                topk_hw = shared_hw
        else:
            _, topk_hw = importance.topk(k, dim=-1)
        topk_hw, _ = topk_hw.sort(dim=-1)
        frame_offsets = torch.arange(t, device=x.device).view(1, t, 1) * hw
        flat_indices = (topk_hw + frame_offsets).reshape(B, t * k)
    elif importance.ndim == 2:
        if importance.shape[1] != hw:
            raise ValueError(
                f"shared importance shape {tuple(importance.shape)} does not match pn={pn}"
            )
        _, topk_hw = importance.topk(k, dim=-1)
        topk_hw, _ = topk_hw.sort(dim=-1)
        frame_offsets = torch.arange(t, device=x.device).view(1, t, 1) * hw
        flat_indices = (topk_hw.unsqueeze(1) + frame_offsets).reshape(B, t * k)
    else:
        raise ValueError(f"expected 2D or 3D attention importance, got {importance.ndim}D")

    pruned_x = x.gather(1, flat_indices.unsqueeze(-1).expand(-1, -1, C))
    pruned_rope = rope_cache[:, :, :, :, flat_indices[0], :]
    return pruned_x, pruned_rope, flat_indices, topk_hw


# ============================================================
# diff: Anchor-Propagate Token Pruning (APTP)
# Single principle: "propagate the update, not the value"
# ============================================================

def compute_diffusion_importance(x, pn, per_frame=False):
    """
    Anchor-coverage importance scoring for diffusion/delta-propagation recovery.

    Tokens at feature boundaries (high Laplacian) are critical anchors.
    Spatial diversity penalty prevents anchor clustering, ensuring even
    coverage for delta interpolation.
    """
    B, L, C = x.shape
    t, h, w = pn
    hw = h * w

    x_2d = x.view(B * t, C, h, w).float()

    padded = F.pad(x_2d, (1, 1, 1, 1), mode='replicate')
    lap = x_2d - F.avg_pool2d(padded, 3, stride=1, padding=0)
    S_lap = (lap ** 2).sum(dim=1).view(B, t, hw)

    x_frames = x.view(B, t, hw, C).float()
    dissim = _compute_temporal_scores(x_frames)

    if per_frame:
        S_spa = _norm01(S_lap)
        if dissim is not None:
            S_tmp = torch.zeros(B, t, hw, device=x.device, dtype=S_lap.dtype)
            S_tmp[:, 0] = dissim[:, 0]
            if t > 2:
                S_tmp[:, 1:-1] = 0.5 * (dissim[:, :-1] + dissim[:, 1:])
            S_tmp[:, -1] = dissim[:, -1]
            S_tmp = _norm01(S_tmp)
            base_imp = (S_spa.pow(2) + S_tmp.pow(2)).pow(0.5)
        else:
            base_imp = S_spa
        imp_2d = base_imp.view(B * t, 1, h, w)
        imp_smooth = F.avg_pool2d(F.pad(imp_2d, (2, 2, 2, 2), mode='replicate'), 5, 1, 0)
        diversity = 1.0 - 0.3 * _norm01(imp_smooth.view(B, t, hw))
        importance = base_imp * diversity
    else:
        S_spa = _norm01(S_lap.mean(dim=1))
        if dissim is not None:
            S_tmp = _norm01(dissim.mean(dim=1))
            base_imp = (S_spa.pow(2) + S_tmp.pow(2)).pow(0.5)
        else:
            base_imp = S_spa
        imp_2d = base_imp.view(B, 1, h, w)
        imp_smooth = F.avg_pool2d(F.pad(imp_2d, (2, 2, 2, 2), mode='replicate'), 5, 1, 0)
        diversity = 1.0 - 0.3 * _norm01(imp_smooth.view(B, hw))
        importance = base_imp * diversity

    return importance


def _bilateral_weights(non_sel, sel, w, n_neighbors, device, feat_non=None, feat_sel=None, sigma_f=0.5):
    """Bilateral weights: spatial IDW * feature similarity."""
    non_h = (non_sel // w).float()
    non_w_c = (non_sel % w).float()
    sel_h = (sel // w).float()
    sel_w_c = (sel % w).float()

    dh = non_h.unsqueeze(1) - sel_h.unsqueeze(0)
    dw = non_w_c.unsqueeze(1) - sel_w_c.unsqueeze(0)
    dist = (dh ** 2 + dw ** 2).sqrt()

    nn = min(n_neighbors, dist.shape[1])
    _, nn_idx = dist.topk(nn, dim=-1, largest=False)
    nn_dist = dist.gather(1, nn_idx).clamp_min(0.5)
    d_nearest = nn_dist[:, 0]

    spatial_w = 1.0 / nn_dist

    if sigma_f > 0 and feat_non is not None and feat_sel is not None:
        feat_sel_nn = feat_sel[nn_idx.reshape(-1)].view(len(non_sel), nn, -1)
        feat_non_exp = feat_non.unsqueeze(1).expand_as(feat_sel_nn)
        feat_dist = (feat_non_exp - feat_sel_nn).pow(2).mean(dim=-1).sqrt()
        scale = (sigma_f * feat_dist.mean().clamp_min(1e-6))
        feat_w = torch.exp(-feat_dist / scale)
        weights = spatial_w * feat_w
    else:
        weights = spatial_w

    weights = weights / weights.sum(dim=-1, keepdim=True)
    return {'nn_idx': nn_idx, 'w': weights}, d_nearest



def delta_propagation_recover(pruned_output, flat_indices, full_input, pn,
                               topk_hw, n_neighbors=8, per_frame=False,
                               sign_correct_frac=0.0):
    """
    Residual Completion (RC), Eq. (9) to Eq. (11). Propagates only the increment
    a scale contributes, not the full feature, so the state a skipped position
    accumulated over earlier scales is preserved.

    Selected:     result = transformer_output (exact)
    Non-selected: result = current_input + IDW_interpolated(delta_at_anchors)
    """
    B, L, C = full_input.shape
    t, h, w = pn
    hw = h * w

    idx_expand = flat_indices.unsqueeze(-1).expand(-1, -1, C)
    input_at_sel = full_input.gather(1, idx_expand)
    delta_sel = pruned_output.float() - input_at_sel.float()

    if per_frame:
        k = topk_hw.shape[2]
        result = full_input.clone()
        result.scatter_(1, idx_expand, pruned_output)

        delta_pf = delta_sel.view(B, t, k, C)
        result_out = result.view(B, t, hw, C)

        for fi in range(t):
            kept = topk_hw[0, fi]
            is_sel = torch.zeros(hw, dtype=torch.bool, device=full_input.device)
            is_sel[kept] = True
            non_sel = (~is_sel).nonzero(as_tuple=True)[0]
            if len(non_sel) == 0:
                continue
            wt = _idw_weights(non_sel, kept, w, n_neighbors, full_input.device)
            fd = delta_pf[:, fi]
            nd = fd[:, wt['nn_idx'].reshape(-1)].view(B, len(non_sel), -1, C)
            interp_delta = (nd * wt['w'].unsqueeze(0).unsqueeze(-1)).sum(dim=2)
            base = full_input.view(B, t, hw, C)[:, fi, non_sel]
            result_out[:, fi, non_sel] = (base.float() + interp_delta).to(result.dtype)

        return result_out.reshape(B, t * hw, C)
    else:
        k = topk_hw.shape[1]
        kept = topk_hw[0]
        is_sel = torch.zeros(hw, dtype=torch.bool, device=full_input.device)
        is_sel[kept] = True
        non_sel = (~is_sel).nonzero(as_tuple=True)[0]

        result = full_input.clone()
        result.scatter_(1, idx_expand, pruned_output)

        if len(non_sel) == 0:
            return result

        wt = _idw_weights(non_sel, kept, w, n_neighbors, full_input.device)
        delta_pf = delta_sel.view(B, t, k, C)
        input_pf = full_input.view(B, t, hw, C)
        result_pf = result.view(B, t, hw, C)

        for fi in range(t):
            fd = delta_pf[:, fi]
            nd = fd[:, wt['nn_idx'].reshape(-1)].view(B, len(non_sel), -1, C)
            interp_delta = (nd * wt['w'].unsqueeze(0).unsqueeze(-1)).sum(dim=2)
            base = input_pf[:, fi, non_sel]
            result_pf[:, fi, non_sel] = (base.float() + interp_delta).to(result.dtype)

        if sign_correct_frac > 0 and len(non_sel) > 0:
            n_c = max(1, int(C * sign_correct_frac))
            nn = wt['nn_idx'].shape[1]
            for fi in range(t):
                feat_non = result_pf[:, fi, non_sel].float()
                feat_sel = result_pf[:, fi, kept].float()
                nn_feat = feat_sel[:, wt['nn_idx'].reshape(-1)].view(B, len(non_sel), nn, C)
                feat_abs = feat_non.abs()
                tau_val = feat_abs.kthvalue(n_c, dim=-1, keepdim=True).values
                uncertain = feat_abs <= tau_val
                mean_signs = nn_feat.sign().mean(dim=2)
                strong = mean_signs.abs() >= 0.5
                majority_sign = mean_signs.sign()
                should_flip = uncertain & (feat_non.sign() != majority_sign) & strong
                corrected = torch.where(should_flip, majority_sign * feat_abs, feat_non)
                result_pf[:, fi, non_sel] = corrected.to(result.dtype)

        return result_pf.reshape(B, t * hw, C)


def recover_attention_tokens(
    pruned_output,
    flat_indices,
    cache,
    full_input,
    pn,
    topk_hw,
    per_frame=False,
    mode='pu',
    n_neighbors=4,
    residual_scale=1.0,
):
    """Recover attention-selected tokens with a single configurable interface."""
    if mode == 'pu':
        return partial_update_recover(pruned_output, flat_indices, cache)
    if mode == 'interp':
        return partial_update_recover_interp(
            pruned_output,
            flat_indices,
            cache,
            pn,
            topk_hw,
            per_frame=per_frame,
            n_neighbors=n_neighbors,
        )
    if mode == 'interp_residual':
        return partial_update_recover_interp_residual(
            pruned_output,
            flat_indices,
            cache,
            pn,
            topk_hw,
            per_frame=per_frame,
            n_neighbors=n_neighbors,
            residual_scale=residual_scale,
        )
    if mode == 'delta_prop':
        return delta_propagation_recover(
            pruned_output,
            flat_indices,
            full_input,
            pn,
            topk_hw,
            n_neighbors=n_neighbors,
            per_frame=per_frame,
        )
    raise ValueError(f"unsupported attention recovery mode={mode!r}")


def build_context_tokens(x_cond, x_uncond, rope_cache, pn, flat_indices, pool):
    """Context-preserving pruning: pool the *skipped* positions of every frame into
    pool x pool windows and return them as extra tokens (cond, uncond, rope).
    The extra tokens are processed with the selected ones and dropped afterwards, so
    selected tokens still attend to a low-resolution summary of the skipped area."""
    import math
    B, L, C = x_cond.shape
    t, h, w = pn
    dev = x_cond.device
    sel = torch.zeros(L, dtype=torch.bool, device=dev)
    sel[flat_indices[0]] = True                      # cond/uncond share one layout
    skip = (~sel).view(t, 1, h, w).float()            # [t,1,h,w]
    hp, wp = math.ceil(h / pool), math.ceil(w / pool)
    ph, pw = hp * pool - h, wp * pool - w
    cnt = F.avg_pool2d(F.pad(skip, (0, pw, 0, ph)), pool, pool) * (pool * pool)   # [t,1,hp,wp]
    keep = (cnt[:, 0] > 0).reshape(-1)                                            # [t*hp*wp]
    def pooled(x):
        xf = x.float().view(B, t, h, w, C).permute(0, 1, 4, 2, 3).reshape(B * t, C, h, w)
        xf = xf * skip.repeat(B, 1, 1, 1)
        sm = F.avg_pool2d(F.pad(xf, (0, pw, 0, ph)), pool, pool) * (pool * pool)   # [B*t,C,hp,wp]
        mean = sm / cnt.repeat(B, 1, 1, 1).clamp_min(1.0)
        mean = mean.view(B, t, C, hp * wp).permute(0, 1, 3, 2).reshape(B, t * hp * wp, C)
        return mean[:, keep].to(x.dtype)
    ti = torch.arange(t, device=dev).view(t, 1, 1)
    hi = (torch.arange(hp, device=dev) * pool + pool // 2).clamp_max(h - 1).view(1, hp, 1)
    wi = (torch.arange(wp, device=dev) * pool + pool // 2).clamp_max(w - 1).view(1, 1, wp)
    centre = (ti * h * w + hi * w + wi).reshape(-1)[keep]
    rope_ctx = rope_cache[:, :, :, :, centre, :]
    return pooled(x_cond), pooled(x_uncond), rope_ctx
