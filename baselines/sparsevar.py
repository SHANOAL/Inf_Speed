"""SparseVAR-style spatial-semantic token sparsity for InfinityStar.

This adapts SparseVAR's two ingredients to the video setting:
  1. high-frequency tokens from an aligned previous-scale MSE/change map;
  2. low-frequency anchor tokens, used to recover skipped low-frequency tokens.
"""

import torch
import torch.nn.functional as F


def _norm01(x):
    mn = x.min(dim=-1, keepdim=True).values
    mx = x.max(dim=-1, keepdim=True).values
    return (x - mn) / (mx - mn + 1e-8)


def _align_score(prev_score, prev_pn, pn, B, device):
    if prev_score is None or prev_pn is None:
        return None
    t0, h0, w0 = [int(v) for v in prev_pn]
    t, h, w = [int(v) for v in pn]
    if prev_score.shape[-1] == h * w:
        return prev_score.to(device=device)
    if prev_score.shape[-1] == t0 * h0 * w0:
        s = prev_score.to(device=device).view(B, t0, h0 * w0).mean(dim=1)
    elif prev_score.shape[-1] == h0 * w0:
        s = prev_score.to(device=device)
    else:
        return None
    s = s.view(B, 1, h0, w0)
    if (h0, w0) != (h, w):
        s = F.interpolate(s, size=(h, w), mode="bilinear", align_corners=False)
    return s.reshape(B, h * w)


def _anchor_positions(h, w, stride, device):
    stride = max(1, int(stride))
    ys = torch.arange(stride // 2, h, stride, device=device)
    xs = torch.arange(stride // 2, w, stride, device=device)
    if ys.numel() == 0:
        ys = torch.tensor([h // 2], device=device)
    if xs.numel() == 0:
        xs = torch.tensor([w // 2], device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    anchors = (yy.reshape(-1) * w + xx.reshape(-1)).long()
    return anchors.unique(sorted=True)


def _nearest_anchor_map(h, w, anchors, device):
    hw = h * w
    all_pos = torch.arange(hw, device=device)
    ay = (anchors // w).float()
    ax = (anchors % w).float()
    py = (all_pos // w).float()
    px = (all_pos % w).float()
    dist = (py[:, None] - ay[None]) ** 2 + (px[:, None] - ax[None]) ** 2
    return anchors[dist.argmin(dim=1)]


def sparsevar_select_tokens(
    x,
    rope_cache,
    pn,
    keep_ratio,
    prev_mse=None,
    prev_pn=None,
    anchor_stride=4,
):
    B, L, C = x.shape
    t, h, w = [int(v) for v in pn]
    hw = h * w
    budget = max(1, int(hw * keep_ratio))

    anchors = _anchor_positions(h, w, anchor_stride, x.device)
    if anchors.numel() > budget:
        # Keep a uniform subset if the anchor grid alone exceeds the budget.
        pick = torch.linspace(0, anchors.numel() - 1, budget, device=x.device).round().long()
        anchors = anchors[pick].unique(sorted=True)

    score = _align_score(prev_mse, prev_pn, pn, B, x.device)
    if score is None:
        xf = x.view(B, t, hw, C).float()
        score = (xf - xf.mean(dim=2, keepdim=True)).pow(2).mean(dim=-1).mean(dim=1)
    score = _norm01(score)

    selected = []
    high_budget = max(0, budget - anchors.numel())
    for b in range(B):
        mask = torch.zeros(hw, dtype=torch.bool, device=x.device)
        mask[anchors] = True
        if high_budget > 0:
            cand = score[b].masked_fill(mask, -1.0)
            _, high = cand.topk(high_budget, dim=-1)
            mask[high] = True
        idx = mask.nonzero(as_tuple=True)[0]
        if idx.numel() < budget:
            cand = score[b].masked_fill(mask, -1.0)
            _, extra = cand.topk(budget - idx.numel(), dim=-1)
            idx = torch.cat([idx, extra])
        selected.append(idx.sort().values[:budget])
    topk_hw = torch.stack(selected, dim=0)

    offsets = torch.arange(t, device=x.device).unsqueeze(1) * hw
    flat_idx = (topk_hw.unsqueeze(1) + offsets.unsqueeze(0)).reshape(B, t * topk_hw.shape[1])
    pruned_x = x.gather(1, flat_idx.unsqueeze(-1).expand(-1, -1, C))
    pruned_rope = rope_cache[:, :, :, :, flat_idx[0], :]

    anchor_map = _nearest_anchor_map(h, w, anchors, x.device)
    info = {
        "anchors": anchors,
        "anchor_map": anchor_map,
        "topk_hw": topk_hw,
        "flat_idx": flat_idx,
        "pn": tuple(pn),
    }
    return pruned_x, pruned_rope, flat_idx, topk_hw, info


def sparsevar_anchor_recover(pruned_output, info, base):
    B, L, C = base.shape
    t, h, w = [int(v) for v in info["pn"]]
    hw = h * w
    topk_hw = info["topk_hw"]
    flat_idx = info["flat_idx"]
    anchor_map = info["anchor_map"]

    full = base.clone()
    full.scatter_(1, flat_idx.unsqueeze(-1).expand(-1, -1, C), pruned_output)

    selected_mask = torch.zeros(B, hw, dtype=torch.bool, device=base.device)
    selected_mask.scatter_(1, topk_hw, True)
    anchor_src = anchor_map.unsqueeze(0).expand(B, -1)

    for b in range(B):
        low = (~selected_mask[b]).nonzero(as_tuple=True)[0]
        if low.numel() == 0:
            continue
        src_hw = anchor_src[b, low]
        for frame in range(t):
            dst = frame * hw + low
            src = frame * hw + src_hw
            full[b, dst] = full[b, src]
    return full


def sparsevar_change_map(output, base, pn):
    B, L, C = output.shape
    t, h, w = [int(v) for v in pn]
    diff = (output.float() - base.float()).pow(2).mean(dim=-1)
    return diff.view(B, t, h * w).mean(dim=1)
