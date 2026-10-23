"""Chunked ToMe adaptation for high-resolution InfinityStar tokens.

The matching primitive follows ToMe's bipartite soft matching, but applies it
per frame and in spatial chunks to avoid constructing a huge global similarity
matrix for 720p video tokens.
"""

import torch
import torch.nn.functional as F


def _merge_once(x, target_len):
    B, L, C = x.shape
    if target_len >= L or L <= 1:
        idx = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        return x, {"identity": True, "idx": idx, "L": L}

    r = min(L // 2, L - target_len)
    if r <= 0:
        idx = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        return x, {"identity": True, "idx": idx, "L": L}

    metric = F.normalize(x.float(), dim=-1)
    a = metric[:, ::2, :]
    b = metric[:, 1::2, :]
    scores = a @ b.transpose(-1, -2)
    node_max, node_idx = scores.max(dim=-1)
    edge_idx = node_max.argsort(dim=-1, descending=True)

    src_idx = edge_idx[:, :r]
    unm_idx = edge_idx[:, r:]
    dst_idx = node_idx.gather(dim=-1, index=src_idx)

    src = x[:, ::2, :]
    dst = x[:, 1::2, :].clone()
    unm = src.gather(1, unm_idx.unsqueeze(-1).expand(-1, -1, C))
    src_sel = src.gather(1, src_idx.unsqueeze(-1).expand(-1, -1, C))
    dst = dst.scatter_reduce(
        1,
        dst_idx.unsqueeze(-1).expand(-1, -1, C),
        src_sel,
        reduce="mean",
        include_self=True,
    )
    merged = torch.cat([unm, dst], dim=1)

    even_pos = 2 * unm_idx
    odd_pos = 2 * torch.arange(dst.shape[1], device=x.device).view(1, -1).expand(B, -1) + 1
    keep_idx = torch.cat([even_pos, odd_pos], dim=1).clamp_max(L - 1)
    info = {
        "identity": False,
        "L": L,
        "src_idx": src_idx,
        "unm_idx": unm_idx,
        "dst_idx": dst_idx,
        "dst_len": dst.shape[1],
        "keep_idx": keep_idx,
    }
    return merged, info


def _unmerge_once(x, info):
    if info.get("identity", False):
        return x
    B, _, C = x.shape
    L = info["L"]
    unm_len = info["unm_idx"].shape[1]
    dst_len = info["dst_len"]
    unm = x[:, :unm_len, :]
    dst = x[:, unm_len:unm_len + dst_len, :]
    src = dst.gather(1, info["dst_idx"].unsqueeze(-1).expand(-1, -1, C))

    out = torch.zeros(B, L, C, device=x.device, dtype=x.dtype)
    odd_len = out[:, 1::2, :].shape[1]
    out[:, 1::2, :] = dst[:, :odd_len, :]
    even = out[:, ::2, :]
    even.scatter_(1, info["unm_idx"].unsqueeze(-1).expand(-1, -1, C), unm)
    even.scatter_(1, info["src_idx"].unsqueeze(-1).expand(-1, -1, C), src)
    out[:, ::2, :] = even
    return out


def tome_merge_tokens(x, rope_cache, pn, keep_ratio, chunk_size=1024):
    B, L, C = x.shape
    t, h, w = [int(v) for v in pn]
    hw = h * w
    target_hw = max(1, int(hw * keep_ratio))
    chunk_size = max(16, int(chunk_size))

    frame_outputs = []
    frame_infos = []
    frame_keep_indices = []
    xf = x.view(B, t, hw, C)

    for frame in range(t):
        chunks = []
        infos = []
        keep_parts = []
        start = 0
        while start < hw:
            end = min(hw, start + chunk_size)
            chunk = xf[:, frame, start:end, :]
            chunk_len = end - start
            # Allocate the spatial budget proportionally to this chunk.
            target = max(1, int(round(chunk_len * target_hw / hw)))
            step_infos = []
            survivor = torch.arange(chunk_len, device=x.device).unsqueeze(0).expand(B, -1)
            cur = chunk
            while cur.shape[1] > target:
                next_target = max(target, (cur.shape[1] + 1) // 2)
                cur, info = _merge_once(cur, next_target)
                if info.get("identity", False):
                    break
                step_infos.append(info)
                survivor = survivor.gather(1, info["keep_idx"])
            chunks.append(cur)
            infos.append(step_infos)
            keep_parts.append(survivor + start + frame * hw)
            start = end
        frame_outputs.append(torch.cat(chunks, dim=1))
        frame_infos.append(infos)
        frame_keep_indices.append(torch.cat(keep_parts, dim=1))

    merged = torch.cat(frame_outputs, dim=1)
    flat_idx = torch.cat(frame_keep_indices, dim=1)
    pruned_rope = rope_cache[:, :, :, :, flat_idx[0], :]
    info = {
        "pn": tuple(pn),
        "chunk_size": chunk_size,
        "frame_infos": frame_infos,
        "frame_lengths": [[out.shape[1] for out in frame_outputs]],
        "flat_idx": flat_idx,
    }
    return merged, pruned_rope, flat_idx, info


def tome_apply_merge(x, info):
    B, L, C = x.shape
    t, h, w = [int(v) for v in info["pn"]]
    hw = h * w
    chunk_size = info["chunk_size"]
    xf = x.view(B, t, hw, C)
    outputs = []
    for frame in range(t):
        start = 0
        chunk_id = 0
        while start < hw:
            end = min(hw, start + chunk_size)
            cur = xf[:, frame, start:end, :]
            for step_info in info["frame_infos"][frame][chunk_id]:
                cur, _ = _merge_once_with_info(cur, step_info)
            outputs.append(cur)
            start = end
            chunk_id += 1
    return torch.cat(outputs, dim=1)


def _merge_once_with_info(x, info):
    if info.get("identity", False):
        return x, info
    B, _, C = x.shape
    src = x[:, ::2, :]
    dst = x[:, 1::2, :].clone()
    unm = src.gather(1, info["unm_idx"].unsqueeze(-1).expand(-1, -1, C))
    src_sel = src.gather(1, info["src_idx"].unsqueeze(-1).expand(-1, -1, C))
    dst = dst.scatter_reduce(
        1,
        info["dst_idx"].unsqueeze(-1).expand(-1, -1, C),
        src_sel,
        reduce="mean",
        include_self=True,
    )
    return torch.cat([unm, dst], dim=1), info


def tome_unmerge_tokens(x, info):
    B, _, C = x.shape
    t, h, w = [int(v) for v in info["pn"]]
    hw = h * w
    chunk_size = info["chunk_size"]
    outputs = []
    offset = 0
    for frame in range(t):
        start = 0
        chunk_id = 0
        while start < hw:
            end = min(hw, start + chunk_size)
            step_infos = info["frame_infos"][frame][chunk_id]
            cur_len = end - start
            for step_info in step_infos:
                cur_len = len_after_merge(cur_len, step_info)
            cur = x[:, offset:offset + cur_len, :]
            offset += cur_len
            for step_info in reversed(step_infos):
                cur = _unmerge_once(cur, step_info)
            outputs.append(cur)
            start = end
            chunk_id += 1
    return torch.cat(outputs, dim=1)


def len_after_merge(length, info):
    if info.get("identity", False):
        return length
    return info["unm_idx"].shape[1] + info["dst_len"]
