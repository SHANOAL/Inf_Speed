"""IDS temporal term: the column-sum variant scores the previous position of a moving
object, while the anchor variant follows the object to its current position."""
import torch
import torch.nn as nn
from infinity.models.st_token_pruning import compute_attention_importance


class _Attn(nn.Module):
    def __init__(self, C=64, H=4):
        super().__init__()
        self.num_heads, self.head_dim, self.num_key_value_heads = H, C // H, H
        self.q_proj = nn.Linear(C, C, bias=False)
        self.k_proj = nn.Linear(C, C, bias=False)
        self.q_proj.weight.data.copy_(torch.eye(C) * 3)
        self.k_proj.weight.data.copy_(torch.eye(C) * 3)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()


def test_anchor_follows_motion():
    torch.manual_seed(0)
    C, t, h, w = 64, 2, 8, 8
    hw = h * w
    att = _Attn(C)
    obj = torch.randn(C)
    obj = obj / obj.norm() * 4

    def clip(pos):
        x = torch.randn(1, t * hw, C) * 0.3
        for f in range(t):
            x[0, f * hw + pos] += obj
        return x

    old, new = 9, 50
    prev, cur = clip(old), clip(new)
    prev_imp = torch.zeros(1, hw)
    prev_imp[0, old] = 1.0
    kw = dict(w_spatial=0.0, w_temporal=1.0, w_text=0.0, fusion_p=1.0)
    col = compute_attention_importance(cur, (t, h, w), att, prev_clip_hidden=prev, temporal_mode='column', **kw)
    anc = compute_attention_importance(cur, (t, h, w), att, prev_clip_hidden=prev, temporal_mode='anchor',
                                       prev_importance=prev_imp, temporal_anchors=4, **kw)
    assert col.argmax().item() == old
    assert anc.argmax().item() == new


if __name__ == '__main__':
    test_anchor_follows_motion()
    print('ok')
