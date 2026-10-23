"""Context-preserving pruning: pooled tokens equal the mean of the skipped positions in
each pool x pool window and carry the RoPE of the window centre."""
import torch
from infinity.models.st_token_pruning import build_context_tokens


def test_context_tokens_pooling():
    torch.manual_seed(0)
    t, h, w, C, pool = 3, 9, 10, 8, 4
    L = t * h * w
    x, xu = torch.randn(1, L, C), torch.randn(1, L, C)
    idx = torch.stack([torch.randperm(h * w)[:30] + f * h * w for f in range(t)]).reshape(1, -1).sort().values
    rope = torch.arange(L).float().view(1, 1, 1, 1, L, 1).expand(2, 1, 1, 1, L, 4).contiguous()
    c, u, r = build_context_tokens(x, xu, rope, (t, h, w), idx, pool)
    sel = torch.zeros(L, dtype=torch.bool)
    sel[idx[0]] = True
    hp, wp = -(-h // pool), -(-w // pool)
    windows = []
    for f in range(t):
        for i in range(hp):
            for j in range(wp):
                pos = [f * h * w + a * w + b for a in range(i * pool, min((i + 1) * pool, h))
                       for b in range(j * pool, min((j + 1) * pool, w)) if not sel[f * h * w + a * w + b]]
                if pos:
                    centre = f * h * w + min(i * pool + pool // 2, h - 1) * w + min(j * pool + pool // 2, w - 1)
                    windows.append((pos, centre))
    assert c.shape == (1, len(windows), C) and u.shape == c.shape and r.shape[4] == len(windows)
    for n, (pos, centre) in enumerate(windows):
        assert torch.allclose(c[0, n], x[0, pos].mean(0), atol=1e-5)
        assert int(r[0, 0, 0, 0, n, 0]) == centre


if __name__ == '__main__':
    test_context_tokens_pooling()
    print('ok')
