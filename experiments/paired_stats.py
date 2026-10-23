"""Paired fidelity statistics between methods evaluated against the same dense reference.

Each input is a results.json written by tools/eval/eval_full_reference_metrics.py.
Videos are paired by file name, so all methods must use the same prompts and seeds,
and the dense reference must be generated on the same GPU type (sampling diverges
across hardware).

    python experiments/paired_stats.py ours=out/fidelity/ours/results.json \
        faststar=out/fidelity/faststar/results.json [more=...]

The first method is compared against every other one.
"""
import json
import sys

import numpy as np
from scipy.stats import wilcoxon

METRICS = (('PSNR', 1), ('SSIM', 1), ('LPIPS', -1))


def load(path):
    return {r['video']: r for r in json.load(open(path))['per_video']}


def main(argv):
    named = [a.split('=', 1) for a in argv]
    if len(named) < 2:
        sys.exit(__doc__)
    runs = {name: load(path) for name, path in named}
    print(f"{'method':14s} {'n':>5s} " + ' '.join(f'{m:>8s}' for m, _ in METRICS))
    for name, r in runs.items():
        print(f'{name:14s} {len(r):5d} ' + ' '.join(f"{np.mean([v[m] for v in r.values()]):8.4f}" for m, _ in METRICS))
    rng = np.random.default_rng(0)
    first, ref = named[0][0], runs[named[0][0]]
    print(f'\n{first} minus other (paired; bootstrap 95% CI, Wilcoxon p, win rate):')
    for name, other in list(runs.items())[1:]:
        keys = sorted(set(ref) & set(other))
        cols = []
        for m, sign in METRICS:
            d = np.array([ref[k][m] - other[k][m] for k in keys])
            boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            p = wilcoxon(d).pvalue if np.any(d) else 1.0
            cols.append(f'{m} {d.mean():+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.1e} win={100 * np.mean(d * sign > 0):.1f}%')
        print(f'  vs {name:12s} n={len(keys)} | ' + ' | '.join(cols))


if __name__ == '__main__':
    main(sys.argv[1:])
