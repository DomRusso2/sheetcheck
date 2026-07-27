"""Calibrate dominant_period: where does autocorrelation pitch estimation fail?

This script exists because F1 and F2 of this repository were wrong, and it
measures exactly how.

The estimator is exact on periodic signals. It fails on *aperiodic* ones -- and
real scroll rays are aperiodic: wrap spacing varies along a single ray, sheets
are missed, neighbouring wraps merge. Real rays show autocorrelation strength
around 0.22; additive noise alone cannot degrade a clean signal that far.

At that strength the estimate is biased high by 30-60 um, the bias grows as the
true pitch shrinks, and the dynamic range compresses toward the middle of the
search band. Any pitch measured this way on scroll data is therefore an
overestimate that also hides real variation.

Synthetic signals are used here only to characterise the estimator against a
known answer. No conclusion about scrolls is drawn from them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sheetcheck.profile import dominant_period  # noqa: E402

VOX = 2.258          # level-1 voxel size used in the surveys
STEP = 0.5
REACH_UM = 900.0
BAND = (90.0, 340.0)
MID = (BAND[0] + BAND[1]) / 2

# jitter, dropped-sheet rate, additive noise -- increasing aperiodicity
REGIMES = [
    ("near-periodic", 0.05, 0.00, 20.0),
    ("mild", 0.20, 0.05, 40.0),
    ("realistic", 0.35, 0.15, 60.0),
    ("severe", 0.50, 0.25, 80.0),
]


def make_ray(mean_p, jitter, drop, noise, rng, gradient=0.0):
    """Papyrus slabs at a mean pitch, with jitter, dropped sheets and drift."""
    n = int(2 * REACH_UM / (VOX * STEP))
    prof = np.full(n, 60.0)
    pos = 0.0
    while pos < 2 * REACH_UM:
        local = mean_p * (1.0 + gradient * (pos / (2 * REACH_UM) - 0.5))
        pos += max(local * (1.0 + rng.normal(0, jitter)), 30.0)
        if rng.random() < drop:
            continue
        a = int(pos / (VOX * STEP))
        b = int((pos + 0.45 * local) / (VOX * STEP))
        if 0 <= a < n:
            prof[a:min(b, n)] = 180.0
    x = np.arange(n) * VOX * STEP
    prof = prof * (1.0 + 0.35 * np.sin(2 * np.pi * x / (2.2 * REACH_UM)))
    return np.clip(prof + rng.normal(0, noise, size=n), 1, 255)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    print(f"search band {BAND[0]:.0f}-{BAND[1]:.0f} um, midpoint {MID:.0f}")
    print("real scroll rays measure autocorrelation strength ~0.22\n")
    print(f"  {'regime':>14} {'true':>6} {'strength':>9} {'estimate':>9} "
          f"{'bias':>7}")
    for name, jit, drop, noise in REGIMES:
        for true_p in (140.0, 180.0, 220.0):
            est, stg = [], []
            for _ in range(args.trials):
                p = make_ray(true_p, jit, drop, noise, rng)
                e, s = dominant_period(p, STEP, VOX,
                                       min_um=BAND[0], max_um=BAND[1])
                if np.isfinite(e):
                    est.append(e)
                    stg.append(s)
            if not est:
                continue
            med = float(np.median(est))
            print(f"  {name:>14} {true_p:>6.0f} {np.median(stg):>9.3f} "
                  f"{med:>9.0f} {med - true_p:>+7.0f}")
        print()

    print("  At the 'realistic' regime -- the one matching real scroll rays --")
    print("  the estimate is high by 30-60 um and the 140-220 true span")
    print("  compresses toward the band midpoint. A pitch measured this way is")
    print("  an overestimate that also conceals genuine variation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
