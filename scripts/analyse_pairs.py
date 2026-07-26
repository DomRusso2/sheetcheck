"""Analyse saved holonomy pairs: how separable is a sheet switch?

A switch should roughly double the radial gap.  The question is where the
clean-trace distribution's upper tail sits relative to 2x the local pitch --
that difference is the whole detection margin.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

p = Path(__file__).resolve().parents[1] / "results" / "m1_pairs.npz"
d = np.load(p)
gap = d["gap_um"]
sheets = d["sheets"]

print(f"n = {len(gap)} pairs from a trusted trace\n")
print("radial/chord gap percentiles (um):")
for q in (1, 5, 25, 50, 75, 90, 95, 99):
    print(f"  p{q:<3d} = {np.percentile(gap, q):7.1f}")

pitch = float(np.median(gap))
print(f"\nusing median gap as local pitch estimate: {pitch:.1f} um")
print(f"a one-winding switch would sit near {2*pitch:.0f} um\n")

print("false-positive rate if we threshold gap/pitch:")
for t in (1.3, 1.4, 1.5, 1.6, 1.75, 2.0):
    fp = float(np.mean(gap > t * pitch))
    print(f"  threshold {t:.2f}x pitch ({t*pitch:6.0f} um): FP = {fp:6.2%}")

print("\nair-gap count on the same pairs:")
vals, counts = np.unique(sheets, return_counts=True)
for v, c in zip(vals, counts):
    m = sheets == v
    print(f"  {int(v)} extra sheets: n={c:4d}  median gap {np.median(gap[m]):6.1f} um")
print(f"\n  fraction reporting >0 extra sheets: {np.mean(sheets > 0):.1%}")
print("  (with pairing now geometrically exact, these are gap-counter false")
print("   positives rather than real switches -- their chords say one pitch)")
