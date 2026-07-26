"""M1 foundation test: is the traced surface a consistent spiral?

Three predictions, each falsifiable:

1. The unwrapped azimuth should span roughly the number of turns implied by the
   segment's arc length divided by its circumference.
2. Radius must increase monotonically with unwrapped azimuth (a spiral).
3. The slope dr/dtheta times 2*pi must equal the winding pitch measured
   independently by ray-marching (~208-220 um for PHerc1667).

If (3) holds, the along-surface half of the holonomy check is trustworthy and
the detector can be built on it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sheetcheck.axis import umbilicus_from_surface  # noqa: E402
from sheetcheck.io import BUCKET, Surface  # noqa: E402
from sheetcheck.winding import cylindrical, turns_spanned, unwrap_grid  # noqa: E402

SCROLL = "PHerc1667"
SEGMENT = "20260108140509-w011_20260108140509268_flatboi"
MESH = "20260108140509-on-20251217075048-2.399um.tifxyz"
VOX_UM = 2.399  # level-0 voxel size for this volume


def main() -> int:
    surf = Surface.load(f"{BUCKET}/{SCROLL}/segments/{SEGMENT}/mesh/{MESH}")
    pts, valid = surf.points, surf.valid
    print(f"surface grid {surf.shape}, valid {valid.mean():.1%}")

    origin, axis, fits = umbilicus_from_surface(pts[valid])
    print(f"umbilicus {origin.round(1)}  axis {axis.round(4)}")

    flat = pts.reshape(-1, 3)
    r, th, h = cylindrical(flat, origin, axis)
    r = r.reshape(valid.shape)
    th = th.reshape(valid.shape)
    h = h.reshape(valid.shape)

    thu, ok = unwrap_grid(th, valid, axis_u=1)
    ok &= valid
    print(f"unwrapped cells: {ok.sum()} ({ok.mean():.1%})")

    turns = turns_spanned(thu, ok)
    print(f"\n[1] turns spanned = {turns:.2f}")
    arc_mm = surf.shape[1] * surf.grid_step_vox * VOX_UM / 1000.0
    r_med_mm = np.median(r[ok]) * VOX_UM / 1000.0
    print(f"    arc length {arc_mm:.1f} mm at median radius {r_med_mm:.2f} mm "
          f"-> expected {arc_mm/(2*np.pi*r_med_mm):.2f} turns")

    # [2] radius vs unwrapped azimuth
    tt = thu[ok]
    rr = r[ok]
    order = np.argsort(tt)
    tt, rr = tt[order], rr[order]

    nb = 40
    edges = np.linspace(tt.min(), tt.max(), nb + 1)
    bt, br = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (tt >= a) & (tt < b)
        if m.sum() > 200:
            bt.append(0.5 * (a + b))
            br.append(np.median(rr[m]))
    bt, br = np.array(bt), np.array(br)
    print(f"\n[2] binned radius profile ({len(bt)} bins):")
    step = max(1, len(bt) // 10)
    for i in range(0, len(bt), step):
        print(f"    theta {bt[i]:+7.2f} rad  ({bt[i]/(2*np.pi):+5.2f} turns)"
              f"   r = {br[i]*VOX_UM/1000:6.3f} mm")
    dr = np.diff(br)
    print(f"    monotonic increasing: {np.mean(dr > 0):.0%} of bins")

    # [3] pitch from the spiral slope
    A = np.column_stack([bt, np.ones_like(bt)])
    slope, intercept = np.linalg.lstsq(A, br, rcond=None)[0]
    pitch_um = slope * 2 * np.pi * VOX_UM
    resid = br - (A @ [slope, intercept])
    print(f"\n[3] dr/dtheta = {slope:.2f} vox/rad")
    print(f"    pitch = 2*pi*dr/dtheta = {pitch_um:.1f} um")
    print(f"    linear-fit residual rms = {np.std(resid)*VOX_UM:.1f} um")
    print("    independent ray-march estimates: 208 um (radial), 220 um (normal)")

    agree = 0.80 <= pitch_um / 214.0 <= 1.25
    print(f"\nM1 FOUNDATION: {'PASS' if agree else 'FAIL'} "
          f"(spiral slope vs ray-march pitch)")
    return 0 if agree else 1


if __name__ == "__main__":
    raise SystemExit(main())
