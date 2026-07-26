"""P3 gate: can an off-papyrus run localise an injected sheet switch?

Two numbers decide it:

  specificity -- on an untouched trace, what fraction of cells score as
                 unsupported?  This is the false-positive rate.
  recall      -- inject a switch as a ramp across the gap; are the ramp cells
                 flagged?

Unlike the azimuth-based detector this uses no global winding coordinate, so
scroll deformation cannot break it.  Everything is measured per ray against
that ray's own contrast.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sheetcheck.io import BUCKET, Surface, Volume  # noqa: E402
from sheetcheck.support import ramp_switch, support_scores  # noqa: E402

SCROLL = "PHerc1667"
SEGMENT = "20260108140509-w011_20260108140509268_flatboi"
MESH = "20260108140509-on-20251217075048-2.399um.tifxyz"
VOLUME = "20251217075048-2.399um-0.2m-78keV-masked.zarr"


def sample_rows(surf, vol, normals, ok, rows, ucols, offs, pts_override=None):
    """Return support scores for a block of grid cells (rows x ucols)."""
    src = surf.points if pts_override is None else pts_override
    sel_v, sel_u = np.meshgrid(rows, ucols, indexing="ij")
    m = ok[sel_v, sel_u]
    if m.sum() < 10:
        return None
    p0 = src[sel_v, sel_u][m]
    nn = normals[sel_v, sel_u][m]
    pl = vol.to_level(p0)
    rays = pl[:, None, :] + offs[None, :, None] * nn[:, None, :]
    block, blo = vol.read_box(rays.reshape(-1, 3).min(axis=0),
                              rays.reshape(-1, 3).max(axis=0))
    if block.size == 0:
        return None
    prof = Volume.sample_box(block, blo, rays)
    score, valid = support_scores(prof, len(offs) // 2)
    grid = np.full(sel_v.shape, np.nan)
    gv = np.full(sel_v.shape, False)
    grid[m] = score
    gv[m] = valid
    return grid, gv, block.nbytes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", type=int, default=6)
    ap.add_argument("--rows", type=int, default=12)
    ap.add_argument("--cols", type=int, default=140)
    ap.add_argument("--reach-um", type=float, default=260.0)
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--thresh", type=float, default=0.15)
    ap.add_argument("--ramp", type=int, default=12)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    surf = Surface.load(f"{BUCKET}/{SCROLL}/segments/{SEGMENT}/mesh/{MESH}",
                        cache_dir=str(root / "cache"))
    vol = Volume(f"{BUCKET}/{SCROLL}/volumes/{VOLUME}", level=args.level)
    vox = vol.voxel_size_um
    normals, ok = surf.normals()
    offs = np.arange(-args.reach_um / vox, args.reach_um / vox + 1e-9, 0.5)
    pitch_vox = 210.0 / vox
    print(f"voxel {vox:.3f} um; ray +/-{args.reach_um:.0f} um; "
          f"1 pitch = {pitch_vox:.0f} vox")
    print(f"ramp width {args.ramp} cols "
          f"({args.ramp * surf.grid_step_vox * vox / 1000:.2f} mm of arc)\n")

    rng = np.random.default_rng(args.seed)
    vi, _ = np.nonzero(ok)
    clean_scores, inj_scores, ramp_flags, ctrl_flags = [], [], [], []
    mb = 0.0
    t0 = time.time()

    for b in range(args.bands):
        cv = int(rng.choice(vi))
        rows = np.arange(max(cv - args.rows // 2, 1),
                         min(cv + args.rows // 2, surf.shape[0] - 1))
        cols_all = np.nonzero(ok[rows].any(axis=0))[0]
        if len(cols_all) < args.cols + args.ramp + 10:
            continue
        start = int(rng.integers(0, len(cols_all) - args.cols - args.ramp - 5))
        ucols = cols_all[start:start + args.cols]

        res = sample_rows(surf, vol, normals, ok, rows, ucols, offs)
        if res is None:
            continue
        g, gv, nb = res
        mb += nb / 1e6
        clean_scores.extend(g[gv].tolist())

        # Inject a ramp beginning a third of the way into this window.
        u0 = int(ucols[len(ucols) // 3])
        inj_pts = ramp_switch(surf.points, ok, normals, u0, args.ramp,
                              pitch_vox)
        res2 = sample_rows(surf, vol, normals, ok, rows, ucols, offs,
                           pts_override=inj_pts)
        if res2 is None:
            continue
        g2, gv2, nb2 = res2
        mb += nb2 / 1e6
        inj_scores.extend(g2[gv2].tolist())

        in_ramp = (ucols >= u0) & (ucols < u0 + args.ramp)
        for r in range(g2.shape[0]):
            row_valid = gv2[r]
            flagged = np.zeros(g2.shape[1], dtype=bool)
            flagged[row_valid] = g2[r][row_valid] < args.thresh
            if in_ramp.sum():
                ramp_flags.append(float(flagged[in_ramp].mean()))
            ctrl = row_valid & ~in_ramp
            if ctrl.sum() > 5:
                ctrl_flags.append(float(flagged[ctrl].mean()))

        print(f"  band {b+1}/{args.bands}: clean {len(clean_scores)}, "
              f"inj {len(inj_scores)}, {mb:.0f} MB, {time.time()-t0:.0f}s")

    c = np.array(clean_scores)
    i = np.array(inj_scores)
    if len(c) < 200:
        print("too few samples")
        return 1

    print(f"\n=== support score on an untouched trace (n={len(c)}) ===")
    for q in (1, 5, 10, 25, 50, 75):
        print(f"  p{q:<3d} = {np.percentile(c, q):+.3f}")
    print(f"  fraction below {args.thresh}: {np.mean(c < args.thresh):.2%} "
          f"<- false-positive rate")

    print("\n=== detection of an injected ramp ===")
    rf = np.array(ramp_flags)
    cf = np.array(ctrl_flags)
    print(f"  flagged inside ramp   : {rf.mean():.1%}  (recall)")
    print(f"  flagged outside ramp  : {cf.mean():.1%}  (false positive)")

    print(f"\n  {'thresh':>7} {'recall':>8} {'FP':>8}")
    for t in (0.05, 0.10, 0.15, 0.25, 0.35, 0.50):
        r = float(np.mean(i[np.isfinite(i)] < t))
        f = float(np.mean(c < t))
        print(f"  {t:>7.2f} {r:>7.1%} {f:>7.1%}")

    np.savez(root / "results" / "p3_support.npz", clean=c, injected=i,
             ramp_flags=rf, ctrl_flags=cf)

    ok_gate = rf.mean() >= 0.60 and cf.mean() <= 0.15
    print(f"\nP3 GATE: {'PASS' if ok_gate else 'FAIL'} "
          f"(recall >=60% and FP <=15%)")
    return 0 if ok_gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
