"""P2 gate: does local CT quality predict where tracing succeeds?

Open Problem 1 asks, verbatim, for "scan-quality metrics to identify
problematic regions".  A traced surface carries its own label for this: cells
where tracing succeeded hold coordinates, and interior holes -- invalid cells
surrounded by valid ones -- are places the tracer gave up while working in
that neighbourhood.

So: interpolate a 3D position for each interior hole from its valid
neighbours, measure local CT quality there, and compare against quality at
successfully traced cells.  If the two separate, scan quality predicts
traceability and the metric is worth building.  If they overlap, it does not.

Quality is measured with primitives already validated in this project:
  planarity    -- structure-tensor sheet-likeness (0.88 on clean papyrus)
  period       -- autocorrelation strength of the wrap periodicity
  contrast     -- spread of intensity along the surface normal
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sheetcheck.io import BUCKET, Surface, Volume  # noqa: E402
from sheetcheck.orient import sheet_normals, structure_tensor  # noqa: E402
from sheetcheck.profile import dominant_period  # noqa: E402

SCROLL = "PHerc1667"
SEGMENT = ("20260612121456-w011_20260108140509268_merged_v4_flatboi"
           "_straightened_v4")
MESH = "20260612121456-on-20251217075048-2.399um.tifxyz"
VOLUME = "20251217075048-2.399um-0.2m-78keV-masked.zarr"


def interior_holes(valid: np.ndarray, min_support: int = 6) -> np.ndarray:
    """Invalid cells whose neighbourhood is mostly valid: tracing failed there.

    Cells outside the traced region entirely are excluded -- they are not
    failures, just outside the working area.
    """
    inv = ~valid
    support = ndimage.uniform_filter(valid.astype(np.float32), size=7)
    return inv & (support >= min_support / 49.0 * 7.0 / 7.0) & (support > 0.55)


def interpolate_positions(points: np.ndarray, valid: np.ndarray,
                          targets: np.ndarray) -> np.ndarray:
    """Fill positions at ``targets`` from a local mean of valid neighbours."""
    out = np.zeros(points.shape, dtype=np.float64)
    w = valid.astype(np.float32)
    for c in range(3):
        num = ndimage.uniform_filter(np.where(valid, points[..., c], 0.0)
                                     .astype(np.float32), size=9)
        den = ndimage.uniform_filter(w, size=9)
        out[..., c] = np.where(den > 1e-6, num / np.maximum(den, 1e-6), np.nan)
    return out[targets]


def measure(vol, pts, nrm, offs, step_vox):
    """Return (planarity, period strength, contrast) at given points."""
    rays = pts[:, None, :] + offs[None, :, None] * nrm[:, None, :]
    lo = rays.reshape(-1, 3).min(axis=0) - 8
    hi = rays.reshape(-1, 3).max(axis=0) + 8
    block, blo = vol.read_box(lo, hi)
    if block.size == 0 or min(block.shape) < 16:
        return None
    J = structure_tensor(block, grad_sigma=1.0, tensor_sigma=2.5)
    _, planar = sheet_normals(J, pts - blo)
    prof = Volume.sample_box(block, blo, rays)

    strength, contrast = [], []
    for p in prof:
        if np.count_nonzero(p) < len(p) * 0.5:
            strength.append(np.nan)
            contrast.append(np.nan)
            continue
        _, s = dominant_period(p, step_vox, vol.voxel_size_um)
        strength.append(s)
        inside = p[p > 0]
        contrast.append(float(inside.std() / max(inside.mean(), 1e-6)))
    return planar, np.array(strength), np.array(contrast), block.nbytes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches", type=int, default=14)
    ap.add_argument("--per-patch", type=int, default=30)
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--reach-um", type=float, default=700.0)
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    surf = Surface.load(f"{BUCKET}/{SCROLL}/segments/{SEGMENT}/mesh/{MESH}",
                        cache_dir=str(root / "cache"))
    vol = Volume(f"{BUCKET}/{SCROLL}/volumes/{VOLUME}", level=args.level)
    vox = vol.voxel_size_um
    print(f"grid {surf.shape}, valid {surf.valid.mean():.1%}, "
          f"voxel {vox:.3f} um (level {args.level})")

    holes = interior_holes(surf.valid)
    print(f"interior holes (tracing failed): {holes.sum()} cells "
          f"({holes.mean():.2%} of grid)")
    if holes.sum() < 500:
        print("too few interior holes to evaluate")
        return 1

    normals, nok = surf.normals()
    # Holes have no normal of their own; borrow the local mean direction.
    nsm = np.stack([ndimage.uniform_filter(
        np.where(nok, normals[..., c], 0.0).astype(np.float32), size=9)
        for c in range(3)], axis=-1)
    nn = np.linalg.norm(nsm, axis=-1, keepdims=True)
    nsm = nsm / np.where(nn > 1e-6, nn, 1.0)

    hole_pos = np.full(surf.points.shape, np.nan)
    hole_pos[holes] = interpolate_positions(surf.points, surf.valid, holes)

    offs = np.arange(-args.reach_um / vox, args.reach_um / vox + 1e-9, 0.5)
    rng = np.random.default_rng(args.seed)
    hv, hu = np.nonzero(holes)
    gv, gu = np.nonzero(nok & surf.valid)

    acc = {"hole": [[], [], []], "traced": [[], [], []]}
    mb = 0.0
    t0 = time.time()
    for pi in range(args.patches):
        c = int(rng.integers(0, len(hv)))
        cv, cu = int(hv[c]), int(hu[c])
        r = 12
        vs = slice(max(cv - r, 0), cv + r + 1)
        us = slice(max(cu - r, 0), cu + r + 1)

        for label, mask, src in (("hole", holes[vs, us], hole_pos[vs, us]),
                                 ("traced", (nok & surf.valid)[vs, us],
                                  surf.points[vs, us])):
            sel = mask & np.all(np.isfinite(src), axis=-1)
            if sel.sum() < 5:
                continue
            pts = src[sel]
            nrm = nsm[vs, us][sel]
            if len(pts) > args.per_patch:
                s = rng.choice(len(pts), args.per_patch, replace=False)
                pts, nrm = pts[s], nrm[s]
            res = measure(vol, vol.to_level(pts), nrm, offs, 0.5)
            if res is None:
                continue
            pl, st, ct, nb = res
            mb += nb / 1e6
            acc[label][0].extend(np.atleast_1d(pl).tolist())
            acc[label][1].extend(st.tolist())
            acc[label][2].extend(ct.tolist())
        print(f"  patch {pi+1}/{args.patches}: "
              f"hole {len(acc['hole'][0])}, traced {len(acc['traced'][0])}, "
              f"{mb:.0f} MB, {time.time()-t0:.0f}s")

    names = ["planarity", "period strength", "contrast"]
    print("\n=== CT quality: traced regions vs tracing failures ===")
    print(f"  {'metric':>16} {'traced median':>14} {'hole median':>13} {'AUC':>7}")
    aucs = []
    for i, nm in enumerate(names):
        a = np.array(acc["traced"][i], dtype=float)
        b = np.array(acc["hole"][i], dtype=float)
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if len(a) < 30 or len(b) < 30:
            print(f"  {nm:>16}  insufficient samples")
            continue
        # Mann-Whitney AUC: P(traced sample scores higher than a hole sample)
        allv = np.concatenate([a, b])
        ranks = np.argsort(np.argsort(allv)) + 1
        ra = ranks[:len(a)].sum()
        auc = (ra - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))
        aucs.append(auc)
        print(f"  {nm:>16} {np.median(a):>14.3f} {np.median(b):>13.3f} "
              f"{auc:>7.3f}")

    np.savez(root / "results" / "p2_quality.npz",
             **{f"{k}_{n}": np.array(acc[k][i], dtype=float)
                for k in acc for i, n in enumerate(["pl", "st", "ct"])})

    best = max((abs(a - 0.5) for a in aucs), default=0.0)
    print(f"\n  best separation: AUC deviates {best:.3f} from chance")
    ok = best >= 0.10
    print(f"\nP2 GATE: {'PASS' if ok else 'FAIL'} "
          f"(CT quality predicts traceability)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
