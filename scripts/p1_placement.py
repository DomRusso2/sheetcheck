"""P1 gate: does surface placement predict ink detectability?

Open Problem 6 lists six candidate causes for ink models failing to
generalise -- scan quality, *surface misplacement*, label mismatch,
architecture limits, ink morphology, or plain signal absence -- and says the
project cannot tell them apart.  Surface misplacement is the one that is
purely geometric, and it is measurable.

The published ink raster for a segment is pixel-exact with its tifxyz grid at
20x resolution (1975x736 grid <-> 39500x14720 raster), so every mesh cell has
both a 3D position and an ink score.  For each sampled cell this measures:

  offset  -- signed distance from the traced point to the centre of the
             papyrus sheet it sits on, along the surface normal
  ink     -- mean and local contrast of the published ink output over the
             20x20 pixel block that cell owns

Ink *contrast* matters more than ink mean: letters create local variance,
whereas a misplaced surface renders flat and featureless regardless of
whether text is present underneath.
"""

from __future__ import annotations

import argparse
import io as _io
import sys
import time
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sheetcheck.io import BUCKET, Surface, Volume, _read_bytes  # noqa: E402
from sheetcheck.profile import find_sheets  # noqa: E402

SCROLL = "PHerc1667"
SEGMENT = "20260108140509-w011_20260108140509268_flatboi"
MESH = "20260108140509-on-20251217075048-2.399um.tifxyz"
VOLUME = "20251217075048-2.399um-0.2m-78keV-masked.zarr"
INK = ("PHerc1667-20260108140509-2.399um-0.22m-78keV-volume-20251217075048"
       "-20260417190342-new_canon_autoresearch_recipe-tile256-stride128.tif")


def load_ink(cache_dir: Path) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    npy = cache_dir / "ink_w011_2399.npy"
    if npy.exists():
        return np.load(npy, mmap_mode="r")
    raw = _read_bytes(f"{BUCKET}/{SCROLL}/segments/{SEGMENT}/ink-detection/{INK}")
    arr = tifffile.imread(_io.BytesIO(raw))
    np.save(npy, arr)
    return np.load(npy, mmap_mode="r")


def signed_offset(profile, step_vox, vox_um, centre_idx):
    """Signed distance (um) from the ray centre to the nearest sheet centre."""
    sheets = find_sheets(profile, step_vox, vox_um, min_thickness_um=25.0)
    if len(sheets) == 0:
        return np.nan
    k = int(np.argmin(np.abs(sheets)))
    return float(sheets[k]) * vox_um


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches", type=int, default=14)
    ap.add_argument("--rays", type=int, default=60)
    ap.add_argument("--reach-um", type=float, default=160.0)
    ap.add_argument("--ink-window", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    cache = root / "cache"
    surf = Surface.load(f"{BUCKET}/{SCROLL}/segments/{SEGMENT}/mesh/{MESH}",
                        cache_dir=str(cache))
    vol = Volume(f"{BUCKET}/{SCROLL}/volumes/{VOLUME}", level=0)
    vox = vol.voxel_size_um

    t0 = time.time()
    ink = load_ink(cache)
    print(f"ink raster {ink.shape} {ink.dtype}  ({time.time()-t0:.0f}s)")
    print(f"tifxyz grid {surf.shape}, step {surf.grid_step_vox:.0f} vox")
    sy = ink.shape[0] / surf.shape[0]
    sx = ink.shape[1] / surf.shape[1]
    print(f"raster/grid ratio = {sy:.2f} x {sx:.2f}  "
          f"({'exact' if abs(sy-round(sy))<1e-6 and abs(sx-round(sx))<1e-6 else 'NOT exact'})")
    sy, sx = int(round(sy)), int(round(sx))

    normals, ok = surf.normals()
    rng = np.random.default_rng(args.seed)
    vi, ui = np.nonzero(ok)
    offs = np.arange(-args.reach_um / vox, args.reach_um / vox + 1e-9, 0.5)

    rec_off, rec_mean, rec_std, rec_surfval = [], [], [], []
    for pi in range(args.patches):
        c = int(rng.integers(0, len(vi)))
        cv, cu = int(vi[c]), int(ui[c])
        r = 7
        vs = slice(max(cv - r, 0), cv + r + 1)
        us = slice(max(cu - r, 0), cu + r + 1)
        m = ok[vs, us]
        if m.sum() < 6:
            continue
        gv, gu = np.nonzero(m)
        gv = gv + max(cv - r, 0)
        gu = gu + max(cu - r, 0)
        p0 = surf.points[gv, gu]
        nn = normals[gv, gu]
        if len(p0) > args.rays:
            s = rng.choice(len(p0), args.rays, replace=False)
            p0, nn, gv, gu = p0[s], nn[s], gv[s], gu[s]

        rays = p0[:, None, :] + offs[None, :, None] * nn[:, None, :]
        block, blo = vol.read_box(rays.reshape(-1, 3).min(axis=0),
                                  rays.reshape(-1, 3).max(axis=0))
        if block.size == 0:
            continue
        prof = Volume.sample_box(block, blo, rays)
        centre = len(offs) // 2

        for k in range(len(p0)):
            pr = prof[k]
            if np.count_nonzero(pr) < len(pr) * 0.7:
                continue
            off = signed_offset(pr, 0.5, vox, centre)
            if not np.isfinite(off):
                continue
            # Ink contrast must be measured over a letter-sized neighbourhood.
            # A cell owns only a 20x20 px block (48 um); letters are 1-2 mm,
            # i.e. 400-800 px, so a per-cell block sits entirely inside one
            # stroke and carries no letter structure at all.
            cy, cx = int(gv[k]) * sy + sy // 2, int(gu[k]) * sx + sx // 2
            h = args.ink_window // 2
            y0, y1 = max(cy - h, 0), min(cy + h, ink.shape[0])
            x0, x1 = max(cx - h, 0), min(cx + h, ink.shape[1])
            tile = np.asarray(ink[y0:y1, x0:x1], dtype=np.float32)
            if tile.size < 64:
                continue
            rec_off.append(off)
            rec_mean.append(float(tile.mean()))
            rec_std.append(float(tile.std()))
            rec_surfval.append(float(pr[centre]))
        print(f"  patch {pi+1}/{args.patches}: {len(rec_off)} samples "
              f"({time.time()-t0:.0f}s)")

    off = np.array(rec_off)
    imean = np.array(rec_mean)
    istd = np.array(rec_std)
    sval = np.array(rec_surfval)
    if len(off) < 50:
        print("too few samples")
        return 1

    print(f"\n=== placement of a published trace (n={len(off)}) ===")
    a = np.abs(off)
    print(f"  |offset| from sheet centre: median {np.median(a):.1f} um  "
          f"p90 {np.percentile(a,90):.1f}  max {a.max():.1f}")
    print(f"  signed offset: median {np.median(off):+.1f} um "
          f"(bias => trace sits consistently to one side)")
    print(f"  fraction beyond 30 um: {np.mean(a > 30):.1%}")

    print(f"\n=== ink vs placement ===")
    print(f"  {'|offset| um':>12} {'n':>5} {'ink mean':>10} {'ink contrast':>13}")
    edges = [0, 10, 20, 30, 45, 60, 1e9]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (a >= lo) & (a < hi)
        if m.sum() < 8:
            continue
        lbl = f"{lo:.0f}-{hi:.0f}" if hi < 1e8 else f">{lo:.0f}"
        print(f"  {lbl:>12} {m.sum():>5d} {imean[m].mean():>10.1f} "
              f"{istd[m].mean():>13.2f}")

    def safe_corr(x, y):
        if len(x) < 8 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    c_mean = safe_corr(a, imean)
    c_std = safe_corr(a, istd)
    print(f"\n  corr(|offset|, ink mean)     = {c_mean:+.3f}")
    print(f"  corr(|offset|, ink contrast) = {c_std:+.3f}")
    print(f"  corr(CT value at surface, ink contrast) = "
          f"{safe_corr(sval, istd):+.3f}")

    np.savez(root / "results" / "p1_placement.npz",
             offset=off, ink_mean=imean, ink_std=istd, surf_val=sval)

    strong = (np.isfinite(c_std) and abs(c_std) >= 0.15) or \
             (np.isfinite(c_mean) and abs(c_mean) >= 0.15)
    print(f"\nP1 GATE: {'PASS' if strong else 'FAIL'} "
          f"(placement relates to ink detectability)")
    return 0 if strong else 1


if __name__ == "__main__":
    raise SystemExit(main())
