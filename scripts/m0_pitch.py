"""M0 gate: measure inter-sheet (winding) pitch by ray-marching surface normals.

If the geometry plumbing is right, the spacing between consecutive papyrus
sheets along a traced surface's normal should reproduce the collection-wide
winding pitch reported by the winding-ruler atlas (median 187.3 um, IQR
181.5-193.4).  If it does not, something upstream is wrong and there is no
point building a detector on top of it.

Rays are cast in local patches so that one bounding-box fetch serves many
rays; sampling S3 one ray at a time is ~100x slower.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sheetcheck.io import BUCKET, Surface, Volume  # noqa: E402
from sheetcheck.profile import dominant_period  # noqa: E402

SCROLL = "PHerc1667"
SEGMENT = "20260108140509-w011_20260108140509268_flatboi"
MESH = "20260108140509-on-20251217075048-2.399um.tifxyz"
VOLUME = "20251217075048-2.399um-0.2m-78keV-masked.zarr"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches", type=int, default=12,
                    help="number of local neighbourhoods to sample")
    ap.add_argument("--rays-per-patch", type=int, default=24)
    ap.add_argument("--patch-radius", type=int, default=6,
                    help="half-width of a patch, in grid cells")
    ap.add_argument("--reach-vox", type=float, default=260.0)
    ap.add_argument("--step-vox", type=float, default=0.5)
    ap.add_argument("--level", type=int, default=1,
                    help="pyramid level; 1 is plenty to resolve a ~187um pitch "
                         "and fetches 8x less data than level 0")
    ap.add_argument("--min-strength", type=float, default=0.15,
                    help="minimum normalised autocorrelation peak to trust a ray")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    mesh_path = f"{BUCKET}/{SCROLL}/segments/{SEGMENT}/mesh/{MESH}"
    vol_path = f"{BUCKET}/{SCROLL}/volumes/{VOLUME}"

    t0 = time.time()
    surf = Surface.load(mesh_path)
    print(f"surface {MESH}")
    print(f"  grid={surf.shape} valid={surf.valid.mean()*100:.1f}% "
          f"grid_step={surf.grid_step_vox:.2f} vox  ({time.time()-t0:.1f}s)")

    vol = Volume(vol_path, level=args.level)
    vox_um = vol.voxel_size_um
    print(f"volume shape={vol.shape} chunks={vol.array.chunks} "
          f"dtype={vol.array.dtype}  {vox_um:.4f} um/vox (level {args.level})")

    normals, ok = surf.normals()
    print(f"  cells with usable normals: {ok.sum()}")

    rng = np.random.default_rng(args.seed)
    vi, ui = np.nonzero(ok)
    r = args.patch_radius
    offs = np.arange(-args.reach_vox, args.reach_vox + 1e-9, args.step_vox)

    all_pitch: list[float] = []
    per_ray_sheets: list[int] = []
    t1 = time.time()
    fetched_mb = 0.0

    for pi in range(args.patches):
        c = rng.integers(0, len(vi))
        cv, cu = int(vi[c]), int(ui[c])
        vs = slice(max(cv - r, 0), cv + r + 1)
        us = slice(max(cu - r, 0), cu + r + 1)
        m = ok[vs, us]
        if m.sum() < 4:
            continue
        pts = surf.points[vs, us][m]
        nrm = normals[vs, us][m]
        if len(pts) > args.rays_per_patch:
            sel = rng.choice(len(pts), args.rays_per_patch, replace=False)
            pts, nrm = pts[sel], nrm[sel]
        # tifxyz stores level-0 coordinates; offsets below are in level voxels.
        pts = vol.to_level(pts)

        rays = pts[:, None, :] + offs[None, :, None] * nrm[:, None, :]
        lo = rays.reshape(-1, 3).min(axis=0)
        hi = rays.reshape(-1, 3).max(axis=0)
        block, blo = vol.read_box(lo, hi)
        if block.size == 0:
            continue
        fetched_mb += block.nbytes / 1e6
        prof = Volume.sample_box(block, blo, rays)

        for p in prof:
            if np.count_nonzero(p) < len(p) * 0.6:
                continue  # ray leaves the masked scroll
            period, strength = dominant_period(p, args.step_vox, vox_um)
            if np.isfinite(period) and strength >= args.min_strength:
                all_pitch.append(period)
                per_ray_sheets.append(strength)

        print(f"  patch {pi+1}/{args.patches}: {len(all_pitch)} pitch samples, "
              f"{fetched_mb:.0f} MB fetched, {time.time()-t1:.0f}s")

    g = np.array(all_pitch)
    s = np.array(per_ray_sheets)

    print(f"\n=== pitch from {len(g)} per-ray period estimates ===")
    if len(g) < 20:
        print("TOO FEW SAMPLES -- gate FAILED")
        return 1

    # Is autocorrelation strength a usable confidence measure?  If it is, the
    # spread should collapse as the strength floor rises -- which would give
    # the detector a principled per-location reliability gate.
    print(f"  {'min strength':>12} {'n':>5} {'median':>9} {'IQR':>19} {'rel IQR':>9}")
    best = None
    for floor in (0.0, 0.15, 0.25, 0.35, 0.45, 0.55):
        sel = s >= floor
        if sel.sum() < 15:
            continue
        gg = g[sel]
        q1, med, q3 = np.percentile(gg, [25, 50, 75])
        rel = (q3 - q1) / med
        print(f"  {floor:>12.2f} {sel.sum():>5d} {med:>8.1f}u "
              f"{q1:>8.1f} - {q3:>6.1f}u {rel:>8.1%}")
        best = (floor, med, q1, q3, rel, int(sel.sum()))

    print("\n  winding-ruler reference: median 187.3 um, IQR 181.5 - 193.4 (rel IQR 6.4%)")

    floor, med, q1, q3, rel, n = best
    ok_gate = 165.0 <= med <= 215.0
    print(f"\nM0 GATE (median in [165, 215]): {'PASS' if ok_gate else 'FAIL'}")
    print(f"  strictest usable floor {floor:.2f}: median {med:.1f} um, "
          f"rel IQR {rel:.1%}, n={n}")
    return 0 if ok_gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
