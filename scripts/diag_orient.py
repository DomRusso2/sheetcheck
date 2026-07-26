"""Decisive check: mesh normals vs. CT structure-tensor sheet normals.

If the mesh-derived normal is a good estimate of the true sheet normal, rays
cast along it cross the wraps perpendicularly and the measured pitch is
unbiased.  If it is systematically oblique, every pitch estimate is inflated
by 1/cos(theta) and no amount of estimator tuning will fix it.

Also re-measures pitch along BOTH normals so the effect is quantified rather
than argued.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sheetcheck.io import BUCKET, Surface, Volume  # noqa: E402
from sheetcheck.orient import (  # noqa: E402
    align_sign,
    angle_between,
    sheet_normals,
    structure_tensor,
)
from sheetcheck.profile import dominant_period  # noqa: E402

SCROLL = "PHerc1667"
SEGMENT = "20260108140509-w011_20260108140509268_flatboi"
MESH = "20260108140509-on-20251217075048-2.399um.tifxyz"
VOLUME = "20251217075048-2.399um-0.2m-78keV-masked.zarr"

LEVEL = 1
REACH = 130.0
STEP = 0.5
PATCHES = 10
RAYS = 24
RADIUS = 5


def main() -> int:
    surf = Surface.load(f"{BUCKET}/{SCROLL}/segments/{SEGMENT}/mesh/{MESH}")
    vol = Volume(f"{BUCKET}/{SCROLL}/volumes/{VOLUME}", level=LEVEL)
    vox_um = vol.voxel_size_um
    mesh_n, ok = surf.normals()
    print(f"level {LEVEL}: {vox_um:.3f} um/vox; expected pitch 187 um "
          f"= {187/vox_um:.1f} vox\n")

    rng = np.random.default_rng(11)
    vi, ui = np.nonzero(ok)
    offs = np.arange(-REACH, REACH + 1e-9, STEP)

    angles, planars = [], []
    pitch_mesh, pitch_st = [], []

    for pi in range(PATCHES):
        c = int(rng.integers(0, len(vi)))
        cv, cu = int(vi[c]), int(ui[c])
        vs = slice(max(cv - RADIUS, 0), cv + RADIUS + 1)
        us = slice(max(cu - RADIUS, 0), cu + RADIUS + 1)
        m = ok[vs, us]
        if m.sum() < 6:
            continue
        pts0 = surf.points[vs, us][m]
        nm = mesh_n[vs, us][m]
        if len(pts0) > RAYS:
            s = rng.choice(len(pts0), RAYS, replace=False)
            pts0, nm = pts0[s], nm[s]
        pts = vol.to_level(pts0)

        # Block must cover the rays AND a margin for the structure tensor.
        rays_m = pts[:, None, :] + offs[None, :, None] * nm[:, None, :]
        lo = rays_m.reshape(-1, 3).min(axis=0) - 8
        hi = rays_m.reshape(-1, 3).max(axis=0) + 8
        block, blo = vol.read_box(lo, hi)
        if block.size == 0 or min(block.shape) < 12:
            continue

        J = structure_tensor(block)
        ns, planar = sheet_normals(J, pts - blo)
        ns = align_sign(ns, nm)
        good = (planar > 0.0) & (np.linalg.norm(ns, axis=-1) > 0.5)
        if not good.any():
            continue

        ang = angle_between(nm[good], ns[good])
        angles.extend(ang.tolist())
        planars.extend(planar[good].tolist())

        # Pitch along each normal, from the same block.
        for nvec, sink in ((nm, pitch_mesh), (ns, pitch_st)):
            r = pts[:, None, :] + offs[None, :, None] * nvec[:, None, :]
            prof = Volume.sample_box(block, blo, r)
            for k, p in enumerate(prof):
                if not good[k] or np.count_nonzero(p) < len(p) * 0.6:
                    continue
                per, strength = dominant_period(p, STEP, vox_um)
                if np.isfinite(per) and strength >= 0.2:
                    sink.append(per)

        print(f"  patch {pi+1}/{PATCHES}: n={len(angles)} angles, "
              f"mesh {len(pitch_mesh)} / st {len(pitch_st)} pitch")

    a = np.array(angles)
    print(f"\nangle(mesh normal, structure-tensor normal), n={len(a)}:")
    for q in (10, 25, 50, 75, 90):
        print(f"  p{q:<3d} = {np.percentile(a, q):6.1f} deg")
    print(f"  planarity: median {np.median(planars):.2f}")

    print("\npitch estimates:")
    for label, arr in (("mesh normal", np.array(pitch_mesh)),
                       ("struct-tensor normal", np.array(pitch_st))):
        if len(arr) < 10:
            print(f"  {label:<22}: too few ({len(arr)})")
            continue
        q1, med, q3 = np.percentile(arr, [25, 50, 75])
        print(f"  {label:<22}: n={len(arr):<4d} median {med:6.1f} um  "
              f"IQR {q1:.0f}-{q3:.0f}  rel {100*(q3-q1)/med:.0f}%")
    print("\n  reference: median 187.3 um, IQR 181.5-193.4 (rel 6.4%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
