"""P0 gate: is the recto/verso ply distinguishable in CT by fiber orientation?

Papyrus is two plies laid crosswise -- recto with fibers running along the
roll (circumferential once wound) and verso with fibers along the scroll axis.
Ink sits on the recto.  A trace resting on the verso looks geometrically
perfect and yields no ink, which is currently indistinguishable from "this
scroll has no readable ink".

The test: walk along the surface normal across a sheet and measure the
in-plane fiber direction at each depth.  If the plies are resolvable, the
fiber orientation should rotate by ~90 degrees as the sample crosses from one
ply to the other.  If it does not, this whole approach is dead and it is
better to find that out now.

Fiber direction is the structure tensor's *smallest* eigenvector: intensity
varies least along a fiber.  The normal is the largest.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sheetcheck.axis import umbilicus_from_surface  # noqa: E402
from sheetcheck.io import BUCKET, Surface, Volume  # noqa: E402
from sheetcheck.orient import structure_tensor  # noqa: E402

SCROLL = "PHerc1667"
SEGMENT = "20260108140509-w011_20260108140509268_flatboi"
MESH = "20260108140509-on-20251217075048-2.399um.tifxyz"
VOLUME = "20251217075048-2.399um-0.2m-78keV-masked.zarr"


def eig_frame(J: np.ndarray, idx: np.ndarray):
    """Return (normal, fiber, planarity, linearity) at integer voxel indices."""
    shp = np.array(J.shape[:3], dtype=np.int64)
    inside = np.all((idx >= 0) & (idx < shp), axis=-1)
    n = np.zeros(idx.shape[:-1] + (3,))
    f = np.zeros(idx.shape[:-1] + (3,))
    pl = np.zeros(idx.shape[:-1])
    li = np.zeros(idx.shape[:-1])
    if not inside.any():
        return n, f, pl, li, inside
    sel = idx[inside]
    Jm = J[sel[..., 0], sel[..., 1], sel[..., 2]].astype(np.float64)
    w, v = np.linalg.eigh(Jm)          # ascending
    n[inside] = v[..., :, -1]          # largest  -> across the sheet
    f[inside] = v[..., :, 0]           # smallest -> along the fibres
    hi, mid, lo = w[..., -1], w[..., -2], w[..., 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        pl[inside] = np.where(hi > 1e-12, (hi - mid) / hi, 0.0)
        li[inside] = np.where(hi > 1e-12, (mid - lo) / hi, 0.0)
    return n, f, pl, li, inside


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches", type=int, default=6)
    ap.add_argument("--rays", type=int, default=40)
    ap.add_argument("--level", type=int, default=0,
                    help="ply separation is ~50um; level 0 (2.4um) is needed")
    ap.add_argument("--span-um", type=float, default=90.0)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()

    cache = str(Path(__file__).resolve().parents[1] / "cache")
    surf = Surface.load(f"{BUCKET}/{SCROLL}/segments/{SEGMENT}/mesh/{MESH}",
                        cache_dir=cache)
    vol = Volume(f"{BUCKET}/{SCROLL}/volumes/{VOLUME}", level=args.level)
    vox = vol.voxel_size_um
    origin, axis, _ = umbilicus_from_surface(surf.points[surf.valid])
    normals, ok = surf.normals()
    print(f"voxel {vox:.3f} um; sampling +/-{args.span_um:.0f} um across the sheet")
    print(f"a ply is ~50 um = {50/vox:.0f} voxels\n")

    offs = np.arange(-args.span_um / vox, args.span_um / vox + 1e-9, 1.0)
    rng = np.random.default_rng(args.seed)
    vi, ui = np.nonzero(ok)

    prof_ang: list[np.ndarray] = []
    prof_pl: list[np.ndarray] = []
    prof_li: list[np.ndarray] = []
    t0 = time.time()

    for pi in range(args.patches):
        c = int(rng.integers(0, len(vi)))
        cv, cu = int(vi[c]), int(ui[c])
        r = 5
        vs = slice(max(cv - r, 0), cv + r + 1)
        us = slice(max(cu - r, 0), cu + r + 1)
        m = ok[vs, us]
        if m.sum() < 4:
            continue
        p0 = surf.points[vs, us][m]
        nn = normals[vs, us][m]
        if len(p0) > args.rays:
            s = rng.choice(len(p0), args.rays, replace=False)
            p0, nn = p0[s], nn[s]

        rays = p0[:, None, :] + offs[None, :, None] * nn[:, None, :]
        lo = rays.reshape(-1, 3).min(axis=0) - 10
        hi = rays.reshape(-1, 3).max(axis=0) + 10
        block, blo = vol.read_box(lo, hi)
        if block.size == 0 or min(block.shape) < 16:
            continue
        J = structure_tensor(block, grad_sigma=1.0, tensor_sigma=2.0)

        idx = np.rint(rays - blo).astype(np.int64)
        nvec, fvec, pl, li, inside = eig_frame(J, idx)

        # Local scroll frame at each ray's base point.
        rel = p0 - origin
        along = (rel @ axis)[:, None] * axis
        radial = rel - along
        rn = np.linalg.norm(radial, axis=1, keepdims=True)
        e_r = radial / np.where(rn > 1e-9, rn, 1.0)
        e_t = np.cross(np.broadcast_to(axis, e_r.shape), e_r)

        # Angle of the fibre within the sheet plane: 0 deg = circumferential
        # (recto-like), 90 deg = axial (verso-like).
        comp_t = np.abs(np.einsum("rdk,rk->rd", fvec, e_t))
        comp_a = np.abs(np.einsum("rdk,k->rd", fvec, axis))
        ang = np.degrees(np.arctan2(comp_a, comp_t))

        good = inside & (pl > 0.0)
        ang = np.where(good, ang, np.nan)
        prof_ang.append(ang)
        prof_pl.append(np.where(inside, pl, np.nan))
        prof_li.append(np.where(inside, li, np.nan))
        print(f"  patch {pi+1}/{args.patches}: {good.sum()} samples "
              f"({block.nbytes/1e6:.0f} MB, {time.time()-t0:.0f}s)")

    if not prof_ang:
        print("no usable patches")
        return 1

    A = np.concatenate(prof_ang, axis=0)
    P = np.concatenate(prof_pl, axis=0)
    L = np.concatenate(prof_li, axis=0)
    depth_um = offs * vox

    print(f"\n=== fibre orientation vs depth across the sheet (n={A.shape[0]} rays) ===")
    print(f"  {'depth um':>9} {'median ang':>11} {'IQR':>15} {'planarity':>10} "
          f"{'linearity':>10}")
    step = max(1, len(depth_um) // 18)
    for k in range(0, len(depth_um), step):
        col = A[:, k]
        col = col[np.isfinite(col)]
        if len(col) < 10:
            continue
        q1, med, q3 = np.percentile(col, [25, 50, 75])
        print(f"  {depth_um[k]:>9.1f} {med:>10.1f}d {q1:>6.1f}-{q3:<6.1f} "
              f"{np.nanmedian(P[:, k]):>10.2f} {np.nanmedian(L[:, k]):>10.2f}")

    inner = np.nanmedian(A[:, depth_um < -20.0])
    outer = np.nanmedian(A[:, depth_um > 20.0])
    swing = abs(outer - inner)
    print(f"\n  median fibre angle inner side  (< -20um): {inner:.1f} deg")
    print(f"  median fibre angle outer side  (> +20um): {outer:.1f} deg")
    print(f"  swing across the sheet: {swing:.1f} deg  (need ~90 for two plies)")

    ok_gate = swing >= 25.0
    print(f"\nP0 GATE: {'PASS' if ok_gate else 'FAIL'} "
          f"(ply orientation contrast detectable)")
    return 0 if ok_gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
