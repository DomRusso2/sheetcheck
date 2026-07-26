"""Diagnostic: are the surface normals actually perpendicular to the wraps?

For a rolled scroll the sheet normal must be close to radial about the scroll
axis (the umbilicus).  If computed normals are strongly oblique to the radial
direction, rays cast along them travel through the sheets at a shallow angle
and any pitch read off them is inflated by 1/cos(theta) -- which would explain
both a too-large period and weak periodicity.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sheetcheck.io import BUCKET, Surface  # noqa: E402
from sheetcheck.axis import radial_frame, umbilicus_from_surface  # noqa: E402

SCROLL = "PHerc1667"
SEGMENT = "20260108140509-w011_20260108140509268_flatboi"
MESH = "20260108140509-on-20251217075048-2.399um.tifxyz"


def fit_axis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit the scroll axis as a line through per-z-slab centroids of the surface.

    Returns (origin, unit direction).  This is a coarse umbilicus estimate but
    is enough to test normal orientation.
    """
    z = points[:, 0]
    edges = np.linspace(z.min(), z.max(), 25)
    cents = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (z >= a) & (z < b)
        if m.sum() > 50:
            cents.append(points[m].mean(axis=0))
    C = np.array(cents)
    origin = C.mean(axis=0)
    _, _, vt = np.linalg.svd(C - origin)
    d = vt[0]
    return origin, d / np.linalg.norm(d)


def main() -> int:
    surf = Surface.load(f"{BUCKET}/{SCROLL}/segments/{SEGMENT}/mesh/{MESH}")
    normals, ok = surf.normals()
    pts = surf.points[ok]
    nrm = normals[ok]

    rng = np.random.default_rng(3)
    sel = rng.choice(len(pts), size=min(40000, len(pts)), replace=False)
    pts, nrm = pts[sel], nrm[sel]

    all_pts = surf.points[surf.valid]

    origin_c, axis_c = fit_axis(all_pts)
    print(f"[centroid fit]  origin={origin_c.round(1)}  dir={axis_c.round(4)}")

    origin, axis, fits = umbilicus_from_surface(all_pts)
    print(f"[circle fit  ]  origin={origin.round(1)}  dir={axis.round(4)}")
    radii = np.array([f[1] for f in fits])
    rms = np.array([f[2] for f in fits])
    print(f"  per-slab circle radius: median {np.median(radii):.0f} vox "
          f"({np.median(radii)*2.399/1000:.2f} mm), spread "
          f"{radii.min():.0f}-{radii.max():.0f}")
    print(f"  per-slab fit rms residual: median {np.median(rms):.1f} vox\n")

    radial_all, rr_all = radial_frame(pts, origin, axis)
    good = rr_all > 1e-6
    radial = radial_all[good]
    rr = rr_all[good]
    n = nrm[good]

    cos = np.abs(np.sum(n * radial, axis=1))
    ang = np.degrees(np.arccos(np.clip(cos, 0, 1)))

    print(f"angle between surface normal and radial direction (n={len(ang)}):")
    for q in (5, 25, 50, 75, 95):
        print(f"  p{q:<3d} = {np.percentile(ang, q):6.1f} deg")
    print(f"  mean 1/cos(theta) inflation = {np.mean(1.0/np.clip(cos,0.05,1)):.2f}x")

    print(f"\nradius from axis: median {np.median(rr):.0f} vox "
          f"({np.median(rr)*2.399/1000:.2f} mm), "
          f"range {rr.min():.0f}-{rr.max():.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
