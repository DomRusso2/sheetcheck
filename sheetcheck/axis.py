"""Umbilicus (scroll axis) estimation from a traced surface.

A single traced winding is an *arc*, not a closed loop, so its centroid is not
the scroll centre -- fitting the axis to per-slab centroids gives a badly wrong
radial direction.  An algebraic circle fit recovers the centre of curvature
from an arc correctly, which is what the winding geometry actually needs.
"""

from __future__ import annotations

import numpy as np


def fit_circle_2d(y: np.ndarray, x: np.ndarray) -> tuple[float, float, float]:
    """Taubin algebraic circle fit. Returns ``(cy, cx, radius)``.

    Taubin is used rather than the simpler Kasa fit because Kasa is strongly
    biased when the data covers only a short arc, which is exactly our case.
    """
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    my, mx = y.mean(), x.mean()
    u, v = y - my, x - mx

    z = u * u + v * v
    zm = z.mean()
    z0 = (z - zm) / (2.0 * np.sqrt(zm)) if zm > 0 else z
    M = np.column_stack([z0, u, v])
    _, _, vt = np.linalg.svd(M, full_matrices=False)
    a = vt[-1]
    a0 = a[0] / (2.0 * np.sqrt(zm)) if zm > 0 else a[0]
    coeff = np.array([a0, a[1], a[2], -zm * a0])
    if abs(coeff[0]) < 1e-12:
        raise ValueError("degenerate circle fit (points are collinear)")
    cy = -coeff[1] / (2 * coeff[0])
    cx = -coeff[2] / (2 * coeff[0])
    r = np.sqrt(cy * cy + cx * cx - coeff[3] / coeff[0])
    return float(cy + my), float(cx + mx), float(r)


def umbilicus_from_surface(
    points: np.ndarray,
    n_slabs: int = 24,
    min_pts: int = 200,
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float, float]]]:
    """Estimate the scroll axis by circle-fitting each z-slab of a surface.

    ``points`` is ``(N, 3)`` in (z, y, x).  Returns ``(origin, direction, fits)``
    where ``fits`` holds the per-slab ``(z, radius, rms_residual)`` so callers
    can judge how well the arc behaved.
    """
    z = points[:, 0]
    edges = np.linspace(z.min(), z.max(), n_slabs + 1)
    centres, fits = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (z >= a) & (z < b)
        if m.sum() < min_pts:
            continue
        p = points[m]
        try:
            cy, cx, r = fit_circle_2d(p[:, 1], p[:, 2])
        except (ValueError, np.linalg.LinAlgError):
            continue
        rr = np.hypot(p[:, 1] - cy, p[:, 2] - cx)
        rms = float(np.sqrt(np.mean((rr - r) ** 2)))
        zc = float(p[:, 0].mean())
        centres.append([zc, cy, cx])
        fits.append((zc, r, rms))

    if len(centres) < 2:
        raise ValueError("not enough slabs for an axis fit")

    C = np.array(centres)
    origin = C.mean(axis=0)
    _, _, vt = np.linalg.svd(C - origin)
    d = vt[0]
    if d[0] < 0:
        d = -d
    return origin, d / np.linalg.norm(d), fits


def radial_frame(
    points: np.ndarray, origin: np.ndarray, axis: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Unit radial vectors and radii for points about an axis."""
    rel = points - origin
    along = (rel @ axis)[:, None] * axis
    radial = rel - along
    r = np.linalg.norm(radial, axis=1)
    safe = np.where(r > 1e-9, r, 1.0)
    return radial / safe[:, None], r
