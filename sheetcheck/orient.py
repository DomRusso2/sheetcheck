"""Local sheet orientation from the CT volume via the structure tensor.

The mesh-derived normal (finite differences of the tifxyz grid) describes the
*traced* surface.  The structure tensor describes the *actual* papyrus in the
scan.  Where the two disagree, either the trace is off the sheet or the sheet
is genuinely ambiguous -- and that disagreement is the signal the detector is
ultimately built on, so it needs its own well-tested estimator.

For a locally planar (sheet-like) structure the structure tensor has one large
eigenvalue whose eigenvector is the sheet normal, and two small ones spanning
the sheet plane.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def structure_tensor(
    block: np.ndarray,
    grad_sigma: float = 1.0,
    tensor_sigma: float = 3.0,
) -> np.ndarray:
    """Return the per-voxel structure tensor of ``block`` as shape ``(...,3,3)``."""
    b = np.asarray(block, dtype=np.float32)
    if grad_sigma > 0:
        b = ndimage.gaussian_filter(b, grad_sigma)
    gz, gy, gx = np.gradient(b)

    comps = {}
    g = (gz, gy, gx)
    for i in range(3):
        for j in range(i, 3):
            comps[(i, j)] = ndimage.gaussian_filter(g[i] * g[j], tensor_sigma)

    J = np.empty(b.shape + (3, 3), dtype=np.float32)
    for i in range(3):
        for j in range(i, 3):
            J[..., i, j] = comps[(i, j)]
            J[..., j, i] = comps[(i, j)]
    return J


def sheet_normals(
    J: np.ndarray, coords_local: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Sheet normals and planarity at integer ``coords_local`` into ``J``.

    Planarity is ``(l2 - l1) / l2`` for eigenvalues ``l2 >= l1 >= l0``: it is
    near 1 for a clean sheet and near 0 for isotropic haze, so it doubles as a
    local confidence that a sheet orientation exists at all.
    """
    idx = np.rint(coords_local).astype(np.int64)
    shp = np.array(J.shape[:3], dtype=np.int64)
    inside = np.all((idx >= 0) & (idx < shp), axis=-1)

    n = np.zeros(coords_local.shape[:-1] + (3,), dtype=np.float64)
    planar = np.zeros(coords_local.shape[:-1], dtype=np.float64)
    if not inside.any():
        return n, planar

    sel = idx[inside]
    Jm = J[sel[..., 0], sel[..., 1], sel[..., 2]].astype(np.float64)
    w, v = np.linalg.eigh(Jm)  # ascending eigenvalues
    principal = v[..., :, -1]
    l_hi, l_mid = w[..., -1], w[..., -2]
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(l_hi > 1e-12, (l_hi - l_mid) / l_hi, 0.0)

    n[inside] = principal
    planar[inside] = np.clip(p, 0.0, 1.0)
    return n, planar


def align_sign(a: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Flip ``a`` so it points the same way as ``ref`` (orientation is arbitrary)."""
    s = np.sign(np.sum(a * ref, axis=-1, keepdims=True))
    s[s == 0] = 1.0
    return a * s


def angle_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Unsigned angle in degrees between unit vectors, treating them as axes."""
    c = np.abs(np.sum(a * b, axis=-1))
    return np.degrees(np.arccos(np.clip(c, 0.0, 1.0)))
