"""Winding geometry of a traced surface.

The core consistency idea behind sheet-switch detection: a scroll is one sheet
wound as a spiral, so for any two points on a correct trace the number of wraps
between them measured *along the surface* must equal the number of papyrus
sheets between them measured *through the volume*.  A sheet switch breaks that
equality by exactly +/-1.

This module supplies the along-the-surface half: cylindrical coordinates about
the umbilicus, and a continuously unwrapped azimuth accumulated over the tifxyz
grid.
"""

from __future__ import annotations

import numpy as np


def axis_frame(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build two unit vectors spanning the plane perpendicular to ``axis``."""
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    seed = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(seed, a)) > 0.9:
        seed = np.array([0.0, 0.0, 1.0])
    e1 = seed - np.dot(seed, a) * a
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    return e1, e2 / np.linalg.norm(e2)


def cylindrical(
    points: np.ndarray, origin: np.ndarray, axis: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(radius, theta, height)`` for points about an axis.

    ``theta`` is wrapped to (-pi, pi]; use :func:`unwrap_grid` to accumulate it
    across a surface grid.
    """
    e1, e2 = axis_frame(axis)
    rel = np.asarray(points, dtype=np.float64) - origin
    h = rel @ axis
    p1 = rel @ e1
    p2 = rel @ e2
    return np.hypot(p1, p2), np.arctan2(p2, p1), h


def unwrap_grid(
    theta: np.ndarray, valid: np.ndarray, axis_u: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate wrapped azimuth continuously across a tifxyz grid.

    Unwrapping runs along the winding direction (``axis_u``, the grid columns)
    independently for each row, then rows are offset by whole turns so they
    agree with a reference row.  Cells that could not be reached from a valid
    run are left masked.

    Returns ``(theta_unwrapped, ok)``.
    """
    th = np.array(theta, dtype=np.float64, copy=True)
    v = np.asarray(valid, dtype=bool)
    if axis_u == 0:
        th, v = th.T, v.T

    out = np.full(th.shape, np.nan)
    for i in range(th.shape[0]):
        row_valid = v[i]
        if row_valid.sum() < 2:
            continue
        idx = np.nonzero(row_valid)[0]
        # Only unwrap across contiguous runs; a gap of unknown width could hide
        # any number of whole turns, so runs are handled separately.
        splits = np.nonzero(np.diff(idx) > 1)[0]
        for run in np.split(idx, splits + 1):
            if len(run) < 2:
                continue
            out[i, run] = np.unwrap(th[i, run])

    # Reconcile rows: neighbouring rows of the same sheet must not differ by a
    # whole turn, so snap each row to the previous one modulo 2*pi.
    ref = None
    for i in range(out.shape[0]):
        row = out[i]
        m = np.isfinite(row)
        if m.sum() < 2:
            continue
        if ref is None:
            ref = i
            continue
        prev = out[ref]
        both = m & np.isfinite(prev)
        if both.sum() >= 2:
            shift = np.round(np.median(prev[both] - row[both]) / (2 * np.pi))
            out[i] = row + shift * 2 * np.pi
        ref = i

    ok = np.isfinite(out)
    if axis_u == 0:
        out, ok = out.T, ok.T
    return out, ok


def turns_spanned(theta_unwrapped: np.ndarray, ok: np.ndarray) -> float:
    """Total number of wraps the surface covers."""
    if ok.sum() == 0:
        return 0.0
    t = theta_unwrapped[ok]
    return float((t.max() - t.min()) / (2 * np.pi))
