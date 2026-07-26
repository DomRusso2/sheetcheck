"""Synthetic sheet-switch injection, for measuring recall.

A real sheet switch is a trace that follows one wrap, then at some azimuth
steps onto the neighbouring wrap and continues there.  Injecting one means
displacing every grid cell past a chosen column radially outward (or inward)
by the local winding pitch.

Two modes:

``fraction``
    Displace by ``f`` times the pitch.  Purely geometric and cheap, so the
    magnitude can be swept continuously to find the detection floor -- f=1.0
    is a full switch, smaller values model a trace drifting off its sheet.

``snap``
    Ray-march and move each point onto the actual next papyrus sheet.  Slower,
    but it lands the injected surface on real material rather than at a
    nominal offset, which is the realism check on the ``fraction`` results.
"""

from __future__ import annotations

import numpy as np

from .io import Surface, Volume
from .profile import find_sheets


def _outward_normals(surf: Surface, origin: np.ndarray,
                     axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Surface normals with a consistent outward (away from axis) sign."""
    n, ok = surf.normals()
    rel = surf.points - origin
    along = np.einsum("...k,k->...", rel, axis)[..., None] * axis
    radial = rel - along
    rn = np.linalg.norm(radial, axis=-1, keepdims=True)
    radial = radial / np.where(rn > 1e-9, rn, 1.0)
    sign = np.sign(np.einsum("...k,...k->...", n, radial))
    sign[sign == 0] = 1.0
    return n * sign[..., None], ok


def inject_switch_fraction(
    surf: Surface,
    origin: np.ndarray,
    axis: np.ndarray,
    u_switch: int,
    pitch_um: float,
    vox_um: float,
    fraction: float = 1.0,
    direction: int = +1,
) -> tuple[Surface, np.ndarray]:
    """Displace all cells with ``u > u_switch`` by ``fraction`` of a pitch.

    Returns the modified surface and a boolean mask of displaced cells.
    """
    n_out, n_ok = _outward_normals(surf, origin, axis)
    pts = surf.points.copy()

    moved = np.zeros(surf.valid.shape, dtype=bool)
    moved[:, u_switch + 1:] = True
    moved &= surf.valid & n_ok

    shift_vox = direction * fraction * pitch_um / vox_um
    pts[moved] = pts[moved] + shift_vox * n_out[moved]

    out = Surface(points=pts, valid=surf.valid.copy(), meta=dict(surf.meta),
                  name=f"{surf.name}+inject{fraction:.2f}")
    return out, moved


def inject_switch_snap(
    surf: Surface,
    vol: Volume,
    origin: np.ndarray,
    axis: np.ndarray,
    u_switch: int,
    vox_um: float,
    direction: int = +1,
    reach_um: float = 700.0,
    max_cells: int = 20000,
    rng: np.random.Generator | None = None,
) -> tuple[Surface, np.ndarray]:
    """Move cells past ``u_switch`` onto the actual next papyrus sheet.

    Cells whose next sheet cannot be located are left in place and excluded
    from the returned mask, so recall is only ever scored against cells that
    genuinely moved onto neighbouring material.
    """
    rng = rng or np.random.default_rng(0)
    n_out, n_ok = _outward_normals(surf, origin, axis)
    pts = surf.points.copy()

    cand = np.zeros(surf.valid.shape, dtype=bool)
    cand[:, u_switch + 1:] = True
    cand &= surf.valid & n_ok

    vi, ui = np.nonzero(cand)
    if len(vi) > max_cells:
        sel = rng.choice(len(vi), max_cells, replace=False)
        vi, ui = vi[sel], ui[sel]

    reach_vox = reach_um / vol.voxel_size_um
    step = 0.5
    offs = np.arange(0.0, reach_vox + 1e-9, step) * direction
    moved = np.zeros(surf.valid.shape, dtype=bool)

    order = np.argsort(vi * 100000 + ui)
    vi, ui = vi[order], ui[order]
    B = 256
    for s in range(0, len(vi), B):
        bv, bu = vi[s:s + B], ui[s:s + B]
        p0 = vol.to_level(surf.points[bv, bu])
        nn = n_out[bv, bu]
        rays = p0[:, None, :] + offs[None, :, None] * nn[:, None, :]
        lo = rays.reshape(-1, 3).min(axis=0)
        hi = rays.reshape(-1, 3).max(axis=0)
        block, blo = vol.read_box(lo, hi)
        if block.size == 0:
            continue
        prof = Volume.sample_box(block, blo, rays)
        for k in range(len(bv)):
            pr = prof[k]
            if np.count_nonzero(pr) < len(pr) * 0.6:
                continue
            # find_sheets measures from the ray centre; re-reference to the start
            sheets = find_sheets(pr, step, vol.voxel_size_um,
                                 min_thickness_um=25.0)
            if len(sheets) == 0:
                continue
            start = sheets + 0.5 * (len(pr) - 1) * step
            nxt = start[start > 0.35 * len(pr) * step * 0.5]
            if len(nxt) == 0:
                continue
            d_l0 = float(nxt[0]) * (2**vol.level) * direction
            pts[bv[k], bu[k]] = surf.points[bv[k], bu[k]] + d_l0 * n_out[bv[k], bu[k]]
            moved[bv[k], bu[k]] = True

    out = Surface(points=pts, valid=surf.valid.copy(), meta=dict(surf.meta),
                  name=f"{surf.name}+snap")
    return out, moved
