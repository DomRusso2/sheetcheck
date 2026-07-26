"""End-to-end sheet-switch detection on a traced surface.

Primary signal is the *radial* component of the chord joining a surface point
to its partner exactly one turn later.  On a correct trace that equals the
local winding pitch; if the trace jumped a winding in between, it doubles.

The air-gap count along the same chord is carried as a corroborating signal
only.  Measured on a trusted trace it fires on 22% of clean pairs, so it is
not safe as a trigger, but it is informative once a pair has already been
flagged on geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .axis import umbilicus_from_surface
from .holonomy import PairSample, count_sheets_between, find_one_turn_pairs
from .io import Surface, Volume
from .winding import cylindrical, unwrap_grid


@dataclass
class Detection:
    """A flagged site: where the trace appears to change winding."""

    va: int
    ua: int
    point: np.ndarray
    ratio: float            # radial gap / local pitch
    radial_um: float
    pitch_um: float
    extra_sheets: int


@dataclass
class DetectResult:
    pairs: list[PairSample]
    pitch_um: float
    ratios: np.ndarray = field(default_factory=lambda: np.array([]))
    detections: list[Detection] = field(default_factory=list)


class PitchMap:
    """Local winding pitch sampled from the CT volume, independent of the trace.

    Deriving the pitch scale from the trace's own one-turn gaps is circular: a
    switch displaces a whole neighbourhood, the normaliser moves with it, and
    the anomaly normalises itself away.  Measured on injected data that alone
    dropped recall from what should be ~100% down to 29%, and made recall
    *fall* as the injected displacement grew.

    The pitch is a property of the scroll, so it is measured from the CT by
    autocorrelating intensity along the surface normal.  Per-ray estimates are
    noisy, so a site's pitch is the median over its k nearest sampled sites.
    """

    def __init__(self, sites: np.ndarray, pitch: np.ndarray, k: int = 9):
        from scipy.spatial import cKDTree

        self.sites = sites
        self.pitch = pitch
        self.k = min(k, len(pitch)) if len(pitch) else 0
        self._tree = cKDTree(sites) if len(sites) else None

    @property
    def ok(self) -> bool:
        return self._tree is not None and self.k > 0

    def at(self, point: np.ndarray) -> float:
        if not self.ok:
            return float("nan")
        _, idx = self._tree.query(np.asarray(point, dtype=np.float64), k=self.k)
        idx = np.atleast_1d(idx)
        return float(np.median(self.pitch[idx]))


def build_pitch_map(
    surf: Surface,
    vol: Volume,
    n_sites: int = 240,
    reach_um: float = 900.0,
    step_vox: float = 0.5,
    min_strength: float = 0.12,
    seed: int = 0,
    patches: int = 12,
    auto_scale: bool = True,
) -> PitchMap:
    """Sample CT-derived winding pitch across a surface.

    Site count scales with surface area when ``auto_scale`` is set: a fixed
    budget that is dense on a single winding becomes hopelessly sparse on a
    full-scroll mesh (130 sites across a 2061 x 30097 grid), and then the
    "local" pitch is not local at all -- which inflates the false-positive
    rate because pitch genuinely varies 150-320 um within one scroll.
    """
    if auto_scale:
        cells = int(surf.valid.sum())
        scale = max(1.0, (cells / 1.2e6) ** 0.5)
        n_sites = int(n_sites * min(scale, 12.0))
        patches = int(patches * min(scale, 12.0))
    from .profile import dominant_period

    normals, ok = surf.normals()
    vi, ui = np.nonzero(ok)
    if len(vi) == 0:
        return PitchMap(np.zeros((0, 3)), np.zeros(0))

    rng = np.random.default_rng(seed)
    offs = np.arange(-reach_um / vol.voxel_size_um,
                     reach_um / vol.voxel_size_um + 1e-9, step_vox)
    per_patch = max(n_sites // max(patches, 1), 1)

    sites, pitches = [], []
    for _ in range(patches):
        c = int(rng.integers(0, len(vi)))
        cv, cu = int(vi[c]), int(ui[c])
        r = 6
        vs = slice(max(cv - r, 0), cv + r + 1)
        us = slice(max(cu - r, 0), cu + r + 1)
        m = ok[vs, us]
        if m.sum() < 4:
            continue
        p0 = surf.points[vs, us][m]
        nn = normals[vs, us][m]
        if len(p0) > per_patch:
            sel = rng.choice(len(p0), per_patch, replace=False)
            p0, nn = p0[sel], nn[sel]

        pl = vol.to_level(p0)
        rays = pl[:, None, :] + offs[None, :, None] * nn[:, None, :]
        block, blo = vol.read_box(rays.reshape(-1, 3).min(axis=0),
                                  rays.reshape(-1, 3).max(axis=0))
        if block.size == 0:
            continue
        prof = Volume.sample_box(block, blo, rays)
        for k in range(len(p0)):
            pr = prof[k]
            if np.count_nonzero(pr) < len(pr) * 0.6:
                continue
            per, strength = dominant_period(pr, step_vox, vol.voxel_size_um)
            if np.isfinite(per) and strength >= min_strength:
                sites.append(p0[k])
                pitches.append(per)

    return PitchMap(np.array(sites, dtype=np.float64),
                    np.array(pitches, dtype=np.float64))


def robust_local_pitch(pairs: list[PairSample], va: int,
                       window: int = 250, q: float = 30.0) -> float:
    """Fallback pitch scale: a low quantile of nearby one-turn radial gaps.

    Switches roughly double the gap, so the distribution is bimodal at P and
    2P.  A low quantile recovers P even when a large minority of nearby pairs
    are switched, unlike the median.
    """
    near = [abs(p.radial_um) for p in pairs if abs(p.va - va) <= window]
    if len(near) < 8:
        near = [abs(p.radial_um) for p in pairs]
    return float(np.percentile(near, q)) if near else float("nan")


def detect(
    surf: Surface,
    vol: Volume,
    n_pairs: int = 400,
    threshold: float = 1.6,
    seed: int = 0,
    count_gaps: bool = True,
    origin: np.ndarray | None = None,
    axis: np.ndarray | None = None,
    lo_threshold: float = 0.5,
    pitch_map: "PitchMap | None" = None,
    min_radius_um: float = 1500.0,
) -> DetectResult:
    """Run the one-turn holonomy check and flag sites above ``threshold``.

    ``pitch_map`` supplies the CT-derived local pitch.  It is built from the
    *original* surface and should be reused across variants of a trace, since
    the scroll's pitch does not change when a trace is wrong about it.
    """
    if origin is None or axis is None:
        origin, axis, _ = umbilicus_from_surface(surf.points[surf.valid])

    flat = surf.points.reshape(-1, 3)
    rad, th, h = cylindrical(flat, origin, axis)
    rad = rad.reshape(surf.valid.shape)
    th = th.reshape(surf.valid.shape)
    h = h.reshape(surf.valid.shape)

    vox_um_l0 = vol.voxel_size_um / (2**vol.level)

    # Mask the core BEFORE unwrapping, not just when selecting pairs: a row
    # that passes through the umbilicus unwraps incorrectly there, and the
    # error then propagates along the rest of that row.
    usable = surf.valid & (rad * vox_um_l0 >= min_radius_um)
    thu, ok = unwrap_grid(th, usable, axis_u=1)
    ok &= usable
    rng = np.random.default_rng(seed)
    pairs = find_one_turn_pairs(thu, h, ok, surf.points, vox_um_l0,
                                n_pairs, rng, origin, axis,
                                min_radius_um=min_radius_um)
    if count_gaps and pairs:
        count_sheets_between(vol, pairs, vox_um_l0)

    pitch = float(np.median(np.abs([p.radial_um for p in pairs]))) if pairs \
        else float("nan")
    res = DetectResult(pairs=pairs, pitch_um=pitch)
    ratios = np.full(len(pairs), np.nan)

    for i, p in enumerate(pairs):
        lp = pitch_map.at(p.pa) if (pitch_map is not None and pitch_map.ok) \
            else robust_local_pitch(pairs, p.va)
        if not np.isfinite(lp) or lp <= 0:
            continue
        ratio = abs(p.radial_um) / lp
        ratios[i] = ratio
        # Two-sided. A trace can jump to either neighbouring wrap: outward
        # doubles the one-turn gap, inward collapses it toward zero. A
        # one-sided high test is blind to half of all real switches -- the
        # injection sweep exposed this by showing recall *fall* as the
        # injected displacement grew.
        if ratio >= threshold or ratio <= lo_threshold:
            res.detections.append(Detection(
                va=p.va, ua=p.ua, point=p.pa, ratio=ratio,
                radial_um=abs(p.radial_um), pitch_um=lp,
                extra_sheets=p.sheets_between,
            ))
    res.ratios = ratios
    return res
