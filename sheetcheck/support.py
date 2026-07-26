"""Is the traced surface actually sitting on papyrus?

A sheet switch is a trace leaving one wrap and joining the next, and to do that
it must cross the air gap between them.  While crossing, the traced surface is
in the gap -- not on any sheet.  That is a purely local, near-binary signature:
no umbilicus, no azimuth, no global winding coordinate, and therefore immune to
the scroll deformation (radius wandering ~4x the pitch) that defeats
azimuth-based holonomy.

Each surface point is classified by comparing the CT value at the point against
a threshold derived from its *own* normal profile, so the test self-calibrates
to local brightness and needs no global intensity model.
"""

from __future__ import annotations

import numpy as np

from .profile import otsu_threshold, smooth


def profile_levels(
    profile: np.ndarray,
    min_gap_frac: float = 0.12,
    min_contrast: float = 0.25,
) -> tuple[float, float, bool]:
    """Estimate a ray's (gap_level, papyrus_level, has_gap_structure).

    Otsu alone is not safe here.  In compressed regions a ray may cross almost
    no air at all, and Otsu will then happily split *within* the papyrus
    distribution, putting the threshold above the surface value and making
    every point look like it is floating in a gap.  That is exactly what made
    an earlier version of this metric report a negative median support on a
    trace that produced readable Greek.

    So the split is validated before use: the dark class must occupy a real
    fraction of the ray, and the two classes must be genuinely separated.
    Rays that fail are reported as having no measurable gap structure, which
    is itself the "compressed region" signal Open Problem 1 asks for rather
    than a value to be silently trusted.
    """
    inside = profile > 0
    if inside.sum() < 24:
        return float("nan"), float("nan"), False
    s = smooth(np.asarray(profile, dtype=np.float64))
    vals = s[inside]

    thr = otsu_threshold(vals)
    lo = vals[vals <= thr]
    hi = vals[vals > thr]
    if len(lo) < min_gap_frac * len(vals) or len(hi) < 8:
        return float("nan"), float("nan"), False

    gap = float(np.median(lo))
    pap = float(np.median(hi))
    if pap <= 1e-6 or (pap - gap) / pap < min_contrast:
        return gap, pap, False
    return gap, pap, True


def support_scores(
    profiles: np.ndarray, centre: int
) -> tuple[np.ndarray, np.ndarray]:
    """Score how well each ray's centre point is supported by papyrus.

    ``score`` is normalised so 0.0 is the ray's air-gap level and 1.0 is its
    papyrus level, which makes it comparable across regions of differing
    density.  A correctly placed trace should score near 1.

    ``valid`` marks rays with measurable gap structure; scores elsewhere are
    NaN rather than a number that looks meaningful but is not.
    """
    n = len(profiles)
    score = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)

    for i, p in enumerate(profiles):
        gap, pap, ok = profile_levels(p)
        if not ok:
            continue
        s = smooth(np.asarray(p, dtype=np.float64))
        score[i] = (s[centre] - gap) / (pap - gap)
        valid[i] = True
    return score, valid


def runs_below(mask: np.ndarray, min_len: int = 2) -> list[tuple[int, int]]:
    """Contiguous runs of True in a 1-D mask, at least ``min_len`` long.

    A switch shows up as a *run* of unsupported cells, not an isolated one;
    isolated low scores are noise or local damage.
    """
    if mask.size == 0:
        return []
    d = np.diff(mask.astype(np.int8))
    starts = list(np.nonzero(d == 1)[0] + 1)
    ends = list(np.nonzero(d == -1)[0] + 1)
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(len(mask))
    return [(a, b) for a, b in zip(starts, ends) if b - a >= min_len]


def ramp_switch(points: np.ndarray, valid: np.ndarray, normals: np.ndarray,
                u0: int, width: int, shift_vox: float) -> np.ndarray:
    """Inject a realistic sheet switch: a ramp across the gap over ``width`` columns.

    A tracer does not teleport between wraps; it drifts across the gap over
    some arc.  Cells partway up the ramp are exactly the ones sitting in the
    air gap, which is what the support test should catch.
    """
    out = points.copy()
    nu = points.shape[1]
    for k in range(width):
        u = u0 + k
        if u >= nu:
            break
        f = (k + 1) / width
        m = valid[:, u]
        out[m, u] = points[m, u] + f * shift_vox * normals[m, u]
    if u0 + width < nu:
        m = valid[:, u0 + width:]
        out[:, u0 + width:][m] = (points[:, u0 + width:][m]
                                  + shift_vox * normals[:, u0 + width:][m])
    return out
