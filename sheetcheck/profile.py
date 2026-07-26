"""Sheet detection along a 1D intensity profile through the scroll.

A ray cast along a traced surface's normal alternates between papyrus (bright)
and air gaps between wraps (dark).  Detecting *runs* of papyrus rather than
intensity maxima matters: a single physical sheet is two plies (recto + verso)
and often shows two maxima, so peak-picking systematically reports roughly half
the true winding pitch.
"""

from __future__ import annotations

import numpy as np


def otsu_threshold(x: np.ndarray, bins: int = 64) -> float:
    """Otsu's between-class variance threshold. Parameter-free and robust here."""
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-6:
        return lo
    hist, edges = np.histogram(x, bins=bins, range=(lo, hi))
    hist = hist.astype(np.float64)
    p = hist / hist.sum()
    centres = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(p)
    w1 = 1.0 - w0
    m0 = np.cumsum(p * centres)
    mt = m0[-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        between = (mt * w0 - m0) ** 2 / (w0 * w1)
    between[~np.isfinite(between)] = -1.0
    return float(centres[int(np.argmax(between))])


def smooth(profile: np.ndarray, width: int = 5) -> np.ndarray:
    if width <= 1:
        return profile
    k = np.hanning(width + 2)[1:-1]
    k = k / k.sum()
    return np.convolve(profile, k, mode="same")


def find_sheets(
    profile: np.ndarray,
    step_vox: float,
    vox_um: float,
    min_thickness_um: float = 25.0,
    threshold: float | None = None,
) -> np.ndarray:
    """Return sheet centre positions, as signed offsets in voxels from the ray centre.

    ``profile`` is sampled uniformly at ``step_vox`` spacing and is assumed to
    be centred on the surface point.
    """
    n = len(profile)
    if n < 5:
        return np.array([])
    s = smooth(np.asarray(profile, dtype=np.float64))

    inside = s > 0
    if inside.sum() < 5:
        return np.array([])
    thr = otsu_threshold(s[inside]) if threshold is None else threshold

    hot = (s > thr) & inside
    if not hot.any():
        return np.array([])

    # Contiguous runs of papyrus.
    edges = np.diff(hot.astype(np.int8))
    starts = list(np.nonzero(edges == 1)[0] + 1)
    ends = list(np.nonzero(edges == -1)[0] + 1)
    if hot[0]:
        starts.insert(0, 0)
    if hot[-1]:
        ends.append(n)

    min_len = max(1, int(round(min_thickness_um / (vox_um * step_vox))))
    centre = (n - 1) / 2.0
    out = []
    for a, b in zip(starts, ends):
        if b - a < min_len:
            continue
        w = s[a:b] - thr
        w = np.clip(w, 0, None)
        if w.sum() <= 0:
            continue
        pos = np.arange(a, b)
        out.append(float((pos * w).sum() / w.sum()))
    if not out:
        return np.array([])
    return (np.array(sorted(out)) - centre) * step_vox


def dominant_period(
    profile: np.ndarray,
    step_vox: float,
    vox_um: float,
    min_um: float = 90.0,
    max_um: float = 340.0,
) -> tuple[float, float]:
    """Estimate the winding pitch as the dominant period of a normal-ray profile.

    Returns ``(period_um, strength)`` where ``strength`` is the normalised
    autocorrelation peak in ``[0, 1]``.

    This replaces per-sheet segmentation for pitch estimation.  Individual
    sheets split (recto/verso) and merge (touching wraps) often enough that
    counting them gives a bimodal spacing distribution, but the *periodicity*
    of the profile survives both failure modes -- a split adds a harmonic, a
    merge drops one cycle's amplitude, and neither moves the fundamental.
    """
    x = np.asarray(profile, dtype=np.float64)
    inside = x > 0
    if inside.sum() < 16:
        return float("nan"), 0.0
    x = x[inside]

    # Detrend before correlating.  Scroll profiles carry a strong slow
    # brightness envelope (density varies across the wrap stack); left in, that
    # low-frequency component dominates the autocorrelation at every lag in the
    # search band and the periodic term is invisible underneath it.
    from scipy import ndimage

    span_samples = max_um / (step_vox * vox_um)
    x = x - ndimage.gaussian_filter1d(x, sigma=max(span_samples / 2.0, 1.0),
                                      mode="nearest")
    if not np.any(np.abs(x) > 1e-9):
        return float("nan"), 0.0

    # Taper so the finite window does not ring in the correlation.
    x = x * np.hanning(len(x))

    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    # Unbias: np.correlate sums N-k terms at lag k, which tilts the estimate
    # toward short lags.
    counts = np.arange(len(x), 0, -1, dtype=np.float64)
    ac = ac / counts * counts[0]
    if ac[0] <= 0:
        return float("nan"), 0.0
    ac = ac / ac[0]

    lag_um = np.arange(len(ac)) * step_vox * vox_um
    band = (lag_um >= min_um) & (lag_um <= max_um)
    if not band.any():
        return float("nan"), 0.0

    idx = np.nonzero(band)[0]
    j = idx[int(np.argmax(ac[idx]))]
    if j <= 0 or j >= len(ac) - 1:
        return float(lag_um[j]), float(max(ac[j], 0.0))

    # Parabolic refinement around the discrete autocorrelation peak.
    y0, y1, y2 = ac[j - 1], ac[j], ac[j + 1]
    denom = y0 - 2 * y1 + y2
    delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
    delta = float(np.clip(delta, -1.0, 1.0))
    period = (j + delta) * step_vox * vox_um
    return float(period), float(max(y1, 0.0))


def pitch_samples(sheets_vox: np.ndarray, vox_um: float) -> np.ndarray:
    """Consecutive sheet-to-sheet spacings, in micrometres.

    Using consecutive differences rather than distance-to-nearest avoids the
    bias from the ray's own sheet, whose near edge sits arbitrarily close to
    the surface point.
    """
    if len(sheets_vox) < 2:
        return np.array([])
    return np.diff(sheets_vox) * vox_um
