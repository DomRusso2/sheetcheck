"""Sheet-switch detection by one-turn holonomy.

The check
---------
Take a point A on the traced surface and the point B that lies exactly one turn
further along the same trace (theta_B = theta_A + 2*pi, same height).  On a
correct trace A and B sit on adjacent wraps, so the straight segment A->B

  * is about one local winding pitch long, and
  * crosses no intervening papyrus sheet.

If the trace jumped a winding somewhere in the turn between them, B lands two
wraps away instead of one: the segment roughly doubles in length and one whole
sheet appears in the middle.  The test is therefore a near-binary count, which
is far more robust than thresholding a continuous displacement.

Why one turn
------------
Real scrolls are heavily crushed -- on PHerc1667 the radius wanders by ~870 um
about a fitted spiral, four times the ~220 um pitch -- so any signal based on
absolute radius or on deviation from a global spiral is swamped by legitimate
deformation.  Comparing points one turn apart cancels it, because A and B sit
in essentially the same deformed neighbourhood.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io import Volume
from .profile import find_sheets


def count_air_gaps(profile: np.ndarray, step_um: float,
                   min_gap_um: float = 30.0) -> int:
    """Count air gaps strictly inside a chord whose endpoints lie on papyrus.

    Counting gaps rather than sheets is much better conditioned for this test.
    A papyrus sheet is ~100 um thick while adjacent wraps are only ~210 um
    apart, so on a one-pitch chord the endpoints' *own* sheets fill most of the
    length and are almost impossible to exclude by a margin rule.  The dark
    runs between them, by contrast, are unambiguously interior: adjacent wraps
    give exactly one, wraps two apart give two.
    """
    from .profile import otsu_threshold, smooth

    x = np.asarray(profile, dtype=np.float64)
    inside = x > 0
    if inside.sum() < 8:
        return -1
    s = smooth(x)
    thr = otsu_threshold(s[inside])

    dark = (s <= thr) & inside
    if not dark.any():
        return 0

    edges = np.diff(dark.astype(np.int8))
    starts = list(np.nonzero(edges == 1)[0] + 1)
    ends = list(np.nonzero(edges == -1)[0] + 1)
    if dark[0]:
        starts.insert(0, 0)
    if dark[-1]:
        ends.append(len(dark))

    min_len = max(1, int(round(min_gap_um / step_um)))
    n = 0
    for a, b in zip(starts, ends):
        # A run touching either end is not interior: it means the endpoint is
        # not actually sitting on papyrus, so the pair is untrustworthy.
        if a == 0 or b == len(dark):
            continue
        if b - a >= min_len:
            n += 1
    return n


@dataclass
class PairSample:
    """One A/B holonomy test site."""

    va: int
    ua: int
    vb: int
    ub: int
    pa: np.ndarray          # level-0 coords
    pb: np.ndarray
    dtheta: float           # actual theta_B - theta_A, radians
    gap_um: float           # |B - A|
    radial_um: float = 0.0  # radial component of the chord -- the pitch test
    tangential_um: float = 0.0
    axial_um: float = 0.0
    sheets_between: int = -1
    expected_between: int = 0


def _interp_partner(theta_row: np.ndarray, cols: np.ndarray, ja: int,
                    target: float) -> float | None:
    """Fractional column where the row's azimuth reaches ``theta[ja] + target``.

    Snapping to the nearest whole grid cell leaves up to half a cell of
    azimuth error, which at scroll radius is hundreds of microns of tangential
    offset -- comparable to the pitch we are trying to measure.  Interpolating
    removes that error instead of tolerating it.
    """
    want = theta_row[ja] + target
    th = theta_row[cols]
    order = np.argsort(th)
    ths, cs = th[order], cols[order].astype(np.float64)
    if want < ths[0] or want > ths[-1]:
        return None
    k = int(np.searchsorted(ths, want))
    if k == 0:
        return float(cs[0])
    t0, t1 = ths[k - 1], ths[k]
    if t1 - t0 < 1e-12:
        return float(cs[k])
    # Only interpolate across genuinely adjacent cells.
    if abs(cs[k] - cs[k - 1]) > 1.5:
        return None
    f = (want - t0) / (t1 - t0)
    return float(cs[k - 1] + f * (cs[k] - cs[k - 1]))


def find_one_turn_pairs(
    theta_u: np.ndarray,
    height: np.ndarray,
    ok: np.ndarray,
    points: np.ndarray,
    vox_um: float,
    n_pairs: int,
    rng: np.random.Generator,
    origin: np.ndarray,
    axis: np.ndarray,
    axial_tol_um: float = 60.0,
    patches: int = 8,
    patch_rows: int = 40,
    min_radius_um: float = 1500.0,
) -> list[PairSample]:
    """Pick sites A and their interpolated one-turn partners B.

    A points are drawn from a few contiguous grid patches.  Each A/B chord is
    physically short (about one pitch), so co-locating the A points keeps every
    pair inside a small volume block and makes the ray-marching affordable;
    sampling A uniformly scatters them around the full circumference and forces
    a bounding box the size of the whole cross-section.
    """
    nv, _ = theta_u.shape
    rows = [i for i in range(nv) if ok[i].sum() > 8]
    if not rows:
        return []

    if patches:
        chosen: list[int] = []
        for _ in range(patches):
            s = int(rng.integers(0, max(len(rows) - patch_rows, 1)))
            chosen.extend(rows[s:s + patch_rows])
        rows = sorted(set(chosen))

    out: list[PairSample] = []
    tries = 0
    target = 2 * np.pi
    axis = np.asarray(axis, dtype=np.float64)

    while len(out) < n_pairs and tries < n_pairs * 60:
        tries += 1
        i = int(rng.choice(rows))
        cols = np.nonzero(ok[i])[0]
        if len(cols) < 8:
            continue
        ja = int(rng.choice(cols))

        jb_f = _interp_partner(theta_u[i], cols, ja, target)
        if jb_f is None:
            continue
        j0 = int(np.floor(jb_f))
        j1 = min(j0 + 1, points.shape[1] - 1)
        if not (ok[i, j0] and ok[i, j1]):
            continue
        w = jb_f - j0
        pa = points[i, ja].astype(np.float64)
        pb = (1 - w) * points[i, j0] + w * points[i, j1]

        # Decompose the chord in the cylindrical frame at A.  The radial part
        # is the quantity that should equal the local pitch; tangential and
        # axial parts are pairing error and must not inflate the measurement.
        rel = pa - origin
        along = np.dot(rel, axis) * axis
        radial_vec = rel - along
        rn = np.linalg.norm(radial_vec)
        if rn < 1e-6:
            continue
        # Reject the scroll core.  Azimuth is unwrapped by requiring adjacent
        # grid cells to differ by less than pi; a 20-voxel grid step subtends
        # 0.004 rad at the outer radius but 3.3 rad at r~6 voxels, so near the
        # umbilicus the unwrap is meaningless and every derived quantity with
        # it.  Excluding a fixed physical radius is the honest fix.
        if rn * vox_um < min_radius_um:
            continue
        e_r = radial_vec / rn
        e_t = np.cross(axis, e_r)

        chord = pb - pa
        radial = float(np.dot(chord, e_r)) * vox_um
        tangential = float(np.dot(chord, e_t)) * vox_um
        axial = float(np.dot(chord, axis)) * vox_um
        if abs(axial) > axial_tol_um:
            continue

        gap = float(np.linalg.norm(chord)) * vox_um
        out.append(PairSample(i, ja, i, j0, pa, pb, target, gap,
                              radial_um=radial, tangential_um=tangential,
                              axial_um=axial))
    return out


def count_sheets_between(
    vol: Volume,
    pairs: list[PairSample],
    vox_um: float,
    step_vox: float = 0.5,
    margin_frac: float = 0.22,
) -> None:
    """Fill in ``sheets_between`` for each pair by ray-marching A->B.

    Sheets within ``margin_frac`` of either endpoint are ignored: A and B lie
    *on* sheets, and their own plies would otherwise be counted as intervening
    material.
    """
    if not pairs:
        return
    # Group by proximity so one fetched block serves several pairs.
    order = np.argsort([p.pa[0] for p in pairs])
    batch: list[PairSample] = []
    batch_lo = batch_hi = None

    def flush(items, lo, hi):
        if not items:
            return
        block, blo = vol.read_box(lo - 4, hi + 4)
        if block.size == 0:
            return
        for p in items:
            a = vol.to_level(p.pa)
            b = vol.to_level(p.pb)
            L = float(np.linalg.norm(b - a))
            if L < 2:
                p.sheets_between = 0
                continue
            n = max(int(L / step_vox), 8)
            t = np.linspace(0.0, 1.0, n)
            ray = a[None, :] + t[:, None] * (b - a)[None, :]
            prof = Volume.sample_box(block, blo, ray)
            if np.count_nonzero(prof) < len(prof) * 0.6:
                p.sheets_between = -1
                continue
            step_um = (L / (n - 1)) * vol.voxel_size_um
            p.sheets_between = count_air_gaps(prof, step_um) - 1

    for idx in order:
        p = pairs[idx]
        lo = vol.to_level(np.minimum(p.pa, p.pb))
        hi = vol.to_level(np.maximum(p.pa, p.pb))
        if batch_lo is None:
            batch, batch_lo, batch_hi = [p], lo, hi
            continue
        nlo = np.minimum(batch_lo, lo)
        nhi = np.maximum(batch_hi, hi)
        if np.prod(nhi - nlo + 8) > 40e6 or len(batch) >= 64:
            flush(batch, batch_lo, batch_hi)
            batch, batch_lo, batch_hi = [p], lo, hi
        else:
            batch.append(p)
            batch_lo, batch_hi = nlo, nhi
    flush(batch, batch_lo, batch_hi)


def summarise(pairs: list[PairSample]) -> dict:
    usable = [p for p in pairs if p.sheets_between >= 0]
    if not usable:
        return {"n": 0}
    sb = np.array([p.sheets_between for p in usable])
    gaps = np.array([p.gap_um for p in usable])
    return {
        "n": len(usable),
        "rejected": len(pairs) - len(usable),
        "sheets_between_hist": {int(k): int(v) for k, v in
                                zip(*np.unique(sb, return_counts=True))},
        "clean_frac": float(np.mean(sb == 0)),
        "gap_median_um": float(np.median(gaps)),
        "gap_iqr_um": (float(np.percentile(gaps, 25)),
                       float(np.percentile(gaps, 75))),
    }
