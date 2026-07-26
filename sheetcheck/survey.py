"""Discover and measure traced surfaces across the open-data collection.

Mesh directories are named ``<segment_id>-on-<volume_id>-<res>um.tifxyz`` and
volumes ``<volume_id>-<res>um-...zarr``, so a mesh can be paired with the exact
volume it was registered against by matching the embedded volume id.  That
pairing is what makes a like-for-like measurement across scrolls possible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .axis import umbilicus_from_surface
from .io import BUCKET, Surface, Volume, s3_ls
from .orient import sheet_normals, structure_tensor
from .profile import dominant_period, find_sheets
from .support import support_scores

MESH_RE = re.compile(r"-on-(\d+)-([0-9.]+)um\.tifxyz$")
VOL_RE = re.compile(r"^(\d+)-([0-9.]+)um-")


@dataclass
class Pairing:
    scroll: str
    segment: str
    mesh_path: str
    volume_path: str
    volume_id: str
    resolution_um: float


def list_scrolls() -> list[str]:
    out = []
    for p in s3_ls(BUCKET):
        name = p.rstrip("/").split("/")[-1]
        if name.startswith("PHerc"):
            out.append(name)
    return out


def find_pairings(scroll: str, max_segments: int = 3) -> list[Pairing]:
    """Pair each segment's tifxyz mesh with its source volume."""
    try:
        vols = s3_ls(f"{BUCKET}/{scroll}/volumes")
    except Exception:  # noqa: BLE001 - scroll may have no volumes
        return []
    by_id: dict[str, str] = {}
    for v in vols:
        name = v.rstrip("/").split("/")[-1]
        m = VOL_RE.match(name)
        if m and name.endswith(".zarr"):
            by_id.setdefault(m.group(1), v.rstrip("/"))

    try:
        segs = s3_ls(f"{BUCKET}/{scroll}/segments")
    except Exception:  # noqa: BLE001
        return []

    out: list[Pairing] = []
    for seg in segs:
        if len(out) >= max_segments:
            break
        segname = seg.rstrip("/").split("/")[-1]
        try:
            meshes = s3_ls(f"{seg.rstrip('/')}/mesh")
        except Exception:  # noqa: BLE001
            continue
        best = None
        for mp in meshes:
            m = MESH_RE.search(mp.rstrip("/"))
            if not m:
                continue
            vid, res = m.group(1), float(m.group(2))
            if vid not in by_id:
                continue
            # Prefer the finest resolution that still has a matching volume.
            if best is None or res < best[1]:
                best = (mp.rstrip("/"), res, vid)
        if best is not None:
            out.append(Pairing(scroll=scroll, segment=segname,
                               mesh_path=best[0], volume_path=by_id[best[2]],
                               volume_id=best[2], resolution_um=best[1]))
    return out


@dataclass
class SurveyResult:
    scroll: str
    segment: str
    resolution_um: float
    n_rays: int = 0
    stats: dict = field(default_factory=dict)


def _pct(a, qs=(10, 50, 90)):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return {f"p{q}": float("nan") for q in qs}
    return {f"p{q}": float(np.percentile(a, q)) for q in qs}


def bootstrap_ci(a, stat=np.median, n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float]:
    """Bootstrap a statistic and its confidence interval.

    Resampling the pooled rays from one pass is both cheaper and sounder than
    re-running the whole survey under several seeds and taking the range: the
    range of a handful of draws is a very high-variance estimator, which is
    what made an earlier convergence check appear to get *worse* with more
    sampling.
    """
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) < 8:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    draws = stat(a[idx], axis=1)
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(stat(a)), float(lo), float(hi)


def survey_surface(
    surf: Surface,
    vol: Volume,
    n_patches: int = 10,
    per_patch: int = 30,
    reach_um: float = 700.0,
    step_vox: float = 0.5,
    seed: int = 0,
) -> dict:
    """Measure geometry and placement statistics for one traced surface."""
    normals, ok = surf.normals()
    if ok.sum() < 50:
        return {}
    vox = vol.voxel_size_um
    offs = np.arange(-reach_um / vox, reach_um / vox + 1e-9, step_vox)
    centre = len(offs) // 2

    rng = np.random.default_rng(seed)
    vi, ui = np.nonzero(ok)
    origin, axis, _ = umbilicus_from_surface(surf.points[surf.valid])

    sup, pit, pst, off, pla, rad = [], [], [], [], [], []
    n_rays = 0
    for _ in range(n_patches):
        c = int(rng.integers(0, len(vi)))
        cv, cu = int(vi[c]), int(ui[c])
        r = 6
        vs = slice(max(cv - r, 0), cv + r + 1)
        us = slice(max(cu - r, 0), cu + r + 1)
        m = ok[vs, us]
        if m.sum() < 5:
            continue
        p0 = surf.points[vs, us][m]
        nn = normals[vs, us][m]
        if len(p0) > per_patch:
            s = rng.choice(len(p0), per_patch, replace=False)
            p0, nn = p0[s], nn[s]

        pl = vol.to_level(p0)
        rays = pl[:, None, :] + offs[None, :, None] * nn[:, None, :]
        block, blo = vol.read_box(rays.reshape(-1, 3).min(axis=0) - 6,
                                  rays.reshape(-1, 3).max(axis=0) + 6)
        if block.size == 0 or min(block.shape) < 12:
            continue
        prof = Volume.sample_box(block, blo, rays)

        keep = np.count_nonzero(prof, axis=1) > prof.shape[1] * 0.6
        if not keep.any():
            continue
        prof = prof[keep]
        n_rays += len(prof)

        s_sc, s_ok = support_scores(prof, centre)
        sup.extend(np.where(s_ok, s_sc, np.nan).tolist())

        for p in prof:
            per, strength = dominant_period(p, step_vox, vox)
            pit.append(per)
            pst.append(strength)
            sheets = find_sheets(p, step_vox, vox, min_thickness_um=25.0)
            if len(sheets):
                off.append(abs(float(sheets[np.argmin(np.abs(sheets))])) * vox)
            else:
                off.append(np.nan)

        J = structure_tensor(block, grad_sigma=1.0, tensor_sigma=2.5)
        _, planar = sheet_normals(J, pl[keep] - blo)
        pla.extend(np.atleast_1d(planar).tolist())

        rel = p0[keep] - origin
        along = (rel @ axis)[:, None] * axis
        rad.extend((np.linalg.norm(rel - along, axis=1) * vox / 1000.0).tolist())

    sup = np.array(sup, dtype=float)
    raw = {"support": sup, "pitch_um": np.array(pit, dtype=float),
           "period_strength": np.array(pst, dtype=float),
           "offset_um": np.array(off, dtype=float),
           "planarity": np.array(pla, dtype=float),
           "radius_mm": np.array(rad, dtype=float)}

    out = {
        "n_rays": n_rays,
        "gap_structure_frac": (float(np.mean(np.isfinite(sup)))
                               if len(sup) else float("nan")),
    }
    for k, v in raw.items():
        med, lo, hi = bootstrap_ci(v)
        out[k] = _pct(v)
        out[k].update({"median": med, "ci_lo": lo, "ci_hi": hi,
                       "n": int(np.isfinite(v).sum())})
    out["_raw"] = {k: v.tolist() for k, v in raw.items()}
    return out
