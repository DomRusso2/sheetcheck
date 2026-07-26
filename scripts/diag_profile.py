"""Diagnostic: look at raw normal-ray intensity profiles.

Before trusting any pitch statistic, check the thing it is computed from:
does a ray cast along the traced surface normal actually cross alternating
papyrus and air gaps?  If it does not, the fault is upstream (normals,
coordinate order, or the surface itself), not in the estimator.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sheetcheck.io import BUCKET, Surface, Volume  # noqa: E402

SCROLL = "PHerc1667"
SEGMENT = "20260108140509-w011_20260108140509268_flatboi"
MESH = "20260108140509-on-20251217075048-2.399um.tifxyz"
VOLUME = "20251217075048-2.399um-0.2m-78keV-masked.zarr"

LEVEL = 1
REACH = 120.0   # level-1 voxels  (~576 um each way)
STEP = 1.0


def spark(vals: np.ndarray) -> str:
    chars = " .:-=+*#%@"
    v = np.asarray(vals, dtype=np.float64)
    lo, hi = v.min(), v.max()
    if hi - lo < 1e-9:
        return "?" * len(v)
    idx = ((v - lo) / (hi - lo) * (len(chars) - 1)).round().astype(int)
    return "".join(chars[i] for i in idx)


def main() -> int:
    surf = Surface.load(f"{BUCKET}/{SCROLL}/segments/{SEGMENT}/mesh/{MESH}")
    vol = Volume(f"{BUCKET}/{SCROLL}/volumes/{VOLUME}", level=LEVEL)
    vox_um = vol.voxel_size_um
    normals, ok = surf.normals()

    print(f"voxel {vox_um:.3f} um at level {LEVEL}; "
          f"expected pitch ~187 um = {187.0/vox_um:.1f} voxels")
    print(f"ray: +/-{REACH} vox = +/-{REACH*vox_um:.0f} um, step {STEP} vox\n")

    rng = np.random.default_rng(7)
    vi, ui = np.nonzero(ok)
    c = rng.integers(0, len(vi))
    cv, cu = int(vi[c]), int(ui[c])
    r = 4
    vs, us = slice(cv - r, cv + r + 1), slice(cu - r, cu + r + 1)
    m = ok[vs, us]
    pts = vol.to_level(surf.points[vs, us][m])[:8]
    nrm = normals[vs, us][m][:8]

    offs = np.arange(-REACH, REACH + 1e-9, STEP)
    rays = pts[:, None, :] + offs[None, :, None] * nrm[:, None, :]
    lo, hi = rays.reshape(-1, 3).min(axis=0), rays.reshape(-1, 3).max(axis=0)
    block, blo = vol.read_box(lo, hi)
    print(f"fetched block {block.shape} = {block.nbytes/1e6:.0f} MB\n")
    prof = Volume.sample_box(block, blo, rays)

    centre = len(offs) // 2
    for i, p in enumerate(prof):
        at_surface = p[centre]
        nz = np.count_nonzero(p)
        print(f"ray {i}: value_at_surface={at_surface:6.1f}  "
              f"nonzero={nz}/{len(p)}  min={p.min():.0f} max={p.max():.0f} "
              f"mean={p.mean():.1f}")
        print("   " + spark(p))

    # Where is the surface sitting relative to the intensity distribution?
    print("\nintensity at the traced surface point vs. along-ray distribution:")
    at = prof[:, centre]
    print(f"  at surface : mean {at.mean():.1f}")
    print(f"  whole ray  : mean {prof[prof>0].mean():.1f}  "
          f"p10 {np.percentile(prof[prof>0],10):.0f}  "
          f"p90 {np.percentile(prof[prof>0],90):.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
