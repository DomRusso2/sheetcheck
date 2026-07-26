"""Independent pitch estimate: count wraps along long axis-aligned radial rays.

The autocorrelation estimator works on short local windows and could carry a
systematic bias.  This measures the same quantity a completely different way:
cast a long ray outward from the umbilicus, count how many papyrus sheets it
crosses, and divide the radial distance by the count.  Over ~40 wraps,
miscounting one or two barely moves the answer, so it is a strong cross-check.

Rays are cast along +/-y and +/-x so the fetched bounding box stays thin.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sheetcheck.axis import umbilicus_from_surface  # noqa: E402
from sheetcheck.io import BUCKET, Surface, Volume  # noqa: E402
from sheetcheck.profile import find_sheets  # noqa: E402

SCROLL = "PHerc1667"
SEGMENT = "20260108140509-w011_20260108140509268_flatboi"
MESH = "20260108140509-on-20251217075048-2.399um.tifxyz"
VOLUME = "20251217075048-2.399um-0.2m-78keV-masked.zarr"

LEVEL = 1
R_START = 250.0    # level-1 voxels; skip the disordered core
R_END = 2200.0
STEP = 0.5


def main() -> int:
    surf = Surface.load(f"{BUCKET}/{SCROLL}/segments/{SEGMENT}/mesh/{MESH}")
    vol = Volume(f"{BUCKET}/{SCROLL}/volumes/{VOLUME}", level=LEVEL)
    vox_um = vol.voxel_size_um

    origin0, axis, _ = umbilicus_from_surface(surf.points[surf.valid])
    origin = vol.to_level(origin0)
    print(f"umbilicus (level {LEVEL}) = {origin.round(1)}, axis {axis.round(3)}")
    print(f"voxel {vox_um:.3f} um; ray {R_START:.0f}-{R_END:.0f} vox "
          f"= {(R_END-R_START)*vox_um/1000:.2f} mm\n")

    dirs = {
        "+y": np.array([0.0, 1.0, 0.0]),
        "-y": np.array([0.0, -1.0, 0.0]),
        "+x": np.array([0.0, 0.0, 1.0]),
        "-x": np.array([0.0, 0.0, -1.0]),
    }
    zs = np.linspace(origin[0] - 2500, origin[0] + 2500, 5)
    r = np.arange(R_START, R_END + 1e-9, STEP)

    results = []
    for z in zs:
        for name, d in dirs.items():
            base = np.array([z, origin[1], origin[2]])
            pts = base[None, :] + r[:, None] * d[None, :]
            lo = pts.min(axis=0) - 4
            hi = pts.max(axis=0) + 4
            block, blo = vol.read_box(lo, hi)
            if block.size == 0:
                continue
            prof = Volume.sample_box(block, blo, pts)
            occupied = np.count_nonzero(prof) / len(prof)
            if occupied < 0.5:
                continue
            sheets = find_sheets(prof, STEP, vox_um, min_thickness_um=20.0)
            if len(sheets) < 8:
                continue
            span_um = (sheets[-1] - sheets[0]) * vox_um
            pitch = span_um / (len(sheets) - 1)
            results.append(pitch)
            print(f"  z={z:8.0f} {name}: {len(sheets):3d} sheets over "
                  f"{span_um/1000:5.2f} mm -> pitch {pitch:6.1f} um "
                  f"(occupied {occupied:.0%}, {block.nbytes/1e6:.0f} MB)")

    a = np.array(results)
    print(f"\n=== radial wrap-counting, n={len(a)} rays ===")
    if len(a) < 4:
        print("too few usable rays")
        return 1
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    print(f"  median {med:.1f} um   IQR {q1:.1f}-{q3:.1f}  "
          f"(rel {100*(q3-q1)/med:.0f}%)")
    print("  reference (across-scroll medians): 187.3 um, IQR 181.5-193.4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
