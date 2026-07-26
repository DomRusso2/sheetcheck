"""M6: isolate voxel size using the OME-Zarr pyramid.

The 2%-vs-100% gap-structure gap between the 7.91 um and 1.129 um scans is
confounded -- it varies scroll and resolution together, and no scroll in the
bucket is published at both a coarse and a fine resolution.

Reading one volume at successive pyramid levels breaks the confound directly:
scroll, segment, trace and scan are all fixed, and only the effective voxel
size changes.

Caveat worth stating plainly: pyramid downsampling is not the same as scanning
at a coarser resolution.  It applies no extra detector blur or noise, so it is
an *optimistic* bound -- a real coarse scan can only do worse than this.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sheetcheck.io import Surface, Volume  # noqa: E402
from sheetcheck.survey import find_pairings, survey_surface  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scroll", default="PHerc1667")
    ap.add_argument("--levels", type=int, nargs="*", default=[1, 2, 3, 4])
    ap.add_argument("--patches", type=int, default=32)
    ap.add_argument("--per-patch", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    pairings = find_pairings(args.scroll, max_segments=1)
    if not pairings:
        print(f"no pairing for {args.scroll}")
        return 1
    pr = pairings[0]
    surf = Surface.load(pr.mesh_path, cache_dir=str(root / "cache"))
    print(f"{args.scroll} / {pr.segment[:46]}")
    print(f"base mesh {pr.resolution_um} um, grid {surf.shape}\n")

    rows = []
    t0 = time.time()
    for lvl in args.levels:
        vol = Volume(pr.volume_path, level=lvl)
        try:
            vox = vol.voxel_size_um
            st = survey_surface(surf, vol, n_patches=args.patches,
                                per_patch=args.per_patch, seed=args.seed)
        except Exception as e:  # noqa: BLE001
            print(f"  level {lvl}: FAILED {type(e).__name__}: {e}")
            continue
        if not st:
            print(f"  level {lvl}: no usable rays")
            continue
        st.pop("_raw", None)
        st["level"] = lvl
        st["voxel_um"] = vox
        rows.append(st)
        print(f"  level {lvl}  voxel {vox:>6.2f} um  rays {st['n_rays']:>4d}  "
              f"gap-structure {st['gap_structure_frac']:>6.1%}   "
              f"({time.time()-t0:.0f}s)")

    if len(rows) < 2:
        print("need at least two levels")
        return 1

    print(f"\n=== voxel size vs measurable structure "
          f"(scroll/segment/scan fixed) ===")
    hdr = (f"{'voxel um':>9}{'rays':>6}{'gapstruct':>11}"
           f"{'pitch um':>20}{'planarity':>18}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        def ci(k, fmt=".2f"):
            d = r[k]
            return f"{d['median']:{fmt}} [{d['ci_lo']:{fmt}},{d['ci_hi']:{fmt}}]"
        print(f"{r['voxel_um']:>9.2f}{r['n_rays']:>6d}"
              f"{r['gap_structure_frac']:>10.1%}"
              f"{ci('pitch_um', '.0f'):>20}{ci('planarity'):>18}")

    print("\n  gap structure vs voxel size:")
    for r in rows:
        f = r["gap_structure_frac"]
        print(f"    {r['voxel_um']:>6.2f} um  {f:>6.1%}  " + "#" * int(round(f * 40)))

    # How many voxels span one wrap at each level?  That is the quantity that
    # should govern whether the gap is resolvable at all.
    print("\n  voxels per wrap (pitch / voxel size):")
    for r in rows:
        p = r["pitch_um"]["median"]
        if np.isfinite(p):
            print(f"    {r['voxel_um']:>6.2f} um -> {p / r['voxel_um']:>5.1f} "
                  f"voxels per wrap, gap structure {r['gap_structure_frac']:.0%}")

    out = root / "results" / "m6_pyramid.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nsaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
