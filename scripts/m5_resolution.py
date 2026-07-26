"""M5: isolate the effect of scan resolution, with the scroll held fixed.

The cross-scroll survey found that only 2% of rays on the 7.91 um PHerc0172
scan have a separable papyrus/air split, against 100% on every 1.129 um scan.
That comparison is confounded: it varies resolution *and* scroll at once.

PHerc1667's segment 20240304141531 is registered against four volumes
(7.91, 3.24, 2.399, 1.129 um), so the same physical papyrus can be measured at
four resolutions with everything else fixed.  If gap structure degrades with
voxel size here, resolution is the cause rather than scroll identity.

Note: the tifxyz grids differ in size between resolutions, so patches are not
cell-paired across runs -- this compares the same surface, not the same cells.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sheetcheck.io import BUCKET, Surface, Volume, s3_ls  # noqa: E402
from sheetcheck.survey import MESH_RE, VOL_RE, survey_surface  # noqa: E402

SCROLL = "PHerc1667"
SEGMENT = "20240304141531-w013_20240304141531_flatboi"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches", type=int, default=32)
    ap.add_argument("--per-patch", type=int, default=25)
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    cache = str(root / "cache")

    vols = {}
    for v in s3_ls(f"{BUCKET}/{SCROLL}/volumes"):
        name = v.rstrip("/").split("/")[-1]
        m = VOL_RE.match(name)
        if m and name.endswith(".zarr"):
            vols[m.group(1)] = v.rstrip("/")

    meshes = []
    for mp in s3_ls(f"{BUCKET}/{SCROLL}/segments/{SEGMENT}/mesh"):
        m = MESH_RE.search(mp.rstrip("/"))
        if m and m.group(1) in vols:
            meshes.append((float(m.group(2)), mp.rstrip("/"), vols[m.group(1)]))
    meshes.sort(reverse=True)

    print(f"{SCROLL} / {SEGMENT[:44]}")
    print(f"resolutions with a matching volume: "
          f"{[r for r, _, _ in meshes]}\n")
    if len(meshes) < 2:
        print("need at least two resolutions")
        return 1

    rows = []
    t0 = time.time()
    for res, mesh_path, vol_path in meshes:
        try:
            surf = Surface.load(mesh_path, cache_dir=cache)
            vol = Volume(vol_path, level=args.level)
            st = survey_surface(surf, vol, n_patches=args.patches,
                                per_patch=args.per_patch, seed=args.seed)
        except Exception as e:  # noqa: BLE001
            print(f"  {res:>6.3f} um  FAILED: {type(e).__name__}: {e}")
            continue
        if not st:
            print(f"  {res:>6.3f} um  no usable rays")
            continue
        st["resolution_um"] = res
        st["grid"] = list(surf.shape)
        st.pop("_raw", None)
        rows.append(st)
        print(f"  {res:>6.3f} um  grid {surf.shape}  rays {st['n_rays']:>4d}  "
              f"gap-structure {st['gap_structure_frac']:>5.1%}   "
              f"({time.time()-t0:.0f}s)")

    if not rows:
        return 1

    print(f"\n=== resolution series, scroll and segment held fixed ===")
    hdr = (f"{'voxel um':>9}{'rays':>6}{'gapstruct':>11}"
           f"{'pitch um':>20}{'planarity':>18}{'offset um':>18}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        def ci(k, fmt=".2f"):
            d = r[k]
            return f"{d['median']:{fmt}} [{d['ci_lo']:{fmt}},{d['ci_hi']:{fmt}}]"
        print(f"{r['resolution_um']:>9.3f}{r['n_rays']:>6d}"
              f"{r['gap_structure_frac']:>10.1%}"
              f"{ci('pitch_um', '.0f'):>20}{ci('planarity'):>18}"
              f"{ci('offset_um', '.0f'):>18}")

    g = [(r["resolution_um"], r["gap_structure_frac"]) for r in rows]
    print("\n  gap structure vs voxel size:")
    for res, frac in g:
        bar = "#" * int(round(frac * 40))
        print(f"    {res:>6.3f} um  {frac:>6.1%}  {bar}")

    fine = [f for res, f in g if res <= 2.5]
    coarse = [f for res, f in g if res >= 7.0]
    if fine and coarse:
        print(f"\n  fine (<=2.5um) mean {np.mean(fine):.1%} vs "
              f"coarse (>=7um) mean {np.mean(coarse):.1%}")
        print("  -> resolution is causal here; the scroll is held fixed")

    out = root / "results" / "m5_resolution.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nsaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
