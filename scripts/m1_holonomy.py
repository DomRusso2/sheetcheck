"""M1: run the one-turn holonomy check on a trusted trace.

PHerc1667 has been read end to end, so its published traces are about as close
to ground truth as this project has.  Running the detector here measures the
FALSE POSITIVE rate: on a correct trace, points one turn apart should sit one
local pitch away from each other with no intervening sheet.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sheetcheck.axis import umbilicus_from_surface  # noqa: E402
from sheetcheck.holonomy import (  # noqa: E402
    count_sheets_between,
    find_one_turn_pairs,
    summarise,
)
from sheetcheck.io import BUCKET, Surface, Volume  # noqa: E402
from sheetcheck.winding import cylindrical, unwrap_grid  # noqa: E402

SCROLL = "PHerc1667"
SEGMENT = "20260108140509-w011_20260108140509268_flatboi"
MESH = "20260108140509-on-20251217075048-2.399um.tifxyz"
VOLUME = "20251217075048-2.399um-0.2m-78keV-masked.zarr"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=200)
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    surf = Surface.load(f"{BUCKET}/{SCROLL}/segments/{SEGMENT}/mesh/{MESH}")
    vol = Volume(f"{BUCKET}/{SCROLL}/volumes/{VOLUME}", level=args.level)
    vox_um_l0 = vol.voxel_size_um / (2**args.level)

    origin, axis, _ = umbilicus_from_surface(surf.points[surf.valid])
    flat = surf.points.reshape(-1, 3)
    r, th, h = cylindrical(flat, origin, axis)
    th = th.reshape(surf.valid.shape)
    h = h.reshape(surf.valid.shape)
    thu, ok = unwrap_grid(th, surf.valid, axis_u=1)
    ok &= surf.valid
    print(f"surface {surf.shape}, unwrapped {ok.mean():.1%}  "
          f"({time.time()-t0:.0f}s)")

    rng = np.random.default_rng(args.seed)
    pairs = find_one_turn_pairs(thu, h, ok, surf.points, vox_um_l0,
                                args.pairs, rng, origin, axis)
    print(f"one-turn pairs found: {len(pairs)}")
    if not pairs:
        print("no pairs -- cannot evaluate")
        return 1

    def stat(name, arr):
        print(f"  {name:<12} median {np.median(arr):>7.0f} um  "
              f"IQR {np.percentile(arr,25):>6.0f}-{np.percentile(arr,75):<6.0f}")

    stat("|A-B|", np.array([p.gap_um for p in pairs]))
    stat("radial", np.abs([p.radial_um for p in pairs]))
    stat("tangential", np.abs([p.tangential_um for p in pairs]))
    stat("axial", np.abs([p.axial_um for p in pairs]))
    print("  (radial should sit near the local pitch, ~208-222 um;")
    print("   tangential/axial are pairing error and should be small)")

    t1 = time.time()
    count_sheets_between(vol, pairs, vox_um_l0)
    print(f"  ray-marched in {time.time()-t1:.0f}s")

    s = summarise(pairs)
    print(f"\n=== holonomy on a TRUSTED trace (false-positive test) ===")
    print(f"  usable pairs   : {s['n']}  (rejected {s.get('rejected', 0)})")
    print(f"  gap median     : {s['gap_median_um']:.0f} um  "
          f"IQR {s['gap_iqr_um'][0]:.0f}-{s['gap_iqr_um'][1]:.0f}")
    print(f"  sheets between : {s['sheets_between_hist']}")
    print(f"  clean (0 sheets between): {s['clean_frac']:.1%}")

    # Do the two independent signals agree?  Gap length and intervening-sheet
    # count are measured differently; if they track each other, the detector is
    # seeing real structure rather than noise.
    usable = [p for p in pairs if p.sheets_between >= 0]
    sb = np.array([p.sheets_between for p in usable])
    gp = np.array([p.gap_um for p in usable])
    print("\n  gap length vs intervening sheets (do the two signals agree?)")
    print(f"  {'sheets':>7} {'n':>5} {'median gap':>12} {'gap/pitch':>10}")
    pitch = 214.0
    for k in sorted(set(sb.tolist())):
        m = sb == k
        print(f"  {k:>7d} {m.sum():>5d} {np.median(gp[m]):>10.0f} um "
              f"{np.median(gp[m])/pitch:>9.2f}x")
    if len(set(sb.tolist())) > 1:
        corr = float(np.corrcoef(sb, gp)[0, 1])
        print(f"  correlation(sheets, gap) = {corr:+.3f}")

    out = Path(__file__).resolve().parents[1] / "results" / "m1_pairs.npz"
    out.parent.mkdir(exist_ok=True)
    np.savez(out, sheets=sb, gap_um=gp,
             dtheta=np.array([p.dtheta for p in usable]),
             pa=np.array([p.pa for p in usable]),
             pb=np.array([p.pb for p in usable]))
    print(f"\n  saved {len(usable)} pairs -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
