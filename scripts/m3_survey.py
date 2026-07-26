"""M3: measure traced-surface geometry across multiple scrolls.

Everything measured so far came from a single scroll, which is too thin to
publish.  This repeats the measurements across the collection so the numbers
carry an across-scroll spread rather than an anecdote.

Sanity check built in: on a correctly placed trace the support score should
sit near 1.0 (papyrus level), not near 0.0 (air-gap level).  An earlier
version of the metric failed exactly this check, so it is asserted here rather
than assumed.
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

DEFAULT_SCROLLS = [
    "PHerc1667",      # read end to end -- the trusted reference
    "PHerc0172",      # Scroll 5, ~70% auto-unwrapped
    "PHercParis4",    # Scroll 1
    "PHerc0332",      # Scroll 3
    "PHerc0139",
    "PHercParis3",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scrolls", nargs="*", default=DEFAULT_SCROLLS)
    ap.add_argument("--segments-per-scroll", type=int, default=1)
    ap.add_argument("--patches", type=int, default=8)
    ap.add_argument("--per-patch", type=int, default=25)
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    cache = str(root / "cache")
    out_rows = []
    t0 = time.time()

    for scroll in args.scrolls:
        print(f"\n=== {scroll} ===")
        try:
            pairings = find_pairings(scroll,
                                     max_segments=args.segments_per_scroll)
        except Exception as e:  # noqa: BLE001
            print(f"  discovery failed: {type(e).__name__}: {e}")
            continue
        if not pairings:
            print("  no segment/volume pairing found")
            continue

        for pr in pairings:
            print(f"  segment {pr.segment[:52]}")
            print(f"    mesh @ {pr.resolution_um} um  vol {pr.volume_id}")
            try:
                surf = Surface.load(pr.mesh_path, cache_dir=cache)
                vol = Volume(pr.volume_path, level=args.level)
                st = survey_surface(surf, vol, n_patches=args.patches,
                                    per_patch=args.per_patch, seed=args.seed)
            except Exception as e:  # noqa: BLE001
                print(f"    FAILED: {type(e).__name__}: {e}")
                continue
            if not st:
                print("    no usable rays")
                continue

            st.update(scroll=scroll, segment=pr.segment,
                      resolution_um=pr.resolution_um)
            out_rows.append(st)
            print(f"    rays {st['n_rays']}  "
                  f"gap-structure {st['gap_structure_frac']:.0%}  "
                  f"({time.time()-t0:.0f}s)")
            print(f"    support   p50 {st['support']['p50']:+.2f} "
                  f"(p10 {st['support']['p10']:+.2f})")
            print(f"    pitch     p50 {st['pitch_um']['p50']:.0f} um "
                  f"(p10-p90 {st['pitch_um']['p10']:.0f}-{st['pitch_um']['p90']:.0f})")
            print(f"    offset    p50 {st['offset_um']['p50']:.0f} um "
                  f"p90 {st['offset_um']['p90']:.0f}")
            print(f"    planarity p50 {st['planarity']['p50']:.2f}   "
                  f"radius p50 {st['radius_mm']['p50']:.1f} mm")

    if not out_rows:
        print("\nno scrolls measured")
        return 1

    print(f"\n\n=== cross-scroll summary ({len(out_rows)} segments) ===")
    print("values are medians with bootstrap 95% CI\n")
    hdr = (f"{'scroll':<13}{'res':>5}{'rays':>6}{'gapstr':>7}"
           f"{'support':>18}{'pitch um':>20}{'offset um':>18}{'planar':>16}")
    print(hdr)
    print("-" * len(hdr))
    for r in out_rows:
        def ci(k, fmt=".2f"):
            d = r[k]
            return (f"{d['median']:{fmt}} [{d['ci_lo']:{fmt}},"
                    f"{d['ci_hi']:{fmt}}]")
        print(f"{r['scroll']:<13}{r['resolution_um']:>5.2f}{r['n_rays']:>6d}"
              f"{r['gap_structure_frac']:>6.0%}"
              f"{ci('support'):>18}{ci('pitch_um', '.0f'):>20}"
              f"{ci('offset_um', '.0f'):>18}{ci('planarity'):>16}")

    sup = np.array([r["support"]["p50"] for r in out_rows], dtype=float)
    sup = sup[np.isfinite(sup)]
    print("\nsanity: support should sit near 1.0 on correctly placed traces")
    print(f"  across-scroll median of per-segment support: {np.median(sup):+.2f}")
    if np.median(sup) < 0.5:
        print("  WARNING: traces score closer to gap level than papyrus level;")
        print("  the metric is still miscalibrated -- do not trust downstream use")

    pit = np.array([r["pitch_um"]["p50"] for r in out_rows], dtype=float)
    pit = pit[np.isfinite(pit)]
    if len(pit) > 1:
        print(f"\npitch across scrolls: median {np.median(pit):.0f} um, "
              f"range {pit.min():.0f}-{pit.max():.0f}")
        print("  winding-ruler reference (across-scroll medians): 187.3 um")

    outp = root / "results" / "m3_survey.json"
    outp.parent.mkdir(exist_ok=True)
    outp.write_text(json.dumps(out_rows, indent=2))
    print(f"\nsaved -> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
