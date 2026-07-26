"""M7: is the measured pitch a perpendicular spacing or a radial one?

Pitch measured along the surface normal is the perpendicular sheet spacing.
Pitch in the spiral sense is radial advance per turn.  If the surface normal is
oblique to the local radial direction by an angle theta, the two differ by
1/cos(theta) -- and at 30 degrees that is 15%, comparable to this project's
disagreement with the commonly cited 187.3 um.

So before claiming the collection's pitch is larger than reported, measure the
obliquity and report what it implies.  Needs geometry only, no volume reads.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sheetcheck.axis import radial_frame, umbilicus_from_surface  # noqa: E402
from sheetcheck.io import Surface  # noqa: E402
from sheetcheck.orient import angle_between  # noqa: E402
from sheetcheck.survey import find_pairings  # noqa: E402

SCROLLS = ["PHerc1667", "PHerc0172", "PHercParis4", "PHerc0139"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scrolls", nargs="*", default=SCROLLS)
    ap.add_argument("--sample", type=int, default=60000)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    cache = str(root / "cache")
    rows = []

    print(f"{'scroll':<14}{'n':>8}{'median ang':>12}{'p25':>7}{'p75':>7}"
          f"{'median 1/cos':>14}")
    print("-" * 62)
    for scroll in args.scrolls:
        prs = find_pairings(scroll, max_segments=1)
        if not prs:
            print(f"{scroll:<14}  no pairing")
            continue
        surf = Surface.load(prs[0].mesh_path, cache_dir=cache)
        normals, ok = surf.normals()
        if ok.sum() < 100:
            print(f"{scroll:<14}  too few normals")
            continue

        origin, axis, _ = umbilicus_from_surface(surf.points[surf.valid])
        pts = surf.points[ok]
        nrm = normals[ok]
        if len(pts) > args.sample:
            rng = np.random.default_rng(0)
            s = rng.choice(len(pts), args.sample, replace=False)
            pts, nrm = pts[s], nrm[s]

        radial, r = radial_frame(pts, origin, axis)
        good = r > 1e-6
        ang = angle_between(nrm[good], radial[good])
        cosang = np.cos(np.radians(ang))
        inflation = 1.0 / np.clip(cosang, 0.05, 1.0)

        med = float(np.median(ang))
        rows.append({"scroll": scroll, "n": int(good.sum()),
                     "median_angle_deg": med,
                     "p25": float(np.percentile(ang, 25)),
                     "p75": float(np.percentile(ang, 75)),
                     "median_inflation": float(np.median(inflation))})
        print(f"{scroll:<14}{good.sum():>8d}{med:>11.1f}d"
              f"{np.percentile(ang,25):>7.1f}{np.percentile(ang,75):>7.1f}"
              f"{np.median(inflation):>14.3f}")

    if not rows:
        return 1

    infl = np.array([r["median_inflation"] for r in rows])
    print(f"\n  median 1/cos(theta) across scrolls: {np.median(infl):.3f}")
    print("\n  Interpretation -- deliberately NOT resolved here.")
    print("  Two sheets separated perpendicularly by d are crossed by a ray")
    print("  oblique by theta at spacing d/cos(theta), so a radial measurement")
    print("  should exceed a normal one by this factor.  But radial wrap")
    print("  counting on PHerc1667 gives 208 um against 220 um for the")
    print("  normal-ray method -- smaller, not larger.  Both cannot be right,")
    print("  so at least one estimator is not sampling the direction assumed.")
    print("  Until that is settled by measuring both directions on the same")
    print("  rays, this script reports only the obliquity, and no pitch figure")
    print("  here should be described as correcting a published value.")

    out = root / "results" / "m7_normal_vs_radial.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nsaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
