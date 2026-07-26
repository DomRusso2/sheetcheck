"""M4: how much sampling does a stable measurement actually need?

The cross-scroll survey gave support +0.82 at 6 patches and +0.35 at 10 on the
*same segment with the same seed*.  Numbers that move that much with sample
size are not measurements, so before reporting anything the sampling has to be
shown to converge.

This runs one segment at increasing patch counts across several independent
seeds, and reports the spread across seeds at each budget.  The budget where
the across-seed spread drops below a useful tolerance is the one the survey
should use.
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

METRICS = [
    ("support", "p50"),
    ("offset_um", "p50"),
    ("pitch_um", "p50"),
    ("planarity", "p50"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scroll", default="PHerc1667")
    ap.add_argument("--budgets", type=int, nargs="*",
                    default=[8, 16, 32, 64])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--per-patch", type=int, default=25)
    ap.add_argument("--level", type=int, default=1)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    pairings = find_pairings(args.scroll, max_segments=1)
    if not pairings:
        print(f"no pairing for {args.scroll}")
        return 1
    pr = pairings[0]
    print(f"{args.scroll} / {pr.segment[:50]} @ {pr.resolution_um} um")

    surf = Surface.load(pr.mesh_path, cache_dir=str(root / "cache"))
    vol = Volume(pr.volume_path, level=args.level)

    t0 = time.time()
    rows = []
    print(f"\n{'patches':>8} {'seeds':>6} " +
          " ".join(f"{k:>22}" for k, _ in METRICS))
    for budget in args.budgets:
        vals = {k: [] for k, _ in METRICS}
        rays = []
        for s in range(args.seeds):
            st = survey_surface(surf, vol, n_patches=budget,
                                per_patch=args.per_patch, seed=s)
            if not st:
                continue
            rays.append(st["n_rays"])
            for k, q in METRICS:
                vals[k].append(st[k][q])
        if not rays:
            continue
        cells = []
        for k, _ in METRICS:
            a = np.array(vals[k], dtype=float)
            a = a[np.isfinite(a)]
            if len(a) < 2:
                cells.append(f"{'--':>22}")
                continue
            spread = a.max() - a.min()
            rel = spread / abs(np.mean(a)) if abs(np.mean(a)) > 1e-9 else np.nan
            cells.append(f"{np.mean(a):>9.2f} +/-{spread/2:>5.2f}({rel:>4.0%})")
            rows.append({"budget": budget, "metric": k,
                         "mean": float(np.mean(a)), "spread": float(spread)})
        print(f"{budget:>8} {args.seeds:>6} " + " ".join(cells) +
              f"   [{int(np.mean(rays))} rays, {time.time()-t0:.0f}s]")

    print("\n  '+/-' is half the across-seed range; '(%)' is that relative to")
    print("  the mean.  A metric is usable once its relative spread is small")
    print("  compared with the differences it is meant to detect.")

    out = root / "results" / "m4_convergence.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nsaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
