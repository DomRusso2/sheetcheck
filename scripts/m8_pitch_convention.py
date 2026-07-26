"""M8: measure perpendicular and radial pitch on the same rays.

The cross-scroll pitch was measured along the surface normal, which yields the
*perpendicular* sheet spacing.  The commonly cited 187.3 um figure comes from
spiral fitting, which is a *radial* advance per turn.  Those differ whenever
the surface normal is oblique to the radial direction -- and it is, by 21-28
degrees (M7), i.e. 7-13%.

Geometry predicts a specific relationship: two sheets separated perpendicularly
by d are crossed by a ray oblique by theta at spacing d/cos(theta).  So

    pitch_radial  =  pitch_perpendicular / cos(theta)

and the radial figure should be the *larger* of the two.  An earlier radial
wrap-count gave 208 um against 220 um for the normal-ray method -- the wrong
way round -- so one of those estimators was not sampling the direction assumed.

This settles it by casting both rays from the *same* points and comparing the
measured ratio against the predicted 1/cos(theta).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sheetcheck.axis import radial_frame, umbilicus_from_surface  # noqa: E402
from sheetcheck.io import Surface, Volume  # noqa: E402
from sheetcheck.orient import angle_between  # noqa: E402
from sheetcheck.profile import dominant_period  # noqa: E402
from sheetcheck.survey import bootstrap_ci, find_pairings  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scroll", default="PHerc1667")
    ap.add_argument("--patches", type=int, default=32)
    ap.add_argument("--per-patch", type=int, default=25)
    ap.add_argument("--reach-um", type=float, default=900.0)
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--min-strength", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    prs = find_pairings(args.scroll, max_segments=1)
    if not prs:
        print(f"no pairing for {args.scroll}")
        return 1
    pr = prs[0]
    surf = Surface.load(pr.mesh_path, cache_dir=str(root / "cache"))
    vol = Volume(pr.volume_path, level=args.level)
    vox = vol.voxel_size_um
    origin, axis, _ = umbilicus_from_surface(surf.points[surf.valid])
    normals, ok = surf.normals()
    print(f"{args.scroll} / {pr.segment[:46]}  voxel {vox:.3f} um")

    offs = np.arange(-args.reach_um / vox, args.reach_um / vox + 1e-9, 0.5)
    rng = np.random.default_rng(args.seed)
    vi, ui = np.nonzero(ok)

    p_perp, p_rad, thetas = [], [], []
    t0 = time.time()
    for pi in range(args.patches):
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
        if len(p0) > args.per_patch:
            s = rng.choice(len(p0), args.per_patch, replace=False)
            p0, nn = p0[s], nn[s]

        rad_dir, rr = radial_frame(p0, origin, axis)
        good = rr > 1e-6
        if good.sum() < 3:
            continue
        p0, nn, rad_dir = p0[good], nn[good], rad_dir[good]
        # Orient the radial vector to the same side as the normal so the two
        # rays sample the same neighbourhood rather than opposite directions.
        sgn = np.sign(np.sum(nn * rad_dir, axis=1))
        sgn[sgn == 0] = 1.0
        rad_dir = rad_dir * sgn[:, None]

        pl = vol.to_level(p0)
        ray_n = pl[:, None, :] + offs[None, :, None] * nn[:, None, :]
        ray_r = pl[:, None, :] + offs[None, :, None] * rad_dir[:, None, :]
        allpts = np.concatenate([ray_n.reshape(-1, 3), ray_r.reshape(-1, 3)])
        block, blo = vol.read_box(allpts.min(axis=0) - 4, allpts.max(axis=0) + 4)
        if block.size == 0:
            continue
        prof_n = Volume.sample_box(block, blo, ray_n)
        prof_r = Volume.sample_box(block, blo, ray_r)
        ang = angle_between(nn, rad_dir)

        for k in range(len(p0)):
            a, b = prof_n[k], prof_r[k]
            if (np.count_nonzero(a) < len(a) * 0.6
                    or np.count_nonzero(b) < len(b) * 0.6):
                continue
            pa, sa = dominant_period(a, 0.5, vox)
            pb, sb = dominant_period(b, 0.5, vox)
            if not (np.isfinite(pa) and np.isfinite(pb)):
                continue
            if sa < args.min_strength or sb < args.min_strength:
                continue
            p_perp.append(pa)
            p_rad.append(pb)
            thetas.append(ang[k])
        print(f"  patch {pi+1}/{args.patches}: {len(p_perp)} paired "
              f"({time.time()-t0:.0f}s)")

    if len(p_perp) < 30:
        print("too few paired measurements")
        return 1

    P = np.array(p_perp)
    R = np.array(p_rad)
    T = np.array(thetas)
    ratio = R / P
    pred = 1.0 / np.cos(np.radians(T))

    def show(name, a, fmt=".1f"):
        med, lo, hi = bootstrap_ci(a)
        print(f"  {name:<28} {med:{fmt}} [{lo:{fmt}}, {hi:{fmt}}]")

    print(f"\n=== paired pitch measurements (n={len(P)}) ===")
    show("perpendicular (normal ray)", P)
    show("radial (radial ray)", R)
    show("measured ratio R/P", ratio, ".3f")
    show("predicted 1/cos(theta)", pred, ".3f")
    show("obliquity theta (deg)", T)

    r_med, r_lo, r_hi = bootstrap_ci(ratio)
    p_med, p_lo, p_hi = bootstrap_ci(pred)
    print(f"\n  geometry predicts radial > perpendicular by "
          f"{p_med:.3f}x; measured {r_med:.3f}x")
    overlap = not (r_hi < p_lo or p_hi < r_lo)
    print(f"  direction confirmed: {'yes' if r_med > 1.0 else 'no'} "
          f"(radial {'>=' if r_med >= 1.0 else '<'} perpendicular)")
    print(f"  magnitude matches prediction: {'yes' if overlap else 'no'} "
          f"(measured {r_med:.3f} is {100*(r_med-1)/(p_med-1):.0f}% of predicted)")
    if not overlap:
        print("  -> the idealised parallel-plane model over-predicts, as")
        print("     expected for curved, locally non-parallel wraps.")

    print("\n  comparison against the cited 187.3 um spiral pitch:")
    for name, a in (("perpendicular", P), ("radial", R)):
        med, lo, hi = bootstrap_ci(a)
        verdict = "excludes" if lo > 187.3 else "includes"
        print(f"    {name:<14} {med:.0f} [{lo:.0f}, {hi:.0f}]  -> {verdict} 187.3")

    out = root / "results" / "m8_pitch_convention.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(
        {"perp": P.tolist(), "radial": R.tolist(), "theta_deg": T.tolist()},
        indent=2))
    print(f"\nsaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
