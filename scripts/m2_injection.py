"""M2: measure recall by injecting sheet switches of known size and location.

Specificity was measured on a trusted trace; this measures the other half.
A switch is injected at a known column, so every pair is labelled:

  straddling  -- the injection boundary lies between A and B, so the pair
                 should fire
  interior    -- A and B are both past the boundary and displaced equally,
                 so the pair should NOT fire (this is the control that
                 catches a detector which merely notices "something moved")

Sweeping the displacement from 0 to a full pitch gives the detection floor:
how small a drift off the correct sheet is still caught.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sheetcheck.axis import umbilicus_from_surface  # noqa: E402
from sheetcheck.detect import build_pitch_map, detect  # noqa: E402
from sheetcheck.inject import inject_switch_fraction  # noqa: E402
from sheetcheck.io import BUCKET, Surface, Volume  # noqa: E402

SCROLL = "PHerc1667"
# The merged, straightened full-scroll trace: it spans many turns, which is
# what makes an "interior" control possible.  On a 1.5-turn segment every pair
# straddles a mid-segment injection, so there is nothing to compare against.
SEGMENT = ("20260612121456-w011_20260108140509268_merged_v4_flatboi"
           "_straightened_v4")
MESH = "20260612121456-on-20251217075048-2.399um.tifxyz"
VOLUME = "20251217075048-2.399um-0.2m-78keV-masked.zarr"


def classify(pairs, u_switch):
    """Label pairs as straddling / interior / before the injection boundary."""
    strad, interior, before = [], [], []
    for i, p in enumerate(pairs):
        lo, hi = min(p.ua, p.ub), max(p.ua, p.ub)
        if lo <= u_switch < hi:
            strad.append(i)
        elif lo > u_switch:
            interior.append(i)
        else:
            before.append(i)
    return np.array(strad), np.array(interior), np.array(before)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=1200)
    ap.add_argument("--threshold", type=float, default=1.6)
    ap.add_argument("--lo", type=float, default=0.5)
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    cache = str(Path(__file__).resolve().parents[1] / "cache")
    surf = Surface.load(f"{BUCKET}/{SCROLL}/segments/{SEGMENT}/mesh/{MESH}",
                        cache_dir=cache)
    vol = Volume(f"{BUCKET}/{SCROLL}/volumes/{VOLUME}", level=args.level)
    vox_um_l0 = vol.voxel_size_um / (2**args.level)
    origin, axis, _ = umbilicus_from_surface(surf.points[surf.valid])
    print(f"loaded in {time.time()-t0:.0f}s; voxel {vox_um_l0:.3f} um (level 0)")

    print(f"surface grid {surf.shape}, valid {surf.valid.mean():.1%}")
    t1 = time.time()
    pmap = build_pitch_map(surf, vol, seed=args.seed)
    print(f"CT pitch map: {len(pmap.pitch)} sites, "
          f"median {np.median(pmap.pitch) if len(pmap.pitch) else float('nan'):.0f} um "
          f"({time.time()-t1:.0f}s)")

    base = detect(surf, vol, n_pairs=args.pairs, threshold=args.threshold,
                  seed=args.seed, count_gaps=False, origin=origin, axis=axis,
                  pitch_map=pmap)
    pitch = base.pitch_um
    print(f"\nclean trace: {len(base.pairs)} pairs, pitch {pitch:.1f} um, "
          f"flagged {len(base.detections)} "
          f"({len(base.detections)/max(len(base.pairs),1):.2%})")

    cols = np.nonzero(surf.valid.any(axis=0))[0]
    u_switch = int(cols[int(0.55 * len(cols))])
    print(f"injecting at column u={u_switch} of {cols.min()}-{cols.max()}\n")

    rows = []
    for direction in (+1, -1):
        label = "outward" if direction > 0 else "inward"
        print(f"\n--- injection {label} ---")
        print(f"{'frac':>6} {'shift um':>9} {'strad n':>8} {'recall':>8} "
              f"{'interior FP':>12} {'before FP':>10} {'med ratio':>10}")
        for frac in (0.0, 0.25, 0.40, 0.50, 0.60, 0.75, 1.00):
            inj, _ = inject_switch_fraction(
                surf, origin, axis, u_switch, pitch, vox_um_l0,
                fraction=frac, direction=direction)
            res = detect(inj, vol, n_pairs=args.pairs,
                         threshold=args.threshold, seed=args.seed,
                         count_gaps=False, origin=origin, axis=axis,
                         pitch_map=pmap)
            ratios = res.ratios
            fired = (ratios >= args.threshold) | (ratios <= args.lo)
            s, it, bf = classify(res.pairs, u_switch)

            recall = float(fired[s].mean()) if len(s) else float("nan")
            fp_int = float(fired[it].mean()) if len(it) else float("nan")
            fp_bef = float(fired[bf].mean()) if len(bf) else float("nan")
            medr = float(np.nanmedian(ratios[s])) if len(s) else float("nan")
            rows.append((direction, frac, recall, fp_int, fp_bef, len(s), medr))
            print(f"{frac:>6.2f} {frac*pitch:>9.0f} {len(s):>8d} "
                  f"{recall:>7.1%} {fp_int:>11.1%} {fp_bef:>9.1%} "
                  f"{medr:>10.2f}")

    print("\n  frac 0.00 is the null injection: all three columns are the")
    print("  detector's false-positive rate on an untouched trace.")

    # Threshold sweep at a full switch, for an operating-point table.
    inj, _ = inject_switch_fraction(surf, origin, axis, u_switch, pitch,
                                    vox_um_l0, fraction=1.0, direction=+1)
    res = detect(inj, vol, n_pairs=args.pairs, threshold=1.0, seed=args.seed,
                 count_gaps=False, origin=origin, axis=axis, pitch_map=pmap)
    ratios = res.ratios
    s, it, bf = classify(res.pairs, u_switch)
    clean_idx = np.concatenate([it, bf]) if len(it) or len(bf) else np.array([])

    print(f"\noperating points at a full switch (n_strad={len(s)}, "
          f"n_clean={len(clean_idx)}):")
    print(f"  {'thresh':>7} {'recall':>8} {'FP':>8}")
    for t in (1.2, 1.3, 1.4, 1.5, 1.6, 1.75, 2.0):
        r = float((ratios[s] >= t).mean()) if len(s) else float("nan")
        f = float((ratios[clean_idx] >= t).mean()) if len(clean_idx) else float("nan")
        print(f"  {t:>7.2f} {r:>7.1%} {f:>7.1%}")

    out = Path(__file__).resolve().parents[1] / "results" / "m2_injection.npz"
    out.parent.mkdir(exist_ok=True)
    np.savez(out, rows=np.array(rows, dtype=float), pitch=pitch,
             u_switch=u_switch)
    print(f"\nsaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
