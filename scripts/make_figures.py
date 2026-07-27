"""Render the figures used in README.md and FINDINGS.md.

Figures 1-3 replot data already saved in results/ and need no network.
Figure 4 re-reads a handful of CT profiles so the sheet detection can be
*looked at* rather than trusted -- see `--profiles`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "figures"

# Reference data-viz palette, slots 1-2 plus chrome. Using the documented
# palette verbatim rather than inventing hues: its first three categorical
# slots are validated all-pairs in both light and dark modes.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
RED = "#d03b3b"


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": ["DejaVu Sans"],
        "font.size": 9,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def fig_placement_vs_ink():
    """The null: ink detectability is flat against surface placement."""
    d = np.load(RES / "p1_placement.npz")
    off = np.abs(d["offset"])
    ink = d["ink_std"]

    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=150)
    ax.grid(True, axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.scatter(off, ink, s=14, color=BLUE, alpha=0.35, linewidths=0,
               zorder=2, label="sampled mesh cells")

    edges = np.array([0, 10, 20, 30, 45, 60, 90, 160])
    cx, cy = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (off >= lo) & (off < hi)
        if m.sum() >= 8:
            cx.append(0.5 * (lo + hi))
            cy.append(np.median(ink[m]))
    ax.plot(cx, cy, color=ORANGE, linewidth=2.0, marker="o", markersize=6,
            zorder=3, label="binned median")

    r = float(np.corrcoef(off, ink)[0, 1])
    ax.set_xlabel("distance from the papyrus sheet centre  (µm)")
    ax.set_ylabel("ink contrast\n(local std of published ink raster)")
    ax.set_title("Surface placement does not predict ink detectability",
                 color=INK, fontsize=11, loc="left", pad=12)
    ax.text(0.99, 0.95, f"r = {r:+.3f}   n = {len(off)}\nPHerc. 1667",
            transform=ax.transAxes, ha="right", va="top",
            color=INK2, fontsize=9)
    # Sits in the sparse upper-right band, clear of both the dense low-offset
    # cloud and the end of the binned-median line.
    ax.legend(loc="upper right", bbox_to_anchor=(0.99, 0.82), fontsize=8,
              labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(FIG / "placement_vs_ink.png")
    plt.close(fig)
    print("  placement_vs_ink.png")


def fig_pitch():
    """Per-scroll pitch with CIs, against the two reference figures."""
    rows = json.load(open(RES / "m3_survey.json"))
    rows = sorted(rows, key=lambda r: r["pitch_um"]["median"])
    names = [r["scroll"] for r in rows]
    med = np.array([r["pitch_um"]["median"] for r in rows])
    lo = np.array([r["pitch_um"]["ci_lo"] for r in rows])
    hi = np.array([r["pitch_um"]["ci_hi"] for r in rows])
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=150)
    ax.grid(True, axis="x", zorder=0)
    ax.set_axisbelow(True)

    ax.axvline(187.3, color=RED, linewidth=1.6, linestyle="--", zorder=1)
    ax.text(187.3, len(rows) - 0.35, " commonly cited\n 187.3 µm",
            color=RED, fontsize=8, va="top", ha="left")
    ax.axvline(225.0, color=MUTED, linewidth=1.6, linestyle=":", zorder=1)
    ax.text(225.0, -1.42, "winding-sync: 225 µm (13 scrolls) ",
            color=INK2, fontsize=8, va="bottom", ha="right")

    ax.hlines(y, lo, hi, color=BLUE, linewidth=2.0, zorder=3)
    ax.plot(med, y, "o", color=BLUE, markersize=8, zorder=4)
    for i, r in enumerate(rows):
        ax.text(hi[i] + 2.5, i, f"{med[i]:.0f}", color=INK2, fontsize=8,
                va="center")

    ax.set_yticks(y)
    ax.set_yticklabels(names, color=INK2)
    ax.set_ylim(-1.75, len(rows) - 0.1)
    ax.set_xlim(min(lo.min(), 187.3) - 8, hi.max() + 12)
    ax.set_xlabel("winding pitch  (µm, median with bootstrap 95% CI)")
    ax.set_title("Measured winding pitch, four scrolls",
                 color=INK, fontsize=11, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(FIG / "pitch_by_scroll.png")
    plt.close(fig)
    print("  pitch_by_scroll.png")


def fig_resolution():
    """Small multiples: what survives coarsening and what does not.

    Three measures on different scales, so three panels sharing one x axis --
    never two y axes on one plot.
    """
    rows = json.load(open(RES / "m6_pyramid.json"))
    rows = sorted(rows, key=lambda r: r["voxel_um"])
    vox = np.array([r["voxel_um"] for r in rows])

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.2), dpi=150, sharex=True)
    panels = [
        ("gap structure", [r["gap_structure_frac"] * 100 for r in rows],
         None, None, "% of rays with a\nseparable papyrus/air split", BLUE),
        ("pitch", [r["pitch_um"]["median"] for r in rows],
         [r["pitch_um"]["ci_lo"] for r in rows],
         [r["pitch_um"]["ci_hi"] for r in rows], "winding pitch (µm)", BLUE),
        ("planarity", [r["planarity"]["median"] for r in rows],
         [r["planarity"]["ci_lo"] for r in rows],
         [r["planarity"]["ci_hi"] for r in rows],
         "structure-tensor\nplanarity", ORANGE),
    ]
    for ax, (name, v, lo, hi, ylab, colour) in zip(axes, panels):
        ax.grid(True, axis="y", zorder=0)
        ax.set_axisbelow(True)
        v = np.array(v, dtype=float)
        if lo is not None:
            ax.fill_between(vox, lo, hi, color=colour, alpha=0.18, zorder=1)
        ax.plot(vox, v, color=colour, linewidth=2.0, marker="o",
                markersize=7, zorder=3)
        ax.set_ylabel(ylab, fontsize=8)
        ax.set_xscale("log")
        if name == "gap structure":
            ax.set_ylim(0, 108)
        verdict = "unchanged" if name != "planarity" else "changes"
        ax.text(0.99, 0.12 if name == "planarity" else 0.88, verdict,
                transform=ax.transAxes, ha="right",
                va="bottom" if name == "planarity" else "top",
                color=INK2, fontsize=9)

    # A log axis draws its own decade minor ticks, which collide with the
    # explicit per-level labels ("4x10^0" landing on top of "4.52").
    from matplotlib.ticker import NullFormatter, NullLocator
    axes[-1].set_xlabel("effective voxel size  (µm, same scan downsampled)")
    axes[-1].xaxis.set_minor_locator(NullLocator())
    axes[-1].xaxis.set_minor_formatter(NullFormatter())
    axes[-1].set_xticks(vox)
    axes[-1].set_xticklabels([f"{v:.2f}" for v in vox])
    axes[0].set_title(
        "One scroll, one scan, one trace — only the voxel size changes",
        color=INK, fontsize=11, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(FIG / "resolution_series.png")
    plt.close(fig)
    print("  resolution_series.png")


def fig_profiles(n_rays=6):
    """Visual check: real CT profiles with the detected sheets marked.

    The measurement primitives are only trustworthy if the thing they claim to
    find is visibly there, so this plots raw intensity along the surface normal
    and marks what the detector picked out.
    """
    from sheetcheck.io import Surface, Volume  # noqa: E402
    from sheetcheck.profile import find_sheets  # noqa: E402
    from sheetcheck.support import profile_levels  # noqa: E402
    from sheetcheck.survey import find_pairings  # noqa: E402

    pr = find_pairings("PHerc1667", max_segments=1)[0]
    surf = Surface.load(pr.mesh_path, cache_dir=str(ROOT / "cache"))
    vol = Volume(pr.volume_path, level=1)
    vox = vol.voxel_size_um
    normals, ok = surf.normals()

    # One ray per widely separated location. Taking several rays from a single
    # patch samples what is effectively the same ray, which makes a local
    # oddity look like a global one.
    rng = np.random.default_rng(4)
    vi, ui = np.nonzero(ok)
    reach = 700.0 / vox
    offs = np.arange(-reach, reach + 1e-9, 0.5)

    prof = []
    tries = 0
    while len(prof) < n_rays and tries < n_rays * 6:
        tries += 1
        c = int(rng.integers(0, len(vi)))
        p0 = surf.points[vi[c], ui[c]][None, :]
        nn = normals[vi[c], ui[c]][None, :]
        ray = vol.to_level(p0)[:, None, :] + offs[None, :, None] * nn[:, None, :]
        block, blo = vol.read_box(ray.reshape(-1, 3).min(axis=0) - 4,
                                  ray.reshape(-1, 3).max(axis=0) + 4)
        if block.size == 0:
            continue
        p = Volume.sample_box(block, blo, ray)[0]
        if np.count_nonzero(p) < len(p) * 0.7:
            continue
        prof.append(p)
    prof = np.array(prof)

    fig, axes = plt.subplots(len(prof), 1, figsize=(7.2, 1.35 * len(prof)),
                             dpi=150, sharex=True)
    x = offs * vox
    for i, (ax, p) in enumerate(zip(np.atleast_1d(axes), prof)):
        ax.grid(True, axis="y", zorder=0)
        ax.set_axisbelow(True)
        ax.plot(x, p, color=BLUE, linewidth=1.4, zorder=3)
        gap, pap, good = profile_levels(p)
        if good:
            ax.axhline(gap, color=MUTED, linewidth=0.8, linestyle=":", zorder=2)
            ax.axhline(pap, color=MUTED, linewidth=0.8, linestyle=":", zorder=2)
        sheets = find_sheets(p, 0.5, vox, min_thickness_um=25.0) * vox
        for s in sheets:
            ax.axvline(s, color=ORANGE, linewidth=1.2, alpha=0.85, zorder=4)
        ax.axvline(0.0, color=RED, linewidth=1.6, zorder=5)
        ax.set_ylabel("CT", fontsize=8)
        ax.set_yticks([])
        if i == 0:
            ax.text(0.005, 1.06,
                    "blue: CT intensity along the surface normal   |   "
                    "orange: detected sheet centres   |   "
                    "red: the traced surface",
                    transform=ax.transAxes, color=INK2, fontsize=8)

    np.atleast_1d(axes)[-1].set_xlabel(
        "distance along the surface normal  (µm)")
    fig.suptitle("Visual check of the sheet detector on real CT",
                 color=INK, fontsize=11, x=0.005, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(FIG / "profile_check.png")
    plt.close(fig)
    print("  profile_check.png")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", action="store_true",
                    help="also render the CT profile check (needs network)")
    args = ap.parse_args()

    style()
    FIG.mkdir(exist_ok=True)
    print("writing figures/")
    fig_placement_vs_ink()
    fig_pitch()
    fig_resolution()
    if args.profiles:
        fig_profiles()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
