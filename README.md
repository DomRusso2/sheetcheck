# sheetcheck

Geometric measurement of traced papyrus surfaces in Herculaneum scroll CT data.

`sheetcheck` measures how a traced surface (`tifxyz`) sits inside the CT volume
it was traced from: how far it is from the papyrus it claims to follow, how the
wraps are spaced around it, and how well-resolved the sheet structure is in that
neighbourhood. It streams directly from the public Vesuvius Challenge S3 bucket,
so nothing has to be downloaded in bulk.

It was built to answer a specific question from the 2026 Open Problems list --
whether **surface misplacement** explains ink-detection failures -- and along
the way it produced a set of measurements about the published traces that had
not been written down anywhere.

## Why this exists

[Open Problem 6](https://scrollprize.org/2026_open_problems) states that when
ink models fail to generalise, it is unclear whether the cause is *scan quality,
surface misplacement, label mismatch, architecture limits, ink morphology
variation, or fundamental signal absence*. Six hypotheses, no way to tell them
apart.

Surface misplacement is the one that is purely geometric, so it is the one that
can be settled by measurement rather than by training another model. This
repository settles it, and reports several other quantities needed to do so.

## Install

```bash
pip install -e .
```

Python 3.10+. No credentials are needed -- the open-data bucket is public and
is read anonymously.

## Quick start

```bash
python scripts/m3_survey.py --scrolls PHerc1667 --patches 10
```

This discovers a traced segment, pairs it with the exact CT volume it was
registered against, samples rays along the surface normal, and reports the
placement and geometry statistics for that trace.

## What it measures

For each sampled point on a traced surface, `sheetcheck` casts a ray along the
surface normal and measures:

| Quantity | Meaning |
| --- | --- |
| `support` | Where the traced point sits between the ray's air-gap level (0.0) and its papyrus level (1.0) |
| `offset_um` | Distance from the traced point to the centre of the nearest papyrus sheet |
| `pitch_um` | Local wrap-to-wrap spacing, from the autocorrelation period of the ray |
| `gap_structure_frac` | Fraction of rays where papyrus and air are separable at all -- low values mark compressed regions |
| `planarity` | Structure-tensor sheet-likeness of the CT at that point |

Every quantity is normalised against the ray's own contrast, so it is
comparable across regions of differing density and across scrolls scanned at
different energies and resolutions.

## Design notes

Three constraints shaped the implementation, each learned by measurement rather
than assumed:

**Everything is local.** Azimuth about a fitted scroll axis is *not* a usable
winding coordinate. On PHerc. 1667 the radius wanders about a fitted spiral by
roughly four times the winding pitch, so advancing 2*pi in azimuth does not
reliably land one wrap later. Any quantity here that needed a global winding
coordinate was removed.

**Pitch is estimated locally, never assumed.** Within a single scroll the wrap
pitch varies by roughly a factor of two. A global constant produces a detector
that misfires in compressed regions and goes blind in expanded ones.

**Ambiguity is reported, not silently resolved.** A ray that crosses almost no
air has no meaningful papyrus/gap threshold. Rather than let Otsu split the
papyrus distribution against itself and return a confident-looking number,
those rays are reported as having no measurable gap structure.

## Tests

```bash
pytest tests/
```

The suite pins the measurement primitives against synthetic ground truth. Each
test corresponds to a bug that occurred during development -- period estimation
under a brightness envelope, sheet counting across two-ply papyrus, circle
fitting on short arcs, and threshold estimation on rays with no gap.

## Data

Reads the AWS Open Data mirror at `s3://vesuvius-challenge-open-data/`.
Surfaces are `tifxyz` (x/y/z TIFF triplet plus `meta.json`), volumes are
multiscale OME-Zarr. Decoded meshes can be cached locally via
`Surface.load(..., cache_dir=...)`; full-scroll meshes are large (the merged
PHerc. 1667 trace is a 2061 x 30097 grid) and slow to re-fetch.

## Licence

MIT.
