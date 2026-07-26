# Measured geometry of published Herculaneum traces

Every number below is produced by the scripts in `scripts/`, reading only the
public `s3://vesuvius-challenge-open-data/` bucket. Medians carry bootstrap 95%
confidence intervals from resampling the pooled per-ray measurements.

Sampling budget is 32 patches x 25 rays per segment, chosen because central
estimates stop moving there (`scripts/m4_convergence.py`): offset reads
24.07 / 24.31 / 24.81 um at 16 / 32 / 64 patches. Below ~16 patches the same
segment with the same seed can return support +0.82 or +0.35, so any figure
quoted from a smaller budget is noise.

---

## F1. The winding pitch is 213-230 um, not 187 um

| Scroll | Voxel | Pitch (95% CI) |
| --- | --- | --- |
| PHercParis4 | 1.129 um | **213 [204, 222]** |
| PHerc1667 | 1.129 um | **222 [209, 236]** |
| PHerc0172 | 7.910 um | **224 [217, 231]** |
| PHerc0139 | 1.129 um | **230 [225, 242]** |

Median 223 um. **Every interval excludes 187.3 um**, the figure commonly taken
from the winding-ruler pitch atlas.

Three independent estimators agree within PHerc1667, sharing almost no code
path:

| Method | Pitch |
| --- | --- |
| Counting wraps along long radial rays | 208 um |
| Autocorrelation of normal-ray intensity | 220 um |
| Spiral slope `2*pi * dr/dtheta` | 222 um |

The measurement is resolution-independent across an 8x range. Reading one
volume at successive pyramid levels, with scroll, segment, trace and scan all
fixed:

| Effective voxel | Voxels per wrap | Pitch (95% CI) |
| --- | --- | --- |
| 2.26 um | 98.5 | 222 [209, 236] |
| 4.52 um | 49.0 | 221 [210, 239] |
| 9.03 um | 24.6 | 222 [208, 239] |
| 18.06 um | 12.1 | 218 [208, 239] |

Two independently scanned volumes of the same segment agree as well
(2.399 um: 230 [220, 238]; 1.129 um: 222 [209, 236]).

**The measurement convention does not explain the gap.** Pitch along the
surface normal is a *perpendicular* sheet spacing; spiral fitting yields a
*radial* advance per turn. The surface normal is oblique to the radial
direction by 19-28 degrees (M7), so the two could differ by up to ~13% -- the
same order as the disagreement above, which would be a fatal confound.

Measuring both on the *same* rays from the *same* points on PHerc1667
(`scripts/m8_pitch_convention.py`, n=421) settles it:

| Quantity | Value (95% CI) |
| --- | --- |
| Perpendicular (normal ray) | 240.1 [219.5, 250.4] |
| Radial (radial ray) | 238.7 [228.6, 249.7] |
| Ratio radial/perpendicular | 1.024 [1.004, 1.042] |
| Predicted 1/cos(theta) | 1.060 [1.052, 1.068] |

The convention effect is real but only ~2.4%, against a ~28% discrepancy.
**Both conventions exclude 187.3 um.** The direction predicted by geometry is
confirmed (radial >= perpendicular); the magnitude is about 40% of the
idealised parallel-plane prediction, and the two intervals do not overlap --
expected, since real wraps are curved rather than parallel planes, which
dilutes the obliquity effect.

**Why the discrepancy matters.** The winding-ruler figure (median 187.3 um, IQR
181.5-193.4) is a distribution of *per-scroll medians across 35+ scrolls*. It is
not a within-scroll spread, and its narrow IQR describes agreement between
scroll averages rather than precision at any point. Used as a geometric
tolerance it is roughly 10-15% too tight.

## F2. Within one scroll the pitch varies about two-fold

Per-ray p10-p90 spans are 106-338, 118-324, 114-322 and 119-334 um for the four
scrolls above -- roughly 110-335 um throughout. The variation is systematic
rather than noisy, tracking compressed versus expanded regions.

Consequence for tooling: any threshold expressed as a fraction of "the pitch"
must estimate the pitch locally. A global constant misfires in compressed
regions and goes blind in expanded ones.

## F3. Deformation is about four times the pitch

Fitting a spiral to a traced surface on PHerc1667 leaves an rms radius residual
of **869 um**, against a pitch of ~220 um. The binned radius profile is not even
monotonic in azimuth (46% of bins increase).

Consequence: advancing `2*pi` in azimuth about a fitted axis does not reliably
land one wrap later, so azimuth is not a usable winding coordinate on a crushed
scroll. Any quantity requiring a global winding coordinate has to come from the
project's own winding field rather than from a fitted axis.

## F4. Published traces sit 26-37 um from the local sheet centre

| Scroll | Offset (95% CI) |
| --- | --- |
| PHerc0139 | 26 [22, 29] um |
| PHerc1667 | 29 [25, 36] um |
| PHercParis4 | 33 [30, 37] um |
| PHerc0172 | 37 [35, 41] um |

Also resolution-independent (31 [29, 35] at 2.399 um vs 29 [25, 36] at 1.129 um
on the same segment).

A sheet is roughly 100 um thick, so ~30 um off-centre is consistent with a trace
targeting the recto *face* rather than the mid-plane, and is not by itself
evidence of error.

## F5. Surface placement does not predict ink detectability

The published ink raster for a segment is pixel-exact with its tifxyz grid at
20x (see F6), so each mesh cell has both a 3D position and an ink score.

Correlating placement against ink over 554 sampled cells on PHerc1667:

| | correlation |
| --- | --- |
| \|offset\| vs ink mean | +0.033 |
| \|offset\| vs ink contrast | -0.045 |
| CT value at surface vs ink contrast | +0.054 |

Nothing, across offsets spanning 0-150 um. Ink contrast was measured over
512x512 px windows; at the letter scale this is a properly powered test, and an
earlier 20x20 px window was not (letters are 400-800 px, so every small window
sat inside a single stroke and carried no letter structure).

**This addresses Open Problem 6 directly.** That problem lists six candidate
causes for ink models failing to generalise -- scan quality, *surface
misplacement*, label mismatch, architecture limits, ink morphology variation, or
fundamental signal absence -- and states they cannot be told apart. Surface
misplacement is ruled out at the scale published traces actually exhibit. This
is consistent with the ink models sampling a stack of layers around the surface,
which makes them robust to modest offset by construction.

## F6. Ink rasters are pixel-exact with tifxyz grids at 20x

For PHerc1667 segment 20260108140509: the tifxyz grid is 1975 x 736 with
`scale = 0.05` (20 volume voxels per grid cell), and the published ink raster is
39500 x 14720 = exactly 20x in both axes. So mesh cell `(v, u)` owns ink pixels
`[20v : 20v+20, 20u : 20u+20]` with no registration or interpolation.

This is not documented anywhere and makes any geometry/ink correlation study
straightforward.

## F7. There is no classical fiber-orientation signal at 2.399 um

Papyrus is two plies laid crosswise, so the in-plane fiber direction should
rotate ~90 degrees between recto and verso. Sampling structure-tensor
orientation across the sheet on PHerc1667:

- median fiber angle at -20 um: 47.9 deg
- median fiber angle at +20 um: 47.9 deg
- **swing across the sheet: 0.1 deg**

Across the same samples, in-plane *linearity* is 0.10-0.16 while sheet
*planarity* is 0.61-0.80. So the normal direction is well defined but the
in-plane structure is essentially isotropic: the sheet is clean, and the plies
are simply not separable by a structure tensor at this resolution.

(Planarity values are not comparable between experiments here -- they depend on
the structure-tensor smoothing scale and on resolution, per F8. The 0.61-0.80
range is from this experiment specifically.)

Consequence: recto/verso ply classification needs a learned fiber model such as
the project's `fiber_hz_vt`, not classical orientation analysis.

## F8. Planarity is resolution-dependent and must not be compared across scans

Same segment, same scroll, same scan, varying only the pyramid level:

| Effective voxel | Planarity (95% CI) |
| --- | --- |
| 2.26 um | 0.77 [0.74, 0.81] |
| 4.52 um | 0.88 [0.87, 0.90] |
| 9.03 um | 0.92 [0.91, 0.93] |
| 18.06 um | 0.92 [0.91, 0.93] |

Finer voxels give *lower* planarity, because internal fiber structure becomes
resolved and the sheet stops looking ideally flat.

Pitch and offset are resolution-independent (F1, F4); planarity is not. Cross
comparing planarity between scans of different resolution produces a spurious
"scan quality" ranking that is really a resolution ranking.

## F10. Loss of wrap-gap structure is a scroll property, not a sampling artefact

The fraction of normal rays with a separable papyrus/air split is ~100% on
PHerc1667, PHercParis4 and PHerc0139, but **2%** on PHerc0172 (Scroll 5). The
obvious explanation would be that PHerc0172 is the only one scanned coarsely
(7.91 um), so resolution and scroll identity are confounded.

The pyramid series rules that out. Degrading PHerc1667 to 18.06 um -- more than
twice as coarse as the PHerc0172 scan, and only 12 voxels per wrap -- leaves gap
structure at **100.0%**:

| Effective voxel | Voxels per wrap | Gap structure |
| --- | --- | --- |
| 2.26 um | 98.5 | 100.0% |
| 4.52 um | 49.0 | 100.0% |
| 9.03 um | 24.6 | 100.0% |
| 18.06 um | 12.1 | 100.0% |

So sampling density is not what destroys the gap signal. PHerc0172 genuinely
lacks resolvable inter-wrap gaps, which is consistent with it being the scroll
whose upper ~30% is reported as too mangled to trace.

This makes `gap_structure_frac` usable as a scan/scroll quality measure that is
*not* contaminated by resolution -- unlike planarity (F8).

**Caveat.** Pyramid downsampling averages voxels; it adds no detector blur or
photon noise. This experiment therefore excludes *sampling density* as the
cause, but does not fully exclude that physically scanning at 18 um would
degrade the signal by other means.

## F9. Mesh normals agree with CT structure-tensor normals

Finite-difference normals from the tifxyz grid versus structure-tensor normals
from the CT, on PHerc1667: median angle **15.7 deg** (p25 9.6, p75 24.7), with
median planarity 0.88. Mesh normals are good enough to cast measurement rays
along.

---

## Hypotheses tested and rejected

Recorded because knowing which approaches fail has value, and each was tested
before being abandoned.

| Hypothesis | Outcome |
| --- | --- |
| Azimuth holonomy detects sheet switches | Works on an isolated winding (0/113 false positives), fails on a merged multi-turn trace (~35%). Cause is F3 -- azimuth is not a winding coordinate on a crushed scroll |
| Recto/verso ply classification from fiber orientation | No signal (F7) |
| Surface placement predicts ink detectability | No relationship (F5) |
| CT quality predicts where tracing succeeds | Inconclusive; the tested metric was miscalibrated and the trace-failure labels were weak |
| Off-papyrus runs localise sheet switches | Inconclusive; recall 74% against 57% false positives, and the support metric was miscalibrated at the time of the test |

---

## Reproduction

```bash
pip install -e .
pytest tests/                             # 25 tests, synthetic ground truth
python scripts/m3_survey.py --patches 32  # cross-scroll table (F1, F2, F4)
python scripts/m4_convergence.py          # sampling budget justification
python scripts/m5_resolution.py           # two independent scans (F1, F4, F8)
python scripts/m6_pyramid.py              # 8x voxel-size series (F1, F8, F10)
python scripts/m7_normal_vs_radial.py     # normal-vs-radial obliquity (F1)
python scripts/m8_pitch_convention.py     # paired pitch conventions (F1)
python scripts/p1_placement.py            # placement vs ink (F5, F6)
python scripts/p0_ply.py                  # fiber orientation (F7)
python scripts/diag_orient.py             # mesh vs CT normals (F9)
```

No credentials are required. Results land in `results/` as JSON, including the
raw per-ray arrays so the statistics can be recomputed without re-streaming.

## Limitations

- Four scrolls, one segment each. The pitch result replicates across all four,
  but scroll-level statistics are single-segment.
- Only PHerc0172 is measured at a coarse resolution. The pyramid series (F10)
  breaks the resolution/scroll confound for gap structure, but no scroll in the
  bucket is published at both a coarse and a fine *physical* scan, so detector
  effects at coarse resolution remain untested.
- F5 is measured on one scroll's ink raster. It rules out placement as a cause
  of ink failure *on a scroll where ink is already readable*; it does not prove
  placement is irrelevant on scrolls where ink has never been recovered.
- Segments using the older `mesh/intermediate/tifxyz_normalized` layout
  (PHerc0332, PHercParis3) are not covered, since it is unverified whether those
  coordinates are in volume voxel space.
