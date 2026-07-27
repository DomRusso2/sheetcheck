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

## F1 and F2. RETRACTED: the winding-pitch measurements were wrong

**These findings claimed the winding pitch is 213-230 um across four scrolls,
with every confidence interval excluding the commonly cited 187.3 um, and that
pitch varies about two-fold within a scroll. Both are withdrawn.**

The error was found by [IyanDopico](https://github.com/IyanDopico/vesuvius-sheet-tools),
who binned winding gaps by radius on PHerc. Paris 4 (706 human-annotated pairs
in one z window) and got a monotonic 136 -> 259 um, against a flat and
unordered result here over the same radii. Their control on PHerc1218 (9,054
pairs) is flat to 1.001 across 0.2-27 mm, which rules out a radius-dependent
bias in their own ruler.

### What was wrong

`dominant_period` estimates the wrap pitch as the dominant autocorrelation
period of the CT profile along the surface normal. It is exact on periodic
signals -- `scripts/m9_pitch_calibration.py` recovers 136/160/180/200/220 um
with zero bias at every additive-noise level tested.

Real scroll rays are not periodic. Wrap spacing varies along a single ray,
sheets are missed, neighbouring wraps merge. Real rays measure autocorrelation
strength around 0.22; additive noise alone cannot degrade a clean signal that
far. Reproducing that regime with position jitter and dropped sheets:

| true pitch | strength 0.82 | strength 0.36 | strength 0.20 (real data) |
| --- | --- | --- | --- |
| 140 um | 140 | 150 | **197** |
| 180 um | 180 | 191 | **224** |
| 220 um | 220 | 226 | **251** |

At the aperiodicity real data has, the estimate is biased high by 30-60 um, the
bias grows as the true pitch shrinks, and the dynamic range compresses toward
the middle of the search band (215 um). With a 60% pitch gradient imposed along
a ray the estimate barely moves.

That reproduces the published numbers exactly: a true 136 um core reads as
~197 (this repository reported 207), and a true ~175 um reads as ~220 (this
repository reported 213-230 everywhere). The flat radial profile was the
estimator compressing real variation, not an absence of variation.

### What this does not affect

Only F1 and F2 used the autocorrelation estimator. F3-F10 rest on the spiral
fit, structure tensor, Otsu-based level detection and the ink raster alignment,
and are unaffected. In particular F5 -- placement does not predict ink
detectability -- is robust to a scale error in the offset axis: a distorted
x-axis cannot manufacture a null correlation.

### Reproduce the retraction

```bash
python scripts/m9_pitch_calibration.py
```

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

Offset is resolution-independent (F4); planarity is not. (The pitch
measurements that once appeared here are retracted -- see F1/F2.) Cross
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
- `gap_structure_frac` is a per-ray verdict, not a guarantee that a whole ray is
  well-resolved. Visual inspection (`figures/profile_check.png`) shows rays with
  large featureless stretches still passing, because the remainder of the ray
  carries enough contrast to yield a valid papyrus/air split.
- Sample counts here are four scrolls, one segment each, a few hundred rays per
  segment. That is far below what the Challenge team asks for when validating a
  claim against ground-truth meshes, and it is the main thing more compute would
  buy. The pitch result replicates across all four and is independently
  confirmed elsewhere; the per-scroll `support` and `offset` figures are
  single-segment and should be read as such.
