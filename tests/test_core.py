"""Unit tests for the measurement primitives, against synthetic ground truth.

Each test pins a property that a real bug in this project violated at some
point, so these are regression tests rather than decoration.
"""

from __future__ import annotations

import numpy as np
import pytest

from sheetcheck.axis import fit_circle_2d
from sheetcheck.survey import bootstrap_ci
from sheetcheck.profile import (
    dominant_period,
    find_sheets,
    otsu_threshold,
)
from sheetcheck.support import profile_levels, runs_below
from sheetcheck.winding import axis_frame, cylindrical, unwrap_grid


def synth_profile(n=800, pitch_samples=80.0, sheet_frac=0.5,
                  gap_level=40.0, pap_level=140.0, envelope=0.0, seed=0):
    """A scroll-like ray: periodic papyrus slabs separated by air gaps.

    ``envelope`` adds the slow brightness drift that real rays carry and that
    broke the first period estimator.
    """
    rng = np.random.default_rng(seed)
    x = np.arange(n)
    phase = (x % pitch_samples) / pitch_samples
    prof = np.where(phase < sheet_frac, pap_level, gap_level).astype(float)
    if envelope:
        prof = prof + envelope * np.sin(2 * np.pi * x / (n * 1.3))
    return prof + rng.normal(0, 3.0, size=n)


class TestOtsu:
    def test_separates_two_modes(self):
        """The threshold must split the modes -- not necessarily land midway.

        Otsu returns a histogram bin centre, and for two zero-variance point
        masses every value in [lo, hi) separates them equally well, so
        asserting a midpoint would be testing a guarantee Otsu never makes.
        """
        x = np.concatenate([np.full(500, 40.0), np.full(500, 140.0)])
        thr = otsu_threshold(x)
        assert 40.0 <= thr < 140.0
        assert (x <= thr).sum() == 500
        assert (x > thr).sum() == 500

    def test_separates_two_spread_modes(self):
        """Assert separation quality, not threshold position.

        When two modes are far apart the between-class variance is nearly flat
        across a wide band of thresholds (2486 vs 2500 here), so the argmax
        position is arbitrary within it. What must hold is that the split
        assigns each mode to its own class.
        """
        rng = np.random.default_rng(0)
        lo = rng.normal(40, 5, 500)
        hi = rng.normal(140, 5, 500)
        thr = otsu_threshold(np.concatenate([lo, hi]))
        assert (lo <= thr).mean() > 0.95
        assert (hi > thr).mean() > 0.95

    def test_constant_input_is_safe(self):
        assert otsu_threshold(np.full(100, 7.0)) == pytest.approx(7.0)

    def test_empty_input_is_safe(self):
        assert otsu_threshold(np.array([])) == 0.0


class TestPeriod:
    @pytest.mark.parametrize("pitch_samples", [60.0, 80.0, 110.0])
    def test_recovers_known_period(self, pitch_samples):
        # vox_um=1, step=1 -> period in samples equals period in "um"
        prof = synth_profile(pitch_samples=pitch_samples)
        per, strength = dominant_period(prof, step_vox=1.0, vox_um=1.0,
                                        min_um=40.0, max_um=200.0)
        assert per == pytest.approx(pitch_samples, rel=0.10)
        assert strength > 0.2

    def test_survives_brightness_envelope(self):
        """A slow envelope must not capture the autocorrelation peak.

        Leaving this in was the bug that made the first pitch estimates drift
        toward the middle of the search band.
        """
        prof = synth_profile(pitch_samples=80.0, envelope=60.0)
        per, _ = dominant_period(prof, step_vox=1.0, vox_um=1.0,
                                 min_um=40.0, max_um=200.0)
        assert per == pytest.approx(80.0, rel=0.12)

    def test_flat_input_returns_nan(self):
        per, strength = dominant_period(np.full(400, 50.0), 1.0, 1.0)
        assert not np.isfinite(per)
        assert strength == 0.0


class TestFindSheets:
    def test_counts_slabs_not_plies(self):
        """One slab must count once even if it has internal structure.

        Peak-picking split every sheet into recto+verso and reported half the
        true pitch; run detection must not.
        """
        prof = synth_profile(n=800, pitch_samples=80.0)
        sheets = find_sheets(prof, step_vox=1.0, vox_um=1.0,
                             min_thickness_um=10.0)
        assert 8 <= len(sheets) <= 11
        spacing = np.diff(sheets)
        assert np.median(spacing) == pytest.approx(80.0, rel=0.12)

    def test_returns_empty_on_flat(self):
        assert len(find_sheets(np.full(200, 0.0), 1.0, 1.0)) == 0


class TestProfileLevels:
    def test_accepts_bimodal_ray(self):
        gap, pap, ok = profile_levels(synth_profile())
        assert ok
        assert gap == pytest.approx(40.0, abs=12)
        assert pap == pytest.approx(140.0, abs=12)

    def test_rejects_ray_without_gap(self):
        """A ray crossing no air must report 'no gap structure', not a number.

        Trusting Otsu here made a correctly placed trace score as if it were
        floating in a gap.
        """
        rng = np.random.default_rng(1)
        prof = 140.0 + rng.normal(0, 4.0, size=600)
        _, _, ok = profile_levels(prof)
        assert not ok


class TestRuns:
    def test_finds_runs_of_minimum_length(self):
        m = np.array([0, 1, 1, 0, 1, 0, 1, 1, 1], dtype=bool)
        assert runs_below(m, min_len=2) == [(1, 3), (6, 9)]

    def test_handles_edges(self):
        m = np.array([1, 1, 0, 0, 1, 1], dtype=bool)
        assert runs_below(m, min_len=2) == [(0, 2), (4, 6)]


class TestCircleFit:
    def test_recovers_circle_from_short_arc(self):
        """Kasa is badly biased on short arcs; Taubin must not be.

        Fitting the scroll axis to arc centroids gave radii of 6-4800 voxels
        for a single winding, which is what motivated this.
        """
        cy, cx, r = 400.0, -150.0, 1700.0
        t = np.linspace(0.0, 0.5, 400)      # ~29 degrees of arc
        y = cy + r * np.sin(t)
        x = cx + r * np.cos(t)
        fy, fx, fr = fit_circle_2d(y, x)
        assert fr == pytest.approx(r, rel=0.02)
        assert fy == pytest.approx(cy, abs=0.02 * r)
        assert fx == pytest.approx(cx, abs=0.02 * r)

    def test_collinear_points_raise(self):
        with pytest.raises((ValueError, np.linalg.LinAlgError)):
            fit_circle_2d(np.arange(50.0), np.zeros(50))


class TestBootstrap:
    def test_ci_brackets_the_statistic(self):
        rng = np.random.default_rng(0)
        a = rng.normal(24.0, 6.0, 400)
        med, lo, hi = bootstrap_ci(a)
        assert lo < med < hi
        assert med == pytest.approx(24.0, abs=1.5)

    def test_ci_narrows_with_sample_size(self):
        """More data must tighten the interval.

        An earlier convergence check estimated spread as the range of three
        seeds, which does *not* behave this way -- it appeared to widen with
        more sampling, which is what motivated switching to bootstrap.
        """
        rng = np.random.default_rng(1)
        widths = []
        for n in (50, 500, 5000):
            _, lo, hi = bootstrap_ci(rng.normal(0.0, 1.0, n))
            widths.append(hi - lo)
        assert widths[0] > widths[1] > widths[2]

    def test_ignores_nans(self):
        a = np.concatenate([np.full(200, 5.0), np.full(50, np.nan)])
        med, lo, hi = bootstrap_ci(a)
        assert med == pytest.approx(5.0)
        assert np.isfinite(lo) and np.isfinite(hi)

    def test_too_few_samples_returns_nan(self):
        med, lo, hi = bootstrap_ci(np.array([1.0, 2.0]))
        assert not np.isfinite(med)


class TestWinding:
    def test_axis_frame_is_orthonormal(self):
        for axis in (np.array([1.0, 0, 0]), np.array([0.3, -0.5, 0.8])):
            a = axis / np.linalg.norm(axis)
            e1, e2 = axis_frame(a)
            assert np.dot(e1, e2) == pytest.approx(0.0, abs=1e-9)
            assert np.dot(e1, a) == pytest.approx(0.0, abs=1e-9)
            assert np.linalg.norm(e1) == pytest.approx(1.0)
            assert np.linalg.norm(e2) == pytest.approx(1.0)

    def test_cylindrical_round_trip(self):
        origin = np.array([0.0, 0.0, 0.0])
        axis = np.array([1.0, 0.0, 0.0])
        pts = np.array([[0.0, 3.0, 4.0], [10.0, -5.0, 0.0]])
        r, th, h = cylindrical(pts, origin, axis)
        assert r[0] == pytest.approx(5.0)
        assert r[1] == pytest.approx(5.0)
        assert h[0] == pytest.approx(0.0)
        assert h[1] == pytest.approx(10.0)

    def test_unwrap_makes_azimuth_continuous(self):
        nv, nu = 3, 400
        th = np.linspace(0, 6 * np.pi, nu)
        wrapped = np.angle(np.exp(1j * th))
        grid = np.tile(wrapped, (nv, 1))
        valid = np.ones((nv, nu), dtype=bool)
        out, ok = unwrap_grid(grid, valid, axis_u=1)
        assert ok.all()
        span = out[0].max() - out[0].min()
        assert span == pytest.approx(6 * np.pi, rel=0.02)
        assert np.all(np.diff(out[0]) > -1e-6)

    def test_unwrap_does_not_bridge_gaps(self):
        """A gap of unknown width can hide whole turns; runs stay separate."""
        nu = 200
        th = np.zeros((1, nu))
        valid = np.ones((1, nu), dtype=bool)
        valid[0, 90:110] = False
        out, ok = unwrap_grid(th, valid, axis_u=1)
        assert not ok[0, 90:110].any()
