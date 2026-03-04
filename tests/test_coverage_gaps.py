"""Tests targeting specific uncovered lines to improve code coverage."""

import jax.numpy as jnp
import numpy as np
import pytest

from sgp4jax import (
    tle_to_satrec, gcrf_positions, gcrf_positions_multi,
    make_satrec, WGS72, SatRec,
)
from sgp4jax._frames import _rot_x, _leap_seconds, _ut1_to_utc
from sgp4jax._initl import gstime


# ISS TLE
LINE1 = '1 25544U 98067A   20045.18587073  .00000950  00000-0  25302-4 0  9990'
LINE2 = '2 25544  51.6443 242.0161 0004397 264.6060 207.3845 15.49165514212791'

PI = float(jnp.pi)


# ---------------------------------------------------------------
# __init__.py:106-109 — gcrf_positions() single-satellite helper
# ---------------------------------------------------------------

class TestGcrfPositions:
    """Test the gcrf_positions() convenience function."""

    def test_basic_output_shape(self):
        sat = tle_to_satrec(LINE1, LINE2, gravity=WGS72)
        base_jd = float(sat.jdsatepoch) + float(sat.jdsatepochF)
        times = jnp.array([base_jd, base_jd + 1.0 / 24.0, base_jd + 1.0])

        r, v = gcrf_positions(sat, times)

        assert r.shape == (3, 3)
        assert v.shape == (3, 3)
        assert jnp.all(jnp.isfinite(r))
        assert jnp.all(jnp.isfinite(v))

    def test_matches_gcrf_positions_multi(self):
        """Single-sat gcrf_positions should match row 0 of gcrf_positions_multi."""
        from sgp4jax import tles_to_satrec

        sat = tle_to_satrec(LINE1, LINE2, gravity=WGS72)
        base_jd = float(sat.jdsatepoch) + float(sat.jdsatepochF)
        times = jnp.array([base_jd, base_jd + 0.5])

        r_single, v_single = gcrf_positions(sat, times)

        batch = tles_to_satrec([(LINE1, LINE2)], gravity=WGS72)
        r_multi, v_multi = gcrf_positions_multi(batch, times)

        np.testing.assert_allclose(
            np.array(r_single), np.array(r_multi[0]), atol=1e-10)
        np.testing.assert_allclose(
            np.array(v_single), np.array(v_multi[0]), atol=1e-10)

    def test_positions_physically_reasonable(self):
        sat = tle_to_satrec(LINE1, LINE2, gravity=WGS72)
        base_jd = float(sat.jdsatepoch) + float(sat.jdsatepochF)
        times = jnp.array([base_jd])

        r, v = gcrf_positions(sat, times)
        pos_mag = float(jnp.linalg.norm(r[0]))
        # ISS at ~400 km altitude
        assert 6500.0 < pos_mag < 7100.0


# ---------------------------------------------------------------
# _types.py:137-139 — make_satrec() factory
# ---------------------------------------------------------------

class TestMakeSatrec:
    """Test the make_satrec() utility function."""

    def test_defaults_to_zero(self):
        sat = make_satrec()
        for field in SatRec._fields:
            val = float(getattr(sat, field))
            assert val == 0.0, f"Field '{field}' should default to 0.0, got {val}"

    def test_partial_kwargs(self):
        sat = make_satrec(bstar=jnp.array(1e-5), ecco=jnp.array(0.5))
        assert float(sat.bstar) == pytest.approx(1e-5)
        assert float(sat.ecco) == pytest.approx(0.5)
        # Other fields should still be zero
        assert float(sat.inclo) == 0.0

    def test_returns_satrec(self):
        sat = make_satrec()
        assert isinstance(sat, SatRec)
        assert len(sat) == len(SatRec._fields)


# ---------------------------------------------------------------
# _frames.py:91-93 — _rot_x() rotation matrix
# ---------------------------------------------------------------

class TestRotX:
    """Test the _rot_x() rotation matrix helper."""

    def test_identity_at_zero(self):
        R = _rot_x(jnp.array(0.0))
        np.testing.assert_allclose(np.array(R), np.eye(3), atol=1e-15)

    def test_90_degrees(self):
        R = _rot_x(jnp.array(PI / 2))
        # Rotating [0, 1, 0] by 90° about x should give [0, 0, 1]
        v = jnp.array([0.0, 1.0, 0.0])
        result = R @ v
        np.testing.assert_allclose(
            np.array(result), [0.0, 0.0, 1.0], atol=1e-15)

    def test_180_degrees(self):
        R = _rot_x(jnp.array(PI))
        # y → -y, z → -z
        v = jnp.array([0.0, 1.0, 1.0])
        result = R @ v
        np.testing.assert_allclose(
            np.array(result), [0.0, -1.0, -1.0], atol=1e-14)

    def test_orthogonality(self):
        R = _rot_x(jnp.array(0.7))
        RtR = np.array(R.T @ R)
        np.testing.assert_allclose(RtR, np.eye(3), atol=1e-15)

    def test_determinant_one(self):
        R = _rot_x(jnp.array(1.23))
        det = float(jnp.linalg.det(R))
        assert det == pytest.approx(1.0, abs=1e-15)

    def test_x_axis_unchanged(self):
        """Rotation about x should leave x-axis vector unchanged."""
        R = _rot_x(jnp.array(2.5))
        v = jnp.array([1.0, 0.0, 0.0])
        result = R @ v
        np.testing.assert_allclose(np.array(result), [1.0, 0.0, 0.0], atol=1e-15)


# ---------------------------------------------------------------
# _frames.py:600-602 — _leap_seconds() lookup
# ---------------------------------------------------------------

class TestLeapSeconds:
    """Test the _leap_seconds() lookup function."""

    def test_before_first_leap_second(self):
        """Before 1972-01-01, should clamp to first entry (10)."""
        jd_1970 = 2440587.5  # 1970-01-01
        ls = float(_leap_seconds(jnp.array(jd_1970)))
        assert ls == 10.0

    def test_1972_start(self):
        """At 1972-01-01 (first entry), should be 10."""
        jd_1972 = 2441317.5  # 1972-01-01
        ls = float(_leap_seconds(jnp.array(jd_1972)))
        assert ls == 10.0

    def test_mid_2000s(self):
        """In 2006, TAI-UTC was 33 seconds."""
        jd_2006 = 2453736.5  # ~2006-01-01
        ls = float(_leap_seconds(jnp.array(jd_2006)))
        assert ls == 33.0

    def test_2017_onwards(self):
        """After 2017-01-01, TAI-UTC is 37 seconds (last entry)."""
        jd_2020 = 2458849.5  # ~2020-01-01
        ls = float(_leap_seconds(jnp.array(jd_2020)))
        assert ls == 37.0

    def test_monotonically_increasing(self):
        """Leap seconds should never decrease over time."""
        jds = jnp.linspace(2441317.5, 2460000.0, 50)
        ls_vals = [float(_leap_seconds(jd)) for jd in jds]
        for i in range(1, len(ls_vals)):
            assert ls_vals[i] >= ls_vals[i - 1], (
                f"Leap seconds decreased at index {i}")


# ---------------------------------------------------------------
# _frames.py:614-620 — _ut1_to_utc() conversion
# ---------------------------------------------------------------

class TestUt1ToUtc:
    """Test the _ut1_to_utc() UT1 → UTC conversion."""

    def test_returns_adjusted_fraction(self):
        """Output JD integer part should be unchanged; fraction should be adjusted."""
        jd = jnp.array(2458849.0)  # ~2020-01-01
        fr = jnp.array(0.5)
        jd_out, fr_out = _ut1_to_utc(jd, fr)

        # Integer part should pass through unchanged
        assert float(jd_out) == float(jd)
        # Fractional part should be close to input (UT1-UTC < 0.9s)
        assert abs(float(fr_out) - float(fr)) < 1.0 / 86400.0  # < 1 second

    def test_adjustment_magnitude(self):
        """UTC-UT1 offset should be small (< 1 second)."""
        jd = jnp.array(2451545.0)  # J2000.0
        fr = jnp.array(0.0)
        _, fr_out = _ut1_to_utc(jd, fr)
        offset_seconds = abs(float(fr_out) - 0.0) * 86400.0
        # delta_T ~ 64s for 2000, leap seconds ~ 32s, 32.184 offset
        # so UTC-UT1 ≈ 64 - 32.184 - 32 ≈ -0.2s  (should be < 1s)
        assert offset_seconds < 2.0, (
            f"UTC-UT1 offset {offset_seconds:.3f}s is too large")

    def test_different_epochs(self):
        """Should produce finite results across different epochs."""
        test_dates = [
            2441317.5,  # 1972
            2448622.5,  # 1992
            2451545.0,  # 2000
            2458849.5,  # 2020
        ]
        for jd_val in test_dates:
            jd = jnp.array(jd_val)
            fr = jnp.array(0.0)
            jd_out, fr_out = _ut1_to_utc(jd, fr)
            assert jnp.isfinite(fr_out), f"Non-finite result for JD={jd_val}"


# ---------------------------------------------------------------
# _initl.py:17 — gstime() negative wrap-around
# ---------------------------------------------------------------

class TestGstimeNegativeWrap:
    """Test gstime() handles the negative modulo edge case."""

    def test_result_always_positive(self):
        """GMST should always be in [0, 2*pi) regardless of input epoch."""
        twopi = 2.0 * PI
        # Test a range of Julian dates including ones that might produce
        # negative intermediate values
        test_jds = [
            2433281.5,      # ~1950-01-01 (epoch reference)
            2415020.0,      # ~1900-01-01
            2400000.0,      # ~1858
            2433281.5 + 1,  # just after epoch ref
            2451545.0,      # J2000
        ]
        for jd in test_jds:
            gst = gstime(jd)
            assert 0.0 <= gst < twopi, (
                f"gstime({jd}) = {gst} not in [0, 2pi)")

    def test_known_gmst_j2000(self):
        """GMST at J2000.0 should be approximately 4.894961 rad."""
        gst = gstime(2451545.0)
        # Reference: GMST at J2000.0 ≈ 280.46062°≈ 4.8949612 rad
        np.testing.assert_allclose(gst, 4.8949612, rtol=1e-5)

    def test_far_past_epoch(self):
        """Very early Julian dates should still produce valid GMST."""
        gst = gstime(2400000.0)
        twopi = 2.0 * PI
        assert 0.0 <= gst < twopi


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
