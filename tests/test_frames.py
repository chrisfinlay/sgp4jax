"""Tests for frame transformations: TEME → GCRF and ITRF ↔ GCRF.

Compares sgp4jax output against Skyfield reference values.
Requires skyfield in test dependencies.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import sgp4jax
from sgp4jax._frames import teme_to_gcrf, itrf_to_gcrf, gcrf_to_itrf

# ISS TLE
LINE1_ISS = "1 25544U 98067A   20045.18587073  .00000950  00000-0  25302-4 0  9990"
LINE2_ISS = "2 25544  51.6443 242.0161 0004397 264.6060 207.3845 15.49165514212791"

# Deep-space (Molniya-like) TLE
LINE1_DEEP = "1 09880U 77021A   00251.45080028  .00000316  00000-0  10000-3 0  3527"
LINE2_DEEP = "2 09880  64.7791 180.0788 7258491 296.1385  20.2281  2.00879014156621"


def _skyfield_reference(line1, line2, minutes_offsets):
    """Get Skyfield reference GCRF and TEME vectors, plus rotation matrices.

    Returns GCRF/TEME for each time, computed using Skyfield's full pipeline.
    Also returns Skyfield's TEME rotation matrices so we can compare our
    transform applied to the same TEME vectors.
    """
    from skyfield.api import load, EarthSatellite
    from skyfield.sgp4lib import TEME as SfTEME
    from skyfield.functions import _T as sf_transpose
    from skyfield.constants import AU_KM, DAY_S

    ts = load.timescale()
    sat = EarthSatellite(line1, line2, ts=ts)

    results = []
    for dt_min in minutes_offsets:
        jd_epoch = sat.model.jdsatepoch
        fr_epoch = sat.model.jdsatepochF
        jd_target = jd_epoch + fr_epoch + dt_min / 1440.0

        t = ts.ut1_jd(jd_target)

        # Get Skyfield's TEME vectors
        r_teme_au, v_teme_au_day, msg = sat._position_and_velocity_TEME_km(t)
        # These are already in km and km/s despite the variable names
        r_teme_km = r_teme_au  # Actually in km (the method name says "km")
        v_teme_km_s = v_teme_au_day

        # Get Skyfield's GCRF output
        pos = sat.at(t)
        r_gcrf_km = pos.position.km
        v_gcrf_km_s = pos.velocity.km_per_s

        # Get Skyfield's rotation matrix for TEME
        R_teme = SfTEME.rotation_at(t)
        R_teme_to_gcrf = sf_transpose(R_teme)

        results.append({
            'r_teme': r_teme_km,
            'v_teme': v_teme_km_s,
            'r_gcrf': r_gcrf_km,
            'v_gcrf': v_gcrf_km_s,
            'R': np.array(R_teme_to_gcrf),
            'jd_ut1': t.whole,
            'frac_ut1': t.ut1_fraction,
        })

    return results


class TestSkyfieldComparison:
    """Compare sgp4jax GCRF output against Skyfield reference."""

    @pytest.mark.parametrize("dt_min", [0.0, 10.0, 100.0, 360.0, 720.0, 1440.0])
    def test_iss_rotation_matrix(self, dt_min):
        """ISS: rotation matrix matches Skyfield to sub-mm precision."""
        refs = _skyfield_reference(LINE1_ISS, LINE2_ISS, [dt_min])
        ref = refs[0]

        # Apply our transform to Skyfield's TEME vectors at the same UT1 time
        r_teme = jnp.array(ref['r_teme'])
        v_teme = jnp.array(ref['v_teme'])
        jd = jnp.float64(ref['jd_ut1'])
        fr = jnp.float64(ref['frac_ut1'])

        r_gcrf, v_gcrf = teme_to_gcrf(r_teme, v_teme, jd, fr)

        # Reference: apply Skyfield's rotation to same TEME vectors
        r_gcrf_ref = ref['R'] @ ref['r_teme']
        v_gcrf_ref = ref['R'] @ ref['v_teme']

        np.testing.assert_allclose(
            np.array(r_gcrf), r_gcrf_ref, atol=1e-6,
            err_msg=f"ISS position mismatch at t={dt_min} min")
        np.testing.assert_allclose(
            np.array(v_gcrf), v_gcrf_ref, atol=1e-9,
            err_msg=f"ISS velocity mismatch at t={dt_min} min")

    @pytest.mark.parametrize("dt_min", [0.0, 100.0, 720.0, 1440.0])
    def test_deep_space_rotation_matrix(self, dt_min):
        """Deep-space: rotation matrix matches Skyfield to sub-mm precision."""
        refs = _skyfield_reference(LINE1_DEEP, LINE2_DEEP, [dt_min])
        ref = refs[0]

        r_teme = jnp.array(ref['r_teme'])
        v_teme = jnp.array(ref['v_teme'])
        jd = jnp.float64(ref['jd_ut1'])
        fr = jnp.float64(ref['frac_ut1'])

        r_gcrf, v_gcrf = teme_to_gcrf(r_teme, v_teme, jd, fr)

        r_gcrf_ref = ref['R'] @ ref['r_teme']
        v_gcrf_ref = ref['R'] @ ref['v_teme']

        np.testing.assert_allclose(
            np.array(r_gcrf), r_gcrf_ref, atol=1e-6,
            err_msg=f"Deep-space position mismatch at t={dt_min} min")
        np.testing.assert_allclose(
            np.array(v_gcrf), v_gcrf_ref, atol=1e-9,
            err_msg=f"Deep-space velocity mismatch at t={dt_min} min")

    def test_propagate_gcrf_end_to_end(self):
        """propagate_gcrf round-trips correctly (same TEME → same GCRF)."""
        sat = sgp4jax.tle_to_satrec(LINE1_ISS, LINE2_ISS)
        r_teme, v_teme, _ = sgp4jax.propagate(sat, jnp.array(100.0))
        jd = jnp.array(sat.jdsatepoch)
        fr = jnp.array(sat.jdsatepochF) + 100.0 / 1440.0

        # Direct transform
        r1, v1 = teme_to_gcrf(r_teme, v_teme, jd, fr)

        # Via propagate_gcrf
        r2, v2, _ = sgp4jax.propagate_gcrf(sat, jnp.array(100.0))

        np.testing.assert_allclose(np.array(r1), np.array(r2), atol=1e-12)
        np.testing.assert_allclose(np.array(v1), np.array(v2), atol=1e-12)

    def test_propagate_jd_gcrf_end_to_end(self):
        """propagate_jd_gcrf matches manual teme_to_gcrf."""
        sat = sgp4jax.tle_to_satrec(LINE1_ISS, LINE2_ISS)
        jd = jnp.array(sat.jdsatepoch)
        fr = jnp.array(sat.jdsatepochF + 100.0 / 1440.0)

        r_teme, v_teme, _ = sgp4jax.propagate_jd(sat, jd, fr)
        r1, v1 = teme_to_gcrf(r_teme, v_teme, jd, fr)
        r2, v2, _ = sgp4jax.propagate_jd_gcrf(sat, jd, fr)

        np.testing.assert_allclose(np.array(r1), np.array(r2), atol=1e-12)
        np.testing.assert_allclose(np.array(v1), np.array(v2), atol=1e-12)


class TestRotationProperties:
    """Test mathematical properties of the rotation matrix."""

    def _extract_rotation(self, jd, fr):
        """Extract rotation matrix by transforming basis vectors."""
        zero = jnp.zeros(3)
        e1 = jnp.array([1.0, 0.0, 0.0])
        e2 = jnp.array([0.0, 1.0, 0.0])
        e3 = jnp.array([0.0, 0.0, 1.0])

        r1, _ = teme_to_gcrf(e1, zero, jd, fr)
        r2, _ = teme_to_gcrf(e2, zero, jd, fr)
        r3, _ = teme_to_gcrf(e3, zero, jd, fr)
        return jnp.stack([r1, r2, r3], axis=1)

    def test_orthogonality(self):
        """Rotation matrix should be orthogonal (R^T R = I)."""
        sat = sgp4jax.tle_to_satrec(LINE1_ISS, LINE2_ISS)
        jd = jnp.array(sat.jdsatepoch)
        fr = jnp.array(sat.jdsatepochF + 100.0 / 1440.0)
        R = self._extract_rotation(jd, fr)
        I = R.T @ R
        np.testing.assert_allclose(np.array(I), np.eye(3), atol=1e-14)

    def test_determinant_one(self):
        """Rotation matrix should have determinant +1."""
        sat = sgp4jax.tle_to_satrec(LINE1_ISS, LINE2_ISS)
        jd = jnp.array(sat.jdsatepoch)
        fr = jnp.array(sat.jdsatepochF + 100.0 / 1440.0)
        R = self._extract_rotation(jd, fr)
        det = jnp.linalg.det(R)
        np.testing.assert_allclose(float(det), 1.0, atol=1e-14)

    def test_magnitude_preservation(self):
        """Rotation should preserve vector magnitude."""
        sat = sgp4jax.tle_to_satrec(LINE1_ISS, LINE2_ISS)
        r_teme, v_teme, _ = sgp4jax.propagate(sat, jnp.array(100.0))
        jd = jnp.array(sat.jdsatepoch)
        fr = jnp.array(sat.jdsatepochF + 100.0 / 1440.0)

        r_gcrf, v_gcrf = teme_to_gcrf(r_teme, v_teme, jd, fr)

        np.testing.assert_allclose(
            float(jnp.linalg.norm(r_gcrf)),
            float(jnp.linalg.norm(r_teme)),
            rtol=1e-14)
        np.testing.assert_allclose(
            float(jnp.linalg.norm(v_gcrf)),
            float(jnp.linalg.norm(v_teme)),
            rtol=1e-14)


class TestJAXCompatibility:
    """Test JIT, vmap, and grad compatibility."""

    def test_jit(self):
        """teme_to_gcrf should work under JIT."""
        sat = sgp4jax.tle_to_satrec(LINE1_ISS, LINE2_ISS)
        r_teme, v_teme, _ = sgp4jax.propagate(sat, jnp.array(100.0))
        jd = jnp.array(sat.jdsatepoch)
        fr = jnp.array(sat.jdsatepochF + 100.0 / 1440.0)

        jitted = jax.jit(teme_to_gcrf)
        r1, v1 = teme_to_gcrf(r_teme, v_teme, jd, fr)
        r2, v2 = jitted(r_teme, v_teme, jd, fr)

        np.testing.assert_allclose(np.array(r1), np.array(r2), atol=1e-15)
        np.testing.assert_allclose(np.array(v1), np.array(v2), atol=1e-15)

    def test_vmap_over_times(self):
        """propagate_gcrf should work with vmap over time steps."""
        sat = sgp4jax.tle_to_satrec(LINE1_ISS, LINE2_ISS)
        times = jnp.array([0.0, 10.0, 100.0, 360.0])

        batched = jax.vmap(sgp4jax.propagate_gcrf, in_axes=(None, 0))
        r_batch, v_batch, err_batch = batched(sat, times)

        assert r_batch.shape == (4, 3)
        assert v_batch.shape == (4, 3)

        # Verify each matches sequential call
        for i, t in enumerate(times):
            r_seq, v_seq, _ = sgp4jax.propagate_gcrf(sat, t)
            np.testing.assert_allclose(
                np.array(r_batch[i]), np.array(r_seq), atol=1e-12)

    def test_grad_wrt_time(self):
        """Gradient of GCRF position norm w.r.t. time should be finite."""
        sat = sgp4jax.tle_to_satrec(LINE1_ISS, LINE2_ISS)

        def loss(t):
            r, v, err = sgp4jax.propagate_gcrf(sat, t)
            return jnp.sum(r ** 2)

        grad_fn = jax.grad(loss)
        g = grad_fn(jnp.array(100.0))
        assert jnp.isfinite(g)
        assert float(g) != 0.0


# ---------------------------------------------------------------------------
# Known ITRF positions (lat_deg, lon_deg, elev_m) for parameterised tests
# ---------------------------------------------------------------------------

_LOCATIONS = [
    ("equator_greenwich",  0.0,    0.0,    0.0),
    ("london",            51.5,   -0.1,    0.0),
    ("sydney",           -33.9,  151.2,    0.0),
    ("north_pole",        89.9,    0.0,    0.0),
    ("high_altitude",     28.6,   77.2,  500_000.0),  # 500 km above Delhi
]


def _skyfield_itrf_reference(lat_deg, lon_deg, elev_m, jd_ut1):
    """Return (r_itrf_km, r_gcrf_km) for a ground location at jd_ut1 (full JD)."""
    from skyfield.api import load, wgs84
    ts = load.timescale()
    obs = wgs84.latlon(lat_deg, lon_deg, elevation_m=elev_m)
    r_itrf_km = np.array(obs.itrs_xyz.km)
    t = ts.ut1_jd(jd_ut1)
    r_gcrf_km = np.array(obs.at(t).position.km)
    jd = float(t.whole)
    fr = float(t.ut1_fraction)
    return r_itrf_km, r_gcrf_km, jd, fr


class TestITRFGCRF:
    """Compare itrf_to_gcrf / gcrf_to_itrf against Skyfield."""

    # A spread of Julian dates spanning several years
    _JD_LIST = [
        2451545.0,    # J2000.0  (2000-01-01 12:00 UT1)
        2453736.5,    # 2006-01-01
        2456658.5,    # 2014-01-01
        2459945.5,    # 2023-01-01
        2460676.5,    # 2025-01-01
    ]

    @pytest.mark.parametrize("name,lat,lon,elev", _LOCATIONS)
    @pytest.mark.parametrize("jd_ut1", _JD_LIST)
    def test_itrf_to_gcrf_vs_skyfield(self, name, lat, lon, elev, jd_ut1):
        """itrf_to_gcrf matches Skyfield observer GCRF to sub-mm precision."""
        r_itrf_km, r_gcrf_ref, jd, fr = _skyfield_itrf_reference(lat, lon, elev, jd_ut1)

        r_gcrf = itrf_to_gcrf(jnp.array(r_itrf_km), jnp.float64(jd), jnp.float64(fr))

        np.testing.assert_allclose(
            np.array(r_gcrf), r_gcrf_ref, atol=1e-6,
            err_msg=f"itrf_to_gcrf mismatch: {name} at JD {jd_ut1}")

    @pytest.mark.parametrize("name,lat,lon,elev", _LOCATIONS)
    @pytest.mark.parametrize("jd_ut1", _JD_LIST)
    def test_gcrf_to_itrf_round_trip(self, name, lat, lon, elev, jd_ut1):
        """gcrf_to_itrf(itrf_to_gcrf(r)) == r to floating-point precision."""
        r_itrf_km, _, jd, fr = _skyfield_itrf_reference(lat, lon, elev, jd_ut1)
        jd_ = jnp.float64(jd)
        fr_ = jnp.float64(fr)

        r_gcrf = itrf_to_gcrf(jnp.array(r_itrf_km), jd_, fr_)
        r_itrf_back = gcrf_to_itrf(r_gcrf, jd_, fr_)

        np.testing.assert_allclose(
            np.array(r_itrf_back), r_itrf_km, atol=1e-9,
            err_msg=f"round-trip mismatch: {name} at JD {jd_ut1}")

    @pytest.mark.parametrize("name,lat,lon,elev", _LOCATIONS)
    @pytest.mark.parametrize("jd_ut1", _JD_LIST)
    def test_gcrf_to_itrf_vs_skyfield(self, name, lat, lon, elev, jd_ut1):
        """gcrf_to_itrf recovers the original ITRF position from Skyfield GCRF."""
        r_itrf_ref, r_gcrf_km, jd, fr = _skyfield_itrf_reference(lat, lon, elev, jd_ut1)

        r_itrf = gcrf_to_itrf(jnp.array(r_gcrf_km), jnp.float64(jd), jnp.float64(fr))

        np.testing.assert_allclose(
            np.array(r_itrf), r_itrf_ref, atol=1e-6,
            err_msg=f"gcrf_to_itrf mismatch: {name} at JD {jd_ut1}")

    def test_magnitude_preserved(self):
        """ITRF ↔ GCRF rotation preserves vector magnitude."""
        r_itrf_km, _, jd, fr = _skyfield_itrf_reference(51.5, -0.1, 0.0, 2451545.0)
        r_gcrf = itrf_to_gcrf(jnp.array(r_itrf_km), jnp.float64(jd), jnp.float64(fr))
        np.testing.assert_allclose(
            float(jnp.linalg.norm(r_gcrf)),
            np.linalg.norm(r_itrf_km),
            rtol=1e-14)

    def test_jit_compatible(self):
        """itrf_to_gcrf and gcrf_to_itrf are already JIT-compiled."""
        r_itrf_km, _, jd, fr = _skyfield_itrf_reference(0.0, 0.0, 0.0, 2451545.0)
        r = jnp.array(r_itrf_km)
        jd_ = jnp.float64(jd)
        fr_ = jnp.float64(fr)

        r1 = itrf_to_gcrf(r, jd_, fr_)
        r2 = jax.jit(itrf_to_gcrf)(r, jd_, fr_)
        np.testing.assert_allclose(np.array(r1), np.array(r2), atol=1e-15)

        r3 = gcrf_to_itrf(r1, jd_, fr_)
        r4 = jax.jit(gcrf_to_itrf)(r1, jd_, fr_)
        np.testing.assert_allclose(np.array(r3), np.array(r4), atol=1e-15)

    def test_vmap_over_times(self):
        """itrf_to_gcrf works under vmap over time axis."""
        from skyfield.api import load, wgs84
        ts = load.timescale()
        obs = wgs84.latlon(51.5, -0.1, elevation_m=0)
        r_itrf_km = jnp.array(obs.itrs_xyz.km)

        jd_vals = jnp.array([2451545.0, 2453736.5, 2456658.5], dtype=jnp.float64)
        fr_vals = jnp.zeros(3, dtype=jnp.float64)

        r_batch = jax.vmap(itrf_to_gcrf, in_axes=(None, 0, 0))(r_itrf_km, jd_vals, fr_vals)
        assert r_batch.shape == (3, 3)

        for i in range(3):
            r_single = itrf_to_gcrf(r_itrf_km, jd_vals[i], fr_vals[i])
            np.testing.assert_allclose(
                np.array(r_batch[i]), np.array(r_single), atol=1e-12)
