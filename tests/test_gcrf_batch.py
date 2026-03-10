"""Tests for gcrf_positions_multi_leo, gcrf_positions_multi_sdp4_nr, gcrf_positions_mixed."""

import jax.numpy as jnp
import numpy as np
import pytest

from sgp4jax import (
    tle_to_satrec, tles_to_satrec,
    gcrf_positions_multi,
    gcrf_positions_multi_leo,
    gcrf_positions_multi_sdp4_nr,
    gcrf_positions_mixed,
    WGS84,
)


# ---------------------------------------------------------------------------
# TLEs
# ---------------------------------------------------------------------------

# Near-earth (method=0) — ISS
_LEO_L1  = '1 25544U 98067A   20045.18587073  .00000950  00000-0  25302-4 0  9990'
_LEO_L2  = '2 25544  51.6443 242.0161 0004397 264.6060 207.3845 15.49165514212791'

# Near-earth (method=0) — Sentinel-1A
_LEO2_L1 = '1 39634U 14016A   20045.50000000  .00000023  00000-0  14064-4 0  9994'
_LEO2_L2 = '2 39634  98.1825 145.6352 0001346  88.3457 271.7897 14.59198523314592'

# Deep-space irez=0 — GPS NAVSTAR 53
_GPS_L1  = '1 28129U 03058A   06175.57071136 -.00000104  00000-0  10000-3 0   459'
_GPS_L2  = '2 28129  54.7298 324.8098 0048506 266.2640  93.1663  2.00562768 18443'

# Deep-space irez=0 — SL-12 R/B
_DS_L1   = '1 20413U 83020D   05363.79166667  .00000000  00000-0  00000+0 0  7041'
_DS_L2   = '2 20413  12.3514 187.4253 7864447 196.3027 356.5478  0.24690082  7978'

# Deep-space irez=1 — ITALSAT 2 (GEO)
_GEO_L1  = '1 24208U 96044A   06177.04061740 -.00000094  00000-0  10000-3 0  1600'
_GEO_L2  = '2 24208   3.8536  80.0121 0026640 311.0977  48.3000  1.00778054 36119'

# Deep-space irez=2 — MOLNIYA 2-14
_MOL_L1  = '1 08195U 75081A   06176.33215444  .00000099  00000-0  11873-3 0   813'
_MOL_L2  = '2 08195  64.1586 279.0717 6877146 264.7651  20.2257  2.00491383225656'


def _epoch_jd(l1, l2):
    sat = tle_to_satrec(l1, l2, gravity=WGS84)
    return float(sat.jdsatepoch) + float(sat.jdsatepochF)


# Five observation times, one per quarter-day, anchored at each group's epoch
_TIMES_JD_LEO = jnp.array([_epoch_jd(_LEO_L1, _LEO_L2) + d for d in [0.0, 0.25, 0.5, 0.75, 1.0]])
_TIMES_JD_GPS = jnp.array([_epoch_jd(_GPS_L1, _GPS_L2) + d for d in [0.0, 0.25, 0.5, 0.75, 1.0]])
# Mixed test: use GPS-era times so all 2006-epoch satellites are near their epoch
_TIMES_JD_MIX = jnp.array([_epoch_jd(_GPS_L1, _GPS_L2) + d for d in [0.0, 0.25, 0.5, 0.75, 1.0]])


# ---------------------------------------------------------------------------
# Output shape tests
# ---------------------------------------------------------------------------

class TestOutputShape:
    def test_leo_shape(self):
        batch = tles_to_satrec([[_LEO_L1, _LEO_L2], [_LEO2_L1, _LEO2_L2]], gravity=WGS84)
        r, v = gcrf_positions_multi_leo(batch, _TIMES_JD_LEO)
        assert r.shape == (2, 5, 3)
        assert v.shape == (2, 5, 3)

    def test_sdp4_nr_shape(self):
        batch = tles_to_satrec([[_GPS_L1, _GPS_L2], [_DS_L1, _DS_L2]], gravity=WGS84)
        r, v = gcrf_positions_multi_sdp4_nr(batch, _TIMES_JD_GPS)
        assert r.shape == (2, 5, 3)
        assert v.shape == (2, 5, 3)

    def test_mixed_shape_heterogeneous(self):
        tles = [
            [_LEO_L1, _LEO_L2],
            [_GPS_L1, _GPS_L2],
            [_GEO_L1, _GEO_L2],
            [_MOL_L1, _MOL_L2],
        ]
        batch = tles_to_satrec(tles, gravity=WGS84)
        r, v = gcrf_positions_mixed(batch, _TIMES_JD_MIX)
        assert r.shape == (4, 5, 3)
        assert v.shape == (4, 5, 3)

    def test_mixed_shape_single_sat(self):
        batch = tles_to_satrec([[_GEO_L1, _GEO_L2]], gravity=WGS84)
        r, v = gcrf_positions_mixed(batch, _TIMES_JD_MIX)
        assert r.shape == (1, 5, 3)

    def test_mixed_shape_single_time(self):
        tles = [[_LEO_L1, _LEO_L2], [_GPS_L1, _GPS_L2]]
        batch = tles_to_satrec(tles, gravity=WGS84)
        r, v = gcrf_positions_mixed(batch, jnp.array([_TIMES_JD_MIX[0]]))
        assert r.shape == (2, 1, 3)


# ---------------------------------------------------------------------------
# Correctness: specialised functions must agree with gcrf_positions_multi
# (full propagator) for matching orbit types
# ---------------------------------------------------------------------------

class TestCorrectness:
    """For irez=0 satellites, propagate_leo and propagate_sdp4_nr are exact
    matches for propagate, so the GCRF batch functions must agree exactly."""

    def test_leo_matches_full(self):
        """gcrf_positions_multi_leo == gcrf_positions_multi for LEO satellites."""
        batch = tles_to_satrec([[_LEO_L1, _LEO_L2], [_LEO2_L1, _LEO2_L2]], gravity=WGS84)
        r_leo, v_leo = gcrf_positions_multi_leo(batch, _TIMES_JD_LEO)
        r_ref, v_ref = gcrf_positions_multi(batch, _TIMES_JD_LEO)
        np.testing.assert_allclose(np.array(r_leo), np.array(r_ref), atol=0,
                                   err_msg="position mismatch")
        np.testing.assert_allclose(np.array(v_leo), np.array(v_ref), atol=0,
                                   err_msg="velocity mismatch")

    def test_sdp4_nr_matches_full(self):
        """gcrf_positions_multi_sdp4_nr == gcrf_positions_multi for irez=0 satellites."""
        batch = tles_to_satrec([[_GPS_L1, _GPS_L2], [_DS_L1, _DS_L2]], gravity=WGS84)
        r_nr, v_nr = gcrf_positions_multi_sdp4_nr(batch, _TIMES_JD_GPS)
        r_ref, v_ref = gcrf_positions_multi(batch, _TIMES_JD_GPS)
        np.testing.assert_allclose(np.array(r_nr), np.array(r_ref), atol=0,
                                   err_msg="position mismatch")
        np.testing.assert_allclose(np.array(v_nr), np.array(v_ref), atol=0,
                                   err_msg="velocity mismatch")

    def test_mixed_homogeneous_leo(self):
        """gcrf_positions_mixed dispatches LEO group to propagate_leo — matches full."""
        batch = tles_to_satrec([[_GPS_L1, _GPS_L2], [_GEO_L1, _GEO_L2]], gravity=WGS84)
        r_mix, v_mix = gcrf_positions_mixed(batch, _TIMES_JD_MIX)
        r_ref, v_ref = gcrf_positions_multi(batch, _TIMES_JD_MIX)
        # GPS (irez=0): sdp4_nr == full; GEO (irez=1): full == full
        np.testing.assert_allclose(np.array(r_mix), np.array(r_ref), atol=0,
                                   err_msg="position mismatch")

    def test_mixed_each_satellite_matches_individual(self):
        """Each satellite in a mixed batch matches its individual gcrf_positions_multi call."""
        tles = [
            [_GPS_L1,  _GPS_L2],   # irez=0
            [_GEO_L1,  _GEO_L2],   # irez=1
            [_MOL_L1,  _MOL_L2],   # irez=2
        ]
        batch = tles_to_satrec(tles, gravity=WGS84)
        r_mix, v_mix = gcrf_positions_mixed(batch, _TIMES_JD_MIX)

        for i, tle_pair in enumerate(tles):
            single = tles_to_satrec([tle_pair], gravity=WGS84)
            r_ref, v_ref = gcrf_positions_multi(single, _TIMES_JD_MIX)
            np.testing.assert_allclose(
                np.array(r_mix[i]), np.array(r_ref[0]), atol=0,
                err_msg=f"position mismatch for satellite index {i}")
            np.testing.assert_allclose(
                np.array(v_mix[i]), np.array(v_ref[0]), atol=0,
                err_msg=f"velocity mismatch for satellite index {i}")


# ---------------------------------------------------------------------------
# Ordering preservation
# ---------------------------------------------------------------------------

class TestOrdering:
    def test_leo_ordering(self):
        """Result order matches input TLE order."""
        fwd = tles_to_satrec([[_LEO_L1, _LEO_L2], [_LEO2_L1, _LEO2_L2]], gravity=WGS84)
        rev = tles_to_satrec([[_LEO2_L1, _LEO2_L2], [_LEO_L1, _LEO_L2]], gravity=WGS84)
        r_fwd, _ = gcrf_positions_multi_leo(fwd, _TIMES_JD_LEO)
        r_rev, _ = gcrf_positions_multi_leo(rev, _TIMES_JD_LEO)
        np.testing.assert_allclose(np.array(r_fwd[0]), np.array(r_rev[1]), atol=0)
        np.testing.assert_allclose(np.array(r_fwd[1]), np.array(r_rev[0]), atol=0)

    def test_mixed_ordering(self):
        """gcrf_positions_mixed preserves input order across orbit type groups."""
        fwd = tles_to_satrec([[_GPS_L1, _GPS_L2], [_GEO_L1, _GEO_L2]], gravity=WGS84)
        rev = tles_to_satrec([[_GEO_L1, _GEO_L2], [_GPS_L1, _GPS_L2]], gravity=WGS84)
        r_fwd, _ = gcrf_positions_mixed(fwd, _TIMES_JD_MIX)
        r_rev, _ = gcrf_positions_mixed(rev, _TIMES_JD_MIX)
        np.testing.assert_allclose(np.array(r_fwd[0]), np.array(r_rev[1]), atol=0)
        np.testing.assert_allclose(np.array(r_fwd[1]), np.array(r_rev[0]), atol=0)


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

class TestCaching:
    def test_group_propagator_gcrf_cache_hit(self):
        from sgp4jax._propagation_mixed import _group_propagator_gcrf
        fn1 = _group_propagator_gcrf(0, 0)
        fn2 = _group_propagator_gcrf(0, 0)
        assert fn1 is fn2

    def test_group_propagator_gcrf_distinct(self):
        from sgp4jax._propagation_mixed import _group_propagator_gcrf
        fn_leo = _group_propagator_gcrf(0, 0)
        fn_gps = _group_propagator_gcrf(1, 0)
        fn_geo = _group_propagator_gcrf(1, 1)
        assert fn_leo is not fn_gps
        assert fn_gps is not fn_geo


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
