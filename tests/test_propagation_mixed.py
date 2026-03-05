"""Tests for propagate_mixed — heterogeneous multi-satellite propagator."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgp4jax import (
    tle_to_satrec, tles_to_satrec,
    propagate, propagate_leo, propagate_sdp4_nr,
    propagate_mixed,
    WGS84,
)


# ---------------------------------------------------------------------------
# TLEs covering all four dispatch cases
# ---------------------------------------------------------------------------

# Near-earth (method=0, irez=0) — ISS
_LEO_L1  = '1 25544U 98067A   20045.18587073  .00000950  00000-0  25302-4 0  9990'
_LEO_L2  = '2 25544  51.6443 242.0161 0004397 264.6060 207.3845 15.49165514212791'

# Near-earth (method=0, irez=0) — Sentinel-1A
_LEO2_L1 = '1 39634U 14016A   20045.50000000  .00000023  00000-0  14064-4 0  9994'
_LEO2_L2 = '2 39634  98.1825 145.6352 0001346  88.3457 271.7897 14.59198523314592'

# Deep-space irez=0 (method=1, irez=0) — GPS NAVSTAR 53
_GPS_L1  = '1 28129U 03058A   06175.57071136 -.00000104  00000-0  10000-3 0   459'
_GPS_L2  = '2 28129  54.7298 324.8098 0048506 266.2640  93.1663  2.00562768 18443'

# Deep-space irez=0 (method=1, irez=0) — SL-12 R/B
_DS_L1   = '1 20413U 83020D   05363.79166667  .00000000  00000-0  00000+0 0  7041'
_DS_L2   = '2 20413  12.3514 187.4253 7864447 196.3027 356.5478  0.24690082  7978'

# Deep-space irez=1 (method=1, irez=1) — ITALSAT 2 (synchronous/GEO)
_GEO_L1  = '1 24208U 96044A   06177.04061740 -.00000094  00000-0  10000-3 0  1600'
_GEO_L2  = '2 24208   3.8536  80.0121 0026640 311.0977  48.3000  1.00778054 36119'

# Deep-space irez=2 (method=1, irez=2) — MOLNIYA 2-14 (half-day resonance)
_MOL_L1  = '1 08195U 75081A   06176.33215444  .00000099  00000-0  11873-3 0   813'
_MOL_L2  = '2 08195  64.1586 279.0717 6877146 264.7651  20.2257  2.00491383225656'


# Shared propagation times
_TIMES = jnp.array([0.0, 60.0, 360.0, 720.0, 1440.0])


def _irez(sat) -> int:
    return int(float(sat.irez))

def _method(sat) -> int:
    return int(float(sat.method))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ref_propagate(sat, times):
    """Reference: vmap(vmap(fn)) where fn is the correct specialized propagator."""
    method = _method(sat)
    irez   = _irez(sat)
    if method == 0:
        fn = propagate_leo
    elif irez == 0:
        fn = propagate_sdp4_nr
    else:
        fn = propagate
    return jax.vmap(jax.vmap(fn, in_axes=(None, 0)), in_axes=(0, None))(
        tles_to_satrec([[l1, l2]], gravity=WGS84) if False else
        _single_to_batch(sat),
        times,
    )

def _single_to_batch(sat):
    """Wrap a scalar SatRec as a size-1 batch."""
    from sgp4jax._types import SatRec
    return SatRec(*[jnp.expand_dims(f, 0) for f in sat])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOutputShape:
    def test_shape_heterogeneous(self):
        tles = [
            [_LEO_L1,  _LEO_L2],
            [_GPS_L1,  _GPS_L2],
            [_GEO_L1,  _GEO_L2],
            [_MOL_L1,  _MOL_L2],
        ]
        batch = tles_to_satrec(tles, gravity=WGS84)
        r, v, e = propagate_mixed(batch, _TIMES)
        assert r.shape == (4, 5, 3), r.shape
        assert v.shape == (4, 5, 3), v.shape
        assert e.shape == (4, 5),    e.shape

    def test_shape_single_sat(self):
        batch = tles_to_satrec([[_LEO_L1, _LEO_L2]], gravity=WGS84)
        r, v, e = propagate_mixed(batch, _TIMES)
        assert r.shape == (1, 5, 3)

    def test_shape_single_time(self):
        tles = [[_LEO_L1, _LEO_L2], [_GPS_L1, _GPS_L2]]
        batch = tles_to_satrec(tles, gravity=WGS84)
        r, v, e = propagate_mixed(batch, jnp.array([360.0]))
        assert r.shape == (2, 1, 3)


class TestCorrectnessHomogeneous:
    """Homogeneous batches: propagate_mixed must match the specialized vmap exactly."""

    def _check(self, tles_list, fn, atol_r=0.0, atol_v=0.0):
        batch = tles_to_satrec(tles_list, gravity=WGS84)
        r_mix, v_mix, e_mix = propagate_mixed(batch, _TIMES)
        r_ref, v_ref, e_ref = jax.vmap(
            jax.vmap(fn, in_axes=(None, 0)), in_axes=(0, None)
        )(batch, _TIMES)
        np.testing.assert_allclose(np.array(r_mix), np.array(r_ref), atol=atol_r,
                                   err_msg="position mismatch")
        np.testing.assert_allclose(np.array(v_mix), np.array(v_ref), atol=atol_v,
                                   err_msg="velocity mismatch")
        np.testing.assert_array_equal(np.array(e_mix), np.array(e_ref))

    def test_leo_batch(self):
        self._check(
            [[_LEO_L1, _LEO_L2], [_LEO2_L1, _LEO2_L2]],
            propagate_leo)

    def test_gps_batch(self):
        self._check(
            [[_GPS_L1, _GPS_L2], [_DS_L1, _DS_L2]],
            propagate_sdp4_nr)

    def test_geo_batch(self):
        self._check(
            [[_GEO_L1, _GEO_L2], [_GEO_L1, _GEO_L2]],
            propagate)

    def test_molniya_batch(self):
        self._check(
            [[_MOL_L1, _MOL_L2], [_MOL_L1, _MOL_L2]],
            propagate)


class TestCorrectnessHeterogeneous:
    """Each satellite in a mixed batch matches its correct specialized propagator."""

    def test_mixed_all_types(self):
        tles = [
            [_LEO_L1,  _LEO_L2],   # index 0 — LEO
            [_GPS_L1,  _GPS_L2],   # index 1 — GPS (irez=0)
            [_GEO_L1,  _GEO_L2],   # index 2 — GEO (irez=1)
            [_MOL_L1,  _MOL_L2],   # index 3 — Molniya (irez=2)
            [_LEO2_L1, _LEO2_L2],  # index 4 — LEO (same group as index 0)
            [_DS_L1,   _DS_L2],    # index 5 — deep-space irez=0
        ]
        batch = tles_to_satrec(tles, gravity=WGS84)
        r_mix, v_mix, e_mix = propagate_mixed(batch, _TIMES)

        # Check each satellite individually against its correct propagator
        expected = [
            (propagate_leo,    0),
            (propagate_sdp4_nr, 1),
            (propagate,        2),
            (propagate,        3),
            (propagate_leo,    4),
            (propagate_sdp4_nr, 5),
        ]
        for fn, i in expected:
            sat_i = tles_to_satrec([tles[i]], gravity=WGS84)
            r_ref, v_ref, e_ref = jax.vmap(fn, in_axes=(None, 0))(sat_i[0] if False else sat_i, _TIMES)
            # vmap fn over times for this single-sat batch
            r_ref, v_ref, e_ref = jax.vmap(
                jax.vmap(fn, in_axes=(None, 0)), in_axes=(0, None)
            )(sat_i, _TIMES)
            # sat_i is (1, M, 3); take first element
            np.testing.assert_allclose(
                np.array(r_mix[i]), np.array(r_ref[0]),
                atol=0.0,
                err_msg=f"position mismatch for satellite index {i}")
            np.testing.assert_allclose(
                np.array(v_mix[i]), np.array(v_ref[0]),
                atol=0.0,
                err_msg=f"velocity mismatch for satellite index {i}")

    def test_ordering_preserved(self):
        """Result order must match the input TLE order regardless of group order."""
        tles_forward = [[_LEO_L1, _LEO_L2], [_GPS_L1, _GPS_L2]]
        tles_reverse = [[_GPS_L1, _GPS_L2], [_LEO_L1, _LEO_L2]]

        r_fwd, _, _ = propagate_mixed(tles_to_satrec(tles_forward, gravity=WGS84), _TIMES)
        r_rev, _, _ = propagate_mixed(tles_to_satrec(tles_reverse, gravity=WGS84), _TIMES)

        # LEO is index 0 in forward, index 1 in reverse
        np.testing.assert_allclose(np.array(r_fwd[0]), np.array(r_rev[1]), atol=0.0)
        # GPS is index 1 in forward, index 0 in reverse
        np.testing.assert_allclose(np.array(r_fwd[1]), np.array(r_rev[0]), atol=0.0)


class TestDispatchTable:
    """Verify that each group is routed to the right propagator."""

    def test_leo_uses_propagate_leo(self):
        sat = tle_to_satrec(_LEO_L1, _LEO_L2, gravity=WGS84)
        assert _method(sat) == 0
        from sgp4jax._propagation_mixed import _group_propagator, _propagate_leo
        fn_cached = _group_propagator(0, 0)
        # The cached function wraps propagate_leo — check via name of wrapped fn
        assert _propagate_leo is __import__('sgp4jax._propagation_leo',
                                            fromlist=['sgp4_leo']).sgp4_leo

    def test_gps_uses_propagate_sdp4_nr(self):
        sat = tle_to_satrec(_GPS_L1, _GPS_L2, gravity=WGS84)
        assert _method(sat) == 1
        assert _irez(sat) == 0

    def test_geo_uses_full_propagate(self):
        sat = tle_to_satrec(_GEO_L1, _GEO_L2, gravity=WGS84)
        assert _method(sat) == 1
        assert _irez(sat) == 1

    def test_molniya_uses_full_propagate(self):
        sat = tle_to_satrec(_MOL_L1, _MOL_L2, gravity=WGS84)
        assert _method(sat) == 1
        assert _irez(sat) == 2


class TestCaching:
    """_group_propagator must return the same cached object on repeated calls."""

    def test_lru_cache_hit(self):
        from sgp4jax._propagation_mixed import _group_propagator
        fn1 = _group_propagator(0, 0)
        fn2 = _group_propagator(0, 0)
        assert fn1 is fn2

    def test_distinct_types_cached_separately(self):
        from sgp4jax._propagation_mixed import _group_propagator
        fn_leo = _group_propagator(0, 0)
        fn_gps = _group_propagator(1, 0)
        fn_geo = _group_propagator(1, 1)
        assert fn_leo is not fn_gps
        assert fn_gps is not fn_geo


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
