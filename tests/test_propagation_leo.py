"""Tests for the LEO-dedicated SGP4 propagator (sgp4_leo / propagate_leo)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgp4jax import tle_to_satrec, propagate, propagate_leo, propagate_jd_leo, WGS84


# ---------------------------------------------------------------------------
# Test TLEs
# ---------------------------------------------------------------------------

# ISS - canonical LEO
_ISS_L1 = '1 25544U 98067A   20045.18587073  .00000950  00000-0  25302-4 0  9990'
_ISS_L2 = '2 25544  51.6443 242.0161 0004397 264.6060 207.3845 15.49165514212791'

# Sentinel-1A - sun-synchronous LEO
_SEN_L1 = '1 39634U 14016A   20045.50000000  .00000023  00000-0  14064-4 0  9994'
_SEN_L2 = '2 39634  98.1825 145.6352 0001346  88.3457 271.7897 14.59198523314592'

# Iridium - LEO constellation
_IRID_L1 = '1 24792U 97020D   20045.50000000  .00000095  00000-0  27765-4 0  9991'
_IRID_L2 = '2 24792  86.3948  50.0427 0002078 272.3438  87.7373 14.34216813208614'

# Vanguard 1 - low-drag near-earth
_VAN_L1 = '1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753'
_VAN_L2 = '2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667'

# NOAA 15 - polar LEO
_NOAA_L1 = '1 25338U 98030A   20045.50000000  .00000038  00000-0  30124-4 0  9993'
_NOAA_L2 = '2 25338  98.5524 128.2183 0010978 333.4490  26.6088 14.25852820113983'

# Aqua - sun-synchronous LEO
_AQUA_L1 = '1 27424U 02022A   20045.50000000  .00000088  00000-0  38440-4 0  9993'
_AQUA_L2 = '2 27424  98.2056  99.3499 0002163  96.1019 264.0390 14.57110877963506'

# Terra - sun-synchronous LEO
_TERRA_L1 = '1 25994U 99068A   20045.50000000  .00000057  00000-0  28869-4 0  9994'
_TERRA_L2 = '2 25994  98.2068  65.9660 0001315  91.7532 268.3835 14.57132869068082'

# SPOT 5 - LEO
_SPOT_L1 = '1 27421U 02021A   20045.50000000  .00000079  00000-0  45100-4 0  9996'
_SPOT_L2 = '2 27421  98.7274 145.4893 0001248  93.3879 266.7444 14.29821940924786'

# Envisat - LEO
_ENVISAT_L1 = '1 27386U 02009A   20045.50000000  .00000030  00000-0  11524-4 0  9992'
_ENVISAT_L2 = '2 27386  98.5490 128.4490 0001243 101.6888 258.4446 14.37965756943714'

# Jason-2 - LEO
_JASON_L1 = '1 33105U 08032A   20045.50000000  .00000052  00000-0  27427-4 0  9993'
_JASON_L2 = '2 33105  66.0408 312.7891 0008498 302.2368  57.7887 12.87642536328614'

LEO_TLES = [
    ("ISS",      _ISS_L1,     _ISS_L2),
    ("Sentinel", _SEN_L1,     _SEN_L2),
    ("Iridium",  _IRID_L1,    _IRID_L2),
    ("Vanguard", _VAN_L1,     _VAN_L2),
    ("NOAA15",   _NOAA_L1,    _NOAA_L2),
    ("Aqua",     _AQUA_L1,    _AQUA_L2),
    ("Terra",    _TERRA_L1,   _TERRA_L2),
    ("SPOT5",    _SPOT_L1,    _SPOT_L2),
    ("Envisat",  _ENVISAT_L1, _ENVISAT_L2),
    ("Jason2",   _JASON_L1,   _JASON_L2),
]

# GEO satellite — deep-space, sgp4_leo results will diverge
_GEO_L1 = '1 00045U 60007A   00182.79010492 -.00000016  00000-0  10000-3 0  9998'
_GEO_L2 = '2 00045   8.4272  13.6326 7142230 209.5880  16.2441  2.24890932 27606'


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLEOCorrectness:
    """sgp4_leo must match sgp4 exactly for near-earth satellites."""

    @pytest.mark.parametrize("name,l1,l2", LEO_TLES)
    def test_matches_full_sgp4(self, name, l1, l2):
        sat = tle_to_satrec(l1, l2, gravity=WGS84)
        for tsince in [0.0, 10.0, 60.0, 360.0, 720.0, 1440.0]:
            t = jnp.array(tsince)
            r_full, v_full, err_full = propagate(sat, t)
            r_leo,  v_leo,  err_leo  = propagate_leo(sat, t)

            if int(err_full) != 0:
                continue  # skip error cases

            assert int(err_leo) == 0, f"{name} t={tsince}: leo error={int(err_leo)}"
            np.testing.assert_allclose(
                np.array(r_leo), np.array(r_full),
                atol=1e-10,
                err_msg=f"{name} position mismatch at t={tsince}")
            np.testing.assert_allclose(
                np.array(v_leo), np.array(v_full),
                atol=1e-13,
                err_msg=f"{name} velocity mismatch at t={tsince}")


class TestDeepSpaceRejection:
    """sgp4_leo must give different results for a deep-space satellite."""

    def test_geo_diverges(self):
        """GEO satellite: sgp4_leo and sgp4 results should differ (deep-space corrections skipped)."""
        sat = tle_to_satrec(_GEO_L1, _GEO_L2, gravity=WGS84)
        t = jnp.array(360.0)
        r_full, v_full, err_full = propagate(sat, t)
        r_leo,  v_leo,  err_leo  = propagate_leo(sat, t)

        if int(err_full) != 0 or int(err_leo) != 0:
            pytest.skip("Satellite produced error — cannot compare")

        pos_diff = float(jnp.linalg.norm(r_leo - r_full))
        assert pos_diff > 1.0, (
            f"Expected sgp4_leo to diverge from sgp4 for GEO sat, "
            f"but position difference is only {pos_diff:.6f} km")


class TestJITCompatibility:
    """sgp4_leo must work correctly under jax.jit."""

    def test_jit_matches_eager(self):
        sat = tle_to_satrec(_ISS_L1, _ISS_L2, gravity=WGS84)
        t = jnp.array(120.0)
        r_eager, v_eager, err_eager = propagate_leo(sat, t)
        r_jit,   v_jit,   err_jit   = jax.jit(propagate_leo)(sat, t)

        np.testing.assert_array_equal(np.array(r_eager), np.array(r_jit))
        np.testing.assert_array_equal(np.array(v_eager), np.array(v_jit))
        assert int(err_eager) == int(err_jit)


class TestVmapCompatibility:
    """sgp4_leo must work correctly under jax.vmap."""

    def test_vmap_over_tsince(self):
        sat = tle_to_satrec(_ISS_L1, _ISS_L2, gravity=WGS84)
        tsinces = jnp.array([0.0, 30.0, 60.0, 120.0, 360.0])

        # Batched call
        r_batch, v_batch, err_batch = jax.vmap(
            propagate_leo, in_axes=(None, 0)
        )(sat, tsinces)

        # Individual calls
        for i, tsince in enumerate(tsinces):
            r_i, v_i, err_i = propagate_leo(sat, tsince)
            np.testing.assert_allclose(
                np.array(r_batch[i]), np.array(r_i), atol=0.0,
                err_msg=f"vmap position mismatch at index {i}")
            np.testing.assert_allclose(
                np.array(v_batch[i]), np.array(v_i), atol=0.0,
                err_msg=f"vmap velocity mismatch at index {i}")


class TestADCompatibility:
    """sgp4_leo must be differentiable."""

    def test_jacfwd_wrt_time(self):
        """Forward-mode Jacobian w.r.t. tsince runs without error."""
        sat = tle_to_satrec(_ISS_L1, _ISS_L2, gravity=WGS84)
        t = jnp.array(120.0)

        def propagate_rv(tsince):
            r, v, _ = propagate_leo(sat, tsince)
            return r, v

        jac_r, jac_v = jax.jacfwd(propagate_rv)(t)
        assert jac_r.shape == (3,), f"Unexpected Jacobian shape: {jac_r.shape}"
        assert jac_v.shape == (3,), f"Unexpected Jacobian shape: {jac_v.shape}"
        assert not jnp.any(jnp.isnan(jac_r)), "NaN in position Jacobian"
        assert not jnp.any(jnp.isnan(jac_v)), "NaN in velocity Jacobian"

    def test_grad_wrt_time(self):
        """Reverse-mode gradient of scalar loss w.r.t. tsince."""
        sat = tle_to_satrec(_ISS_L1, _ISS_L2, gravity=WGS84)
        t = jnp.array(120.0)

        def loss(tsince):
            r, v, _ = propagate_leo(sat, tsince)
            return jnp.sum(r ** 2)

        g = jax.grad(loss)(t)
        assert not jnp.isnan(g), "NaN gradient"


class TestPropagateJdLeo:
    """propagate_jd_leo wrapper should agree with propagate_leo."""

    def test_jd_wrapper(self):
        from sgp4.api import Satrec as RefSatrec, WGS84 as REF_WGS84
        sat = tle_to_satrec(_ISS_L1, _ISS_L2, gravity=WGS84)
        ref = RefSatrec.twoline2rv(_ISS_L1, _ISS_L2, REF_WGS84)

        jd = jnp.array(ref.jdsatepoch)
        fr = jnp.array(ref.jdsatepochF + 0.5)  # 12 hours after epoch

        r_jd, v_jd, err_jd = propagate_jd_leo(sat, jd, fr)
        tsince = (jd - sat.jdsatepoch) * 1440.0 + (fr - sat.jdsatepochF) * 1440.0
        r_ts, v_ts, err_ts = propagate_leo(sat, tsince)

        np.testing.assert_array_equal(np.array(r_jd), np.array(r_ts))
        np.testing.assert_array_equal(np.array(v_jd), np.array(v_ts))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
