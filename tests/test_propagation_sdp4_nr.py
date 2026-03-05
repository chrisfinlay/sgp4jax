"""Tests for the deep-space no-resonance SGP4 propagator (sgp4_sdp4_nr)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgp4jax import tle_to_satrec, propagate, propagate_sdp4_nr, propagate_jd_sdp4_nr, WGS84


# ---------------------------------------------------------------------------
# Test TLEs — irez=0 deep-space (no resonance)
# ---------------------------------------------------------------------------

# NAVSTAR 53 / USA-175 — GPS Block IIA, labeled "12h non-resonant GPS (ecc < 0.5)"
# in the SGP4 verification dataset. n≈2.006 rev/day, e=0.0048, irez=0.
_GPS_L1 = '1 28129U 03058A   06175.57071136 -.00000104  00000-0  10000-3 0   459'
_GPS_L2 = '2 28129  54.7298 324.8098 0048506 266.2640  93.1663  2.00562768 18443'

# STR#3 SDP4 — original deep-space test case, n≈2.285 rev/day, e=0.73.
# Outside both resonance bands → irez=0.
_STR3_L1 = '1 11801U          80230.29629788  .01431103  00000-0  14311-1      13'
_STR3_L2 = '2 11801  46.7916 230.4354 7318036  47.4722  10.4117  2.28537848    13'

# SL-6 R/B(2) — deep-space, n≈4.885 rev/day (period≈295 min).
# nm > 9.24e-3 → outside irez=2 band → irez=0.
_SL6_L1 = '1 16925U 86065D   06151.67415771  .02550794 -30915-6  18784-3 0  4486'
_SL6_L2 = '2 16925  62.0906 295.0239 5596327 245.1593  47.9690  4.88511875148616'

# SL-12 R/B — very low mean motion, n≈0.247 rev/day (period≈5827 min).
# nm < 3.49e-3 → below irez=1 band → irez=0.
_SL12_L1 = '1 20413U 83020D   05363.79166667  .00000000  00000-0  00000+0 0  7041'
_SL12_L2 = '2 20413  12.3514 187.4253 7864447 196.3027 356.5478  0.24690082  7978'

# ARIANE 44L R/B — n≈2.259 rev/day, e=0.726. nm≈9.85e-3 > 9.24e-3.
# Outside irez=2 band → irez=0.
_ARIANE_L1 = '1 23177U 94040C   06175.45752052  .00000386  00000-0  76590-3 0    95'
_ARIANE_L2 = '2 23177   7.0496 179.8238 7258491 296.0482   8.3061  2.25906668 97438'

# H-2 R/B — deep-space, n≈3.795 rev/day (period≈380 min).
# nm≈0.01654 > 9.24e-3 → outside irez=2 band → irez=0.
_H2_L1 = '1 28623U 05006B   06177.81079184  .00637644  69054-6  96390-3 0  6000'
_H2_L2 = '2 28623  28.5200 114.9834 6249053 170.2550 212.8965  3.79477162 12753'

# irez=0 deep-space TLE collection
NR_TLES = [
    ("GPS_NAVSTAR53", _GPS_L1,    _GPS_L2),
    ("STR3_SDP4",     _STR3_L1,   _STR3_L2),
    ("SL6_RB",        _SL6_L1,    _SL6_L2),
    ("SL12_RB",       _SL12_L1,   _SL12_L2),
    ("ARIANE44L",     _ARIANE_L1, _ARIANE_L2),
    ("H2_RB",         _H2_L1,     _H2_L2),
]

# ---------------------------------------------------------------------------
# Resonant TLEs — should diverge from sgp4_sdp4_nr
# ---------------------------------------------------------------------------

# ITALSAT 2 — GEO, 24-hour synchronous resonance (irez=1).
# n≈1.008 rev/day, period≈1428 min, nm≈4.40e-3 ∈ (3.49e-3, 5.24e-3).
_GEO_L1 = '1 24208U 96044A   06177.04061740 -.00000094  00000-0  10000-3 0  1600'
_GEO_L2 = '2 24208   3.8536  80.0121 0026640 311.0977  48.3000  1.00778054 36119'

# MOLNIYA 2-14 — 12-hour resonance (irez=2).
# n≈2.005 rev/day, e=0.688 ≥ 0.5, nm≈8.75e-3 ∈ [8.26e-3, 9.24e-3].
_MOLNIYA_L1 = '1 08195U 75081A   06176.33215444  .00000099  00000-0  11873-3 0   813'
_MOLNIYA_L2 = '2 08195  64.1586 279.0717 6877146 264.7651  20.2257  2.00491383225656'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _irez(satrec) -> int:
    return int(float(satrec.irez))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSDP4NRCorrectness:
    """sgp4_sdp4_nr must match sgp4 exactly for irez=0 deep-space satellites."""

    @pytest.mark.parametrize("name,l1,l2", NR_TLES)
    def test_matches_full_sgp4(self, name, l1, l2):
        sat = tle_to_satrec(l1, l2, gravity=WGS84)
        assert _irez(sat) == 0, (
            f"{name}: expected irez=0 but got irez={_irez(sat)}. "
            "TLE may not be a no-resonance deep-space object.")

        for tsince in [0.0, 10.0, 60.0, 360.0, 720.0, 1440.0]:
            t = jnp.array(tsince)
            r_full, v_full, err_full = propagate(sat, t)
            r_nr,   v_nr,   err_nr   = propagate_sdp4_nr(sat, t)

            if int(err_full) != 0:
                continue  # skip error cases

            assert int(err_nr) == 0, f"{name} t={tsince}: sdp4_nr error={int(err_nr)}"
            np.testing.assert_allclose(
                np.array(r_nr), np.array(r_full),
                atol=1e-10,
                err_msg=f"{name} position mismatch at t={tsince}")
            np.testing.assert_allclose(
                np.array(v_nr), np.array(v_full),
                atol=1e-13,
                err_msg=f"{name} velocity mismatch at t={tsince}")


class TestResonanceRejection:
    """sgp4_sdp4_nr must give wrong (diverging) results for resonant satellites."""

    def test_geo_diverges(self):
        """GEO (irez=1): resonance terms skipped → results differ from full sgp4."""
        sat = tle_to_satrec(_GEO_L1, _GEO_L2, gravity=WGS84)
        assert _irez(sat) == 1, f"Expected irez=1 for GEO, got {_irez(sat)}"

        # Resonance builds slowly — use 2 days to ensure > 1 km divergence
        t = jnp.array(2880.0)
        r_full, v_full, err_full = propagate(sat, t)
        r_nr,   v_nr,   err_nr   = propagate_sdp4_nr(sat, t)

        if int(err_full) != 0 or int(err_nr) != 0:
            pytest.skip("Satellite produced error — cannot compare")

        pos_diff = float(jnp.linalg.norm(r_nr - r_full))
        assert pos_diff > 1.0, (
            f"Expected sgp4_sdp4_nr to diverge from sgp4 for GEO (irez=1), "
            f"but position difference is only {pos_diff:.6f} km")

    def test_molniya_diverges(self):
        """Molniya (irez=2): resonance terms skipped → results differ from full sgp4."""
        sat = tle_to_satrec(_MOLNIYA_L1, _MOLNIYA_L2, gravity=WGS84)
        assert _irez(sat) == 2, f"Expected irez=2 for Molniya, got {_irez(sat)}"

        # Resonance builds slowly — use 5 days to ensure > 1 km divergence
        t = jnp.array(7200.0)
        r_full, v_full, err_full = propagate(sat, t)
        r_nr,   v_nr,   err_nr   = propagate_sdp4_nr(sat, t)

        if int(err_full) != 0 or int(err_nr) != 0:
            pytest.skip("Satellite produced error — cannot compare")

        pos_diff = float(jnp.linalg.norm(r_nr - r_full))
        assert pos_diff > 1.0, (
            f"Expected sgp4_sdp4_nr to diverge from sgp4 for Molniya (irez=2), "
            f"but position difference is only {pos_diff:.6f} km")


class TestJITCompatibility:
    """sgp4_sdp4_nr must work correctly under jax.jit."""

    def test_jit_matches_eager(self):
        sat = tle_to_satrec(_GPS_L1, _GPS_L2, gravity=WGS84)
        t = jnp.array(120.0)
        r_eager, v_eager, err_eager = propagate_sdp4_nr(sat, t)
        r_jit,   v_jit,   err_jit   = jax.jit(propagate_sdp4_nr)(sat, t)

        np.testing.assert_array_equal(np.array(r_eager), np.array(r_jit))
        np.testing.assert_array_equal(np.array(v_eager), np.array(v_jit))
        assert int(err_eager) == int(err_jit)


class TestVmapCompatibility:
    """sgp4_sdp4_nr must work correctly under jax.vmap."""

    def test_vmap_over_tsince(self):
        sat = tle_to_satrec(_GPS_L1, _GPS_L2, gravity=WGS84)
        tsinces = jnp.array([0.0, 30.0, 60.0, 120.0, 360.0])

        r_batch, v_batch, err_batch = jax.vmap(
            propagate_sdp4_nr, in_axes=(None, 0)
        )(sat, tsinces)

        for i, tsince in enumerate(tsinces):
            r_i, v_i, err_i = propagate_sdp4_nr(sat, tsince)
            np.testing.assert_allclose(
                np.array(r_batch[i]), np.array(r_i), atol=0.0,
                err_msg=f"vmap position mismatch at index {i}")
            np.testing.assert_allclose(
                np.array(v_batch[i]), np.array(v_i), atol=0.0,
                err_msg=f"vmap velocity mismatch at index {i}")


class TestADCompatibility:
    """sgp4_sdp4_nr must be differentiable."""

    def test_jacfwd_wrt_time(self):
        """Forward-mode Jacobian w.r.t. tsince runs without error."""
        sat = tle_to_satrec(_GPS_L1, _GPS_L2, gravity=WGS84)
        t = jnp.array(120.0)

        def propagate_rv(tsince):
            r, v, _ = propagate_sdp4_nr(sat, tsince)
            return r, v

        jac_r, jac_v = jax.jacfwd(propagate_rv)(t)
        assert jac_r.shape == (3,), f"Unexpected Jacobian shape: {jac_r.shape}"
        assert jac_v.shape == (3,), f"Unexpected Jacobian shape: {jac_v.shape}"
        assert not jnp.any(jnp.isnan(jac_r)), "NaN in position Jacobian"
        assert not jnp.any(jnp.isnan(jac_v)), "NaN in velocity Jacobian"

    def test_grad_wrt_time(self):
        """Reverse-mode gradient of scalar loss w.r.t. tsince."""
        sat = tle_to_satrec(_GPS_L1, _GPS_L2, gravity=WGS84)
        t = jnp.array(120.0)

        def loss(tsince):
            r, v, _ = propagate_sdp4_nr(sat, tsince)
            return jnp.sum(r ** 2)

        g = jax.grad(loss)(t)
        assert not jnp.isnan(g), "NaN gradient"


class TestPropagateJdSdp4Nr:
    """propagate_jd_sdp4_nr wrapper should agree with propagate_sdp4_nr."""

    def test_jd_wrapper(self):
        from sgp4.api import Satrec as RefSatrec, WGS84 as REF_WGS84
        sat = tle_to_satrec(_GPS_L1, _GPS_L2, gravity=WGS84)
        ref = RefSatrec.twoline2rv(_GPS_L1, _GPS_L2, REF_WGS84)

        jd = jnp.array(ref.jdsatepoch)
        fr = jnp.array(ref.jdsatepochF + 0.5)  # 12 hours after epoch

        r_jd, v_jd, err_jd = propagate_jd_sdp4_nr(sat, jd, fr)
        tsince = (jd - sat.jdsatepoch) * 1440.0 + (fr - sat.jdsatepochF) * 1440.0
        r_ts, v_ts, err_ts = propagate_sdp4_nr(sat, tsince)

        np.testing.assert_array_equal(np.array(r_jd), np.array(r_ts))
        np.testing.assert_array_equal(np.array(v_jd), np.array(v_ts))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
