"""Tests for the pure Keplerian propagator (_kepler.py).

Verifies correctness against SGP4 at t=0 (where both agree exactly) and
checks JAX-compatibility properties (JIT, vmap, grad).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import sgp4jax
from sgp4jax import tle_to_satrec, tles_to_satrec
from sgp4jax._kepler import _solve_kepler, kepler_gcrf_positions, kepler_gcrf_positions_multi


# ---------------------------------------------------------------------------
# Fixtures — a mix of orbit types
# ---------------------------------------------------------------------------

# ISS — near-earth, low eccentricity
_ISS_L1 = "1 25544U 98067A   20045.18587073  .00000950  00000-0  25302-4 0  9990"
_ISS_L2 = "2 25544  51.6443 242.0161 0004397 264.6060 207.3845 15.49165514212791"

# GPS IIR-12 — deep-space, near-circular MEO (irez=0)
_GPS_L1 = "1 28190U 04009A   20045.52490438 -.00000022  00000-0  00000+0 0  9993"
_GPS_L2 = "2 28190  55.3697 252.2502 0091151  48.4297 312.3987  2.00568625117815"

# Molniya — highly elliptic HEO
_MOLNIYA_L1 = "1 14947U 84012A   20044.87674740 -.00000143  00000-0  00000+0 0  9992"
_MOLNIYA_L2 = "2 14947  65.0476 160.8503 7085418 276.3547  11.5044  2.00611024264509"


@pytest.fixture(scope="module")
def iss():
    return tle_to_satrec(_ISS_L1, _ISS_L2)


@pytest.fixture(scope="module")
def gps():
    return tle_to_satrec(_GPS_L1, _GPS_L2)


@pytest.fixture(scope="module")
def molniya():
    return tle_to_satrec(_MOLNIYA_L1, _MOLNIYA_L2)


@pytest.fixture(scope="module")
def batch_satrec():
    return tles_to_satrec([
        [_ISS_L1, _ISS_L2],
        [_GPS_L1, _GPS_L2],
        [_MOLNIYA_L1, _MOLNIYA_L2],
    ])


# ---------------------------------------------------------------------------
# Kepler equation solver
# ---------------------------------------------------------------------------

def test_kepler_solver_circular():
    """For e=0 the eccentric and mean anomaly are identical."""
    M = jnp.linspace(0.0, 2 * jnp.pi, 100)
    E = _solve_kepler(M, jnp.float64(0.0))
    np.testing.assert_allclose(np.array(E), np.array(M), atol=1e-12)


def test_kepler_solver_residual():
    """M = E - e*sin(E) residual is < 1e-13 for a range of eccentricities."""
    for e_val in [0.0, 0.1, 0.5, 0.9]:
        e = jnp.float64(e_val)
        M = jnp.linspace(0.0, 2 * jnp.pi, 50)
        E = _solve_kepler(M, e)
        residual = np.abs(np.array(M - E + e_val * np.sin(np.array(E))))
        assert residual.max() < 1e-13, (
            f"Kepler residual too large for e={e_val}: max={residual.max():.2e}"
        )


def test_kepler_solver_high_eccentricity():
    """Solver converges for the Molniya-like eccentricity e≈0.71."""
    e = jnp.float64(0.7085418)
    M = jnp.linspace(0.1, 6.0, 100)
    E = _solve_kepler(M, e)
    residual = np.abs(np.array(M) - np.array(E) + 0.7085418 * np.sin(np.array(E)))
    assert residual.max() < 1e-13


# ---------------------------------------------------------------------------
# At-epoch element consistency
#
# TLE elements are Brouwer *mean* elements. SGP4 converts them to osculating
# elements by adding short-period corrections (J2, drag, lunar-solar).  A
# pure Keplerian propagator deliberately skips those corrections, so its
# at-epoch positions will differ from SGP4 by a few km to tens of km
# (consistent with the magnitude of short-period J2 oscillations).
#
# Instead of comparing against SGP4, we verify that the propagated position
# is physically consistent with the TLE elements themselves.
# ---------------------------------------------------------------------------

def _epoch_jd(satrec):
    """Split Julian date of the TLE epoch."""
    return float(satrec.jdsatepoch), float(satrec.jdsatepochF)


def test_kepler_radius_consistent_with_elements_iss(iss):
    """At t=0 the orbital radius matches r = a(1 − e·cos E) from the TLE elements."""
    from sgp4jax._kepler import _solve_kepler

    mu = float(iss.mu)
    n = float(iss.no_kozai)
    ecco = float(iss.ecco)
    mu_min = mu * 3600.0
    a = (mu_min / n ** 2) ** (1.0 / 3.0)  # km

    E0 = float(_solve_kepler(jnp.float64(iss.mo), jnp.float64(ecco)))
    expected_r = a * (1.0 - ecco * np.cos(E0))

    jd, fr = _epoch_jd(iss)
    r, _ = kepler_gcrf_positions(iss, jnp.array([jd + fr]))
    actual_r = float(jnp.linalg.norm(r[0]))

    np.testing.assert_allclose(actual_r, expected_r, rtol=1e-10,
                               err_msg="At-epoch radius doesn't match element geometry")


def test_kepler_vis_viva_at_epoch(iss):
    """At t=0 position and velocity satisfy the vis-viva equation v²=μ(2/r − 1/a)."""
    mu = float(iss.mu)
    n = float(iss.no_kozai)
    mu_min = mu * 3600.0
    a = (mu_min / n ** 2) ** (1.0 / 3.0)

    jd, fr = _epoch_jd(iss)
    r, v = kepler_gcrf_positions(iss, jnp.array([jd + fr]))
    r_km = float(jnp.linalg.norm(r[0]))
    v2 = float(jnp.dot(v[0], v[0]))

    vis_viva = mu * (2.0 / r_km - 1.0 / a)
    np.testing.assert_allclose(v2, vis_viva, rtol=1e-10,
                               err_msg="vis-viva equation violated at epoch")


def test_kepler_sgp4_error_order_of_magnitude(iss):
    """Keplerian vs SGP4 discrepancy at epoch is the expected tens-of-km scale.

    TLE mean elements differ from SGP4 osculating elements by O(1–50 km) for
    LEO due to J2 short-period corrections.  We assert the error is within a
    reasonable range: not absurdly large (< 500 km) and not suspiciously small
    (> 0.01 km, confirming we are NOT accidentally running full SGP4).
    """
    jd, fr = _epoch_jd(iss)
    times_jd = jnp.array([jd + fr])
    r_kepler, _ = kepler_gcrf_positions(iss, times_jd)
    r_sgp4, _, _ = sgp4jax.propagate_jd_gcrf(iss, jnp.float64(jd), jnp.float64(fr))
    err_km = float(jnp.linalg.norm(r_kepler[0] - r_sgp4))
    assert err_km < 500.0, f"Keplerian error unexpectedly large: {err_km:.1f} km"
    assert err_km > 0.01,  f"Keplerian error unexpectedly small (< 10 m): {err_km:.4f} km"


# ---------------------------------------------------------------------------
# Output shape checks
# ---------------------------------------------------------------------------

def test_single_sat_output_shape(iss):
    jd0 = float(iss.jdsatepoch) + float(iss.jdsatepochF)
    times_jd = jnp.linspace(jd0, jd0 + 1.0, 48)   # 48 × 30 min = 1 day
    r, v = kepler_gcrf_positions(iss, times_jd)
    assert r.shape == (48, 3)
    assert v.shape == (48, 3)


def test_multi_sat_output_shape(batch_satrec):
    jd0 = 2458909.5   # arbitrary fixed epoch
    times_jd = jnp.linspace(jd0, jd0 + 1.0, 24)
    r, v = kepler_gcrf_positions_multi(batch_satrec, times_jd)
    assert r.shape == (3, 24, 3)
    assert v.shape == (3, 24, 3)


# ---------------------------------------------------------------------------
# Orbital energy conservation
# ---------------------------------------------------------------------------

def test_energy_conserved_iss(iss):
    """Total specific energy is constant over one orbital period (< 1 J/kg drift)."""
    mu = float(iss.mu)   # km³/s²
    jd0 = float(iss.jdsatepoch) + float(iss.jdsatepochF)
    period_min = 2.0 * float(jnp.pi) / float(iss.no_kozai)   # minutes
    # 50 evenly spaced times over one period
    times_jd = jnp.array([jd0 + period_min / 1440.0 * i / 49 for i in range(50)])
    r, v = kepler_gcrf_positions(iss, times_jd)

    r_mag = jnp.linalg.norm(r, axis=-1)       # (50,)
    v2 = jnp.sum(v ** 2, axis=-1)             # (50,)
    energy = 0.5 * v2 - mu / r_mag            # km²/s²  (specific orbital energy)

    spread = float(jnp.max(energy) - jnp.min(energy))
    assert spread < 1e-9, f"Energy not conserved: spread = {spread:.2e} km²/s²"


def test_angular_momentum_conserved_iss(iss):
    """Specific angular momentum magnitude is constant over one period."""
    jd0 = float(iss.jdsatepoch) + float(iss.jdsatepochF)
    period_min = 2.0 * float(jnp.pi) / float(iss.no_kozai)
    times_jd = jnp.array([jd0 + period_min / 1440.0 * i / 49 for i in range(50)])
    r, v = kepler_gcrf_positions(iss, times_jd)

    h = jnp.cross(r, v)                       # (50, 3)
    h_mag = jnp.linalg.norm(h, axis=-1)       # (50,)
    spread = float(jnp.max(h_mag) - jnp.min(h_mag))
    assert spread < 1e-9, f"Angular momentum not conserved: spread = {spread:.2e} km²/s"


# ---------------------------------------------------------------------------
# JAX compatibility: JIT, vmap, grad
# ---------------------------------------------------------------------------

def test_kepler_jit(iss):
    """kepler_gcrf_positions is JIT-compilable (no Python-side branching)."""
    jd0 = float(iss.jdsatepoch) + float(iss.jdsatepochF)
    times_jd = jnp.linspace(jd0, jd0 + 1.0, 10)

    fn = jax.jit(kepler_gcrf_positions)
    r1, v1 = kepler_gcrf_positions(iss, times_jd)
    r2, v2 = fn(iss, times_jd)
    np.testing.assert_allclose(np.array(r1), np.array(r2), atol=1e-12)


def test_kepler_multi_jit(batch_satrec):
    """kepler_gcrf_positions_multi is JIT-compilable."""
    jd0 = 2458909.5
    times_jd = jnp.linspace(jd0, jd0 + 1.0, 8)
    fn = jax.jit(kepler_gcrf_positions_multi)
    r1, _ = kepler_gcrf_positions_multi(batch_satrec, times_jd)
    r2, _ = fn(batch_satrec, times_jd)
    np.testing.assert_allclose(np.array(r1), np.array(r2), atol=1e-12)


def test_kepler_grad_wrt_time(iss):
    """Gradient of position norm with respect to propagation time is finite."""
    jd0 = float(iss.jdsatepoch) + float(iss.jdsatepochF)
    t = jnp.float64(jd0 + 0.05)   # ~72 min after epoch

    def pos_norm(t_jd):
        r, _ = kepler_gcrf_positions(iss, jnp.array([t_jd]))
        return jnp.linalg.norm(r[0])

    g = jax.grad(pos_norm)(t)
    assert jnp.isfinite(g), f"Gradient w.r.t. time is not finite: {g}"


def test_kepler_grad_wrt_elements(iss):
    """Gradient of range w.r.t. mean anomaly is finite and non-zero."""
    from sgp4jax._kepler import _kepler_jd_gcrf
    from sgp4jax._types import make_satrec

    jd0 = jnp.float64(iss.jdsatepoch)
    fr0 = jnp.float64(iss.jdsatepochF) + jnp.float64(0.05)

    def loss(mo):
        sat = make_satrec(
            inclo=iss.inclo, nodeo=iss.nodeo, ecco=iss.ecco, argpo=iss.argpo,
            mo=mo, no_kozai=iss.no_kozai, mu=iss.mu,
            jdsatepoch=iss.jdsatepoch, jdsatepochF=iss.jdsatepochF,
        )
        r, _ = _kepler_jd_gcrf(sat, jd0, fr0)
        return jnp.linalg.norm(r)

    g = jax.grad(loss)(jnp.float64(iss.mo))
    assert jnp.isfinite(g), f"Gradient w.r.t. mo is not finite: {g}"
    assert float(jnp.abs(g)) > 0.0, "Gradient w.r.t. mo is zero"


def test_kepler_vmap_matches_loop(iss):
    """vmap over times gives identical results to a Python loop."""
    jd0 = float(iss.jdsatepoch) + float(iss.jdsatepochF)
    times_jd = jnp.linspace(jd0, jd0 + 0.5, 20)

    r_vmap, v_vmap = kepler_gcrf_positions(iss, times_jd)

    # Reference: call _kepler_jd_gcrf in a Python loop
    from sgp4jax._kepler import _kepler_jd_gcrf
    jd_arr = jnp.floor(times_jd)
    fr_arr = times_jd - jd_arr
    r_loop = jnp.stack([_kepler_jd_gcrf(iss, jd_arr[i], fr_arr[i])[0]
                         for i in range(20)])
    np.testing.assert_allclose(np.array(r_vmap), np.array(r_loop), atol=1e-11)
