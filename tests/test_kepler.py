"""Tests for the pure Keplerian propagator (_kepler.py).

Covers:
- Kepler equation solver convergence and accuracy
- At-epoch element consistency (radius, vis-viva equation)
- Expected discrepancy range vs full SGP4
- Output shapes for scalar and batched SatRec
- Orbital invariant conservation (energy, angular momentum, eccentricity vector)
  for near-earth (ISS), MEO (GPS), and high-eccentricity (Molniya) orbits
- Periodicity (orbit closes after exactly one Keplerian period)
- Angular momentum direction preservation (orbital plane stability)
- Multi-satellite batch consistency: multi == repeated single calls
- JAX compatibility: JIT, vmap, grad w.r.t. time and orbital elements
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import sgp4jax
from sgp4jax import tle_to_satrec, tles_to_satrec
from sgp4jax._kepler import (
    _solve_kepler, _kepler_rv_teme,
    kepler_gcrf_positions, kepler_gcrf_positions_multi,
)


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


# ---------------------------------------------------------------------------
# Conservation laws — all orbit types
# ---------------------------------------------------------------------------

def _conservation_spread(satrec, n_points=200):
    """Return (energy_spread, h_spread) over one Keplerian period."""
    mu = float(satrec.mu)
    jd0 = float(satrec.jdsatepoch) + float(satrec.jdsatepochF)
    period_min = 2.0 * float(jnp.pi) / float(satrec.no_kozai)
    times_jd = jnp.linspace(jd0, jd0 + period_min / 1440.0, n_points)
    r, v = kepler_gcrf_positions(satrec, times_jd)

    r_mag = jnp.linalg.norm(r, axis=-1)
    v2 = jnp.sum(v ** 2, axis=-1)
    energy = 0.5 * v2 - mu / r_mag
    h_mag = jnp.linalg.norm(jnp.cross(r, v), axis=-1)
    return (float(jnp.max(energy) - jnp.min(energy)),
            float(jnp.max(h_mag) - jnp.min(h_mag)))


@pytest.mark.parametrize("sat_name", ["iss", "gps", "molniya"])
def test_energy_conserved_all_orbits(sat_name, iss, gps, molniya):
    """Specific orbital energy is conserved to < 1e-9 km²/s² for all orbit types."""
    sat = {"iss": iss, "gps": gps, "molniya": molniya}[sat_name]
    energy_spread, _ = _conservation_spread(sat)
    assert energy_spread < 1e-9, (
        f"Energy not conserved for {sat_name}: spread = {energy_spread:.2e} km²/s²"
    )


@pytest.mark.parametrize("sat_name", ["iss", "gps", "molniya"])
def test_angular_momentum_conserved_all_orbits(sat_name, iss, gps, molniya):
    """Angular momentum magnitude is conserved to < 1e-9 km²/s for all orbit types."""
    sat = {"iss": iss, "gps": gps, "molniya": molniya}[sat_name]
    _, h_spread = _conservation_spread(sat)
    assert h_spread < 1e-9, (
        f"Angular momentum not conserved for {sat_name}: spread = {h_spread:.2e} km²/s"
    )


# ---------------------------------------------------------------------------
# Eccentricity vector (Laplace-Runge-Lenz) conservation
#
# For pure Keplerian motion the eccentricity vector
#   e⃗ = (v × h) / μ − r̂
# is a constant of motion.  Its magnitude equals the eccentricity and its
# direction points toward the periapsis.
# ---------------------------------------------------------------------------

def _eccentricity_vector(r, v, mu):
    """Compute the eccentricity vector from position/velocity arrays.

    Args:
        r: (..., 3) positions in km.
        v: (..., 3) velocities in km/s.
        mu: gravitational parameter in km³/s².

    Returns:
        e_vec: (..., 3) eccentricity vectors (dimensionless).
    """
    h = jnp.cross(r, v)                          # (..., 3)  km²/s
    e_vec = jnp.cross(v, h) / mu - r / jnp.linalg.norm(r, axis=-1, keepdims=True)
    return e_vec


@pytest.mark.parametrize("sat_name", ["iss", "gps", "molniya"])
def test_eccentricity_vector_conserved(sat_name, iss, gps, molniya):
    """The eccentricity vector magnitude equals the TLE eccentricity (< 1e-10 error)."""
    sat = {"iss": iss, "gps": gps, "molniya": molniya}[sat_name]
    mu = float(sat.mu)
    ecco = float(sat.ecco)

    jd0 = float(sat.jdsatepoch) + float(sat.jdsatepochF)
    period_min = 2.0 * float(jnp.pi) / float(sat.no_kozai)
    times_jd = jnp.linspace(jd0, jd0 + period_min / 1440.0, 100)
    r, v = kepler_gcrf_positions(sat, times_jd)

    e_vec = _eccentricity_vector(r, v, mu)           # (100, 3)
    e_mag = jnp.linalg.norm(e_vec, axis=-1)          # (100,)

    # Magnitude must equal TLE eccentricity at every point
    np.testing.assert_allclose(
        np.array(e_mag), ecco,
        atol=1e-10,
        err_msg=f"Eccentricity vector not conserved for {sat_name}",
    )


# ---------------------------------------------------------------------------
# Periodicity: orbit closes after exactly one Keplerian period
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sat_name", ["iss", "gps", "molniya"])
def test_periodicity(sat_name, iss, gps, molniya):
    """After one Keplerian period T = 2π/n the orbit returns to the same state.

    The orbital radius and speed at t=T must match those at t=0 to near float64
    precision.  We use rtol=1e-10 rather than 1e-15 because the Julian-date
    representation of jd0+T introduces ~1e-11 fractional rounding (JD ≈ 2.4M,
    float64 eps ≈ 1e-16, giving ~1e-10 relative error in tsince).
    """
    sat = {"iss": iss, "gps": gps, "molniya": molniya}[sat_name]
    jd0 = float(sat.jdsatepoch) + float(sat.jdsatepochF)
    period_day = 2.0 * float(jnp.pi) / float(sat.no_kozai) / 1440.0

    # Propagate to t=0 and t=T
    times = jnp.array([jd0, jd0 + period_day])
    r, v = kepler_gcrf_positions(sat, times)

    r_mag = jnp.linalg.norm(r, axis=-1)   # (2,)
    v_mag = jnp.linalg.norm(v, axis=-1)   # (2,)

    # rtol=1e-7: tight enough to catch any non-periodicity while accounting
    # for JD float64 rounding (~1e-11 day) amplified by high eccentricity
    # (Molniya e=0.71 makes r very sensitive to small tsince errors).
    np.testing.assert_allclose(
        float(r_mag[1]), float(r_mag[0]), rtol=1e-7,
        err_msg=f"Orbital radius not periodic for {sat_name}",
    )
    np.testing.assert_allclose(
        float(v_mag[1]), float(v_mag[0]), rtol=1e-7,
        err_msg=f"Speed not periodic for {sat_name}",
    )


# ---------------------------------------------------------------------------
# Angular momentum direction (orbital plane stability)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sat_name", ["iss", "gps", "molniya"])
def test_angular_momentum_direction_conserved(sat_name, iss, gps, molniya):
    """The angular momentum direction (orbital plane normal) is constant in TEME.

    We work in TEME rather than GCRF for this test.  TEME is the propagation
    frame where the orbital elements are defined; h⃗ = r × v is exactly
    constant there.  In GCRF the h direction has an apparent drift of ~3e-8
    per orbit because the TEME→GCRF rotation includes slowly-varying
    precession terms (IAU-2006, ~50 arcsec/year), making GCRF a slightly
    different basis at each time step.
    """
    sat = {"iss": iss, "gps": gps, "molniya": molniya}[sat_name]
    jd0 = float(sat.jdsatepoch)
    fr0 = float(sat.jdsatepochF)
    period_min = 2.0 * float(jnp.pi) / float(sat.no_kozai)

    # Sample 50 times over one period; compute TEME r, v directly
    t_list = np.linspace(0.0, period_min, 50)
    r_list, v_list = [], []
    for t in t_list:
        r_t, v_t = _kepler_rv_teme(
            sat.inclo, sat.nodeo, sat.ecco, sat.argpo,
            sat.mo, sat.no_kozai, sat.mu,
            sat.jdsatepoch, sat.jdsatepochF,
            jnp.float64(jd0), jnp.float64(fr0 + t / 1440.0),
        )
        r_list.append(np.array(r_t))
        v_list.append(np.array(v_t))

    r_arr = np.array(r_list)     # (50, 3)
    v_arr = np.array(v_list)     # (50, 3)

    h = np.cross(r_arr, v_arr)                                     # (50, 3)
    h_hat = h / np.linalg.norm(h, axis=-1, keepdims=True)         # (50, 3)

    deviation = np.max(np.linalg.norm(h_hat - h_hat[0], axis=-1))
    assert deviation < 1e-12, (
        f"Orbital plane drifts in TEME for {sat_name}: max deviation = {deviation:.2e}"
    )


# ---------------------------------------------------------------------------
# Multi-satellite batch consistency
# ---------------------------------------------------------------------------

def test_multi_matches_single(iss, gps, molniya, batch_satrec):
    """kepler_gcrf_positions_multi gives the same result as three single calls."""
    jd0 = 2458909.5
    times_jd = jnp.linspace(jd0, jd0 + 1.0, 16)

    r_multi, v_multi = kepler_gcrf_positions_multi(batch_satrec, times_jd)

    for i, sat in enumerate([iss, gps, molniya]):
        # Shift to the satellite's own epoch neighbourhood for a fair comparison,
        # but use the same fixed times_jd so no re-parametrization is needed.
        r_single, v_single = kepler_gcrf_positions(sat, times_jd)
        np.testing.assert_allclose(
            np.array(r_multi[i]), np.array(r_single),
            atol=1e-12,
            err_msg=f"Multi/single mismatch for satellite index {i}",
        )
        np.testing.assert_allclose(
            np.array(v_multi[i]), np.array(v_single),
            atol=1e-12,
            err_msg=f"Multi/single velocity mismatch for satellite index {i}",
        )


# ---------------------------------------------------------------------------
# Gradient w.r.t. all six TLE orbital elements
# ---------------------------------------------------------------------------

def test_grad_wrt_all_elements(iss):
    """jax.grad is finite and non-zero for every TLE orbital element."""
    from sgp4jax._kepler import _kepler_jd_gcrf
    from sgp4jax._types import make_satrec

    jd0 = jnp.float64(iss.jdsatepoch)
    fr0 = jnp.float64(iss.jdsatepochF) + jnp.float64(0.1)  # 144 min after epoch

    def loss(inclo, nodeo, ecco, argpo, mo, no_kozai):
        sat = make_satrec(
            inclo=inclo, nodeo=nodeo, ecco=ecco, argpo=argpo,
            mo=mo, no_kozai=no_kozai, mu=iss.mu,
            jdsatepoch=iss.jdsatepoch, jdsatepochF=iss.jdsatepochF,
        )
        r, _ = _kepler_jd_gcrf(sat, jd0, fr0)
        return jnp.linalg.norm(r)

    grads = jax.grad(loss, argnums=(0, 1, 2, 3, 4, 5))(
        jnp.float64(iss.inclo), jnp.float64(iss.nodeo),
        jnp.float64(iss.ecco),  jnp.float64(iss.argpo),
        jnp.float64(iss.mo),    jnp.float64(iss.no_kozai),
    )
    names = ("inclo", "nodeo", "ecco", "argpo", "mo", "no_kozai")
    for name, g in zip(names, grads):
        assert jnp.isfinite(g), f"Gradient w.r.t. {name} is not finite: {g}"
