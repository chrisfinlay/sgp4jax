"""Pure two-body Keplerian propagator — no perturbations.

Uses only the six classical orbital elements at the TLE epoch:

* ``inclo``    — inclination (rad)
* ``nodeo``    — right ascension of ascending node (rad)
* ``ecco``     — eccentricity
* ``argpo``    — argument of perigee (rad)
* ``mo``       — mean anomaly at epoch (rad)
* ``no_kozai`` — mean motion at epoch (rad/min)

All perturbations are ignored (atmospheric drag, J2, lunar/solar forces,
resonances).  The orbit is a pure Keplerian ellipse.  This is useful as a
fast, differentiable baseline and for gradient-based orbit determination
where perturbation accuracy is not required.

Both functions are fully JIT-compilable and differentiable with respect to
all orbital elements, making them suitable for ``jax.grad``, ``jax.jacobian``,
and ``jax.vmap``.
"""

from __future__ import annotations

import jax
from jax import vmap
import jax.numpy as jnp
import jax.typing

from sgp4jax._types import SatRec
from sgp4jax._frames import teme_to_gcrf


# ---------------------------------------------------------------------------
# Kepler equation solver
# ---------------------------------------------------------------------------

def _solve_kepler(
    M: jax.typing.ArrayLike,
    ecco: jax.typing.ArrayLike,
    n_iter: int = 20,
) -> jax.Array:
    """Solve Kepler's equation  M = E − e·sin(E)  for eccentric anomaly E.

    Uses Newton-Raphson iteration via :func:`jax.lax.scan`.  The scan
    produces a compact, fixed-size computation graph regardless of
    *n_iter*, giving faster JIT compilation than a Python ``for`` loop.
    Reverse-mode AD (``jax.grad``, ``jax.jacobian``) is fully supported.

    Convergence is quadratic: float64 machine precision is reached in
    fewer than 10 iterations for *e* < 0.99 and fewer than 20 for
    *e* < 0.9999, covering all TLE eccentricities.

    Args:
        M: Mean anomaly in radians.  May be outside [0, 2π].
        ecco: Eccentricity  0 ≤ e < 1.
        n_iter: Number of Newton-Raphson iterations (default 20).

    Returns:
        Eccentric anomaly E in radians.
    """
    def step(E: jax.Array, _: None) -> tuple[jax.Array, None]:
        return E + (M - E + ecco * jnp.sin(E)) / (1.0 - ecco * jnp.cos(E)), None

    E, _ = jax.lax.scan(step, jnp.asarray(M, dtype=jnp.float64), None, length=n_iter)
    return E


# ---------------------------------------------------------------------------
# Core propagation (single satellite, single time)
# ---------------------------------------------------------------------------

def _kepler_rv_teme(
    inclo: jax.typing.ArrayLike,
    nodeo: jax.typing.ArrayLike,
    ecco: jax.typing.ArrayLike,
    argpo: jax.typing.ArrayLike,
    mo: jax.typing.ArrayLike,
    no_kozai: jax.typing.ArrayLike,
    mu: jax.typing.ArrayLike,
    jdsatepoch: jax.typing.ArrayLike,
    jdsatepochF: jax.typing.ArrayLike,
    jd: jax.typing.ArrayLike,
    fr: jax.typing.ArrayLike,
) -> tuple[jax.Array, jax.Array]:
    """Propagate one satellite to one time with pure two-body Keplerian motion.

    Solves Kepler's equation for the eccentric anomaly, constructs position
    and velocity in perifocal (PQW) coordinates, then rotates to the TEME
    frame using the Euler sequence R_z(Ω) R_x(i) R_z(ω).

    Args:
        inclo: Inclination (rad).
        nodeo: Right ascension of ascending node (rad).
        ecco: Eccentricity.
        argpo: Argument of perigee (rad).
        mo: Mean anomaly at epoch (rad).
        no_kozai: Mean motion at epoch (rad/min).
        mu: Gravitational parameter (km³/s²).
        jdsatepoch: TLE epoch Julian date, whole part.
        jdsatepochF: TLE epoch Julian date, fractional part.
        jd: Target Julian date, whole part.
        fr: Target Julian date, fractional part.

    Returns:
        r_teme: Position in TEME frame, shape ``(3,)``, in km.
        v_teme: Velocity in TEME frame, shape ``(3,)``, in km/s.
    """
    # --- Time since TLE epoch (minutes) ---
    tsince = (jd - jdsatepoch + fr - jdsatepochF) * 1440.0  # type: ignore[operator]

    # --- Semi-major axis via Kepler's third law  n² a³ = μ ---
    # no_kozai in rad/min → convert μ from km³/s² to km³/min² (*3600).
    mu_min = mu * 3600.0                           # km³/min²
    a = (mu_min / no_kozai ** 2) ** (1.0 / 3.0)   # km

    # --- Propagate mean anomaly, solve for eccentric anomaly ---
    M = mo + no_kozai * tsince                     # rad
    E = _solve_kepler(M, ecco)

    # --- Eccentric → true anomaly ---
    cosE = jnp.cos(E)
    sinE = jnp.sin(E)
    nu = jnp.arctan2(jnp.sqrt(1.0 - ecco ** 2) * sinE, cosE - ecco)

    # --- Orbital distance and semi-latus rectum ---
    r_mag = a * (1.0 - ecco * cosE)    # km
    p = a * (1.0 - ecco ** 2)          # km  (semi-latus rectum)

    # --- Perifocal position (km) and velocity (km/s) ---
    cos_nu = jnp.cos(nu)
    sin_nu = jnp.sin(nu)
    sqrt_mu_p = jnp.sqrt(mu / p)       # km/s  (μ in km³/s²)

    # --- Rotation: perifocal → TEME via P̂ and Q̂ unit vectors ---
    # Standard result of  R_z(Ω) R_x(i) R_z(ω):
    ci = jnp.cos(inclo);  si = jnp.sin(inclo)
    cO = jnp.cos(nodeo);  sO = jnp.sin(nodeo)
    cw = jnp.cos(argpo);  sw = jnp.sin(argpo)

    P = jnp.array([cO * cw - sO * sw * ci,
                   sO * cw + cO * sw * ci,
                   sw * si])

    Q = jnp.array([-(cO * sw + sO * cw * ci),
                   -(sO * sw - cO * cw * ci),
                   cw * si])

    r_teme = (r_mag * cos_nu) * P + (r_mag * sin_nu) * Q
    v_teme = (sqrt_mu_p * (-sin_nu)) * P + (sqrt_mu_p * (ecco + cos_nu)) * Q

    return r_teme, v_teme  # type: ignore[return-value]


def _kepler_jd_gcrf(
    satrec: SatRec,
    jd: jax.typing.ArrayLike,
    fr: jax.typing.ArrayLike,
) -> tuple[jax.Array, jax.Array]:
    """Keplerian propagation for one satellite and one Julian date → GCRF.

    Thin wrapper around :func:`_kepler_rv_teme` and :func:`teme_to_gcrf`
    that reads the required fields from *satrec*.  Designed for use inside
    ``jax.vmap``.

    Args:
        satrec: Scalar SatRec from :func:`tle_to_satrec`.
        jd: Julian date, whole part (scalar).
        fr: Julian date, fractional part (scalar).

    Returns:
        r_gcrf: GCRF position ``(3,)`` in km.
        v_gcrf: GCRF velocity ``(3,)`` in km/s.
    """
    r_teme, v_teme = _kepler_rv_teme(
        satrec.inclo, satrec.nodeo, satrec.ecco, satrec.argpo,
        satrec.mo, satrec.no_kozai, satrec.mu,
        satrec.jdsatepoch, satrec.jdsatepochF, jd, fr,
    )
    r_gcrf, v_gcrf = teme_to_gcrf(r_teme, v_teme, jd, fr)
    return r_gcrf, v_gcrf


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def kepler_gcrf_positions(
    satrec: SatRec,
    times_jd: jax.typing.ArrayLike,
) -> tuple[jax.Array, jax.Array]:
    """Propagate one satellite to multiple times using pure Keplerian motion.

    No perturbations are applied.  Only the six TLE orbital elements
    (``inclo``, ``nodeo``, ``ecco``, ``argpo``, ``mo``, ``no_kozai``) and
    the epoch (``jdsatepoch``, ``jdsatepochF``) are used.

    The function is JIT-compiled and differentiable with respect to all
    fields of *satrec* and with respect to *times_jd*.

    Args:
        satrec: Scalar SatRec from :func:`tle_to_satrec`.
        times_jd: 1-D array of UTC Julian dates, shape ``(n_times,)``.

    Returns:
        r_gcrf: GCRF positions, shape ``(n_times, 3)``, in km.
        v_gcrf: GCRF velocities, shape ``(n_times, 3)``, in km/s.

    Note:
        The TEME → GCRF rotation uses the input Julian dates as UT1
        (UTC is an approximation accurate to < 1 s).  For sub-arcsecond
        frame accuracy call :func:`utc_to_ut1` first and pass the UT1 dates.
    """
    times_jd = jnp.asarray(times_jd)
    jd_arr = jnp.floor(times_jd)
    fr_arr = times_jd - jd_arr
    r, v = vmap(_kepler_jd_gcrf, (None, 0, 0))(satrec, jd_arr, fr_arr)
    return r, v


def kepler_gcrf_positions_multi(
    satrec: SatRec,
    times_jd: jax.typing.ArrayLike,
) -> tuple[jax.Array, jax.Array]:
    """Propagate multiple satellites to multiple times using pure Keplerian motion.

    No perturbations are applied.  The outer ``vmap`` iterates over satellites,
    the inner over times.

    The function is JIT-compiled and differentiable with respect to all
    fields of *satrec* and with respect to *times_jd*.

    Args:
        satrec: Batched SatRec from :func:`tles_to_satrec` with leading
            dimension ``n_sat``.
        times_jd: 1-D array of UTC Julian dates, shape ``(n_times,)``.

    Returns:
        r_gcrf: GCRF positions, shape ``(n_sat, n_times, 3)``, in km.
        v_gcrf: GCRF velocities, shape ``(n_sat, n_times, 3)``, in km/s.

    Note:
        The TEME → GCRF rotation uses the input Julian dates as UT1
        (UTC is an approximation accurate to < 1 s).  For sub-arcsecond
        frame accuracy call :func:`utc_to_ut1` first and pass the UT1 dates.
    """
    times_jd = jnp.asarray(times_jd)
    jd_arr = jnp.floor(times_jd)
    fr_arr = times_jd - jd_arr
    r, v = vmap(
        vmap(_kepler_jd_gcrf, (None, 0, 0)),
        (0, None, None),
    )(satrec, jd_arr, fr_arr)
    return r, v
