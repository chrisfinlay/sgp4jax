"""JAX-compatible SGP4 satellite propagation."""

import jax
jax.config.update("jax_enable_x64", True)

from jax import vmap
import jax.typing
import jax.numpy as jnp
from sgp4jax._constants import GravityConstants, WGS72OLD, WGS72, WGS84
from sgp4jax._types import SatRec, make_satrec
from sgp4jax._tle import tle_to_satrec
from sgp4jax._propagation import sgp4 as propagate
from sgp4jax._propagation_leo import sgp4_leo as propagate_leo
from sgp4jax._propagation_sdp4_nr import sgp4_sdp4_nr as propagate_sdp4_nr
from sgp4jax._propagation_mixed import propagate_mixed, gcrf_positions_mixed
from sgp4jax._frames import teme_to_gcrf, itrf_to_gcrf, gcrf_to_itrf
from sgp4jax._iers import update_iers_table, load_iers_table, utc_to_ut1
from sgp4jax._kepler import kepler_gcrf_positions, kepler_gcrf_positions_multi
from sgp4jax._sgp4init import sgp4init as _sgp4init
from sgp4jax._covariance import (
    ric_rotation,
    cov_ric_to_teme, cov_teme_to_ric,
    cov_elements_to_teme, cov_teme_to_elements,
    cov_ric_to_elements, cov_elements_to_ric,
    elements_jacobian,
    elements7_jacobian,
    cov_elements7_to_teme, cov_teme_to_elements7,
    cov_elements7_to_ric, cov_ric_to_elements7,
)

__all__ = [
    "SatRec", "make_satrec",
    "WGS72OLD", "WGS72", "WGS84",
    "tle_to_satrec", "tles_to_satrec", "satrec_from_elements",
    "propagate", "propagate_jd",
    "propagate_leo", "propagate_jd_leo",
    "propagate_sdp4_nr", "propagate_jd_sdp4_nr",
    "propagate_mixed",
    "teme_to_gcrf",
    "itrf_to_gcrf", "gcrf_to_itrf",
    "update_iers_table", "load_iers_table", "utc_to_ut1",
    "propagate_gcrf", "propagate_jd_gcrf",
    "gcrf_positions", "gcrf_positions_multi",
    "gcrf_positions_multi_leo", "gcrf_positions_multi_sdp4_nr",
    "gcrf_positions_mixed",
    "kepler_gcrf_positions", "kepler_gcrf_positions_multi",
    "ric_rotation",
    "cov_ric_to_teme", "cov_teme_to_ric",
    "cov_elements_to_teme", "cov_teme_to_elements",
    "cov_ric_to_elements", "cov_elements_to_ric",
    "elements_jacobian",
    "elements7_jacobian",
    "cov_elements7_to_teme", "cov_teme_to_elements7",
    "cov_elements7_to_ric", "cov_ric_to_elements7",
]


def tles_to_satrec(tles: list[list[str]], gravity: GravityConstants | None = None) -> SatRec:
    """Parse an array of TLEs and return a batched SatRec.

    Args:
        tles: TLE lines with shape ``(n_sat, 2)``.  Each row contains
            ``[line1, line2]`` as strings.
        gravity: Gravity constants (default WGS72).

    Returns:
        A single :class:`SatRec` whose fields are stacked arrays with
        a leading dimension of ``n_sat``, ready for use with
        ``jax.vmap``.
    """
    satrecs = [tle_to_satrec(l1, l2, gravity=gravity) for l1, l2 in tles]
    return SatRec(*[jnp.stack(vals) for vals in zip(*satrecs)])


def satrec_from_elements(
    inclo: float,
    nodeo: float,
    ecco: float,
    argpo: float,
    mo: float,
    no_kozai: float,
    *,
    bstar: float = 0.0,
    ndot: float = 0.0,
    nddot: float = 0.0,
    epoch_jd: float = 2451545.0,
    gravity: GravityConstants | None = None,
) -> SatRec:
    """Initialize a SatRec directly from orbital elements.

    A user-friendly alternative to parsing a TLE string when the orbital
    elements are already known.  Calls the internal ``sgp4init`` routine to
    compute all derived SGP4 quantities, so the returned record is ready for
    use with :func:`propagate`, :func:`propagate_jd`, and all GCRF/covariance
    helpers.

    Args:
        inclo: Inclination (rad).
        nodeo: Right ascension of ascending node (rad).
        ecco: Eccentricity (0 ≤ e < 1).
        argpo: Argument of perigee (rad).
        mo: Mean anomaly at epoch (rad).
        no_kozai: Mean motion at epoch (rad/min).
        bstar: SGP4 drag term B* (km⁻¹).  Default 0.
        ndot: First derivative of mean motion (rad/min²).  Default 0.
        nddot: Second derivative of mean motion (rad/min³).  Default 0.
        epoch_jd: Epoch as a full Julian date (e.g. 2451545.0 = J2000.0).
            Internally split into whole + fractional parts to preserve
            float64 precision.  Default J2000.0.
        gravity: Gravity model constants.  Default :data:`WGS72`.

    Returns:
        Fully initialised :class:`SatRec` ready for propagation.

    Example::

        import sgp4jax, jax.numpy as jnp

        sat = sgp4jax.satrec_from_elements(
            inclo=0.9006,   # 51.6°
            nodeo=1.2217,
            ecco=0.0004,
            argpo=4.6194,
            mo=3.6160,
            no_kozai=0.0672,   # ~15.5 rev/day (ISS-like)
            bstar=2.5e-4,
            epoch_jd=2458924.686,
        )
        r, v, err = sgp4jax.propagate(sat, jnp.array(60.0))
    """
    if gravity is None:
        gravity = WGS72
    jd_whole = float(int(epoch_jd))
    jd_frac = epoch_jd - jd_whole
    epoch = jd_whole + jd_frac - 2433281.5
    return _sgp4init(
        gravity, epoch,
        bstar, ndot, nddot,
        ecco, argpo, inclo, mo, no_kozai, nodeo,
        jnp.array(jd_whole), jnp.array(jd_frac),
    )


def propagate_jd(satrec: SatRec, jd: jax.typing.ArrayLike, fr: jax.typing.ArrayLike) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Propagate satellite to Julian Date (jd + fr) in TEME.

    Converts the Julian Date to minutes-since-epoch and calls :func:`propagate`.

    Args:
        satrec: Initialized SatRec from :func:`tle_to_satrec`.
        jd: Julian date (UTC), integer/whole part (scalar).
        fr: Julian date (UTC), fractional part (scalar).

    Returns:
        r: Position in TEME frame (3,) in km.
        v: Velocity in TEME frame (3,) in km/s.
        error: Error code (0 = success).
    """
    tsince = ((jd - satrec.jdsatepoch) * 1440.0 +
              (fr - satrec.jdsatepochF) * 1440.0)
    return propagate(satrec, tsince)  # type: ignore[no-any-return]


def propagate_jd_leo(satrec: SatRec, jd: jax.typing.ArrayLike, fr: jax.typing.ArrayLike) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Propagate a near-earth satellite to Julian Date (jd + fr) in TEME.

    LEO/near-earth only variant — deep-space code paths are absent.
    See :func:`propagate_leo` for the performance trade-offs and limitations.

    Args:
        satrec: Initialized SatRec from :func:`tle_to_satrec` (near-earth only).
        jd: Julian date (UTC), integer/whole part (scalar).
        fr: Julian date (UTC), fractional part (scalar).

    Returns:
        r: Position in TEME frame (3,) in km.
        v: Velocity in TEME frame (3,) in km/s.
        error: Error code (0 = success).
    """
    tsince = ((jd - satrec.jdsatepoch) * 1440.0 +
              (fr - satrec.jdsatepochF) * 1440.0)
    return propagate_leo(satrec, tsince)  # type: ignore[no-any-return]


def propagate_jd_sdp4_nr(satrec: SatRec, jd: jax.typing.ArrayLike, fr: jax.typing.ArrayLike) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Propagate a deep-space no-resonance (irez=0) satellite to Julian Date (jd + fr) in TEME.

    Deep-space irez=0 only variant — resonance integrator is absent.
    See :func:`propagate_sdp4_nr` for the performance trade-offs and limitations.

    Args:
        satrec: Initialized SatRec from :func:`tle_to_satrec` (deep-space, irez=0 only).
        jd: Julian date (UTC), integer/whole part (scalar).
        fr: Julian date (UTC), fractional part (scalar).

    Returns:
        r: Position in TEME frame (3,) in km.
        v: Velocity in TEME frame (3,) in km/s.
        error: Error code (0 = success).
    """
    tsince = ((jd - satrec.jdsatepoch) * 1440.0 +
              (fr - satrec.jdsatepochF) * 1440.0)
    return propagate_sdp4_nr(satrec, tsince)  # type: ignore[no-any-return]


def propagate_gcrf(satrec: SatRec, tsince: jax.typing.ArrayLike) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Propagate satellite and return GCRF position/velocity.

    Args:
        satrec: Initialized SatRec from :func:`tle_to_satrec`.
        tsince: Time since epoch in minutes (scalar jnp.ndarray).

    Returns:
        r_gcrf: Position in GCRF frame (3,) in km.
        v_gcrf: Velocity in GCRF frame (3,) in km/s.
        error: Error code (0 = success).
    """
    r_teme, v_teme, error = propagate(satrec, tsince)
    jd = jnp.array(satrec.jdsatepoch)
    fr = jnp.array(satrec.jdsatepochF) + tsince / 1440.0
    r_gcrf, v_gcrf = teme_to_gcrf(r_teme, v_teme, jd, fr)
    return r_gcrf, v_gcrf, error


def propagate_jd_gcrf(satrec: SatRec, jd: jax.typing.ArrayLike, fr: jax.typing.ArrayLike) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Propagate satellite to Julian Date (jd + fr) and return GCRF.

    The input time ``jd + fr`` is interpreted as **UTC**, matching the
    time scale of TLE epochs and Measurement Set timestamps.  The same
    UTC value is used for SGP4 propagation and, as an approximation to
    UT1, for the TEME → GCRF frame transformation (the UT1 − UTC
    difference is at most 0.9 s, introducing < 1 m frame error).

    Args:
        satrec: Initialized SatRec from :func:`tle_to_satrec`.
        jd: Julian date (UTC), integer/whole part (scalar jnp.ndarray).
        fr: Julian date (UTC), fractional part (scalar jnp.ndarray).

    Returns:
        r_gcrf: Position in GCRF frame (3,) in km.
        v_gcrf: Velocity in GCRF frame (3,) in km/s.
        error: Error code (0 = success).
    """
    r_teme, v_teme, error = propagate_jd(satrec, jd, fr)
    r_gcrf, v_gcrf = teme_to_gcrf(r_teme, v_teme, jd, fr)
    return r_gcrf, v_gcrf, error


def gcrf_positions(satrec: SatRec, times_jd: jax.typing.ArrayLike) -> tuple[jax.Array, jax.Array]:
    """Propagate a single satellite to multiple UTC Julian dates.

    Args:
        satrec: Scalar SatRec from :func:`tle_to_satrec`.
        times_jd: 1-D array of UTC Julian dates, shape ``(n_times,)``.

    Returns:
        r_gcrf: Positions in GCRF frame, shape ``(n_times, 3)`` in km.
        v_gcrf: Velocities in GCRF frame, shape ``(n_times, 3)`` in km/s.
    """
    jd = jnp.floor(times_jd)
    fr = times_jd - jd
    r, v, _ = vmap(propagate_jd_gcrf, (None, 0, 0))(satrec, jd, fr)
    return r, v


def gcrf_positions_multi(satrec: SatRec, times_jd: jax.typing.ArrayLike) -> tuple[jax.Array, jax.Array]:
    """Propagate multiple satellites to multiple UTC Julian dates.

    Args:
        satrec: Batched SatRec from :func:`tles_to_satrec` with leading
            dimension ``n_sat``.
        times_jd: 1-D array of UTC Julian dates, shape ``(n_times,)``.

    Returns:
        r_gcrf: Positions in GCRF, shape ``(n_sat, n_times, 3)`` in km.
        v_gcrf: Velocities in GCRF, shape ``(n_sat, n_times, 3)`` in km/s.
    """
    jd = jnp.floor(times_jd)
    fr = times_jd - jd
    r, v, _ = vmap(
        vmap(propagate_jd_gcrf, (None, 0, 0)),
        (0, None, None),
    )(satrec, jd, fr)
    return r, v


def _propagate_jd_gcrf_leo(satrec: SatRec, jd: jax.typing.ArrayLike, fr: jax.typing.ArrayLike) -> tuple[jax.Array, jax.Array, jax.Array]:
    r_teme, v_teme, error = propagate_jd_leo(satrec, jd, fr)
    r_gcrf, v_gcrf = teme_to_gcrf(r_teme, v_teme, jd, fr)
    return r_gcrf, v_gcrf, error


def _propagate_jd_gcrf_sdp4_nr(satrec: SatRec, jd: jax.typing.ArrayLike, fr: jax.typing.ArrayLike) -> tuple[jax.Array, jax.Array, jax.Array]:
    r_teme, v_teme, error = propagate_jd_sdp4_nr(satrec, jd, fr)
    r_gcrf, v_gcrf = teme_to_gcrf(r_teme, v_teme, jd, fr)
    return r_gcrf, v_gcrf, error


def gcrf_positions_multi_leo(satrec: SatRec, times_jd: jax.typing.ArrayLike) -> tuple[jax.Array, jax.Array]:
    """Propagate a near-earth (LEO) satellite batch to multiple UTC Julian dates in GCRF.

    Use this for homogeneous batches of near-earth satellites (``method=0``).
    For heterogeneous batches, use :func:`gcrf_positions_mixed`.

    Args:
        satrec: Batched SatRec from :func:`tles_to_satrec`, all near-earth.
        times_jd: 1-D array of UTC Julian dates, shape ``(n_times,)``.

    Returns:
        r_gcrf: Positions in GCRF, shape ``(n_sat, n_times, 3)`` in km.
        v_gcrf: Velocities in GCRF, shape ``(n_sat, n_times, 3)`` in km/s.
    """
    jd = jnp.floor(times_jd)
    fr = times_jd - jd
    r, v, _ = vmap(
        vmap(_propagate_jd_gcrf_leo, (None, 0, 0)),
        (0, None, None),
    )(satrec, jd, fr)
    return r, v


def gcrf_positions_multi_sdp4_nr(satrec: SatRec, times_jd: jax.typing.ArrayLike) -> tuple[jax.Array, jax.Array]:
    """Propagate a deep-space no-resonance satellite batch to multiple UTC Julian dates in GCRF.

    Use this for homogeneous batches of deep-space irez=0 satellites
    (GPS/GLONASS/Galileo/BeiDou MEO constellations).
    For heterogeneous batches, use :func:`gcrf_positions_mixed`.

    Args:
        satrec: Batched SatRec from :func:`tles_to_satrec`, all deep-space irez=0.
        times_jd: 1-D array of UTC Julian dates, shape ``(n_times,)``.

    Returns:
        r_gcrf: Positions in GCRF, shape ``(n_sat, n_times, 3)`` in km.
        v_gcrf: Velocities in GCRF, shape ``(n_sat, n_times, 3)`` in km/s.
    """
    jd = jnp.floor(times_jd)
    fr = times_jd - jd
    r, v, _ = vmap(
        vmap(_propagate_jd_gcrf_sdp4_nr, (None, 0, 0)),
        (0, None, None),
    )(satrec, jd, fr)
    return r, v
