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
from sgp4jax._frames import teme_to_gcrf

__all__ = [
    "SatRec", "make_satrec",
    "WGS72OLD", "WGS72", "WGS84",
    "tle_to_satrec", "tles_to_satrec",
    "propagate", "propagate_jd",
    "teme_to_gcrf",
    "propagate_gcrf", "propagate_jd_gcrf",
    "gcrf_positions", "gcrf_positions_multi",
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


def propagate_jd(satrec: SatRec, jd: jax.typing.ArrayLike, fr: jax.typing.ArrayLike) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Propagate satellite to Julian Date (jd + fr).

    Returns TEME position (km), velocity (km/s), and error code.
    """
    tsince = ((jd - satrec.jdsatepoch) * 1440.0 +
              (fr - satrec.jdsatepochF) * 1440.0)
    return propagate(satrec, tsince)  # type: ignore[no-any-return]


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
