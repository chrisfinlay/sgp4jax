"""JAX-compatible SGP4 satellite propagation."""

import jax
jax.config.update("jax_enable_x64", True)

from sgp4jax._constants import WGS72OLD, WGS72, WGS84
from sgp4jax._types import SatRec, make_satrec
from sgp4jax._tle import tle_to_satrec
from sgp4jax._propagation import sgp4 as propagate
from sgp4jax._frames import teme_to_gcrf
import jax.numpy as jnp

__all__ = [
    "SatRec", "make_satrec",
    "WGS72OLD", "WGS72", "WGS84",
    "tle_to_satrec",
    "propagate", "propagate_jd",
    "teme_to_gcrf",
    "propagate_gcrf", "propagate_jd_gcrf",
]


def propagate_jd(satrec: SatRec, jd: jnp.ndarray, fr: jnp.ndarray):
    """Propagate satellite to Julian Date (jd + fr).

    Returns TEME position (km), velocity (km/s), and error code.
    """
    tsince = ((jd - satrec.jdsatepoch) * 1440.0 +
              (fr - satrec.jdsatepochF) * 1440.0)
    return propagate(satrec, tsince)


def propagate_gcrf(satrec: SatRec, tsince: jnp.ndarray):
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


def propagate_jd_gcrf(satrec: SatRec, jd: jnp.ndarray, fr: jnp.ndarray):
    """Propagate satellite to Julian Date (jd + fr) and return GCRF.

    Args:
        satrec: Initialized SatRec from :func:`tle_to_satrec`.
        jd: Julian date, integer/whole part (scalar jnp.ndarray).
        fr: Julian date, fractional part (scalar jnp.ndarray).

    Returns:
        r_gcrf: Position in GCRF frame (3,) in km.
        v_gcrf: Velocity in GCRF frame (3,) in km/s.
        error: Error code (0 = success).
    """
    r_teme, v_teme, error = propagate_jd(satrec, jd, fr)
    r_gcrf, v_gcrf = teme_to_gcrf(r_teme, v_teme, jd, fr)
    return r_gcrf, v_gcrf, error
