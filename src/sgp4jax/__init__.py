"""JAX-compatible SGP4 satellite propagation."""

import jax
jax.config.update("jax_enable_x64", True)

from sgp4jax._constants import WGS72OLD, WGS72, WGS84
from sgp4jax._types import SatRec, make_satrec
from sgp4jax._tle import tle_to_satrec
from sgp4jax._propagation import sgp4 as propagate
import jax.numpy as jnp

__all__ = [
    "SatRec", "make_satrec",
    "WGS72OLD", "WGS72", "WGS84",
    "tle_to_satrec",
    "propagate", "propagate_jd",
]


def propagate_jd(satrec: SatRec, jd: jnp.ndarray, fr: jnp.ndarray):
    """Propagate satellite to Julian Date (jd + fr)."""
    tsince = ((jd - satrec.jdsatepoch) * 1440.0 +
              (fr - satrec.jdsatepochF) * 1440.0)
    return propagate(satrec, tsince)
