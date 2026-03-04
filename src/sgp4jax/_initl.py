"""_initl - compute auxiliary orbital quantities and Greenwich sidereal time."""

import jax
import jax.numpy as jnp
import jax.typing
from math import pi

twopi = 2.0 * pi


def gstime(jdut1: jax.typing.ArrayLike) -> jax.Array:
    """Greenwich sidereal time from Julian date."""
    tut1 = (jdut1 - 2451545.0) / 36525.0
    temp = (-6.2e-6 * tut1 * tut1 * tut1 + 0.093104 * tut1 * tut1 +
            (876600.0 * 3600 + 8640184.812866) * tut1 + 67310.54841)
    deg2rad = pi / 180.0
    temp = (temp * deg2rad / 240.0) % twopi  # type: ignore[operator]

    temp = jnp.where(temp < 0.0, temp + twopi, temp)

    return temp


def initl(
    xke: jax.typing.ArrayLike,
    j2: jax.typing.ArrayLike,
    ecco: jax.typing.ArrayLike,
    epoch: jax.typing.ArrayLike,
    inclo: jax.typing.ArrayLike,
    no: jax.typing.ArrayLike,
) -> tuple[jax.Array, ...]:
    """Initialize auxiliary orbital quantities.

    Returns:
        (no_unkozai, ainv, ao, con41, con42, cosio, cosio2, eccsq,
         omeosq, posq, rp, rteosq, sinio, gsto)
    """
    x2o3 = 2.0 / 3.0

    # Calculate auxiliary epoch quantities
    eccsq = ecco * ecco
    omeosq = 1.0 - eccsq
    rteosq = jnp.sqrt(omeosq)
    cosio = jnp.cos(inclo)
    cosio2 = cosio * cosio

    # Un-Kozai the mean motion
    ak = (xke / no) ** x2o3
    d1 = 0.75 * j2 * (3.0 * cosio2 - 1.0) / (rteosq * omeosq)
    del_ = d1 / (ak * ak)
    adel = ak * (1.0 - del_ * del_ - del_ *
            (1.0 / 3.0 + 134.0 * del_ * del_ / 81.0))
    del_ = d1 / (adel * adel)
    no = no / (1.0 + del_)

    ao = (xke / no) ** x2o3
    sinio = jnp.sin(inclo)
    po = ao * omeosq
    con42 = 1.0 - 5.0 * cosio2
    con41 = -con42 - cosio2 - cosio2
    ainv = 1.0 / ao
    posq = po * po
    rp = ao * (1.0 - ecco)

    # Improved mode sidereal time
    gsto = gstime(epoch + 2433281.5)

    return (no, ainv, ao, con41, con42, cosio, cosio2, eccsq,  # type: ignore[return-value]
            omeosq, posq, rp, rteosq, sinio, gsto)
