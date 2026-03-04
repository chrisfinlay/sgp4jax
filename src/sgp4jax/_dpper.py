"""_dpper - deep space long period periodic contributions."""

import jax
import jax.numpy as jnp
import jax.typing
from math import pi


twopi = 2.0 * pi

# Shorthand used in signatures below
_A = jax.typing.ArrayLike


def dpper_init(
    e3: _A, ee2: _A, peo: _A, pgho: _A, pho: _A, pinco: _A, plo: _A,
    se2: _A, se3: _A, sgh2: _A, sgh3: _A, sgh4: _A, sh2: _A, sh3: _A,
    si2: _A, si3: _A, sl2: _A, sl3: _A, sl4: _A,
    xgh2: _A, xgh3: _A, xgh4: _A, xh2: _A, xh3: _A,
    xi2: _A, xi3: _A, xl2: _A, xl3: _A, xl4: _A,
    zmol: _A, zmos: _A,
    inclo: _A, ep: _A, inclp: _A, nodep: _A, argpp: _A, mp: _A,
) -> tuple[jax.Array, ...]:
    """Deep space periodics at initialization (init='y')."""
    zns = 1.19459e-5
    zes = 0.01675
    znl = 1.5835218e-4
    zel = 0.05490

    # t = 0 at init
    zm = zmos  # init='y' so zm = zmos
    zf = zm + 2.0 * zes * jnp.sin(zm)
    sinzf = jnp.sin(zf)
    f2 = 0.5 * sinzf * sinzf - 0.25
    f3 = -0.5 * sinzf * jnp.cos(zf)
    ses = se2 * f2 + se3 * f3
    sis = si2 * f2 + si3 * f3
    sls = sl2 * f2 + sl3 * f3 + sl4 * sinzf
    sghs = sgh2 * f2 + sgh3 * f3 + sgh4 * sinzf
    shs = sh2 * f2 + sh3 * f3

    zm = zmol  # init='y' so zm = zmol
    zf = zm + 2.0 * zel * jnp.sin(zm)
    sinzf = jnp.sin(zf)
    f2 = 0.5 * sinzf * sinzf - 0.25
    f3 = -0.5 * sinzf * jnp.cos(zf)
    sel = ee2 * f2 + e3 * f3
    sil = xi2 * f2 + xi3 * f3
    sll = xl2 * f2 + xl3 * f3 + xl4 * sinzf
    sghl = xgh2 * f2 + xgh3 * f3 + xgh4 * sinzf
    shll = xh2 * f2 + xh3 * f3

    pe = ses + sel
    pinc = sis + sil
    pl = sls + sll
    pgh = sghs + sghl
    ph = shs + shll

    # init='y' does NOT apply periodics (only init='n' does)
    return ep, inclp, nodep, argpp, mp  # type: ignore[return-value]


def dpper(
    e3: _A, ee2: _A, peo: _A, pgho: _A, pho: _A, pinco: _A, plo: _A,
    se2: _A, se3: _A, sgh2: _A, sgh3: _A, sgh4: _A, sh2: _A, sh3: _A,
    si2: _A, si3: _A, sl2: _A, sl3: _A, sl4: _A, t: _A,
    xgh2: _A, xgh3: _A, xgh4: _A, xh2: _A, xh3: _A,
    xi2: _A, xi3: _A, xl2: _A, xl3: _A, xl4: _A,
    zmol: _A, zmos: _A,
    inclo: _A, ep: _A, inclp: _A, nodep: _A, argpp: _A, mp: _A,
) -> tuple[jax.Array, ...]:
    """Deep space periodics during propagation (init='n').

    This is the JIT-compatible version using jnp.
    """
    zns = 1.19459e-5
    zes = 0.01675
    znl = 1.5835218e-4
    zel = 0.05490

    # Solar terms
    zm = zmos + zns * t
    zf = zm + 2.0 * zes * jnp.sin(zm)
    sinzf = jnp.sin(zf)
    f2 = 0.5 * sinzf * sinzf - 0.25
    f3 = -0.5 * sinzf * jnp.cos(zf)
    ses = se2 * f2 + se3 * f3
    sis = si2 * f2 + si3 * f3
    sls = sl2 * f2 + sl3 * f3 + sl4 * sinzf
    sghs = sgh2 * f2 + sgh3 * f3 + sgh4 * sinzf
    shs = sh2 * f2 + sh3 * f3

    # Lunar terms
    zm = zmol + znl * t
    zf = zm + 2.0 * zel * jnp.sin(zm)
    sinzf = jnp.sin(zf)
    f2 = 0.5 * sinzf * sinzf - 0.25
    f3 = -0.5 * sinzf * jnp.cos(zf)
    sel = ee2 * f2 + e3 * f3
    sil = xi2 * f2 + xi3 * f3
    sll = xl2 * f2 + xl3 * f3 + xl4 * sinzf
    sghl = xgh2 * f2 + xgh3 * f3 + xgh4 * sinzf
    shll = xh2 * f2 + xh3 * f3

    pe = ses + sel - peo
    pinc = sis + sil - pinco
    pl = sls + sll - plo
    pgh = sghs + sghl - pgho
    ph = shs + shll - pho

    inclp = inclp + pinc
    ep = ep + pe
    sinip = jnp.sin(inclp)
    cosip = jnp.cos(inclp)

    # Apply periodics - two branches based on inclp >= 0.2
    # Branch A: inclp >= 0.2 (direct application)
    ph_over_sinip_a = ph / sinip
    pgh_a = pgh - cosip * ph_over_sinip_a
    argpp_a = argpp + pgh_a
    nodep_a = nodep + ph_over_sinip_a
    mp_a = mp + pl

    # Branch B: Lyddane modification (inclp < 0.2)
    sinop = jnp.sin(nodep)
    cosop = jnp.cos(nodep)
    alfdp = sinip * sinop
    betdp = sinip * cosop
    dalf = ph * cosop + pinc * cosip * sinop
    dbet = -ph * sinop + pinc * cosip * cosop
    alfdp = alfdp + dalf
    betdp = betdp + dbet
    # Sign-preserving modulo (matches reference: nodep % twopi if >= 0 else -(-nodep % twopi))
    nodep_b = jnp.where(nodep >= 0.0,  # type: ignore[operator]
                         nodep % twopi,  # type: ignore[operator]
                         -((-nodep) % twopi))  # type: ignore[operator]
    # Note: AFSPC 'a' mode would add twopi for negative nodep here, but
    # we only support improved 'i' mode which skips that correction.
    xls = mp + argpp + pl + pgh + (cosip - pinc * sinip) * nodep_b
    xnoh = nodep_b
    nodep_b = jnp.arctan2(alfdp, betdp)
    # Note: AFSPC 'a' mode would add twopi for negative nodep here too,
    # but improved 'i' mode skips it.
    # Wrapping correction (not opsmode-guarded, always applies)
    nodep_b = jnp.where(jnp.abs(xnoh - nodep_b) > pi,
                         jnp.where(nodep_b < xnoh, nodep_b + twopi, nodep_b - twopi),
                         nodep_b)
    mp_b = mp + pl
    argpp_b = xls - mp_b - cosip * nodep_b

    use_a = inclp >= 0.2
    argpp = jnp.where(use_a, argpp_a, argpp_b)
    nodep = jnp.where(use_a, nodep_a, nodep_b)
    mp = jnp.where(use_a, mp_a, mp_b)

    return ep, inclp, nodep, argpp, mp  # type: ignore[return-value]
