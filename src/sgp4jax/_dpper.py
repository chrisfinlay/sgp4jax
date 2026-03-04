"""_dpper - deep space long period periodic contributions."""

import jax.numpy as jnp
from math import pi


twopi = 2.0 * pi


def dpper_init(e3, ee2, peo, pgho, pho, pinco, plo,
               se2, se3, sgh2, sgh3, sgh4, sh2, sh3,
               si2, si3, sl2, sl3, sl4,
               xgh2, xgh3, xgh4, xh2, xh3, xi2, xi3, xl2, xl3, xl4,
               zmol, zmos,
               inclo, ep, inclp, nodep, argpp, mp):
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
    return ep, inclp, nodep, argpp, mp


def dpper(e3, ee2, peo, pgho, pho, pinco, plo,
          se2, se3, sgh2, sgh3, sgh4, sh2, sh3,
          si2, si3, sl2, sl3, sl4, t,
          xgh2, xgh3, xgh4, xh2, xh3, xi2, xi3, xl2, xl3, xl4,
          zmol, zmos,
          inclo, ep, inclp, nodep, argpp, mp):
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
    nodep_b = jnp.where(nodep >= 0.0,
                         nodep % twopi,
                         -((-nodep) % twopi))
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

    return ep, inclp, nodep, argpp, mp
