"""sgp4_sdp4_nr propagation - deep-space, no-resonance (irez=0) only."""

import jax
import jax.numpy as jnp
import jax.typing
from sgp4jax._types import SatRec
from sgp4jax._dpper import dpper


twopi = 2.0 * jnp.pi


def _fmod_python(x: jax.typing.ArrayLike, y: jax.typing.ArrayLike) -> jax.Array:
    """Python-style modulo: result has same sign as y (always positive for positive y)."""
    return x - y * jnp.floor(x / y)  # type: ignore[return-value]


@jax.jit
def sgp4_sdp4_nr(satrec: SatRec, tsince: jax.typing.ArrayLike) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Propagate a deep-space, no-resonance (irez=0) satellite to time tsince.

    This is a stripped-down version of sgp4 for deep-space satellites with
    irez=0 (no resonance). This captures MEO navigation satellites (GPS,
    GLONASS, Galileo, BeiDou MEO) and other deep-space objects that do not
    fall in the 24-hour synchronous (irez=1) or 12-hour Molniya (irez=2)
    resonance bands.

    Simplifications over the full sgp4:

    * The resonance integrator (dspace / lax.scan) is removed entirely.
      For irez=0, dspace reduces to five scalar secular drift terms
      (dedt, didt, domdt, dnodt, dmdt), which are applied directly.
    * Deep-space satellites always have isimp=1, so the full-drag
      computation branch is omitted.
    * All ``jnp.where(is_deep, ...)`` guards collapse to unconditional
      deep-space assignments.
    * dpper (lunar-solar periodics) is retained unchanged.

    .. warning::

        Do not use for near-earth satellites (period < 225 min),
        irez=1 (GEO/synchronous), or irez=2 (Molniya/12h resonant)
        satellites — resonance and drag terms will be missing or zero.

    Parameters
    ----------
    satrec : SatRec
        Initialized SatRec from :func:`~sgp4jax.tle_to_satrec`.
        Deep-space, ``irez=0`` only.
    tsince : array-like
        Time since epoch in minutes (scalar).

    Returns
    -------
    r : jax.Array, shape (3,)
        Position in TEME frame, km.
    v : jax.Array, shape (3,)
        Velocity in TEME frame, km/s.
    error : jax.Array
        Error code (0 = success).
    """
    temp4 = 1.5e-12
    x2o3 = 2.0 / 3.0
    vkmpersec = satrec.radiusearthkm * satrec.xke / 60.0
    t = tsince

    # Secular gravity and atmospheric drag (isimp=1 always for deep-space)
    xmdf = satrec.mo + satrec.mdot * t
    argpdf = satrec.argpo + satrec.argpdot * t
    nodedf = satrec.nodeo + satrec.nodedot * t
    argpm = argpdf
    mm = xmdf
    t2 = t * t
    nodem = nodedf + satrec.nodecf * t2
    tempa = 1.0 - satrec.cc1 * t
    tempe = satrec.bstar * satrec.cc4 * t
    templ = satrec.t2cof * t2

    nm = satrec.no_unkozai
    em = satrec.ecco
    inclm = satrec.inclo

    # Deep-space secular perturbations (irez=0: no resonance, no scan needed)
    em    = em    + satrec.dedt  * t
    inclm = inclm + satrec.didt  * t
    argpm = argpm + satrec.domdt * t
    nodem = nodem + satrec.dnodt * t
    mm    = mm    + satrec.dmdt  * t
    # nm unchanged (irez=0: dndt=0)

    # Error: nm <= 0
    error = jnp.where(nm <= 0.0, 2, 0)

    am = jnp.power(satrec.xke / nm, x2o3) * tempa * tempa
    nm = satrec.xke / jnp.power(am, 1.5)
    em = em - tempe

    # Error: eccentricity out of range
    error = jnp.where((em >= 1.0) | (em < -0.001), 1, error)

    # Clamp eccentricity
    em = jnp.maximum(em, 1.0e-6)

    mm = mm + satrec.no_unkozai * templ
    xlm = mm + argpm + nodem
    emsq = em * em
    temp = 1.0 - emsq

    # Angle normalization (matching Python semantics)
    nodem = jnp.where(nodem >= 0.0,
                      _fmod_python(nodem, twopi),
                      -_fmod_python(-nodem, twopi))
    argpm = _fmod_python(argpm, twopi)
    xlm = _fmod_python(xlm, twopi)
    mm = _fmod_python(xlm - argpm - nodem, twopi)

    # Extra mean quantities
    sinim = jnp.sin(inclm)
    cosim = jnp.cos(inclm)

    # Lunar-solar periodics
    ep = em
    xincp = inclm
    argpp = argpm
    nodep = nodem
    mp = mm
    sinip = sinim
    cosip = cosim

    # Deep space periodics (init='n')
    ep_d, xincp_d, nodep_d, argpp_d, mp_d = dpper(
        satrec.e3, satrec.ee2, satrec.peo, satrec.pgho, satrec.pho,
        satrec.pinco, satrec.plo,
        satrec.se2, satrec.se3, satrec.sgh2, satrec.sgh3, satrec.sgh4,
        satrec.sh2, satrec.sh3,
        satrec.si2, satrec.si3, satrec.sl2, satrec.sl3, satrec.sl4, t,
        satrec.xgh2, satrec.xgh3, satrec.xgh4,
        satrec.xh2, satrec.xh3, satrec.xi2, satrec.xi3,
        satrec.xl2, satrec.xl3, satrec.xl4,
        satrec.zmol, satrec.zmos,
        satrec.inclo, ep, xincp, nodep, argpp, mp)

    # Handle negative inclination from dpper
    xincp_neg = -xincp_d
    nodep_neg = nodep_d + jnp.pi
    argpp_neg = argpp_d - jnp.pi
    is_neg_xincp = xincp_d < 0.0
    xincp_d = jnp.where(is_neg_xincp, xincp_neg, xincp_d)
    nodep_d = jnp.where(is_neg_xincp, nodep_neg, nodep_d)
    argpp_d = jnp.where(is_neg_xincp, argpp_neg, argpp_d)

    # Deep-space always: assign perturbed elements directly
    ep    = ep_d
    xincp = xincp_d
    nodep = nodep_d
    argpp = argpp_d
    mp    = mp_d

    # Error: perturbed eccentricity
    error = jnp.where((ep < 0.0) | (ep > 1.0), 3, error)

    # Long period periodics (always deep-space, recompute from perturbed inclination)
    sinip = jnp.sin(xincp)
    cosip = jnp.cos(xincp)

    # Geometry coefficients from perturbed inclination (unconditional for deep-space)
    aycof  = -0.5 * satrec.j3oj2 * sinip
    xlcof_num = -0.25 * satrec.j3oj2 * sinip * (3.0 + 5.0 * cosip)
    xlcof  = xlcof_num / jnp.where(jnp.abs(cosip + 1.0) > 1.5e-12, 1.0 + cosip, temp4)
    cosisq = cosip * cosip
    con41  = 3.0 * cosisq - 1.0
    x1mth2 = 1.0 - cosisq
    x7thm1 = 7.0 * cosisq - 1.0

    axnl = ep * jnp.cos(argpp)
    temp = 1.0 / (am * (1.0 - ep * ep))
    aynl = ep * jnp.sin(argpp) + temp * aycof
    xl = mp + argpp + nodep + temp * xlcof * axnl

    # Kepler's equation
    u = _fmod_python(xl - nodep, twopi)

    # Newton-Raphson iteration (fixed 10 iterations via fori_loop)
    def kepler_body(i, eo1):
        sineo1 = jnp.sin(eo1)
        coseo1 = jnp.cos(eo1)
        denom = 1.0 - coseo1 * axnl - sineo1 * aynl
        tem5 = (u - aynl * coseo1 + axnl * sineo1 - eo1) / denom
        tem5 = jnp.clip(tem5, -0.95, 0.95)
        return eo1 + tem5

    eo1 = jax.lax.fori_loop(0, 10, kepler_body, u)

    # Short period preliminary quantities
    sineo1 = jnp.sin(eo1)
    coseo1 = jnp.cos(eo1)
    ecose = axnl * coseo1 + aynl * sineo1
    esine = axnl * sineo1 - aynl * coseo1
    el2 = axnl * axnl + aynl * aynl
    pl = am * (1.0 - el2)

    # Error: semi-latus rectum < 0
    error = jnp.where(pl < 0.0, 4, error)

    rl = am * (1.0 - ecose)
    rdotl = jnp.sqrt(am) * esine / rl
    rvdotl = jnp.sqrt(pl) / rl
    betal = jnp.sqrt(1.0 - el2)
    temp = esine / (1.0 + betal)
    sinu = am / rl * (sineo1 - aynl - axnl * temp)
    cosu = am / rl * (coseo1 - axnl + aynl * temp)
    su = jnp.arctan2(sinu, cosu)
    sin2u = (cosu + cosu) * sinu
    cos2u = 1.0 - 2.0 * sinu * sinu
    temp = 1.0 / pl
    temp1 = 0.5 * satrec.j2 * temp
    temp2 = temp1 * temp

    # Short period periodics
    mrt = rl * (1.0 - 1.5 * temp2 * betal * con41) + \
          0.5 * temp1 * x1mth2 * cos2u
    su = su - 0.25 * temp2 * x7thm1 * sin2u
    xnode = nodep + 1.5 * temp2 * cosip * sin2u
    xinc = xincp + 1.5 * temp2 * cosip * sinip * cos2u
    mvt = rdotl - nm * temp1 * x1mth2 * sin2u / satrec.xke
    rvdot = rvdotl + nm * temp1 * (x1mth2 * cos2u +
            1.5 * con41) / satrec.xke

    # Orientation vectors
    sinsu = jnp.sin(su)
    cossu = jnp.cos(su)
    snod = jnp.sin(xnode)
    cnod = jnp.cos(xnode)
    sini = jnp.sin(xinc)
    cosi = jnp.cos(xinc)
    xmx = -snod * cosi
    xmy = cnod * cosi
    ux = xmx * sinsu + cnod * cossu
    uy = xmy * sinsu + snod * cossu
    uz = sini * sinsu
    vx = xmx * cossu - cnod * sinsu
    vy = xmy * cossu - snod * sinsu
    vz = sini * cossu

    # Position and velocity
    _mr = mrt * satrec.radiusearthkm
    r = jnp.array([_mr * ux, _mr * uy, _mr * uz])
    v = jnp.array([(mvt * ux + rvdot * vx) * vkmpersec,
                    (mvt * uy + rvdot * vy) * vkmpersec,
                    (mvt * uz + rvdot * vz) * vkmpersec])

    # Satellite decay check
    error = jnp.where(mrt < 1.0, 6, error)

    # Mask errors with NaN
    nan3 = jnp.array([jnp.nan, jnp.nan, jnp.nan])
    r = jnp.where(error == 0, r, nan3)
    v = jnp.where(error == 0, v, nan3)

    return r, v, error
