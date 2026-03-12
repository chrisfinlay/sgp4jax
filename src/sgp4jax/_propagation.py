"""sgp4 propagation - the JIT-compatible hot path."""

import jax
import jax.numpy as jnp
import jax.typing
from sgp4jax._types import SatRec
from sgp4jax._dspace import dspace
from sgp4jax._dpper import dpper


twopi = 2.0 * jnp.pi


def _fmod_python(x: jax.typing.ArrayLike, y: jax.typing.ArrayLike) -> jax.Array:
    """Python-style modulo: result has same sign as y (always positive for positive y)."""
    return x - y * jnp.floor(x / y)  # type: ignore[return-value]


@jax.jit
def sgp4(satrec: SatRec, tsince: jax.typing.ArrayLike) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Propagate satellite to time tsince (minutes from epoch).

    Args:
        satrec: Initialized SatRec
        tsince: Time since epoch in minutes (scalar jnp.ndarray)

    Returns:
        r: position vector (3,) in km (TEME frame)
        v: velocity vector (3,) in km/s (TEME frame)
        error: error code (0 = success)
    """
    temp4 = 1.5e-12
    x2o3 = 2.0 / 3.0
    vkmpersec = satrec.radiusearthkm * satrec.xke / 60.0
    t = tsince

    # Secular gravity and atmospheric drag
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

    # Non-simplified drag (isimp == 0)
    delomg = satrec.omgcof * t
    delmtemp = 1.0 + satrec.eta * jnp.cos(xmdf)
    delm = satrec.xmcof * (delmtemp * delmtemp * delmtemp - satrec.delmo)
    temp = delomg + delm
    mm_full = xmdf + temp
    argpm_full = argpdf - temp
    t3 = t2 * t
    t4 = t3 * t
    tempa_full = tempa - satrec.d2 * t2 - satrec.d3 * t3 - satrec.d4 * t4
    tempe_full = tempe + satrec.bstar * satrec.cc5 * (jnp.sin(mm_full) - satrec.sinmao)
    templ_full = templ + satrec.t3cof * t3 + t4 * (satrec.t4cof + t * satrec.t5cof)

    is_simple = satrec.isimp == 1.0
    mm = jnp.where(is_simple, mm, mm_full)
    argpm = jnp.where(is_simple, argpm, argpm_full)
    tempa = jnp.where(is_simple, tempa, tempa_full)
    tempe = jnp.where(is_simple, tempe, tempe_full)
    templ = jnp.where(is_simple, templ, templ_full)

    nm = satrec.no_unkozai
    em = satrec.ecco
    inclm = satrec.inclo

    # Deep space
    is_deep = satrec.method == 1.0
    tc = t

    atime_d, em_d, argpm_d, inclm_d, xli_d, mm_d, xni_d, nodem_d, dndt_d, nm_d = dspace(
        satrec.irez, satrec.d2201, satrec.d2211, satrec.d3210,
        satrec.d3222, satrec.d4410, satrec.d4422,
        satrec.d5220, satrec.d5232, satrec.d5421, satrec.d5433,
        satrec.dedt, satrec.del1, satrec.del2, satrec.del3,
        satrec.didt, satrec.dmdt, satrec.dnodt, satrec.domdt,
        satrec.argpo, satrec.argpdot, t, tc, satrec.gsto,
        satrec.xfact, satrec.xlamo, satrec.no_unkozai,
        satrec.atime, em, argpm, inclm, satrec.xli, mm, satrec.xni, nodem, nm)

    em = jnp.where(is_deep, em_d, em)
    argpm = jnp.where(is_deep, argpm_d, argpm)
    inclm = jnp.where(is_deep, inclm_d, inclm)
    mm = jnp.where(is_deep, mm_d, mm)
    nodem = jnp.where(is_deep, nodem_d, nodem)
    nm = jnp.where(is_deep, nm_d, nm)

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

    ep = jnp.where(is_deep, ep_d, ep)
    xincp = jnp.where(is_deep, xincp_d, xincp)
    nodep = jnp.where(is_deep, nodep_d, nodep)
    argpp = jnp.where(is_deep, argpp_d, argpp)
    mp = jnp.where(is_deep, mp_d, mp)

    # Error: perturbed eccentricity
    error = jnp.where(is_deep & ((ep < 0.0) | (ep > 1.0)), 3, error)

    # Long period periodics
    sinip = jnp.where(is_deep, jnp.sin(xincp), sinip)
    cosip = jnp.where(is_deep, jnp.cos(xincp), cosip)

    # For deep space, recompute aycof and xlcof from perturbed inclination
    aycof_d = -0.5 * satrec.j3oj2 * sinip
    xlcof_num = -0.25 * satrec.j3oj2 * sinip * (3.0 + 5.0 * cosip)
    xlcof_d = xlcof_num / jnp.where(jnp.abs(cosip + 1.0) > 1.5e-12, 1.0 + cosip, temp4)

    aycof = jnp.where(is_deep, aycof_d, satrec.aycof)
    xlcof = jnp.where(is_deep, xlcof_d, satrec.xlcof)

    # For deep space, update short-period quantities
    cosisq_d = cosip * cosip
    con41 = jnp.where(is_deep, 3.0 * cosisq_d - 1.0, satrec.con41)
    x1mth2 = jnp.where(is_deep, 1.0 - cosisq_d, satrec.x1mth2)
    x7thm1 = jnp.where(is_deep, 7.0 * cosisq_d - 1.0, satrec.x7thm1)

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
