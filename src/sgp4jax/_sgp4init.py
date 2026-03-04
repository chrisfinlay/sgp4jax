"""sgp4init - initialize SGP4 propagator from orbital elements."""

from math import pi
import jax.typing
import jax.numpy as jnp

from sgp4jax._constants import GravityConstants
from sgp4jax._types import SatRec
from sgp4jax._initl import initl
from sgp4jax._dscom import dscom
from sgp4jax._dpper import dpper_init
from sgp4jax._dsinit import dsinit

twopi = 2.0 * pi

_A = jax.typing.ArrayLike


def sgp4init(
    whichconst: GravityConstants,
    epoch: _A,
    xbstar: _A,
    xndot: _A,
    xnddot: _A,
    xecco: _A,
    xargpo: _A,
    xinclo: _A,
    xmo: _A,
    xno_kozai: _A,
    xnodeo: _A,
    jdsatepoch: _A,
    jdsatepochF: _A,
) -> SatRec:
    """Initialize SGP4 satellite record from orbital elements.

    Args:
        whichconst: GravityConstants namedtuple
        epoch: epoch time in days from jan 0, 1950. 0 hr
        xbstar: drag coefficient
        xndot: first derivative of mean motion
        xnddot: second derivative of mean motion
        xecco: eccentricity
        xargpo: argument of perigee (rad)
        xinclo: inclination (rad)
        xmo: mean anomaly (rad)
        xno_kozai: mean motion (rad/min), Kozai
        xnodeo: RAAN (rad)
        jdsatepoch: Julian date (whole part)
        jdsatepochF: Julian date (fractional part)

    Returns:
        SatRec NamedTuple
    """
    temp4 = 1.5e-12

    tumin, mu, radiusearthkm, xke, j2, j3, j4, j3oj2 = whichconst

    bstar = xbstar
    ndot = xndot
    nddot = xnddot
    ecco = xecco
    argpo = xargpo
    inclo = xinclo
    mo = xmo
    no_kozai = xno_kozai
    nodeo = xnodeo

    ss = 78.0 / radiusearthkm + 1.0
    qzms2ttemp = (120.0 - 78.0) / radiusearthkm
    qzms2t = qzms2ttemp * qzms2ttemp * qzms2ttemp * qzms2ttemp
    x2o3 = 2.0 / 3.0

    (no_unkozai, ainv, ao, con41, con42, cosio, cosio2, eccsq,
     omeosq, posq, rp, rteosq, sinio, gsto) = initl(
        xke, j2, ecco, epoch, inclo, no_kozai)

    a = (no_unkozai * tumin) ** (-2.0 / 3.0)
    alta = a * (1.0 + ecco) - 1.0
    altp = a * (1.0 - ecco) - 1.0

    # Branchless isimp: 1 if rp < 220/RE + 1, else 0
    isimp = jnp.where(rp < 220.0 / radiusearthkm + 1.0, 1.0, 0.0)

    sfour = ss
    qzms24 = qzms2t
    perige = (rp - 1.0) * radiusearthkm

    # Branchless perige adjustments
    sfour_low = jnp.where(perige < 98.0, 20.0, perige - 78.0)
    sfour = jnp.where(perige < 156.0, sfour_low, sfour)  # type: ignore[assignment]
    qzms24temp = (120.0 - sfour) / radiusearthkm
    qzms24_low = qzms24temp * qzms24temp * qzms24temp * qzms24temp
    sfour_adj = sfour / radiusearthkm + 1.0
    qzms24 = jnp.where(perige < 156.0, qzms24_low, qzms24)  # type: ignore[assignment]
    sfour = jnp.where(perige < 156.0, sfour_adj, sfour)  # type: ignore[assignment]

    pinvsq = 1.0 / posq
    tsi = 1.0 / (ao - sfour)
    eta = ao * ecco * tsi
    etasq = eta * eta
    eeta = ecco * eta
    psisq = jnp.abs(1.0 - etasq)
    coef = qzms24 * tsi ** 4.0
    coef1 = coef / psisq ** 3.5
    cc2 = coef1 * no_unkozai * (ao * (1.0 + 1.5 * etasq + eeta *
                  (4.0 + etasq)) + 0.375 * j2 * tsi / psisq * con41 *
                  (8.0 + 3.0 * etasq * (8.0 + etasq)))
    cc1 = bstar * cc2
    cc3_nonzero = -2.0 * coef * tsi * j3oj2 * no_unkozai * sinio / ecco
    cc3 = jnp.where(ecco > 1.0e-4, cc3_nonzero, 0.0)  # type: ignore[operator]
    x1mth2 = 1.0 - cosio2
    cc4 = 2.0 * no_unkozai * coef1 * ao * omeosq * \
                      (eta * (2.0 + 0.5 * etasq) + ecco *
                      (0.5 + 2.0 * etasq) - j2 * tsi / (ao * psisq) *
                      (-3.0 * con41 * (1.0 - 2.0 * eeta + etasq *
                      (1.5 - 0.5 * eeta)) + 0.75 * x1mth2 *
                      (2.0 * etasq - eeta * (1.0 + etasq)) * jnp.cos(2.0 * argpo)))
    cc5 = 2.0 * coef1 * ao * omeosq * (1.0 + 2.75 *
                  (etasq + eeta) + eeta * etasq)
    cosio4 = cosio2 * cosio2
    temp1 = 1.5 * j2 * pinvsq * no_unkozai
    temp2 = 0.5 * temp1 * j2 * pinvsq
    temp3 = -0.46875 * j4 * pinvsq * pinvsq * no_unkozai
    mdot = no_unkozai + 0.5 * temp1 * rteosq * con41 + 0.0625 * \
                      temp2 * rteosq * (13.0 - 78.0 * cosio2 + 137.0 * cosio4)
    argpdot = (-0.5 * temp1 * con42 + 0.0625 * temp2 *
                       (7.0 - 114.0 * cosio2 + 395.0 * cosio4) +
                       temp3 * (3.0 - 36.0 * cosio2 + 49.0 * cosio4))
    xhdot1 = -temp1 * cosio
    nodedot = xhdot1 + (0.5 * temp2 * (4.0 - 19.0 * cosio2) +
                        2.0 * temp3 * (3.0 - 7.0 * cosio2)) * cosio
    xpidot = argpdot + nodedot
    omgcof = bstar * cc3 * jnp.cos(argpo)
    xmcof_nonzero = -x2o3 * coef * bstar / eeta
    xmcof = jnp.where(ecco > 1.0e-4, xmcof_nonzero, 0.0)  # type: ignore[operator]
    nodecf = 3.5 * omeosq * xhdot1 * cc1
    t2cof = 1.5 * cc1

    xlcof_normal = -0.25 * j3oj2 * sinio * (3.0 + 5.0 * cosio) / (1.0 + cosio)
    xlcof_singular = -0.25 * j3oj2 * sinio * (3.0 + 5.0 * cosio) / temp4
    xlcof = jnp.where(jnp.abs(cosio + 1.0) > 1.5e-12, xlcof_normal, xlcof_singular)

    aycof = -0.5 * j3oj2 * sinio
    delmotemp = 1.0 + eta * jnp.cos(mo)
    delmo = delmotemp * delmotemp * delmotemp
    sinmao = jnp.sin(mo)
    x7thm1 = 7.0 * cosio2 - 1.0

    # Deep space initialization (method 1.0 if period >= 225 min)
    is_deep = twopi / no_unkozai >= 225.0
    method = jnp.where(is_deep, 1.0, 0.0)
    isimp = jnp.where(is_deep, 1.0, isimp)

    tc = 0.0
    inclm = inclo

    (snodm, cnodm, sinim, cosim, sinomm,
     cosomm, day, ds_e3, ds_ee2, ds_em,
     ds_emsq, ds_gam, ds_peo, ds_pgho, ds_pho,
     ds_pinco, ds_plo, ds_rtemsq, ds_se2, ds_se3,
     ds_sgh2, ds_sgh3, ds_sgh4, ds_sh2, ds_sh3,
     ds_si2, ds_si3, ds_sl2, ds_sl3, ds_sl4,
     ds_s1, ds_s2, ds_s3, ds_s4, ds_s5,
     ds_s6, ds_s7, ds_ss1, ds_ss2, ds_ss3,
     ds_ss4, ds_ss5, ds_ss6, ds_ss7, ds_sz1,
     ds_sz2, ds_sz3, ds_sz11, ds_sz12, ds_sz13,
     ds_sz21, ds_sz22, ds_sz23, ds_sz31, ds_sz32,
     ds_sz33, ds_xgh2, ds_xgh3, ds_xgh4, ds_xh2,
     ds_xh3, ds_xi2, ds_xi3, ds_xl2, ds_xl3,
     ds_xl4, ds_nm, ds_z1, ds_z2, ds_z3,
     ds_z11, ds_z12, ds_z13, ds_z21, ds_z22,
     ds_z23, ds_z31, ds_z32, ds_z33, ds_zmol,
     ds_zmos
     ) = dscom(
        epoch, ecco, argpo, tc, inclo, nodeo, no_unkozai)

    (dp_ecco, dp_inclo, dp_nodeo, dp_argpo, dp_mo
     ) = dpper_init(
        ds_e3, ds_ee2, ds_peo, ds_pgho, ds_pho, ds_pinco, ds_plo,
        ds_se2, ds_se3, ds_sgh2, ds_sgh3, ds_sgh4, ds_sh2, ds_sh3,
        ds_si2, ds_si3, ds_sl2, ds_sl3, ds_sl4,
        ds_xgh2, ds_xgh3, ds_xgh4, ds_xh2, ds_xh3,
        ds_xi2, ds_xi3, ds_xl2, ds_xl3, ds_xl4,
        ds_zmol, ds_zmos,
        inclo, ecco, inclo, nodeo, argpo, mo)

    # For deep space, use perturbed elements
    ecco = jnp.where(is_deep, dp_ecco, ecco)
    inclo = jnp.where(is_deep, dp_inclo, inclo)
    nodeo = jnp.where(is_deep, dp_nodeo, nodeo)
    argpo = jnp.where(is_deep, dp_argpo, argpo)
    mo = jnp.where(is_deep, dp_mo, mo)

    argpm = 0.0
    nodem = 0.0
    mm = 0.0

    (di_em, di_argpm, di_inclm, di_mm, di_nm, di_nodem,
     di_irez, di_atime,
     di_d2201, di_d2211, di_d3210, di_d3222,
     di_d4410, di_d4422, di_d5220, di_d5232,
     di_d5421, di_d5433, di_dedt, di_didt,
     di_dmdt, di_dndt, di_dnodt, di_domdt,
     di_del1, di_del2, di_del3, di_xfact,
     di_xlamo, di_xli, di_xni
     ) = dsinit(
        xke, cosim, ds_emsq, argpo, ds_s1, ds_s2, ds_s3, ds_s4, ds_s5, sinim,
        ds_ss1, ds_ss2, ds_ss3, ds_ss4, ds_ss5,
        ds_sz1, ds_sz3, ds_sz11, ds_sz13, ds_sz21, ds_sz23, ds_sz31, ds_sz33,
        0.0, tc, gsto, mo, mdot, no_unkozai, nodeo,
        nodedot, xpidot,
        ds_z1, ds_z3, ds_z11, ds_z13, ds_z21, ds_z23, ds_z31, ds_z33,
        ecco, eccsq, ds_em, argpm, inclm, mm, ds_nm, nodem)

    # Select deep space values or near-earth defaults (0.0)
    def ds_sel(deep_val, default=0.0):
        return jnp.where(is_deep, deep_val, default)

    d2201 = ds_sel(di_d2201); d2211 = ds_sel(di_d2211)
    d3210 = ds_sel(di_d3210); d3222 = ds_sel(di_d3222)
    d4410 = ds_sel(di_d4410); d4422 = ds_sel(di_d4422)
    d5220 = ds_sel(di_d5220); d5232 = ds_sel(di_d5232)
    d5421 = ds_sel(di_d5421); d5433 = ds_sel(di_d5433)
    dedt = ds_sel(di_dedt); didt = ds_sel(di_didt)
    dmdt = ds_sel(di_dmdt); dnodt = ds_sel(di_dnodt); domdt = ds_sel(di_domdt)
    del1 = ds_sel(di_del1); del2 = ds_sel(di_del2); del3 = ds_sel(di_del3)
    xfact = ds_sel(di_xfact); xlamo = ds_sel(di_xlamo)
    xli = ds_sel(di_xli); xni = ds_sel(di_xni)
    irez = ds_sel(di_irez); atime = ds_sel(di_atime)
    e3 = ds_sel(ds_e3); ee2 = ds_sel(ds_ee2)
    peo = ds_sel(ds_peo); pgho = ds_sel(ds_pgho); pho = ds_sel(ds_pho)
    pinco = ds_sel(ds_pinco); plo = ds_sel(ds_plo)
    se2 = ds_sel(ds_se2); se3 = ds_sel(ds_se3)
    sgh2 = ds_sel(ds_sgh2); sgh3 = ds_sel(ds_sgh3); sgh4 = ds_sel(ds_sgh4)
    sh2 = ds_sel(ds_sh2); sh3 = ds_sel(ds_sh3)
    si2 = ds_sel(ds_si2); si3 = ds_sel(ds_si3)
    sl2 = ds_sel(ds_sl2); sl3 = ds_sel(ds_sl3); sl4 = ds_sel(ds_sl4)
    xgh2 = ds_sel(ds_xgh2); xgh3 = ds_sel(ds_xgh3); xgh4 = ds_sel(ds_xgh4)
    xh2 = ds_sel(ds_xh2); xh3 = ds_sel(ds_xh3)
    xi2 = ds_sel(ds_xi2); xi3 = ds_sel(ds_xi3)
    xl2 = ds_sel(ds_xl2); xl3 = ds_sel(ds_xl3); xl4 = ds_sel(ds_xl4)
    zmol = ds_sel(ds_zmol); zmos = ds_sel(ds_zmos)

    # Near-earth higher-order drag terms (isimp == 0)
    cc1sq = cc1 * cc1
    d2_ne = 4.0 * ao * tsi * cc1sq
    temp_ne = d2_ne * tsi * cc1 / 3.0
    d3_ne = (17.0 * ao + sfour) * temp_ne
    d4_ne = 0.5 * temp_ne * ao * tsi * (221.0 * ao + 31.0 * sfour) * cc1
    t3cof_ne = d2_ne + 2.0 * cc1sq
    t4cof_ne = 0.25 * (3.0 * d3_ne + cc1 * (12.0 * d2_ne + 10.0 * cc1sq))
    t5cof_ne = 0.2 * (3.0 * d4_ne + 12.0 * cc1 * d3_ne +
                      6.0 * d2_ne * d2_ne + 15.0 * cc1sq * (2.0 * d2_ne + cc1sq))

    is_not_simp = isimp == 0.0
    d2 = jnp.where(is_not_simp, d2_ne, 0.0)
    d3 = jnp.where(is_not_simp, d3_ne, 0.0)
    d4 = jnp.where(is_not_simp, d4_ne, 0.0)
    t3cof = jnp.where(is_not_simp, t3cof_ne, 0.0)
    t4cof = jnp.where(is_not_simp, t4cof_ne, 0.0)
    t5cof = jnp.where(is_not_simp, t5cof_ne, 0.0)

    # Build the SatRec - use jnp.asarray to preserve JAX tracers
    def f(x):
        return jnp.asarray(x, dtype=jnp.float64)

    satrec = SatRec(
        bstar=f(bstar), ecco=f(ecco), argpo=f(argpo), inclo=f(inclo),
        mo=f(mo), no_kozai=f(no_kozai), nodeo=f(nodeo),
        ndot=f(ndot), nddot=f(nddot),
        j2=f(j2), j3=f(j3), j4=f(j4), j3oj2=f(j3oj2),
        xke=f(xke), mu=f(mu), radiusearthkm=f(radiusearthkm), tumin=f(tumin),
        jdsatepoch=f(jdsatepoch), jdsatepochF=f(jdsatepochF),
        no_unkozai=f(no_unkozai), a=f(a), alta=f(alta), altp=f(altp),
        con41=f(con41), cc1=f(cc1), cc4=f(cc4), cc5=f(cc5),
        d2=f(d2), d3=f(d3), d4=f(d4), delmo=f(delmo), eta=f(eta),
        argpdot=f(argpdot), omgcof=f(omgcof), sinmao=f(sinmao),
        t2cof=f(t2cof), t3cof=f(t3cof), t4cof=f(t4cof), t5cof=f(t5cof),
        x1mth2=f(x1mth2), x7thm1=f(x7thm1), mdot=f(mdot),
        nodedot=f(nodedot), xlcof=f(xlcof), xmcof=f(xmcof),
        nodecf=f(nodecf), aycof=f(aycof), gsto=f(gsto),
        d2201=f(d2201), d2211=f(d2211), d3210=f(d3210), d3222=f(d3222),
        d4410=f(d4410), d4422=f(d4422), d5220=f(d5220), d5232=f(d5232),
        d5421=f(d5421), d5433=f(d5433), dedt=f(dedt),
        del1=f(del1), del2=f(del2), del3=f(del3),
        didt=f(didt), dmdt=f(dmdt), dnodt=f(dnodt), domdt=f(domdt),
        e3=f(e3), ee2=f(ee2), peo=f(peo), pgho=f(pgho), pho=f(pho),
        pinco=f(pinco), plo=f(plo), se2=f(se2), se3=f(se3),
        sgh2=f(sgh2), sgh3=f(sgh3), sgh4=f(sgh4), sh2=f(sh2), sh3=f(sh3),
        si2=f(si2), si3=f(si3), sl2=f(sl2), sl3=f(sl3), sl4=f(sl4),
        xfact=f(xfact), xgh2=f(xgh2), xgh3=f(xgh3), xgh4=f(xgh4),
        xh2=f(xh2), xh3=f(xh3), xi2=f(xi2), xi3=f(xi3),
        xl2=f(xl2), xl3=f(xl3), xl4=f(xl4), xlamo=f(xlamo),
        xli=f(xli), xni=f(xni), zmol=f(zmol), zmos=f(zmos), atime=f(atime),
        method=f(method), isimp=f(isimp), irez=f(irez),
    )

    # Run propagation at t=0 to finalize (mirrors reference sgp4init calling sgp4(satrec, 0.0))
    from sgp4jax._propagation import sgp4 as _sgp4
    r, v, error = _sgp4(satrec, jnp.array(0.0))

    # For deep space satellites, the t=0 propagation recomputes some fields
    # from perturbed inclination. We need to capture those updates.
    # Recompute from the perturbed inclination at t=0
    from sgp4jax._dpper import dpper
    ep_ds = f(ecco)
    xincp_ds = f(inclo)
    nodep_ds = f(nodeo)
    argpp_ds = f(argpo)
    mp_ds = f(mo)

    ep_ds, xincp_ds, nodep_ds, argpp_ds, mp_ds = dpper(
        f(e3), f(ee2), f(peo), f(pgho), f(pho), f(pinco), f(plo),
        f(se2), f(se3), f(sgh2), f(sgh3), f(sgh4), f(sh2), f(sh3),
        f(si2), f(si3), f(sl2), f(sl3), f(sl4), jnp.array(0.0),
        f(xgh2), f(xgh3), f(xgh4), f(xh2), f(xh3),
        f(xi2), f(xi3), f(xl2), f(xl3), f(xl4),
        f(zmol), f(zmos),
        f(inclo), f(ecco),
        f(inclo), f(nodeo),
        f(argpo), f(mo))

    # Handle negative inclination
    xincp_neg = xincp_ds < 0.0
    xincp_ds = jnp.where(xincp_neg, -xincp_ds, xincp_ds)
    nodep_ds = jnp.where(xincp_neg, nodep_ds + pi, nodep_ds)
    argpp_ds = jnp.where(xincp_neg, argpp_ds - pi, argpp_ds)

    sinip = jnp.sin(xincp_ds)
    cosip = jnp.cos(xincp_ds)
    new_aycof = -0.5 * j3oj2 * sinip
    new_xlcof_normal = -0.25 * j3oj2 * sinip * (3.0 + 5.0 * cosip) / (1.0 + cosip)
    new_xlcof_singular = -0.25 * j3oj2 * sinip * (3.0 + 5.0 * cosip) / temp4
    new_xlcof = jnp.where(jnp.abs(cosip + 1.0) > 1.5e-12, new_xlcof_normal, new_xlcof_singular)
    cosisq = cosip * cosip
    new_con41 = 3.0 * cosisq - 1.0
    new_x1mth2 = 1.0 - cosisq
    new_x7thm1 = 7.0 * cosisq - 1.0

    satrec = satrec._replace(
        aycof=jnp.where(is_deep, f(new_aycof), satrec.aycof),
        xlcof=jnp.where(is_deep, f(new_xlcof), satrec.xlcof),
        con41=jnp.where(is_deep, f(new_con41), satrec.con41),
        x1mth2=jnp.where(is_deep, f(new_x1mth2), satrec.x1mth2),
        x7thm1=jnp.where(is_deep, f(new_x7thm1), satrec.x7thm1),
    )

    return satrec
