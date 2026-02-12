"""sgp4init - initialize SGP4 propagator from orbital elements."""

from math import cos, fabs, pi, pow, sin, sqrt
import jax.numpy as jnp

from sgp4jax._types import SatRec
from sgp4jax._initl import initl
from sgp4jax._dscom import dscom
from sgp4jax._dpper import dpper_init
from sgp4jax._dsinit import dsinit

twopi = 2.0 * pi


def sgp4init(whichconst, epoch, xbstar, xndot, xnddot, xecco, xargpo,
             xinclo, xmo, xno_kozai, xnodeo, jdsatepoch, jdsatepochF):
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

    a = pow(no_unkozai * tumin, -2.0 / 3.0)
    alta = a * (1.0 + ecco) - 1.0
    altp = a * (1.0 - ecco) - 1.0

    # Initialize all coefficients
    isimp = 0
    method = 0.0  # 0.0 = near-earth
    irez = 0.0

    cc1 = 0.0; cc4 = 0.0; cc5 = 0.0
    d2 = 0.0; d3 = 0.0; d4 = 0.0
    delmo = 0.0; eta = 0.0; argpdot = 0.0; omgcof = 0.0
    sinmao = 0.0; t2cof = 0.0; t3cof = 0.0; t4cof = 0.0; t5cof = 0.0
    x1mth2 = 0.0; x7thm1 = 0.0; mdot = 0.0; nodedot = 0.0
    xlcof = 0.0; xmcof = 0.0; nodecf = 0.0; aycof = 0.0

    # Deep space vars
    d2201 = 0.0; d2211 = 0.0; d3210 = 0.0; d3222 = 0.0
    d4410 = 0.0; d4422 = 0.0; d5220 = 0.0; d5232 = 0.0
    d5421 = 0.0; d5433 = 0.0; dedt = 0.0; del1 = 0.0; del2 = 0.0; del3 = 0.0
    didt = 0.0; dmdt = 0.0; dnodt = 0.0; domdt = 0.0
    e3 = 0.0; ee2 = 0.0; peo = 0.0; pgho = 0.0; pho = 0.0
    pinco = 0.0; plo = 0.0; se2 = 0.0; se3 = 0.0
    sgh2 = 0.0; sgh3 = 0.0; sgh4 = 0.0; sh2 = 0.0; sh3 = 0.0
    si2 = 0.0; si3 = 0.0; sl2 = 0.0; sl3 = 0.0; sl4 = 0.0
    xfact = 0.0; xgh2 = 0.0; xgh3 = 0.0; xgh4 = 0.0
    xh2 = 0.0; xh3 = 0.0; xi2 = 0.0; xi3 = 0.0
    xl2 = 0.0; xl3 = 0.0; xl4 = 0.0; xlamo = 0.0
    xli = 0.0; xni = 0.0; zmol = 0.0; zmos = 0.0; atime = 0.0

    if omeosq >= 0.0 or no_unkozai >= 0.0:

        if rp < 220.0 / radiusearthkm + 1.0:
            isimp = 1
        sfour = ss
        qzms24 = qzms2t
        perige = (rp - 1.0) * radiusearthkm

        if perige < 156.0:
            sfour = perige - 78.0
            if perige < 98.0:
                sfour = 20.0
            qzms24temp = (120.0 - sfour) / radiusearthkm
            qzms24 = qzms24temp * qzms24temp * qzms24temp * qzms24temp
            sfour = sfour / radiusearthkm + 1.0

        pinvsq = 1.0 / posq
        tsi = 1.0 / (ao - sfour)
        eta = ao * ecco * tsi
        etasq = eta * eta
        eeta = ecco * eta
        psisq = fabs(1.0 - etasq)
        coef = qzms24 * pow(tsi, 4.0)
        coef1 = coef / pow(psisq, 3.5)
        cc2 = coef1 * no_unkozai * (ao * (1.0 + 1.5 * etasq + eeta *
                      (4.0 + etasq)) + 0.375 * j2 * tsi / psisq * con41 *
                      (8.0 + 3.0 * etasq * (8.0 + etasq)))
        cc1 = bstar * cc2
        cc3 = 0.0
        if ecco > 1.0e-4:
            cc3 = -2.0 * coef * tsi * j3oj2 * no_unkozai * sinio / ecco
        x1mth2 = 1.0 - cosio2
        cc4 = 2.0 * no_unkozai * coef1 * ao * omeosq * \
                          (eta * (2.0 + 0.5 * etasq) + ecco *
                          (0.5 + 2.0 * etasq) - j2 * tsi / (ao * psisq) *
                          (-3.0 * con41 * (1.0 - 2.0 * eeta + etasq *
                          (1.5 - 0.5 * eeta)) + 0.75 * x1mth2 *
                          (2.0 * etasq - eeta * (1.0 + etasq)) * cos(2.0 * argpo)))
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
        omgcof = bstar * cc3 * cos(argpo)
        xmcof = 0.0
        if ecco > 1.0e-4:
            xmcof = -x2o3 * coef * bstar / eeta
        nodecf = 3.5 * omeosq * xhdot1 * cc1
        t2cof = 1.5 * cc1

        if fabs(cosio + 1.0) > 1.5e-12:
            xlcof = -0.25 * j3oj2 * sinio * (3.0 + 5.0 * cosio) / (1.0 + cosio)
        else:
            xlcof = -0.25 * j3oj2 * sinio * (3.0 + 5.0 * cosio) / temp4

        aycof = -0.5 * j3oj2 * sinio
        delmotemp = 1.0 + eta * cos(mo)
        delmo = delmotemp * delmotemp * delmotemp
        sinmao = sin(mo)
        x7thm1 = 7.0 * cosio2 - 1.0

        # Deep space initialization
        if 2 * pi / no_unkozai >= 225.0:
            method = 1.0  # deep-space
            isimp = 1
            tc = 0.0
            inclm = inclo

            (snodm, cnodm, sinim, cosim, sinomm,
             cosomm, day, e3, ee2, em,
             emsq, gam, peo, pgho, pho,
             pinco, plo, rtemsq, se2, se3,
             sgh2, sgh3, sgh4, sh2, sh3,
             si2, si3, sl2, sl3, sl4,
             s1, s2, s3, s4, s5,
             s6, s7, ss1, ss2, ss3,
             ss4, ss5, ss6, ss7, sz1,
             sz2, sz3, sz11, sz12, sz13,
             sz21, sz22, sz23, sz31, sz32,
             sz33, xgh2, xgh3, xgh4, xh2,
             xh3, xi2, xi3, xl2, xl3,
             xl4, nm, z1, z2, z3,
             z11, z12, z13, z21, z22,
             z23, z31, z32, z33, zmol,
             zmos
             ) = dscom(
                epoch, ecco, argpo, tc, inclo, nodeo, no_unkozai)

            (ecco, inclo, nodeo, argpo, mo
             ) = dpper_init(
                e3, ee2, peo, pgho, pho, pinco, plo,
                se2, se3, sgh2, sgh3, sgh4, sh2, sh3,
                si2, si3, sl2, sl3, sl4,
                xgh2, xgh3, xgh4, xh2, xh3, xi2, xi3, xl2, xl3, xl4,
                zmol, zmos,
                inclo, ecco, inclo, nodeo, argpo, mo)

            argpm = 0.0
            nodem = 0.0
            mm = 0.0

            (em, argpm, inclm, mm, nm, nodem,
             irez, atime,
             d2201, d2211, d3210, d3222,
             d4410, d4422, d5220, d5232,
             d5421, d5433, dedt, didt,
             dmdt, dndt, dnodt, domdt,
             del1, del2, del3, xfact,
             xlamo, xli, xni
             ) = dsinit(
                xke, cosim, emsq, argpo, s1, s2, s3, s4, s5, sinim,
                ss1, ss2, ss3, ss4, ss5,
                sz1, sz3, sz11, sz13, sz21, sz23, sz31, sz33,
                0.0, tc, gsto, mo, mdot, no_unkozai, nodeo,
                nodedot, xpidot,
                z1, z3, z11, z13, z21, z23, z31, z33,
                ecco, eccsq, em, argpm, inclm, mm, nm, nodem)

            irez = float(irez)

        # Near-earth higher-order drag terms
        if isimp != 1:
            cc1sq = cc1 * cc1
            d2 = 4.0 * ao * tsi * cc1sq
            temp = d2 * tsi * cc1 / 3.0
            d3 = (17.0 * ao + sfour) * temp
            d4 = 0.5 * temp * ao * tsi * (221.0 * ao + 31.0 * sfour) * cc1
            t3cof = d2 + 2.0 * cc1sq
            t4cof = 0.25 * (3.0 * d3 + cc1 * (12.0 * d2 + 10.0 * cc1sq))
            t5cof = 0.2 * (3.0 * d4 + 12.0 * cc1 * d3 +
                          6.0 * d2 * d2 + 15.0 * cc1sq * (2.0 * d2 + cc1sq))

    # Build the SatRec
    def f(x):
        return jnp.array(float(x))

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
        method=f(method), isimp=f(float(isimp)), irez=f(float(irez)),
    )

    # Run propagation at t=0 to finalize (mirrors reference sgp4init calling sgp4(satrec, 0.0))
    # For deep-space, this updates aycof, xlcof, con41, x1mth2, x7thm1
    from sgp4jax._propagation import sgp4 as _sgp4
    r, v, error = _sgp4(satrec, jnp.array(0.0))

    # For deep space satellites, the t=0 propagation recomputes some fields
    # from perturbed inclination. We need to capture those updates.
    if method == 1.0:  # deep space
        # Recompute from the perturbed inclination at t=0
        # This mirrors lines 1846-1855 of propagation.py
        from sgp4jax._dpper import dpper
        ep = ecco
        xincp = inclo
        nodep_ = nodeo
        argpp_ = argpo
        mp_ = mo

        ep, xincp, nodep_, argpp_, mp_ = dpper(
            e3, ee2, peo, pgho, pho, pinco, plo,
            se2, se3, sgh2, sgh3, sgh4, sh2, sh3,
            si2, si3, sl2, sl3, sl4, jnp.array(0.0),
            xgh2, xgh3, xgh4, xh2, xh3, xi2, xi3, xl2, xl3, xl4,
            zmol, zmos,
            inclo, jnp.array(float(ecco)),
            jnp.array(float(inclo)), jnp.array(float(nodeo)),
            jnp.array(float(argpo)), jnp.array(float(mo)))

        if float(xincp) < 0.0:
            xincp = -xincp
            nodep_ = nodep_ + pi
            argpp_ = argpp_ - pi

        sinip = jnp.sin(xincp)
        cosip = jnp.cos(xincp)
        new_aycof = -0.5 * j3oj2 * sinip
        cosip_val = float(cosip)
        sinip_val = float(sinip)
        if fabs(cosip_val + 1.0) > 1.5e-12:
            new_xlcof = -0.25 * j3oj2 * sinip_val * (3.0 + 5.0 * cosip_val) / (1.0 + cosip_val)
        else:
            new_xlcof = -0.25 * j3oj2 * sinip_val * (3.0 + 5.0 * cosip_val) / temp4
        cosisq = cosip * cosip
        new_con41 = 3.0 * cosisq - 1.0
        new_x1mth2 = 1.0 - cosisq
        new_x7thm1 = 7.0 * cosisq - 1.0

        satrec = satrec._replace(
            aycof=jnp.array(float(new_aycof)),
            xlcof=jnp.array(float(new_xlcof)),
            con41=jnp.array(float(new_con41)),
            x1mth2=jnp.array(float(new_x1mth2)),
            x7thm1=jnp.array(float(new_x7thm1)),
        )

    return satrec
