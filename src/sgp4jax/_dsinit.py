"""_dsinit - deep space initialization for resonance terms."""

from math import cos, fabs, pi, pow, sin

twopi = 2.0 * pi


def dsinit(xke, cosim, emsq, argpo, s1, s2, s3, s4, s5, sinim,
           ss1, ss2, ss3, ss4, ss5,
           sz1, sz3, sz11, sz13, sz21, sz23, sz31, sz33,
           t, tc, gsto, mo, mdot, no, nodeo, nodedot, xpidot,
           z1, z3, z11, z13, z21, z23, z31, z33,
           ecco, eccsq, em, argpm, inclm, mm, nm, nodem):
    """Deep space initialization for resonance effects.

    Returns tuple of computed coefficients and updated elements.
    """
    q22 = 1.7891679e-6
    q31 = 2.1460748e-6
    q33 = 2.2123015e-7
    root22 = 1.7891679e-6
    root44 = 7.3636953e-9
    root54 = 2.1765803e-9
    rptim = 4.37526908801129966e-3
    root32 = 3.7393792e-7
    root52 = 1.1428639e-7
    x2o3 = 2.0 / 3.0
    znl = 1.5835218e-4
    zns = 1.19459e-5

    # Deep space initialization
    irez = 0
    if 0.0034906585 < nm < 0.0052359877:
        irez = 1
    if 8.26e-3 <= nm <= 9.24e-3 and em >= 0.5:
        irez = 2

    # Solar terms
    ses = ss1 * zns * ss5
    sis = ss2 * zns * (sz11 + sz13)
    sls = -zns * ss3 * (sz1 + sz3 - 14.0 - 6.0 * emsq)
    sghs = ss4 * zns * (sz31 + sz33 - 6.0)
    shs = -zns * ss2 * (sz21 + sz23)
    if inclm < 5.2359877e-2 or inclm > pi - 5.2359877e-2:
        shs = 0.0
    if sinim != 0.0:
        shs = shs / sinim
    sgs = sghs - cosim * shs

    # Lunar terms
    dedt = ses + s1 * znl * s5
    didt = sis + s2 * znl * (z11 + z13)
    dmdt = sls - znl * s3 * (z1 + z3 - 14.0 - 6.0 * emsq)
    sghl = s4 * znl * (z31 + z33 - 6.0)
    shll = -znl * s2 * (z21 + z23)
    if inclm < 5.2359877e-2 or inclm > pi - 5.2359877e-2:
        shll = 0.0
    domdt = sgs + sghl
    dnodt = shs
    if sinim != 0.0:
        domdt = domdt - cosim / sinim * shll
        dnodt = dnodt + shll / sinim

    # Deep space resonance effects
    dndt = 0.0
    theta = (gsto + tc * rptim) % twopi
    em = em + dedt * t
    inclm = inclm + didt * t
    argpm = argpm + domdt * t
    nodem = nodem + dnodt * t
    mm = mm + dmdt * t

    # Initialize resonance coefficients
    d2201 = 0.0; d2211 = 0.0; d3210 = 0.0; d3222 = 0.0
    d4410 = 0.0; d4422 = 0.0; d5220 = 0.0; d5232 = 0.0
    d5421 = 0.0; d5433 = 0.0
    del1 = 0.0; del2 = 0.0; del3 = 0.0
    xfact = 0.0; xlamo = 0.0; xli = 0.0; xni = 0.0; atime = 0.0

    if irez != 0:
        aonv = pow(nm / xke, x2o3)

        # Geopotential resonance for 12-hour orbits
        if irez == 2:
            cosisq = cosim * cosim
            emo = em
            em = ecco
            emsqo = emsq
            emsq = eccsq
            eoc = em * emsq
            g201 = -0.306 - (em - 0.64) * 0.440

            if em <= 0.65:
                g211 = 3.616 - 13.2470 * em + 16.2900 * emsq
                g310 = -19.302 + 117.3900 * em - 228.4190 * emsq + 156.5910 * eoc
                g322 = -18.9068 + 109.7927 * em - 214.6334 * emsq + 146.5816 * eoc
                g410 = -41.122 + 242.6940 * em - 471.0940 * emsq + 313.9530 * eoc
                g422 = -146.407 + 841.8800 * em - 1629.014 * emsq + 1083.4350 * eoc
                g520 = -532.114 + 3017.977 * em - 5740.032 * emsq + 3708.2760 * eoc
            else:
                g211 = -72.099 + 331.819 * em - 508.738 * emsq + 266.724 * eoc
                g310 = -346.844 + 1582.851 * em - 2415.925 * emsq + 1246.113 * eoc
                g322 = -342.585 + 1554.908 * em - 2366.899 * emsq + 1215.972 * eoc
                g410 = -1052.797 + 4758.686 * em - 7193.992 * emsq + 3651.957 * eoc
                g422 = -3581.690 + 16178.110 * em - 24462.770 * emsq + 12422.520 * eoc
                if em > 0.715:
                    g520 = -5149.66 + 29936.92 * em - 54087.36 * emsq + 31324.56 * eoc
                else:
                    g520 = 1464.74 - 4664.75 * em + 3763.64 * emsq

            if em < 0.7:
                g533 = -919.22770 + 4988.6100 * em - 9064.7700 * emsq + 5542.21 * eoc
                g521 = -822.71072 + 4568.6173 * em - 8491.4146 * emsq + 5337.524 * eoc
                g532 = -853.66600 + 4690.2500 * em - 8624.7700 * emsq + 5341.4 * eoc
            else:
                g533 = -37995.780 + 161616.52 * em - 229838.20 * emsq + 109377.94 * eoc
                g521 = -51752.104 + 218913.95 * em - 309468.16 * emsq + 146349.42 * eoc
                g532 = -40023.880 + 170470.89 * em - 242699.48 * emsq + 115605.82 * eoc

            sini2 = sinim * sinim
            f220 = 0.75 * (1.0 + 2.0 * cosim + cosisq)
            f221 = 1.5 * sini2
            f321 = 1.875 * sinim * (1.0 - 2.0 * cosim - 3.0 * cosisq)
            f322 = -1.875 * sinim * (1.0 + 2.0 * cosim - 3.0 * cosisq)
            f441 = 35.0 * sini2 * f220
            f442 = 39.3750 * sini2 * sini2
            f522 = 9.84375 * sinim * (sini2 * (1.0 - 2.0 * cosim - 5.0 * cosisq) +
                    0.33333333 * (-2.0 + 4.0 * cosim + 6.0 * cosisq))
            f523 = sinim * (4.92187512 * sini2 * (-2.0 - 4.0 * cosim +
                   10.0 * cosisq) + 6.56250012 * (1.0 + 2.0 * cosim - 3.0 * cosisq))
            f542 = 29.53125 * sinim * (2.0 - 8.0 * cosim + cosisq *
                   (-12.0 + 8.0 * cosim + 10.0 * cosisq))
            f543 = 29.53125 * sinim * (-2.0 - 8.0 * cosim + cosisq *
                   (12.0 + 8.0 * cosim - 10.0 * cosisq))
            xno2 = nm * nm
            ainv2 = aonv * aonv
            temp1 = 3.0 * xno2 * ainv2
            temp = temp1 * root22
            d2201 = temp * f220 * g201
            d2211 = temp * f221 * g211
            temp1 = temp1 * aonv
            temp = temp1 * root32
            d3210 = temp * f321 * g310
            d3222 = temp * f322 * g322
            temp1 = temp1 * aonv
            temp = 2.0 * temp1 * root44
            d4410 = temp * f441 * g410
            d4422 = temp * f442 * g422
            temp1 = temp1 * aonv
            temp = temp1 * root52
            d5220 = temp * f522 * g520
            d5232 = temp * f523 * g532
            temp = 2.0 * temp1 * root54
            d5421 = temp * f542 * g521
            d5433 = temp * f543 * g533
            xlamo = (mo + nodeo + nodeo - theta - theta) % twopi
            xfact = mdot + dmdt + 2.0 * (nodedot + dnodt - rptim) - no
            em = emo
            emsq = emsqo

        # Synchronous resonance terms
        if irez == 1:
            g200 = 1.0 + emsq * (-2.5 + 0.8125 * emsq)
            g310 = 1.0 + 2.0 * emsq
            g300 = 1.0 + emsq * (-6.0 + 6.60937 * emsq)
            f220 = 0.75 * (1.0 + cosim) * (1.0 + cosim)
            f311 = (0.9375 * sinim * sinim * (1.0 + 3.0 * cosim) -
                    0.75 * (1.0 + cosim))
            f330 = 1.0 + cosim
            f330 = 1.875 * f330 * f330 * f330
            del1 = 3.0 * nm * nm * aonv * aonv
            del2 = 2.0 * del1 * f220 * g200 * q22
            del3 = 3.0 * del1 * f330 * g300 * q33 * aonv
            del1 = del1 * f311 * g310 * q31 * aonv
            xlamo = (mo + nodeo + argpo - theta) % twopi
            xfact = mdot + xpidot - rptim + dmdt + domdt + dnodt - no

        # Initialize the integrator
        xli = xlamo
        xni = no
        atime = 0.0
        nm = no + dndt

    return (em, argpm, inclm, mm, nm, nodem,
            irez, atime,
            d2201, d2211, d3210, d3222,
            d4410, d4422, d5220, d5232,
            d5421, d5433, dedt, didt,
            dmdt, dndt, dnodt, domdt,
            del1, del2, del3, xfact,
            xlamo, xli, xni)
