"""_dsinit - deep space initialization for resonance terms."""

import jax
import jax.numpy as jnp
import jax.typing
from math import pi

twopi = 2.0 * pi

_A = jax.typing.ArrayLike


def dsinit(
    xke: _A, cosim: _A, emsq: _A, argpo: _A, s1: _A, s2: _A, s3: _A, s4: _A, s5: _A, sinim: _A,
    ss1: _A, ss2: _A, ss3: _A, ss4: _A, ss5: _A,
    sz1: _A, sz3: _A, sz11: _A, sz13: _A, sz21: _A, sz23: _A, sz31: _A, sz33: _A,
    t: _A, tc: _A, gsto: _A, mo: _A, mdot: _A, no: _A, nodeo: _A, nodedot: _A, xpidot: _A,
    z1: _A, z3: _A, z11: _A, z13: _A, z21: _A, z23: _A, z31: _A, z33: _A,
    ecco: _A, eccsq: _A, em: _A, argpm: _A, inclm: _A, mm: _A, nm: _A, nodem: _A,
) -> tuple[jax.Array, ...]:
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
    # irez: 0 = no resonance, 1 = synchronous, 2 = half-day
    irez_is_1 = (nm > 0.0034906585) & (nm < 0.0052359877)  # type: ignore[operator]
    irez_is_2 = (nm >= 8.26e-3) & (nm <= 9.24e-3) & (em >= 0.5)  # type: ignore[operator]
    irez = jnp.where(irez_is_2, 2.0, jnp.where(irez_is_1, 1.0, 0.0))

    # Solar terms
    ses = ss1 * zns * ss5
    sis = ss2 * zns * (sz11 + sz13)
    sls = -zns * ss3 * (sz1 + sz3 - 14.0 - 6.0 * emsq)
    sghs = ss4 * zns * (sz31 + sz33 - 6.0)
    shs_base = -zns * ss2 * (sz21 + sz23)
    # Zero out shs for near-polar orbits
    near_polar = (inclm < 5.2359877e-2) | (inclm > pi - 5.2359877e-2)  # type: ignore[operator]
    shs = jnp.where(near_polar, 0.0, shs_base)
    shs = jnp.where((sinim != 0.0) & jnp.logical_not(near_polar), shs / sinim, shs)
    sgs = sghs - cosim * shs

    # Lunar terms
    dedt = ses + s1 * znl * s5
    didt = sis + s2 * znl * (z11 + z13)
    dmdt = sls - znl * s3 * (z1 + z3 - 14.0 - 6.0 * emsq)
    sghl = s4 * znl * (z31 + z33 - 6.0)
    shll_base = -znl * s2 * (z21 + z23)
    shll = jnp.where(near_polar, 0.0, shll_base)
    domdt = sgs + sghl
    dnodt = shs
    domdt = jnp.where(sinim != 0.0, domdt - cosim / sinim * shll, domdt)
    dnodt = jnp.where(sinim != 0.0, dnodt + shll / sinim, dnodt)

    # Deep space resonance effects
    dndt = 0.0
    theta = (gsto + tc * rptim) % twopi  # type: ignore[operator]
    em = em + dedt * t
    inclm = inclm + didt * t
    argpm = argpm + domdt * t
    nodem = nodem + dnodt * t
    mm = mm + dmdt * t

    # Initialize resonance coefficients (defaults for irez == 0)
    d2201_0 = 0.0; d2211_0 = 0.0; d3210_0 = 0.0; d3222_0 = 0.0
    d4410_0 = 0.0; d4422_0 = 0.0; d5220_0 = 0.0; d5232_0 = 0.0
    d5421_0 = 0.0; d5433_0 = 0.0
    del1_0 = 0.0; del2_0 = 0.0; del3_0 = 0.0
    xfact_0 = 0.0; xlamo_0 = 0.0; xli_0 = 0.0; xni_0 = 0.0; atime_0 = 0.0

    # Compute resonance terms (always computed, selected via jnp.where)
    aonv = (nm / xke) ** x2o3

    # --- Half-day resonance (irez == 2) ---
    cosisq = cosim * cosim
    emo = em
    emsqo = emsq
    eoc = ecco * eccsq
    g201 = -0.306 - (ecco - 0.64) * 0.440

    # Two sub-branches based on ecco <= 0.65
    g211_a = 3.616 - 13.2470 * ecco + 16.2900 * eccsq
    g310_a = -19.302 + 117.3900 * ecco - 228.4190 * eccsq + 156.5910 * eoc
    g322_a = -18.9068 + 109.7927 * ecco - 214.6334 * eccsq + 146.5816 * eoc
    g410_a = -41.122 + 242.6940 * ecco - 471.0940 * eccsq + 313.9530 * eoc
    g422_a = -146.407 + 841.8800 * ecco - 1629.014 * eccsq + 1083.4350 * eoc
    g520_a = -532.114 + 3017.977 * ecco - 5740.032 * eccsq + 3708.2760 * eoc

    g211_b = -72.099 + 331.819 * ecco - 508.738 * eccsq + 266.724 * eoc
    g310_b = -346.844 + 1582.851 * ecco - 2415.925 * eccsq + 1246.113 * eoc
    g322_b = -342.585 + 1554.908 * ecco - 2366.899 * eccsq + 1215.972 * eoc
    g410_b = -1052.797 + 4758.686 * ecco - 7193.992 * eccsq + 3651.957 * eoc
    g422_b = -3581.690 + 16178.110 * ecco - 24462.770 * eccsq + 12422.520 * eoc
    g520_b1 = -5149.66 + 29936.92 * ecco - 54087.36 * eccsq + 31324.56 * eoc
    g520_b2 = 1464.74 - 4664.75 * ecco + 3763.64 * eccsq
    g520_b = jnp.where(ecco > 0.715, g520_b1, g520_b2)  # type: ignore[operator]

    low_ecc = ecco <= 0.65  # type: ignore[operator]
    g211 = jnp.where(low_ecc, g211_a, g211_b)
    g310 = jnp.where(low_ecc, g310_a, g310_b)
    g322 = jnp.where(low_ecc, g322_a, g322_b)
    g410 = jnp.where(low_ecc, g410_a, g410_b)
    g422 = jnp.where(low_ecc, g422_a, g422_b)
    g520 = jnp.where(low_ecc, g520_a, g520_b)

    g533_a = -919.22770 + 4988.6100 * ecco - 9064.7700 * eccsq + 5542.21 * eoc
    g521_a = -822.71072 + 4568.6173 * ecco - 8491.4146 * eccsq + 5337.524 * eoc
    g532_a = -853.66600 + 4690.2500 * ecco - 8624.7700 * eccsq + 5341.4 * eoc
    g533_b = -37995.780 + 161616.52 * ecco - 229838.20 * eccsq + 109377.94 * eoc
    g521_b = -51752.104 + 218913.95 * ecco - 309468.16 * eccsq + 146349.42 * eoc
    g532_b = -40023.880 + 170470.89 * ecco - 242699.48 * eccsq + 115605.82 * eoc
    low_ecc2 = ecco < 0.7  # type: ignore[operator]
    g533 = jnp.where(low_ecc2, g533_a, g533_b)
    g521 = jnp.where(low_ecc2, g521_a, g521_b)
    g532 = jnp.where(low_ecc2, g532_a, g532_b)

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
    d2201_2 = temp * f220 * g201
    d2211_2 = temp * f221 * g211
    temp1_2 = temp1 * aonv
    temp = temp1_2 * root32
    d3210_2 = temp * f321 * g310
    d3222_2 = temp * f322 * g322
    temp1_2 = temp1_2 * aonv
    temp = 2.0 * temp1_2 * root44
    d4410_2 = temp * f441 * g410
    d4422_2 = temp * f442 * g422
    temp1_2 = temp1_2 * aonv
    temp = temp1_2 * root52
    d5220_2 = temp * f522 * g520
    d5232_2 = temp * f523 * g532
    temp = 2.0 * temp1_2 * root54
    d5421_2 = temp * f542 * g521
    d5433_2 = temp * f543 * g533
    xlamo_2 = (mo + nodeo + nodeo - theta - theta) % twopi  # type: ignore[operator]
    xfact_2 = mdot + dmdt + 2.0 * (nodedot + dnodt - rptim) - no

    # --- Synchronous resonance (irez == 1) ---
    g200 = 1.0 + emsq * (-2.5 + 0.8125 * emsq)
    g310_1 = 1.0 + 2.0 * emsq
    g300 = 1.0 + emsq * (-6.0 + 6.60937 * emsq)
    f220_1 = 0.75 * (1.0 + cosim) * (1.0 + cosim)
    f311 = (0.9375 * sinim * sinim * (1.0 + 3.0 * cosim) -
            0.75 * (1.0 + cosim))
    f330 = 1.0 + cosim
    f330 = 1.875 * f330 * f330 * f330
    del1_1 = 3.0 * nm * nm * aonv * aonv
    del2_1 = 2.0 * del1_1 * f220_1 * g200 * q22
    del3_1 = 3.0 * del1_1 * f330 * g300 * q33 * aonv
    del1_1 = del1_1 * f311 * g310_1 * q31 * aonv
    xlamo_1 = (mo + nodeo + argpo - theta) % twopi  # type: ignore[operator]
    xfact_1 = mdot + xpidot - rptim + dmdt + domdt + dnodt - no

    # Select based on resonance type
    is_rez1 = irez == 1.0
    is_rez2 = irez == 2.0
    has_rez = is_rez1 | is_rez2

    d2201 = jnp.where(is_rez2, d2201_2, d2201_0)
    d2211 = jnp.where(is_rez2, d2211_2, d2211_0)
    d3210 = jnp.where(is_rez2, d3210_2, d3210_0)
    d3222 = jnp.where(is_rez2, d3222_2, d3222_0)
    d4410 = jnp.where(is_rez2, d4410_2, d4410_0)
    d4422 = jnp.where(is_rez2, d4422_2, d4422_0)
    d5220 = jnp.where(is_rez2, d5220_2, d5220_0)
    d5232 = jnp.where(is_rez2, d5232_2, d5232_0)
    d5421 = jnp.where(is_rez2, d5421_2, d5421_0)
    d5433 = jnp.where(is_rez2, d5433_2, d5433_0)
    del1 = jnp.where(is_rez1, del1_1, del1_0)
    del2 = jnp.where(is_rez1, del2_1, del2_0)
    del3 = jnp.where(is_rez1, del3_1, del3_0)
    xlamo = jnp.where(is_rez2, xlamo_2, jnp.where(is_rez1, xlamo_1, xlamo_0))
    xfact = jnp.where(is_rez2, xfact_2, jnp.where(is_rez1, xfact_1, xfact_0))

    xli = jnp.where(has_rez, xlamo, xli_0)
    xni = jnp.where(has_rez, no, xni_0)
    atime = jnp.where(has_rez, 0.0, atime_0)
    nm = jnp.where(has_rez, no + dndt, nm)

    return (em, argpm, inclm, mm, nm, nodem,  # type: ignore[return-value]
            irez, atime,
            d2201, d2211, d3210, d3222,
            d4410, d4422, d5220, d5232,
            d5421, d5433, dedt, didt,
            dmdt, dndt, dnodt, domdt,
            del1, del2, del3, xfact,
            xlamo, xli, xni)
