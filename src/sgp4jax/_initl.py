"""_initl - compute auxiliary orbital quantities and Greenwich sidereal time."""

from math import atan2, cos, pi, pow, sin, sqrt

twopi = 2.0 * pi


def gstime(jdut1):
    """Greenwich sidereal time from Julian date."""
    tut1 = (jdut1 - 2451545.0) / 36525.0
    temp = (-6.2e-6 * tut1 * tut1 * tut1 + 0.093104 * tut1 * tut1 +
            (876600.0 * 3600 + 8640184.812866) * tut1 + 67310.54841)
    deg2rad = pi / 180.0
    temp = (temp * deg2rad / 240.0) % twopi

    if temp < 0.0:
        temp += twopi

    return temp


def initl(xke, j2, ecco, epoch, inclo, no):
    """Initialize auxiliary orbital quantities.

    Returns:
        (no_unkozai, ainv, ao, con41, con42, cosio, cosio2, eccsq,
         omeosq, posq, rp, rteosq, sinio, gsto)
    """
    x2o3 = 2.0 / 3.0

    # Calculate auxiliary epoch quantities
    eccsq = ecco * ecco
    omeosq = 1.0 - eccsq
    rteosq = sqrt(omeosq)
    cosio = cos(inclo)
    cosio2 = cosio * cosio

    # Un-Kozai the mean motion
    ak = pow(xke / no, x2o3)
    d1 = 0.75 * j2 * (3.0 * cosio2 - 1.0) / (rteosq * omeosq)
    del_ = d1 / (ak * ak)
    adel = ak * (1.0 - del_ * del_ - del_ *
            (1.0 / 3.0 + 134.0 * del_ * del_ / 81.0))
    del_ = d1 / (adel * adel)
    no = no / (1.0 + del_)

    ao = pow(xke / no, x2o3)
    sinio = sin(inclo)
    po = ao * omeosq
    con42 = 1.0 - 5.0 * cosio2
    con41 = -con42 - cosio2 - cosio2
    ainv = 1.0 / ao
    posq = po * po
    rp = ao * (1.0 - ecco)

    # Improved mode sidereal time
    gsto = gstime(epoch + 2433281.5)

    return (no, ainv, ao, con41, con42, cosio, cosio2, eccsq,
            omeosq, posq, rp, rteosq, sinio, gsto)
