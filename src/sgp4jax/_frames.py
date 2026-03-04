"""TEME to GCRF frame transformation.

Implements the IAU-2006 precession, IAU-2000A nutation (678 lunisolar +
687 planetary terms), ICRS-to-J2000 frame bias, and SGP4-specific
GMST1982/GAST equation to convert SGP4 TEME output to the GCRF (≈ICRS)
frame.  Matches Skyfield's ``EarthSatellite._at()`` to sub-millimetre
precision.

The algorithms and nutation coefficient data in this module are derived
from the `Skyfield <https://github.com/skyfielders/python-skyfield>`_
astronomy library by Brandon Rhodes, which is licensed under the MIT
License.  See THIRD_PARTY_NOTICES for the full license text.
"""

from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import jax.typing

# ---------------------------------------------------------------------------
# Load nutation coefficient tables at import time
# ---------------------------------------------------------------------------

_data = np.load(Path(__file__).parent / "data" / "nutation.npz")

_ke0_t = jnp.array(_data["ke0_t"], dtype=jnp.float64)      # (33, 14)
_ke1 = jnp.array(_data["ke1"], dtype=jnp.float64)           # (14,)
_lunisolar_lon = jnp.array(
    _data["lunisolar_longitude_coefficients"], dtype=jnp.float64)  # (678,3)
_lunisolar_obl = jnp.array(
    _data["lunisolar_obliquity_coefficients"], dtype=jnp.float64)  # (678,3)
_nals_t = jnp.array(_data["nals_t"], dtype=jnp.float64)     # (678, 5)
_napl_t = jnp.array(_data["napl_t"], dtype=jnp.float64)     # (687, 14)
_plan_lon = jnp.array(
    _data["nutation_coefficients_longitude"], dtype=jnp.float64)   # (687,2)
_plan_obl = jnp.array(
    _data["nutation_coefficients_obliquity"], dtype=jnp.float64)   # (687,2)
_se0_t_0 = jnp.array(_data["se0_t_0"], dtype=jnp.float64)   # (33,)
_se0_t_1 = jnp.array(_data["se0_t_1"], dtype=jnp.float64)   # (33,)

del _data

# ---------------------------------------------------------------------------
# Constants (matching Skyfield exactly)
# ---------------------------------------------------------------------------

_ASEC2RAD = 4.848136811095359935899141e-6
_ASEC360 = 1296000.0
_DAY_S = 86400.0
_T0 = 2451545.0
_tau = 6.283185307179586476925287

# 1e-7 arcsecond to radian conversion (nutation output units)
_TENTH_USEC_2_RAD = _ASEC2RAD / 1e7

# Complementary-terms sine/cosine coefficients for t^1 term
_se1_0 = -0.87e-6
_se1_1 = +0.00e-6

# Planetary anomaly constants/coefficients (Skyfield nutationlib.py)
_anomaly_constant = jnp.array([
    2.35555598, 6.24006013, 1.627905234, 5.198466741, 2.18243920,
    4.402608842, 3.176146697, 1.753470314, 6.203480913, 0.599546497,
    0.874016757, 5.481293871, 5.321159000, 0.02438175,
])
_anomaly_coefficient = jnp.array([
    8328.6914269554, 628.301955, 8433.466158131, 7771.3771468121,
    -33.757045, 2608.7903141574, 1021.3285546211, 628.3075849991,
    334.0612426700, 52.9690962641, 21.3299104960, 7.4781598567,
    3.8127774000, 0.00000538691,
])

# Fundamental argument polynomial coefficients (Simon et al. 1994)
# Each row: [a0, a1, a2, a3, a4] in arcseconds (a0) or arcsec/century^n
_fa0 = jnp.array([485868.249036, 1287104.79305, 335779.526232,
                   1072260.70369, 450160.398036])
_fa1 = jnp.array([1717915923.2178, 129596581.0481, 1739527262.8478,
                   1602961601.2090, -6962890.5431])
_fa2 = jnp.array([31.8792, -0.5532, -12.7512, -6.3706, 7.4722])
_fa3 = jnp.array([0.051635, 0.000136, -0.001037, 0.006593, 0.007702])
_fa4 = jnp.array([-0.00024470, -0.00001149, 0.00000417, -0.00003169,
                   -0.00005939])


# ---------------------------------------------------------------------------
# Rotation matrices
# ---------------------------------------------------------------------------

def _rot_x(theta: jax.typing.ArrayLike) -> jax.Array:
    """Active rotation about the x-axis."""
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    return jnp.array([[1.0, 0.0, 0.0],
                       [0.0,   c,  -s],
                       [0.0,   s,   c]])


def _rot_z(theta: jax.typing.ArrayLike) -> jax.Array:
    """Active rotation about the z-axis."""
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    return jnp.array([[ c,  -s, 0.0],
                       [ s,   c, 0.0],
                       [0.0, 0.0, 1.0]])


# ---------------------------------------------------------------------------
# Frame bias  (ICRS → J2000)
# ---------------------------------------------------------------------------

def _frame_bias() -> jax.Array:
    """Return the ICRS-to-J2000 frame bias matrix (constant)."""
    xi0 = -0.0166170 * _ASEC2RAD
    eta0 = -0.0068192 * _ASEC2RAD
    da0 = -0.01460 * _ASEC2RAD

    yx = -da0
    zx = xi0
    xy = da0
    zy = eta0
    xz = -xi0
    yz = -eta0

    xx = 1.0 - 0.5 * (yx * yx + zx * zx)
    yy = 1.0 - 0.5 * (yx * yx + zy * zy)
    zz = 1.0 - 0.5 * (zy * zy + zx * zx)

    return jnp.array([[xx, xy, xz],
                       [yx, yy, yz],
                       [zx, zy, zz]])


# Precompute — this is a constant matrix
_B = _frame_bias()


# ---------------------------------------------------------------------------
# Precession  (IAU 2006, Capitaine et al. 2003)
# ---------------------------------------------------------------------------

def _precession_matrix(T: jax.typing.ArrayLike) -> jax.Array:
    """IAU-2006 precession matrix P for TDB centuries *T* from J2000."""
    eps0 = 84381.406

    psia = ((((-0.0000000951 * T
               + 0.000132851) * T
              - 0.00114045) * T
             - 1.0790069) * T
            + 5038.481507) * T

    omegaa = ((((+0.0000003337 * T
                 - 0.000000467) * T
                - 0.00772503) * T
               + 0.0512623) * T
              - 0.025754) * T + eps0

    chia = ((((-0.0000000560 * T
               + 0.000170663) * T
              - 0.00121197) * T
             - 2.3814292) * T
            + 10.556403) * T

    eps0_r = eps0 * _ASEC2RAD
    psia_r = psia * _ASEC2RAD
    omegaa_r = omegaa * _ASEC2RAD
    chia_r = chia * _ASEC2RAD

    sa = jnp.sin(eps0_r)
    ca = jnp.cos(eps0_r)
    sb = jnp.sin(-psia_r)
    cb = jnp.cos(-psia_r)
    sc = jnp.sin(-omegaa_r)
    cc = jnp.cos(-omegaa_r)
    sd = jnp.sin(chia_r)
    cd = jnp.cos(chia_r)

    # R3(chi_a) R1(-omega_a) R3(-psi_a) R1(eps0)
    return jnp.array([
        [cd * cb - sb * sd * cc,
         cd * sb * ca + sd * cc * cb * ca - sa * sd * sc,
         cd * sb * sa + sd * cc * cb * sa + ca * sd * sc],
        [-sd * cb - sb * cd * cc,
         -sd * sb * ca + cd * cc * cb * ca - sa * cd * sc,
         -sd * sb * sa + cd * cc * cb * sa + ca * cd * sc],
        [sb * sc,
         -sc * cb * ca - sa * cc,
         -sc * cb * sa + cc * ca],
    ])


# ---------------------------------------------------------------------------
# Mean obliquity
# ---------------------------------------------------------------------------

def _mean_obliquity(T: jax.typing.ArrayLike) -> jax.Array:
    """Mean obliquity of the ecliptic in arcseconds (TDB centuries *T*)."""
    return ((((-0.0000000434 * T  # type: ignore[return-value]
               - 0.000000576) * T
              + 0.00200340) * T
             - 0.0001831) * T
            - 46.836769) * T + 84381.406


# ---------------------------------------------------------------------------
# Fundamental arguments for lunisolar nutation
# ---------------------------------------------------------------------------

def _fundamental_arguments(t: jax.typing.ArrayLike) -> jax.Array:
    """5 Delaunay arguments in radians for TT centuries *t*."""
    a = _fa4 * t
    a = (a + _fa3) * t
    a = (a + _fa2) * t
    a = (a + _fa1) * t
    a = a + _fa0
    a = jnp.fmod(a, _ASEC360) * _ASEC2RAD
    return a  # (5,)


# ---------------------------------------------------------------------------
# IAU 2000A nutation  (678 lunisolar + 687 planetary terms)
# ---------------------------------------------------------------------------

def _iau2000a(T: jax.typing.ArrayLike) -> tuple[jax.Array, jax.Array]:
    """Return (dpsi, deps) in radians for TT centuries *T* from J2000.

    Full IAU-2000A model with 678 lunisolar and 687 planetary terms.
    """
    t = T

    # --- lunisolar ---
    a = _fundamental_arguments(t)            # (5,)
    arg = _nals_t @ a                        # (678,)
    sarg = jnp.sin(arg)
    carg = jnp.cos(arg)

    dpsi = (jnp.dot(sarg, _lunisolar_lon[:, 0])
            + jnp.dot(sarg, _lunisolar_lon[:, 1]) * t
            + jnp.dot(carg, _lunisolar_lon[:, 2]))
    deps = (jnp.dot(carg, _lunisolar_obl[:, 0])
            + jnp.dot(carg, _lunisolar_obl[:, 1]) * t
            + jnp.dot(sarg, _lunisolar_obl[:, 2]))

    # --- planetary ---
    pa = t * _anomaly_coefficient + _anomaly_constant   # (14,)
    pa = pa.at[-1].multiply(t)  # type: ignore[union-attr]  # general precession term *= t

    parg = _napl_t @ pa  # (687,)
    psarg = jnp.sin(parg)
    pcarg = jnp.cos(parg)

    dpsi = dpsi + jnp.dot(psarg, _plan_lon[:, 0]) + jnp.dot(pcarg, _plan_lon[:, 1])
    deps = deps + jnp.dot(psarg, _plan_obl[:, 0]) + jnp.dot(pcarg, _plan_obl[:, 1])

    # Convert from 1e-7 arcseconds to radians
    dpsi = dpsi * _TENTH_USEC_2_RAD
    deps = deps * _TENTH_USEC_2_RAD
    return dpsi, deps


# ---------------------------------------------------------------------------
# Equation of the equinoxes (including complementary terms)
# ---------------------------------------------------------------------------

def _equation_of_the_equinoxes_complementary_terms(T: jax.typing.ArrayLike) -> jax.Array:
    """Complementary terms of the equation of the equinoxes, in radians.

    *T* is TT centuries from J2000.
    """
    t = T

    # Build 14 fundamental arguments (same as Skyfield nutationlib.py)
    fa = jnp.zeros(14)

    # Moon's mean anomaly
    fa = fa.at[0].set(
        (485868.249036
         + (715923.2178
            + (31.8792
               + (0.051635
                  + (-0.00024470) * t) * t) * t) * t) * _ASEC2RAD
        + jnp.fmod(1325.0 * t, 1.0) * _tau)

    # Sun's mean anomaly
    fa = fa.at[1].set(
        (1287104.793048
         + (1292581.0481
            + (-0.5532
               + (0.000136
                  + (-0.00001149) * t) * t) * t) * t) * _ASEC2RAD
        + jnp.fmod(99.0 * t, 1.0) * _tau)

    # Moon's mean longitude minus ascending node longitude
    fa = fa.at[2].set(
        (335779.526232
         + (295262.8478
            + (-12.7512
               + (-0.001037
                  + (0.00000417) * t) * t) * t) * t) * _ASEC2RAD
        + jnp.fmod(1342.0 * t, 1.0) * _tau)

    # Moon's mean elongation from Sun
    fa = fa.at[3].set(
        (1072260.703692
         + (1105601.2090
            + (-6.3706
               + (0.006593
                  + (-0.00003169) * t) * t) * t) * t) * _ASEC2RAD
        + jnp.fmod(1236.0 * t, 1.0) * _tau)

    # Moon's ascending node longitude
    fa = fa.at[4].set(
        (450160.398036
         + (-482890.5431
            + (7.4722
               + (0.007702
                  + (-0.00005939) * t) * t) * t) * t) * _ASEC2RAD
        + jnp.fmod(-5.0 * t, 1.0) * _tau)

    # Planetary longitudes
    fa = fa.at[5].set(4.402608842 + 2608.7903141574 * t)
    fa = fa.at[6].set(3.176146697 + 1021.3285546211 * t)
    fa = fa.at[7].set(1.753470314 + 628.3075849991 * t)
    fa = fa.at[8].set(6.203480913 + 334.0612426700 * t)
    fa = fa.at[9].set(0.599546497 + 52.9690962641 * t)
    fa = fa.at[10].set(0.874016757 + 21.3299104960 * t)
    fa = fa.at[11].set(5.481293872 + 7.4781598567 * t)
    fa = fa.at[12].set(5.311886287 + 3.8133035638 * t)
    fa = fa.at[13].set((0.024381750 + 0.00000538691 * t) * t)

    fa = jnp.fmod(fa, _tau)

    # t^1 terms (single entry)
    a1 = jnp.dot(_ke1, fa)
    c_terms = _se1_0 * jnp.sin(a1) + _se1_1 * jnp.cos(a1)
    c_terms = c_terms * t

    # t^0 terms (33 entries)
    a0 = _ke0_t @ fa  # (33,)
    c_terms = c_terms + jnp.dot(_se0_t_0, jnp.sin(a0))
    c_terms = c_terms + jnp.dot(_se0_t_1, jnp.cos(a0))

    return c_terms * _ASEC2RAD


# ---------------------------------------------------------------------------
# Nutation matrix
# ---------------------------------------------------------------------------

def _nutation_matrix(mean_obl: jax.typing.ArrayLike, true_obl: jax.typing.ArrayLike, dpsi: jax.typing.ArrayLike) -> jax.Array:
    """Build the 3x3 nutation matrix from obliquity and nutation angles.

    All arguments in radians.
    """
    cobm = jnp.cos(mean_obl)
    sobm = jnp.sin(mean_obl)
    cobt = jnp.cos(true_obl)
    sobt = jnp.sin(true_obl)
    cpsi = jnp.cos(dpsi)
    spsi = jnp.sin(dpsi)

    return jnp.array([
        [cpsi, -spsi * cobm, -spsi * sobm],
        [spsi * cobt,
         cpsi * cobm * cobt + sobm * sobt,
         cpsi * sobm * cobt - cobm * sobt],
        [spsi * sobt,
         cpsi * cobm * sobt - sobm * cobt,
         cpsi * sobm * sobt + cobm * cobt],
    ])


# ---------------------------------------------------------------------------
# GMST 1982  (SGP4-specific, from AIAA 2006-6753 Appendix C)
# ---------------------------------------------------------------------------

def _theta_gmst1982(jd_ut1: jax.typing.ArrayLike, frac_ut1: jax.typing.ArrayLike) -> tuple[jax.Array, jax.Array]:
    """GMST angle in radians and its rate (rad/day of UT1)."""
    t = (jd_ut1 - _T0 + frac_ut1) / 36525.0
    g = 67310.54841 + (8640184.812866 + (0.093104 + (-6.2e-6) * t) * t) * t
    dg = 8640184.812866 + (0.093104 * 2.0 + (-6.2e-6 * 3.0) * t) * t
    theta = jnp.fmod(jnp.fmod(jd_ut1, 1.0) + frac_ut1 + jnp.fmod(g / _DAY_S, 1.0), 1.0) * _tau
    theta_dot = (1.0 + dg / (_DAY_S * 36525.0)) * _tau
    return theta, theta_dot  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# GMST (equinox method, matching Skyfield earthlib.sidereal_time)
# ---------------------------------------------------------------------------

def _earth_rotation_angle(jd_ut1: jax.typing.ArrayLike, frac_ut1: jax.typing.ArrayLike) -> jax.Array:
    """Earth Rotation Angle (fraction of full rotation)."""
    th = 0.7790572732640 + 0.00273781191135448 * (jd_ut1 - _T0 + frac_ut1)
    return jnp.fmod(jnp.fmod(th, 1.0) + jnp.fmod(jd_ut1, 1.0) + frac_ut1, 1.0)


def _gmst_hours(jd_ut1: jax.typing.ArrayLike, frac_ut1: jax.typing.ArrayLike, jd_tdb: jax.typing.ArrayLike, frac_tdb: jax.typing.ArrayLike) -> jax.Array:
    """Greenwich Mean Sidereal Time in hours (equinox method)."""
    theta = _earth_rotation_angle(jd_ut1, frac_ut1)
    t = (jd_tdb - _T0 + frac_tdb) / 36525.0
    st = (0.014506
          + ((((-0.0000000368 * t
                - 0.000029956) * t
               - 0.00000044) * t
              + 1.3915817) * t
             + 4612.156534) * t)
    return jnp.fmod(st / 54000.0 + theta * 24.0, 24.0)


# ---------------------------------------------------------------------------
# GAST  (Greenwich Apparent Sidereal Time)
# ---------------------------------------------------------------------------

def _gast_hours(T_tt: jax.typing.ArrayLike, dpsi: jax.typing.ArrayLike, mean_obl_rad: jax.typing.ArrayLike, jd_ut1: jax.typing.ArrayLike, frac_ut1: jax.typing.ArrayLike, jd_tdb: jax.typing.ArrayLike, frac_tdb: jax.typing.ArrayLike) -> jax.Array:
    """GAST in hours, matching Skyfield's Time.gast property."""
    c_terms = _equation_of_the_equinoxes_complementary_terms(T_tt)
    eq_eq = dpsi * jnp.cos(mean_obl_rad) + c_terms
    gmst = _gmst_hours(jd_ut1, frac_ut1, jd_tdb, frac_tdb)
    return jnp.fmod(gmst + eq_eq / _tau * 24.0, 24.0)


# ---------------------------------------------------------------------------
# TDB - TT approximation (matching Skyfield)
# ---------------------------------------------------------------------------

def _tdb_minus_tt(jd_tt: jax.typing.ArrayLike, frac_tt: jax.typing.ArrayLike) -> jax.Array:
    """TDB - TT in seconds (Fairhead & Bretagnon 1990 approximation)."""
    t = (jd_tt - _T0 + frac_tt) / 36525.0
    # Simplified; the dominant term is sufficient for our precision needs
    return 0.001657 * jnp.sin(628.3076 * t + 6.2401)


# ---------------------------------------------------------------------------
# Public API: TEME → GCRF
# ---------------------------------------------------------------------------

def _delta_t(jd_ut1: jax.typing.ArrayLike, frac_ut1: jax.typing.ArrayLike) -> jax.Array:
    """Approximate delta_t = TT - UT1 in seconds.

    Uses table-based interpolation of observed/predicted values from IERS
    Bulletin A.  Accurate to ~0.1 s for 1972-2030, which is sufficient for
    both frame transformations and UT1 ↔ UTC conversions.
    """
    y = 2000.0 + ((jd_ut1 - _T0) + frac_ut1) / 365.25
    return jnp.interp(y, _DELTA_T_YEARS, _DELTA_T_VALUES)


# Observed delta_T (TT - UT1) values from IERS Bulletin A / USNO.
# Pre-2005 at half-year intervals, 2005-2026 at monthly intervals.
_DELTA_T_YEARS = jnp.array([
    # Pre-2005 (half-year intervals)
    1957.0, 1960.0, 1965.0, 1970.0,
    1972.0, 1972.5, 1973.0, 1973.5, 1974.0, 1974.5,
    1975.0, 1975.5, 1976.0, 1976.5, 1977.0, 1977.5,
    1978.0, 1978.5, 1979.0, 1979.5, 1980.0, 1980.5,
    1981.0, 1981.5, 1982.0, 1982.5, 1983.0, 1983.5,
    1984.0, 1984.5, 1985.0, 1985.5, 1986.0, 1986.5,
    1987.0, 1987.5, 1988.0, 1988.5, 1989.0, 1989.5,
    1990.0, 1990.5, 1991.0, 1991.5, 1992.0, 1992.5,
    1993.0, 1993.5, 1994.0, 1994.5, 1995.0, 1995.5,
    1996.0, 1996.5, 1997.0, 1997.5, 1998.0, 1998.5,
    1999.0, 1999.5, 2000.0, 2000.5, 2001.0, 2001.5,
    2002.0, 2002.5, 2003.0, 2003.5, 2004.0, 2004.5,
    # 2005-2026 (monthly intervals)
    *[2005 + m / 12.0 for m in range(264)],
    # Extrapolation
    2027.5, 2030.0,
], dtype=jnp.float64)

_DELTA_T_VALUES = jnp.array([
    # Pre-2005 (half-year intervals)
    31.1, 33.2, 35.7, 40.2,
    42.15, 42.76, 43.37, 43.96, 44.48, 45.00,
    45.48, 45.99, 46.46, 47.00, 47.52, 48.04,
    48.53, 49.10, 49.59, 50.11, 50.54, 50.98,
    51.38, 51.82, 52.17, 52.58, 52.96, 53.44,
    53.79, 54.09, 54.34, 54.64, 54.87, 55.11,
    55.32, 55.58, 55.82, 56.09, 56.30, 56.57,
    56.86, 57.23, 57.57, 57.96, 58.31, 58.74,
    59.12, 59.59, 59.98, 60.40, 60.79, 61.25,
    61.63, 62.00, 62.30, 62.66, 62.97, 63.29,
    63.47, 63.66, 63.83, 63.98, 64.09, 64.21,
    64.30, 64.41, 64.47, 64.55, 64.57, 64.65,
    # 2005 monthly
    64.688, 64.705, 64.734, 64.758, 64.783, 64.801,
    64.799, 64.787, 64.783, 64.793, 64.810, 64.831,
    # 2006
    64.845, 64.859, 64.889, 64.919, 64.949, 64.980,
    64.991, 65.003, 65.014, 65.038, 65.078, 65.114,
    # 2007
    65.147, 65.183, 65.216, 65.252, 65.294, 65.328,
    65.342, 65.346, 65.350, 65.372, 65.398, 65.431,
    # 2008
    65.458, 65.487, 65.516, 65.546, 65.580, 65.613,
    65.629, 65.637, 65.649, 65.676, 65.710, 65.746,
    # 2009
    65.777, 65.802, 65.825, 65.860, 65.900, 65.934,
    65.951, 65.953, 65.963, 65.985, 66.015, 66.042,
    # 2010
    66.070, 66.095, 66.135, 66.170, 66.208, 66.236,
    66.241, 66.233, 66.235, 66.244, 66.276, 66.306,
    # 2011
    66.325, 66.340, 66.364, 66.398, 66.431, 66.463,
    66.475, 66.476, 66.483, 66.507, 66.539, 66.572,
    # 2012
    66.604, 66.634, 66.657, 66.694, 66.731, 66.759,
    66.771, 66.774, 66.785, 66.811, 66.840, 66.878,
    # 2013
    66.907, 66.943, 66.979, 67.027, 67.074, 67.111,
    67.127, 67.133, 67.146, 67.172, 67.209, 67.247,
    # 2014
    67.281, 67.313, 67.349, 67.391, 67.434, 67.467,
    67.487, 67.499, 67.511, 67.536, 67.572, 67.608,
    # 2015
    67.644, 67.676, 67.714, 67.761, 67.804, 67.841,
    67.862, 67.884, 67.913, 67.956, 68.006, 68.053,
    # 2016
    68.103, 68.158, 68.207, 68.268, 68.321, 68.372,
    68.397, 68.410, 68.430, 68.464, 68.508, 68.554,
    # 2017
    68.592, 68.629, 68.670, 68.715, 68.764, 68.805,
    68.825, 68.838, 68.848, 68.869, 68.901, 68.936,
    # 2018
    68.968, 68.987, 69.020, 69.052, 69.084, 69.107,
    69.113, 69.114, 69.121, 69.136, 69.165, 69.197,
    # 2019
    69.220, 69.245, 69.274, 69.305, 69.335, 69.355,
    69.357, 69.344, 69.338, 69.338, 69.343, 69.354,
    # 2020
    69.361, 69.375, 69.390, 69.409, 69.427, 69.439,
    69.423, 69.392, 69.369, 69.357, 69.359, 69.363,
    # 2021
    69.360, 69.351, 69.355, 69.359, 69.367, 69.368,
    69.350, 69.327, 69.303, 69.289, 69.288, 69.291,
    # 2022
    69.294, 69.292, 69.286, 69.284, 69.281, 69.279,
    69.250, 69.221, 69.197, 69.189, 69.194, 69.204,
    # 2023
    69.204, 69.199, 69.199, 69.209, 69.219, 69.230,
    69.219, 69.198, 69.182, 69.173, 69.172, 69.172,
    # 2024
    69.175, 69.180, 69.187, 69.198, 69.202, 69.205,
    69.187, 69.158, 69.132, 69.125, 69.130, 69.134,
    # 2025
    69.138, 69.136, 69.140, 69.147, 69.154, 69.155,
    69.140, 69.121, 69.099, 69.091, 69.091, 69.105,
    # 2026
    69.110, 69.113, 69.124, 69.139, 69.151, 69.150,
    69.131, 69.105, 69.089, 69.091, 69.095, 69.099,
    # Extrapolation
    69.10, 69.10,
], dtype=jnp.float64)


# ---------------------------------------------------------------------------
# Leap seconds table  (TAI - UTC)
# ---------------------------------------------------------------------------

# Each entry is (JD of midnight UTC when the leap second takes effect,
# cumulative TAI − UTC in seconds after that point).
_LEAP_SECOND_JD = jnp.array([
    2441317.5,  # 1972-01-01  (10)
    2441499.5,  # 1972-07-01  (11)
    2441683.5,  # 1973-01-01  (12)
    2442048.5,  # 1974-01-01  (13)
    2442413.5,  # 1975-01-01  (14)
    2442778.5,  # 1976-01-01  (15)
    2443144.5,  # 1977-01-01  (16)
    2443509.5,  # 1978-01-01  (17)
    2443874.5,  # 1979-01-01  (18)
    2444239.5,  # 1980-01-01  (19)
    2444786.5,  # 1981-07-01  (20)
    2445151.5,  # 1982-07-01  (21)
    2445516.5,  # 1983-07-01  (22)
    2446247.5,  # 1985-07-01  (23)
    2447161.5,  # 1988-01-01  (24)
    2447892.5,  # 1990-01-01  (25)
    2448257.5,  # 1991-01-01  (26)
    2448804.5,  # 1992-07-01  (27)
    2449169.5,  # 1993-07-01  (28)
    2449534.5,  # 1994-07-01  (29)
    2450083.5,  # 1996-01-01  (30)
    2450630.5,  # 1997-07-01  (31)
    2451179.5,  # 1999-01-01  (32)
    2453736.5,  # 2006-01-01  (33)
    2454832.5,  # 2009-01-01  (34)
    2456109.5,  # 2012-07-01  (35)
    2457204.5,  # 2015-07-01  (36)
    2457754.5,  # 2017-01-01  (37)
], dtype=jnp.float64)

_LEAP_SECOND_TAI_UTC = jnp.array([
    10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0,
    20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0,
    30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0,
], dtype=jnp.float64)


def _leap_seconds(jd_utc: jax.typing.ArrayLike) -> jax.Array:
    """Return cumulative TAI − UTC leap seconds at the given UTC Julian date."""
    idx = jnp.searchsorted(_LEAP_SECOND_JD, jd_utc, side='right') - 1
    idx = jnp.clip(idx, 0, len(_LEAP_SECOND_TAI_UTC) - 1)
    return _LEAP_SECOND_TAI_UTC[idx]


def _ut1_to_utc(jd_ut1: jax.typing.ArrayLike, frac_ut1: jax.typing.ArrayLike) -> tuple[jax.Array, jax.Array]:
    """Convert UT1 Julian date to UTC Julian date.

    Uses delta_T and the leap seconds table to compute:
        UTC = UT1 + (delta_T − 32.184 − leap_seconds)

    Returns (jd_ut1, frac_utc) where *frac_utc* is adjusted so that
    ``jd_ut1 + frac_utc`` gives JD(UTC).
    """
    dt = _delta_t(jd_ut1, frac_ut1)
    # TT = UT1 + delta_T;  TAI = TT − 32.184;  UTC = TAI − leap_seconds
    # ⇒ UTC − UT1 = delta_T − 32.184 − leap_seconds
    ls = _leap_seconds(jd_ut1 + frac_ut1)
    utc_minus_ut1 = (dt - 32.184 - ls) / _DAY_S
    frac_utc = frac_ut1 + utc_minus_ut1
    return jd_ut1, frac_utc  # type: ignore[return-value]


@jax.jit
def teme_to_gcrf(
    r_teme: jax.typing.ArrayLike,
    v_teme: jax.typing.ArrayLike,
    jd: jax.typing.ArrayLike,
    fr: jax.typing.ArrayLike,
) -> tuple[jax.Array, jax.Array]:
    """Transform position/velocity from TEME to GCRF (≈ICRS).

    Replicates Skyfield's ``EarthSatellite._at()`` transformation chain:
    ``M = N @ P @ B``, then ``R = rot_z(theta_GMST1982 - GAST) @ M``
    with the final GCRF vectors obtained via ``R^T @ r_TEME``.

    The input Julian date ``jd + fr`` is treated as UT1 time (≈UTC).
    TT and TDB are derived internally using an approximate delta_T model.

    Args:
        r_teme: Position in TEME frame (3,) in km.
        v_teme: Velocity in TEME frame (3,) in km/s.
        jd: Julian date, integer/whole part (scalar).
        fr: Julian date, fractional part (scalar).

    Returns:
        r_gcrf: Position in GCRF frame (3,) in km.
        v_gcrf: Velocity in GCRF frame (3,) in km/s.
    """
    # UT1 time (treat input as UT1 ≈ UTC)
    jd_ut1 = jd
    frac_ut1 = fr

    # Derive TT = UT1 + delta_t
    dt = _delta_t(jd_ut1, frac_ut1)
    frac_tt = frac_ut1 + dt / _DAY_S

    # TDB ≈ TT + small periodic correction
    dt_tdb = _tdb_minus_tt(jd_ut1, frac_tt) / _DAY_S
    frac_tdb = frac_tt + dt_tdb

    # TDB centuries from J2000 (for precession and mean obliquity)
    T_tdb = (jd_ut1 - _T0 + frac_tdb) / 36525.0

    # TT centuries from J2000 (for nutation)
    T_tt = (jd_ut1 - _T0 + frac_tt) / 36525.0

    # Nutation angles (use TT)
    dpsi, deps = _iau2000a(T_tt)

    # Mean and true obliquity (use TDB centuries)
    mean_obl = _mean_obliquity(T_tdb) * _ASEC2RAD
    true_obl = mean_obl + deps

    # Build M = N @ P @ B (precession uses TDB)
    P = _precession_matrix(T_tdb)
    N = _nutation_matrix(mean_obl, true_obl, dpsi)
    M = N @ P @ _B

    # GAST and GMST1982 (use UT1 for Earth rotation, TDB for polynomial)
    gast = _gast_hours(T_tt, dpsi, mean_obl, jd_ut1, frac_ut1, jd_ut1, frac_tdb)
    theta, _ = _theta_gmst1982(jd_ut1, frac_ut1)

    # TEME rotation: angle = theta_GMST1982 - GAST (in radians)
    angle = theta - gast / 24.0 * _tau

    # R_teme = rot_z(angle) @ M  ;  R_teme_to_gcrf = R_teme^T
    R_teme = _rot_z(angle) @ M
    R = R_teme.T

    r_gcrf = R @ r_teme
    v_gcrf = R @ v_teme

    return r_gcrf, v_gcrf
