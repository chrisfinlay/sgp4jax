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

def _rot_x(theta):
    """Active rotation about the x-axis."""
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    return jnp.array([[1.0, 0.0, 0.0],
                       [0.0,   c,  -s],
                       [0.0,   s,   c]])


def _rot_z(theta):
    """Active rotation about the z-axis."""
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    return jnp.array([[ c,  -s, 0.0],
                       [ s,   c, 0.0],
                       [0.0, 0.0, 1.0]])


# ---------------------------------------------------------------------------
# Frame bias  (ICRS → J2000)
# ---------------------------------------------------------------------------

def _frame_bias():
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

def _precession_matrix(T):
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

def _mean_obliquity(T):
    """Mean obliquity of the ecliptic in arcseconds (TDB centuries *T*)."""
    return ((((-0.0000000434 * T
               - 0.000000576) * T
              + 0.00200340) * T
             - 0.0001831) * T
            - 46.836769) * T + 84381.406


# ---------------------------------------------------------------------------
# Fundamental arguments for lunisolar nutation
# ---------------------------------------------------------------------------

def _fundamental_arguments(t):
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

def _iau2000a(T):
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
    pa = pa.at[-1].multiply(t)  # general precession term *= t

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

def _equation_of_the_equinoxes_complementary_terms(T):
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

def _nutation_matrix(mean_obl, true_obl, dpsi):
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

def _theta_gmst1982(jd_ut1, frac_ut1):
    """GMST angle in radians and its rate (rad/day of UT1)."""
    t = (jd_ut1 - _T0 + frac_ut1) / 36525.0
    g = 67310.54841 + (8640184.812866 + (0.093104 + (-6.2e-6) * t) * t) * t
    dg = 8640184.812866 + (0.093104 * 2.0 + (-6.2e-6 * 3.0) * t) * t
    theta = jnp.fmod(jnp.fmod(jd_ut1, 1.0) + frac_ut1 + jnp.fmod(g / _DAY_S, 1.0), 1.0) * _tau
    theta_dot = (1.0 + dg / (_DAY_S * 36525.0)) * _tau
    return theta, theta_dot


# ---------------------------------------------------------------------------
# GMST (equinox method, matching Skyfield earthlib.sidereal_time)
# ---------------------------------------------------------------------------

def _earth_rotation_angle(jd_ut1, frac_ut1):
    """Earth Rotation Angle (fraction of full rotation)."""
    th = 0.7790572732640 + 0.00273781191135448 * (jd_ut1 - _T0 + frac_ut1)
    return jnp.fmod(jnp.fmod(th, 1.0) + jnp.fmod(jd_ut1, 1.0) + frac_ut1, 1.0)


def _gmst_hours(jd_ut1, frac_ut1, jd_tdb, frac_tdb):
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

def _gast_hours(T_tt, dpsi, mean_obl_rad, jd_ut1, frac_ut1, jd_tdb, frac_tdb):
    """GAST in hours, matching Skyfield's Time.gast property."""
    c_terms = _equation_of_the_equinoxes_complementary_terms(T_tt)
    eq_eq = dpsi * jnp.cos(mean_obl_rad) + c_terms
    gmst = _gmst_hours(jd_ut1, frac_ut1, jd_tdb, frac_tdb)
    return jnp.fmod(gmst + eq_eq / _tau * 24.0, 24.0)


# ---------------------------------------------------------------------------
# TDB - TT approximation (matching Skyfield)
# ---------------------------------------------------------------------------

def _tdb_minus_tt(jd_tt, frac_tt):
    """TDB - TT in seconds (Fairhead & Bretagnon 1990 approximation)."""
    t = (jd_tt - _T0 + frac_tt) / 36525.0
    # Simplified; the dominant term is sufficient for our precision needs
    return 0.001657 * jnp.sin(628.3076 * t + 6.2401)


# ---------------------------------------------------------------------------
# Public API: TEME → GCRF
# ---------------------------------------------------------------------------

def _delta_t(jd_ut1, frac_ut1):
    """Approximate delta_t = TT - UT1 in seconds.

    Uses a simple polynomial fit that is accurate to ~0.2 s for 1990-2030.
    This is more than sufficient since the frame transform is insensitive
    to delta_t errors (1 s error → < 1e-12 rotation matrix change).
    """
    # Convert to fractional year
    y = 2000.0 + ((jd_ut1 - _T0) + frac_ut1) / 365.25
    dt = y - 2000.0
    # Polynomial fit for 2005-2050 range (from IERS Conventions)
    return 62.92 + 0.32217 * dt + 0.005589 * dt * dt


@jax.jit
def teme_to_gcrf(r_teme, v_teme, jd, fr):
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
