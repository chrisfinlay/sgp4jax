"""Earth gravity model constants for SGP4."""

from math import sqrt
from typing import NamedTuple


class GravityConstants(NamedTuple):
    """Earth gravity model constants for use with SGP4.

    Attributes:
        tumin: Minutes per canonical time unit (= 1 / xke).
        mu: Gravitational parameter (km³/s²).
        radiusearthkm: Earth equatorial radius (km).
        xke: Square root of Earth's gravitational parameter in canonical
            units (ER^1.5/min).
        j2: Second zonal harmonic (dimensionless).
        j3: Third zonal harmonic (dimensionless).
        j4: Fourth zonal harmonic (dimensionless).
        j3oj2: Ratio j3 / j2 (precomputed for efficiency).
    """

    tumin: float
    mu: float
    radiusearthkm: float
    xke: float
    j2: float
    j3: float
    j4: float
    j3oj2: float


# WGS-72 old constants
_mu = 398600.79964
_re = 6378.135
_xke = 0.0743669161
WGS72OLD = GravityConstants(
    tumin=1.0 / _xke,
    mu=_mu,
    radiusearthkm=_re,
    xke=_xke,
    j2=0.001082616,
    j3=-0.00000253881,
    j4=-0.00000165597,
    j3oj2=-0.00000253881 / 0.001082616,
)

# WGS-72 constants
_mu = 398600.8
_re = 6378.135
_xke = 60.0 / sqrt(_re * _re * _re / _mu)
WGS72 = GravityConstants(
    tumin=1.0 / _xke,
    mu=_mu,
    radiusearthkm=_re,
    xke=_xke,
    j2=0.001082616,
    j3=-0.00000253881,
    j4=-0.00000165597,
    j3oj2=-0.00000253881 / 0.001082616,
)

# WGS-84 constants
_mu = 398600.5
_re = 6378.137
_xke = 60.0 / sqrt(_re * _re * _re / _mu)
WGS84 = GravityConstants(
    tumin=1.0 / _xke,
    mu=_mu,
    radiusearthkm=_re,
    xke=_xke,
    j2=0.00108262998905,
    j3=-0.00000253215306,
    j4=-0.00000161098761,
    j3oj2=-0.00000253215306 / 0.00108262998905,
)
