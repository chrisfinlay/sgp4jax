"""TLE parsing - pure Python, not JIT-compatible."""

from math import pi, pow
from sgp4jax._constants import WGS72
from sgp4jax._sgp4init import sgp4init


def days2mdhms(year, days):
    """Convert day-of-year to month, day, hour, minute, second."""
    second = days * 86400.0
    second = round(second, 6)
    minute, second = divmod(second, 60.0)
    second = round(second, 6)
    minute = int(minute)
    hour, minute = divmod(minute, 60)
    day_of_year, hour = divmod(hour, 24)

    is_leap = year % 400 == 0 or (year % 4 == 0 and year % 100 != 0)
    february_bump = (2 - is_leap) * (day_of_year >= 60 + is_leap)
    august = day_of_year >= 215
    month, day = divmod(2 * (day_of_year - 1 + 30 * august + february_bump), 61)
    month += 1 - august
    day //= 2
    day += 1

    return month, day, int(hour), int(minute), second


def jday(year, mon, day, hr, minute, sec):
    """Calendar date to Julian date."""
    jd = (367.0 * year
         - 7 * (year + ((mon + 9) // 12.0)) * 0.25 // 1.0
         + 275 * mon / 9.0 // 1.0
         + day
         + 1721013.5)
    fr = (sec + minute * 60.0 + hr * 3600.0) / 86400.0
    return jd, fr


def parse_tle(line1, line2):
    """Parse two TLE lines into a dict of orbital elements.

    Returns dict with keys matching sgp4init parameters.
    """
    deg2rad = pi / 180.0
    xpdotp = 1440.0 / (2.0 * pi)

    # Parse line 1
    line = line1.rstrip()
    satnum_str = line[2:7]
    two_digit_year = int(line[18:20])
    epochdays = float(line[20:32])
    ndot = float(line[33:43])
    nddot = float(line[44] + '.' + line[45:50])
    nexp = int(line[50:52])
    bstar = float(line[53] + '.' + line[54:59])
    ibexp = int(line[59:61])

    # Parse line 2
    line = line2.rstrip()
    inclo = float(line[8:16])
    nodeo = float(line[17:25])
    ecco = float('0.' + line[26:33].replace(' ', '0'))
    argpo = float(line[34:42])
    mo = float(line[43:51])
    no_kozai = float(line[52:63])

    # Convert units
    no_kozai = no_kozai / xpdotp  # rad/min
    nddot = nddot * pow(10.0, nexp)
    bstar = bstar * pow(10.0, ibexp)
    ndot = ndot / (xpdotp * 1440.0)
    nddot = nddot / (xpdotp * 1440.0 * 1440)

    # Convert to radians
    inclo = inclo * deg2rad
    nodeo = nodeo * deg2rad
    argpo = argpo * deg2rad
    mo = mo * deg2rad

    # Year
    if two_digit_year < 57:
        year = two_digit_year + 2000
    else:
        year = two_digit_year + 1900

    # Compute epoch Julian date
    mon, day, hr, minute, sec = days2mdhms(year, epochdays)
    jdsatepoch, jdsatepochF = jday(year, mon, day, hr, minute, sec)

    # Rebuild split JD from TLE year and day for precision
    days_i, fraction = divmod(epochdays, 1.0)
    jdsatepoch_precise = year * 365 + (year - 1) // 4 + days_i + 1721044.5
    jdsatepochF_precise = round(fraction, 8)

    # Epoch for sgp4init (days from 0 jan 1950)
    epoch = jdsatepoch_precise + jdsatepochF_precise - 2433281.5

    return {
        'satnum_str': satnum_str,
        'bstar': bstar,
        'ndot': ndot,
        'nddot': nddot,
        'ecco': ecco,
        'argpo': argpo,
        'inclo': inclo,
        'mo': mo,
        'no_kozai': no_kozai,
        'nodeo': nodeo,
        'epoch': epoch,
        'jdsatepoch': jdsatepoch_precise,
        'jdsatepochF': jdsatepochF_precise,
    }


def tle_to_satrec(line1, line2, gravity=None):
    """Parse TLE and initialize a SatRec ready for propagation.

    Args:
        line1: First TLE line
        line2: Second TLE line
        gravity: Gravity constants (default WGS84)

    Returns:
        SatRec NamedTuple
    """
    if gravity is None:
        gravity = WGS72

    params = parse_tle(line1, line2)
    return sgp4init(
        gravity,
        params['epoch'],
        params['bstar'],
        params['ndot'],
        params['nddot'],
        params['ecco'],
        params['argpo'],
        params['inclo'],
        params['mo'],
        params['no_kozai'],
        params['nodeo'],
        params['jdsatepoch'],
        params['jdsatepochF'],
    )
