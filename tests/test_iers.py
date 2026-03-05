"""Tests for IERS UTC→UT1 conversion.

Compares sgp4jax.utc_to_ut1 against Skyfield's t.dut1 (UT1-UTC in seconds).
Requires skyfield and a downloaded IERS table (update_iers_table() or an
existing cache at ~/.cache/sgp4jax/finals2000A.npz).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgp4jax._iers import utc_to_ut1, load_iers_table, update_iers_table, _mjd, _dut1
from sgp4jax._tle import jday


# ---------------------------------------------------------------------------
# Fixture: ensure IERS table is loaded
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def iers_loaded():
    """Download IERS table if not already cached."""
    from sgp4jax import _iers
    if _iers._mjd is None:
        update_iers_table()
    yield


# ---------------------------------------------------------------------------
# Reference helper
# ---------------------------------------------------------------------------

def _sf_dut1(year: int, month: int, day: int) -> float:
    """Return Skyfield's DUT1 = UT1 − UTC (seconds) for midnight UTC."""
    from skyfield.api import load
    ts = load.timescale()
    t = ts.utc(year, month, day)
    return float(t.dut1)


def _our_dut1(year: int, month: int, day: int) -> float:
    """Return our DUT1 = UT1 − UTC (seconds) for midnight UTC."""
    jd, fr = jday(year, month, day, 0, 0, 0.0)
    _, fr_ut1 = utc_to_ut1(jnp.float64(jd), jnp.float64(fr))
    return float((float(fr_ut1) - fr) * 86400.0)


# ---------------------------------------------------------------------------
# Historical dates — Bulletin A IERS values (flag 'I'), stable and final
# ---------------------------------------------------------------------------

_HISTORICAL = [
    (2000, 1, 1),
    (2000, 7, 4),
    (2004, 2, 29),   # leap day
    (2006, 1, 1),    # day after leap second
    (2009, 1, 1),    # day after leap second
    (2012, 7, 1),    # day of leap second insertion
    (2015, 7, 1),    # day of leap second insertion
    (2017, 1, 1),    # last leap second to date
    (2018, 6, 15),
    (2020, 3, 1),
    (2021, 9, 22),
    (2022, 12, 31),
    (2023, 6, 1),
]


@pytest.mark.parametrize("ymd", _HISTORICAL)
def test_dut1_matches_skyfield_historical(ymd):
    """Historical DUT1 matches Skyfield to <0.1 ms."""
    y, m, d = ymd
    our = _our_dut1(y, m, d)
    sf = _sf_dut1(y, m, d)
    np.testing.assert_allclose(
        our, sf, atol=1e-4,
        err_msg=f"DUT1 mismatch on {y}-{m:02d}-{d:02d}: ours={our:.7f} sf={sf:.7f}"
    )


# ---------------------------------------------------------------------------
# Recent dates — Bulletin A values, may vary slightly between downloads
# ---------------------------------------------------------------------------

_RECENT = [
    (2024, 1, 1),
    (2024, 6, 15),
    (2024, 11, 1),
    (2025, 1, 1),
]


@pytest.mark.parametrize("ymd", _RECENT)
def test_dut1_matches_skyfield_recent(ymd):
    """Recent DUT1 agrees with Skyfield to <1 ms (same file, different cache age)."""
    y, m, d = ymd
    our = _our_dut1(y, m, d)
    sf = _sf_dut1(y, m, d)
    np.testing.assert_allclose(
        our, sf, atol=1e-3,
        err_msg=f"DUT1 mismatch on {y}-{m:02d}-{d:02d}: ours={our:.7f} sf={sf:.7f}"
    )


# ---------------------------------------------------------------------------
# Range and continuity checks
# ---------------------------------------------------------------------------

def test_table_coverage():
    """Table covers at least 1973–2025."""
    from sgp4jax import _iers
    mjd_min = float(_iers._mjd.min())
    mjd_max = float(_iers._mjd.max())
    # MJD 41684 = 1973-01-02, MJD 60676 = 2025-01-01
    assert mjd_min <= 41684.0, f"Table starts too late: MJD {mjd_min}"
    assert mjd_max >= 60676.0, f"Table ends too early: MJD {mjd_max}"


def test_dut1_magnitude():
    """DUT1 stays within the IERS-maintained ±0.9 s window for modern dates."""
    jd, fr = jday(2000, 1, 1, 0, 0, 0.0)
    jd_end, fr_end = jday(2026, 1, 1, 0, 0, 0.0)

    # Sample ~100 evenly-spaced days
    n = 100
    jds = jnp.linspace(jd + fr, jd_end + fr_end, n)
    jd_arr = jnp.floor(jds)
    fr_arr = jds - jd_arr

    _, fr_ut1 = jax.vmap(utc_to_ut1)(jd_arr, fr_arr)
    dut1 = (fr_ut1 - fr_arr) * 86400.0
    assert float(jnp.max(jnp.abs(dut1))) < 0.9, \
        f"DUT1 outside ±0.9 s window: max |DUT1| = {float(jnp.max(jnp.abs(dut1))):.3f} s"


def test_dut1_continuous():
    """DUT1 changes smoothly between days except at leap-second boundaries.

    Leap seconds introduce ~1 s jumps; otherwise daily changes are <5 ms.
    """
    from sgp4jax import _iers
    dut1_arr = np.array(_iers._dut1)
    day_to_day = np.abs(np.diff(dut1_arr))

    # Jumps due to leap seconds are ~1.0 s; everything else must be < 5 ms
    non_leap = day_to_day[day_to_day < 0.9]
    assert non_leap.max() < 0.05, \
        f"Unexpectedly large non-leap DUT1 jump: {non_leap.max():.4f} s"

    # Count leap-second events (jumps > 0.9 s): ≥25 within our table coverage
    # (27 total since 1972; table starts 1973 so misses the first two)
    leap_jumps = np.sum(day_to_day > 0.9)
    assert leap_jumps >= 25, f"Found only {leap_jumps} leap-second events, expected ≥ 25"


# ---------------------------------------------------------------------------
# JAX compatibility
# ---------------------------------------------------------------------------

def test_utc_to_ut1_jit():
    """utc_to_ut1 is traceable through jax.jit."""
    jd, fr = jday(2020, 6, 1, 12, 0, 0.0)
    jd_ = jnp.float64(jd)
    fr_ = jnp.float64(fr)

    jd1, fr1 = utc_to_ut1(jd_, fr_)
    jd2, fr2 = jax.jit(utc_to_ut1)(jd_, fr_)

    np.testing.assert_allclose(float(fr1), float(fr2), atol=1e-15)


def test_utc_to_ut1_vmap():
    """utc_to_ut1 works under vmap over a batch of times."""
    dates = [(2000, 1, 1), (2010, 6, 15), (2020, 12, 31)]
    jds = []
    frs = []
    expected = []
    for y, m, d in dates:
        jd_v, fr_v = jday(y, m, d, 0, 0, 0.0)
        jds.append(jd_v)
        frs.append(fr_v)
        expected.append(_our_dut1(y, m, d))

    jd_batch = jnp.array(jds, dtype=jnp.float64)
    fr_batch = jnp.array(frs, dtype=jnp.float64)

    _, fr_ut1_batch = jax.vmap(utc_to_ut1)(jd_batch, fr_batch)
    dut1_batch = (fr_ut1_batch - fr_batch) * 86400.0

    for i, (exp, got) in enumerate(zip(expected, dut1_batch)):
        np.testing.assert_allclose(float(got), exp, atol=1e-12,
                                   err_msg=f"vmap result mismatch at index {i}")


def test_utc_to_ut1_magnitude_preserved():
    """Converting to UT1 and back gives the original UTC (UT1−UTC < 1 s)."""
    jd, fr = jday(2023, 3, 15, 6, 30, 0.0)
    jd_ = jnp.float64(jd)
    fr_ = jnp.float64(fr)
    jd_ut1, fr_ut1 = utc_to_ut1(jd_, fr_)
    diff_sec = abs(float(jd_ut1 + fr_ut1) - float(jd_ + fr_)) * 86400.0
    assert diff_sec < 0.9, f"|UT1 - UTC| = {diff_sec:.4f} s > 0.9 s"


# ---------------------------------------------------------------------------
# Integration: full UTC → UT1 → GCRF pipeline vs Skyfield
# ---------------------------------------------------------------------------

def test_itrf_gcrf_with_utc_input():
    """itrf_to_gcrf with utc_to_ut1 agrees with Skyfield ground-station GCRF."""
    from skyfield.api import load, wgs84
    from sgp4jax._frames import itrf_to_gcrf

    ts = load.timescale()
    obs = wgs84.latlon(51.5, -0.1, elevation_m=0)  # near London
    r_itrf_km = np.array(obs.itrs_xyz.km)

    # Use a fixed UTC date well within Bulletin A IERS coverage
    y, m, d = 2023, 6, 1
    jd_utc, fr_utc = jday(y, m, d, 12, 0, 0.0)
    t_sf = ts.utc(y, m, d, 12)

    # Our pipeline: UTC → UT1 → GCRF
    jd_ut1, fr_ut1 = utc_to_ut1(jnp.float64(jd_utc), jnp.float64(fr_utc))
    r_gcrf_ours = itrf_to_gcrf(jnp.array(r_itrf_km), jd_ut1, fr_ut1)

    # Skyfield reference
    r_gcrf_sf = np.array(obs.at(t_sf).position.km)

    np.testing.assert_allclose(
        np.array(r_gcrf_ours), r_gcrf_sf, atol=1e-6,
        err_msg="UTC→UT1→GCRF pipeline disagrees with Skyfield"
    )
