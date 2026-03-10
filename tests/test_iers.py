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
import sgp4jax._iers as _iers_mod


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


# ---------------------------------------------------------------------------
# Unit tests: parser and error paths (no network required)
# ---------------------------------------------------------------------------

def _make_iers_line(mjd: float, flag: str, dut1: float) -> str:
    """Build a minimal finals2000A.all line matching the fixed-column spec."""
    buf = list(' ' * 80)
    for i, ch in enumerate(f"{mjd:8.2f}"):
        buf[7 + i] = ch
    buf[57] = flag
    for i, ch in enumerate(f"{dut1:10.7f}"):
        buf[58 + i] = ch
    return ''.join(buf)


def test_parse_finals2000A_valid_lines():
    """Parser extracts MJD and DUT1 from I/P-flagged lines."""
    from sgp4jax._iers import _parse_finals2000A
    text = '\n'.join([
        _make_iers_line(50001.0, 'I', 0.1234567),
        _make_iers_line(50002.0, 'P', 0.2345678),
    ])
    mjd, dut1 = _parse_finals2000A(text)
    np.testing.assert_array_equal(mjd, [50001.0, 50002.0])
    np.testing.assert_allclose(dut1, [0.1234567, 0.2345678], atol=1e-9)


def test_parse_finals2000A_skips_invalid_lines():
    """Parser silently skips short lines, bad flags, and non-numeric fields."""
    from sgp4jax._iers import _parse_finals2000A

    # Non-numeric MJD
    bad_mjd = list(_make_iers_line(50003.0, 'I', 0.0))
    bad_mjd[7:15] = list('XXXXXXXX')

    # Invalid flag ('X' is not 'I' or 'P')
    bad_flag = _make_iers_line(50004.0, 'X', 0.0)

    # Non-numeric DUT1
    bad_dut1 = list(_make_iers_line(50005.0, 'I', 0.0))
    bad_dut1[58:68] = list('NOT_FLOAT.')

    text = '\n'.join([
        'too short',                          # len < 68
        ''.join(bad_mjd),
        bad_flag,
        ''.join(bad_dut1),
        _make_iers_line(50006.0, 'I', 0.5),  # only valid line
    ])
    mjd, dut1 = _parse_finals2000A(text)
    assert list(mjd) == [50006.0]
    assert float(dut1[0]) == pytest.approx(0.5, abs=1e-9)


def test_parse_finals2000A_empty_raises():
    """Parser raises ValueError when no valid data lines are found."""
    from sgp4jax._iers import _parse_finals2000A
    with pytest.raises(ValueError, match="No UT1-UTC data"):
        _parse_finals2000A("no valid lines here\nshort\n")


def test_load_iers_table_missing_file():
    """load_iers_table raises FileNotFoundError for a non-existent path."""
    with pytest.raises(FileNotFoundError, match="IERS cache not found"):
        load_iers_table("/nonexistent/sgp4jax_iers_missing_test.npz")


def test_utc_to_ut1_raises_when_not_loaded():
    """utc_to_ut1 raises RuntimeError when the IERS table has not been loaded."""
    old_mjd, old_dut1 = _iers_mod._mjd, _iers_mod._dut1
    try:
        _iers_mod._mjd = None
        _iers_mod._dut1 = None
        with pytest.raises(RuntimeError, match="IERS table not loaded"):
            utc_to_ut1(jnp.float64(2451545.0), jnp.float64(0.0))
    finally:
        _iers_mod._mjd = old_mjd
        _iers_mod._dut1 = old_dut1
