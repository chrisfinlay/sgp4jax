"""IERS Earth Orientation Data — UTC to UT1 conversion.

Downloads and caches the IERS finals2000A.all file (Bulletin A, updated
weekly).  Provides DUT1 = UT1 − UTC interpolated to sub-microsecond
precision, matching Skyfield's ``t.dut1`` values exactly.

Typical workflow::

    import sgp4jax

    # Once (or periodically to refresh predictions):
    sgp4jax.update_iers_table()

    # Every session (automatic if the default cache exists):
    # sgp4jax.load_iers_table()   # called at import time

    # Convert UTC → UT1 before frame transformations:
    jd_ut1, fr_ut1 = sgp4jax.utc_to_ut1(jd_utc, fr_utc)
    r_gcrf = sgp4jax.itrf_to_gcrf(r_itrf, jd_ut1, fr_ut1)

Column layout in finals2000A.all (0-indexed, same as Skyfield)
--------------------------------------------------------------
  [7:15]   MJD (F8.2)
  [57]     UT1 source flag: 'I' = IERS observed, 'P' = prediction
  [58:68]  UT1−UTC (F10.7, seconds)  — Bulletin A value used by Skyfield
"""

from __future__ import annotations

import urllib.request
import urllib.error
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import jax.typing

from sgp4jax._precision import check_jd_fr, require_x64

_IERS_URLS = [
    "https://datacenter.iers.org/data/9/finals2000A.all",
    "https://maia.usno.navy.mil/ser7/finals2000A.all",
]

_DEFAULT_CACHE = Path.home() / ".cache" / "sgp4jax" / "finals2000A.npz"

# Module-level interpolation table (populated by load_iers_table /
# update_iers_table and captured as constants by JAX JIT).
_mjd: jax.Array | None = None
_dut1: jax.Array | None = None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_finals2000A(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse finals2000A.all text into (mjd, dut1_seconds) arrays.

    Uses the Bulletin A UT1−UTC column (flag at index 57, value at
    indices 58-68).  This exactly matches Skyfield's parsing, giving
    agreement to floating-point precision for all covered dates.
    """
    mjd_list: list[float] = []
    dut1_list: list[float] = []

    for line in text.splitlines():
        if len(line) < 68:
            continue
        try:
            mjd = float(line[7:15])
        except ValueError:
            continue

        flag = line[57]
        if flag not in ('I', 'P'):
            continue
        try:
            dut1 = float(line[58:68])
        except ValueError:
            continue

        mjd_list.append(mjd)
        dut1_list.append(dut1)

    if not mjd_list:
        raise ValueError("No UT1-UTC data found in finals2000A.all")

    return (np.array(mjd_list, dtype=np.float64),
            np.array(dut1_list, dtype=np.float64))


def _fetch(url: str | None = None) -> str:
    """Download finals2000A.all, trying each URL in _IERS_URLS."""
    urls = [url] if url else _IERS_URLS
    last_exc: Exception | None = None
    for u in urls:
        try:
            with urllib.request.urlopen(u, timeout=30) as resp:
                return resp.read().decode("ascii", errors="replace")  # type: ignore[no-any-return]
        except (urllib.error.URLError, OSError) as exc:
            last_exc = exc
    raise RuntimeError(
        "Failed to download IERS finals2000A.all. Tried:\n"
        + "\n".join(f"  {u}" for u in urls)
        + (f"\nLast error: {last_exc}" if last_exc else "")
    )


# ---------------------------------------------------------------------------
# Public API: manage the cache
# ---------------------------------------------------------------------------

def update_iers_table(
    cache_path: str | Path | None = None,
    url: str | None = None,
) -> None:
    """Download the latest IERS finals2000A table and refresh the cache.

    Fetches the file from the IERS data centre (falling back to USNO),
    parses the Bulletin A UT1-UTC values, writes a compact ``.npz`` cache,
    and immediately loads the result into the module-level interpolation
    table used by :func:`utc_to_ut1`.

    Parameters
    ----------
    cache_path : str or Path, optional
        Destination for the ``.npz`` cache.  Default:
        ``~/.cache/sgp4jax/finals2000A.npz``.
    url : str, optional
        Override the download URL.
    """
    global _mjd, _dut1

    require_x64("update_iers_table")
    cache_path = Path(cache_path) if cache_path else _DEFAULT_CACHE
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    text = _fetch(url)
    mjd_arr, dut1_arr = _parse_finals2000A(text)

    np.savez(cache_path, mjd=mjd_arr, dut1=dut1_arr)
    _mjd = jnp.array(mjd_arr, dtype=jnp.float64)
    _dut1 = jnp.array(dut1_arr, dtype=jnp.float64)

    # Approximate date range for display
    mjd0, mjd1 = float(mjd_arr[0]), float(mjd_arr[-1])
    y0 = 2000.0 + (mjd0 - 51544.5) / 365.25
    y1 = 2000.0 + (mjd1 - 51544.5) / 365.25
    print(
        f"IERS table updated: {len(mjd_arr)} daily entries "
        f"(MJD {mjd0:.1f}–{mjd1:.1f}, ~{y0:.1f}–{y1:.1f}). "
        f"Cached at {cache_path}"
    )


def load_iers_table(
    cache_path: str | Path | None = None,
) -> None:
    """Load the IERS table from the local ``.npz`` cache.

    Called automatically at module import if the default cache exists.
    Re-call with an explicit *cache_path* to load from a non-default
    location or to reload after :func:`update_iers_table`.

    Parameters
    ----------
    cache_path : str or Path, optional
        Path to the ``.npz`` cache file.  Default:
        ``~/.cache/sgp4jax/finals2000A.npz``.

    Raises
    ------
    FileNotFoundError
        Cache file not found.  Call :func:`update_iers_table` to download it.
    """
    global _mjd, _dut1

    require_x64("load_iers_table")
    cache_path = Path(cache_path) if cache_path else _DEFAULT_CACHE
    if not cache_path.exists():
        raise FileNotFoundError(
            f"IERS cache not found at {cache_path}.\n"
            "Run:  import sgp4jax; sgp4jax.update_iers_table()"
        )

    data = np.load(cache_path)
    _mjd = jnp.array(data["mjd"], dtype=jnp.float64)
    _dut1 = jnp.array(data["dut1"], dtype=jnp.float64)


# ---------------------------------------------------------------------------
# Public API: UTC → UT1
# ---------------------------------------------------------------------------

def utc_to_ut1(
    jd_utc: jax.typing.ArrayLike,
    fr_utc: jax.typing.ArrayLike,
) -> tuple[jax.Array, jax.Array]:
    """Convert a UTC Julian date to UT1 using the IERS finals2000A table.

    Interpolates DUT1 = UT1 − UTC (seconds) from the Bulletin A column
    of the loaded IERS table and adjusts the fractional Julian date.
    Matches Skyfield's ``t.dut1`` values to better than 1 µs for all
    dates within the table's coverage (~1972 to ~1 year ahead).

    The function is JAX-traceable and works inside ``jax.jit`` and
    ``jax.vmap`` after the table has been loaded.

    Parameters
    ----------
    jd_utc : array-like
        UTC Julian date, integer/whole part.
    fr_utc : array-like
        UTC Julian date, fractional part.

    Returns
    -------
    jd_ut1 : jax.Array
        UT1 Julian date, whole part (same value as ``jd_utc``).
    fr_ut1 : jax.Array
        UT1 Julian date, fractional part.

    Raises
    ------
    RuntimeError
        IERS table not loaded — call :func:`update_iers_table` or
        :func:`load_iers_table` first.
    TypeError
        *jd_utc* or *fr_utc* is not float64.
    """
    if _mjd is None or _dut1 is None:
        raise RuntimeError(
            "IERS table not loaded.\n"
            "Run:  import sgp4jax; sgp4jax.update_iers_table()"
        )

    jd_utc, fr_utc = check_jd_fr(
        jd_utc, fr_utc, context="utc_to_ut1", names=("jd_utc", "fr_utc"))

    mjd = jd_utc + fr_utc - 2400000.5
    dut1 = jnp.interp(mjd, _mjd, _dut1)
    fr_ut1 = fr_utc + dut1 / 86400.0
    return jd_utc, fr_ut1


# ---------------------------------------------------------------------------
# Auto-load at import
# ---------------------------------------------------------------------------

try:
    load_iers_table()
except FileNotFoundError:
    pass
