"""Network-dependent IERS tests.

These tests download data from the IERS data centre and are excluded from the
default test run.  Run them explicitly with::

    pytest -m network

or together with the full suite::

    pytest -m ''
"""

import numpy as np
import pytest

pytestmark = pytest.mark.network


def test_fetch_returns_iers_text():
    """_fetch() downloads finals2000A.all and returns non-empty ASCII text."""
    from sgp4jax._iers import _fetch

    text = _fetch()
    assert len(text) > 50_000, f"Expected a large IERS file, got {len(text)} bytes"
    # Bulletin A 'I' (observed) and 'P' (predicted) flags must both be present
    assert 'I' in text
    assert 'P' in text


def test_update_iers_table_downloads(tmp_path):
    """update_iers_table() fetches, parses, and writes a valid .npz cache."""
    from sgp4jax import update_iers_table

    cache = tmp_path / "finals2000A.npz"
    update_iers_table(cache_path=cache)

    assert cache.exists(), "Cache file was not created"
    data = np.load(cache)
    assert set(data.files) >= {"mjd", "dut1"}
    assert len(data["mjd"]) > 10_000, "Expected thousands of daily MJD entries"
    assert np.all(np.diff(data["mjd"]) > 0), "MJD must be monotonically increasing"
    assert np.all(np.abs(data["dut1"]) < 1.0), "All DUT1 values must be within ±1 s"


def test_update_iers_table_refreshes_module_globals(tmp_path):
    """update_iers_table() replaces the module-level _mjd and _dut1 arrays."""
    import sgp4jax._iers as _iers_mod
    from sgp4jax import update_iers_table

    old_mjd = _iers_mod._mjd
    try:
        update_iers_table(cache_path=tmp_path / "tmp.npz")
        assert _iers_mod._mjd is not None
        assert _iers_mod._dut1 is not None
        assert len(_iers_mod._mjd) > 10_000
    finally:
        _iers_mod._mjd = old_mjd  # restore so other tests are unaffected
