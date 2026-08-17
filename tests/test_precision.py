"""Tests for the float64 requirement and its error handling.

sgp4jax does not enable ``jax_enable_x64`` globally.  Instead it checks the
setting on import and validates the dtype of times and coordinates at every
entry point.  These tests cover both halves.
"""

import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import sgp4jax
from sgp4jax import _precision


# ISS TLE
LINE1 = "1 25544U 98067A   20045.18587073  .00000950  00000-0  25302-4 0  9990"
LINE2 = "2 25544  51.6443 242.0161 0004397 264.6060 207.3845 15.49165514212791"


@pytest.fixture(scope="module")
def sat():
    return sgp4jax.tle_to_satrec(LINE1, LINE2)


@pytest.fixture(scope="module")
def sats():
    return sgp4jax.tles_to_satrec([[LINE1, LINE2]])


# ---------------------------------------------------------------------------
# The x64 switch itself
# ---------------------------------------------------------------------------

class TestX64Switch:

    def test_x64_enabled_in_test_session(self):
        """conftest.py enables x64 for the whole suite."""
        assert _precision.x64_enabled() is True
        assert sgp4jax.x64_enabled() is True

    def test_require_x64_passes(self):
        assert _precision.require_x64() is None
        assert sgp4jax.require_x64("my_function") is None

    def test_import_does_not_change_global_config(self):
        """Importing sgp4jax must not flip x64 on behalf of the user."""
        src = (
            "import jax, sys\n"
            "assert not jax.config.jax_enable_x64\n"
            "try:\n"
            "    import sgp4jax\n"
            "except RuntimeError:\n"
            "    pass\n"
            "assert not jax.config.jax_enable_x64, 'sgp4jax enabled x64 globally'\n"
        )
        _run_without_x64(src)

    def test_import_without_x64_raises_with_instructions(self):
        """`import sgp4jax` without x64 fails, and says how to fix it."""
        src = (
            "import sgp4jax\n"
        )
        result = _run_without_x64(src, check=False)
        assert result.returncode != 0, result.stdout
        assert "RuntimeError" in result.stderr
        for hint in ("jax_enable_x64", "JAX_ENABLE_X64", "float64"):
            assert hint in result.stderr

    def test_x64_disabled_after_import_is_caught(self, sat):
        """Turning x64 off after import raises rather than silently downcasting."""
        jax.config.update("jax_enable_x64", False)
        try:
            with pytest.raises(RuntimeError, match="jax_enable_x64"):
                sgp4jax.tle_to_satrec(LINE1, LINE2)
            with pytest.raises(RuntimeError, match="jax_enable_x64"):
                sgp4jax.propagate_jd(sat, 2458893.5, 0.7)
        finally:
            jax.config.update("jax_enable_x64", True)


def _run_without_x64(src: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run *src* in a subprocess with x64 disabled and sgp4jax importable."""
    env = dict(os.environ)
    env.pop("JAX_ENABLE_X64", None)
    pkg_parent = str(Path(sgp4jax.__file__).parents[1])
    env["PYTHONPATH"] = os.pathsep.join(
        [pkg_parent] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    return subprocess.run(
        [sys.executable, "-c", src],
        capture_output=True, text=True, env=env, check=check,
    )


# ---------------------------------------------------------------------------
# Times: absolute Julian dates must be float64
# ---------------------------------------------------------------------------

class TestJulianDateDtype:

    def test_propagate_jd_rejects_float32(self, sat):
        jd = jnp.float32(2458893.5)
        fr = jnp.float32(0.7)
        with pytest.raises(TypeError, match="must be a float64 array"):
            sgp4jax.propagate_jd(sat, jd, fr)

    @pytest.mark.parametrize("fn", [
        sgp4jax.propagate_jd,
        sgp4jax.propagate_jd_leo,
        sgp4jax.propagate_jd_gcrf,
    ])
    def test_jd_propagators_reject_float32(self, sat, fn):
        with pytest.raises(TypeError, match="`jd`"):
            fn(sat, jnp.float32(2458893.5), jnp.float64(0.7))
        with pytest.raises(TypeError, match="`fr`"):
            fn(sat, jnp.float64(2458893.5), jnp.float32(0.7))

    @pytest.mark.parametrize("name", [
        "gcrf_positions",
        "gcrf_positions_multi_leo",
        "kepler_gcrf_positions",
    ])
    def test_times_jd_rejects_float32(self, sat, sats, name):
        fn = getattr(sgp4jax, name)
        satrec = sats if "multi" in name else sat
        times = jnp.linspace(2458893.5, 2458894.5, 4, dtype=jnp.float32)
        with pytest.raises(TypeError, match="`times_jd`"):
            fn(satrec, times)

    def test_gcrf_positions_mixed_rejects_float32(self, sats):
        times = jnp.linspace(2458893.5, 2458894.5, 4, dtype=jnp.float32)
        with pytest.raises(TypeError, match="`times_jd`"):
            sgp4jax.gcrf_positions_mixed(sats, times)

    def test_numpy_float32_rejected(self, sat):
        with pytest.raises(TypeError, match="must be a float64 array"):
            sgp4jax.propagate_jd(sat, np.float32(2458893.5), np.float32(0.7))

    def test_error_message_names_argument_and_function(self, sat):
        with pytest.raises(TypeError) as excinfo:
            sgp4jax.propagate_jd(sat, jnp.float32(2458893.5), jnp.float64(0.7))
        message = str(excinfo.value)
        assert "`jd`" in message
        assert "propagate_jd()" in message
        assert "float64" in message

    def test_python_floats_accepted(self, sat):
        r, v, err = sgp4jax.propagate_jd(sat, 2458893.5, 0.7)
        assert r.dtype == jnp.float64
        assert int(err) == 0

    def test_integer_times_promoted(self, sat):
        """Integer Julian dates are exact, so they are promoted, not rejected."""
        r_int, v_int = sgp4jax.gcrf_positions(
            sat, jnp.array([2458894, 2458895], dtype=jnp.int32))
        r_f64, v_f64 = sgp4jax.gcrf_positions(
            sat, jnp.array([2458894.0, 2458895.0]))
        assert r_int.dtype == jnp.float64
        np.testing.assert_allclose(np.asarray(r_int), np.asarray(r_f64))

    def test_float64_inputs_unchanged(self, sat):
        """Validation must not perturb the propagated result."""
        jd, fr = jnp.float64(2458893.5), jnp.float64(0.7)
        r, v, err = sgp4jax.propagate_jd(sat, jd, fr)
        r_ref, v_ref, _ = sgp4jax.propagate(
            sat, (jd - sat.jdsatepoch) * 1440.0 + (fr - sat.jdsatepochF) * 1440.0)
        np.testing.assert_array_equal(np.asarray(r), np.asarray(r_ref))
        np.testing.assert_array_equal(np.asarray(v), np.asarray(v_ref))


# ---------------------------------------------------------------------------
# Coordinates: frame transforms
# ---------------------------------------------------------------------------

class TestFrameDtype:

    def test_teme_to_gcrf_rejects_float32_coordinates(self):
        r = jnp.array([7000.0, 0.0, 0.0], dtype=jnp.float32)
        v = jnp.array([0.0, 7.5, 0.0], dtype=jnp.float32)
        jd, fr = jnp.float64(2458893.5), jnp.float64(0.7)
        with pytest.raises(TypeError, match="`r_teme`"):
            sgp4jax.teme_to_gcrf(r, v, jd, fr)
        with pytest.raises(TypeError, match="`v_teme`"):
            sgp4jax.teme_to_gcrf(r.astype(jnp.float64), v, jd, fr)

    def test_itrf_to_gcrf_rejects_float32(self):
        r = jnp.array([5000.0, 3000.0, 1000.0], dtype=jnp.float32)
        with pytest.raises(TypeError, match="`r_itrf`"):
            sgp4jax.itrf_to_gcrf(r, jnp.float64(2458893.5), jnp.float64(0.7))

    def test_gcrf_to_itrf_rejects_float32_time(self):
        r = jnp.array([5000.0, 3000.0, 1000.0])
        with pytest.raises(TypeError, match="`fr`"):
            sgp4jax.gcrf_to_itrf(r, jnp.float64(2458893.5), jnp.float32(0.7))

    def test_frame_round_trip_still_works(self):
        r = jnp.array([5000.0, 3000.0, 1000.0])
        jd, fr = jnp.float64(2458893.5), jnp.float64(0.7)
        r_gcrf = sgp4jax.itrf_to_gcrf(r, jd, fr)
        r_back = sgp4jax.gcrf_to_itrf(r_gcrf, jd, fr)
        np.testing.assert_allclose(np.asarray(r_back), np.asarray(r), atol=1e-9)

    def test_frame_transforms_still_jit_and_vmap(self):
        r = jnp.tile(jnp.array([5000.0, 3000.0, 1000.0]), (3, 1))
        jd = jnp.full(3, 2458893.5)
        fr = jnp.linspace(0.1, 0.9, 3)
        out = jax.jit(jax.vmap(sgp4jax.itrf_to_gcrf))(r, jd, fr)
        assert out.shape == (3, 3)
        assert out.dtype == jnp.float64


# ---------------------------------------------------------------------------
# SatRec epoch
# ---------------------------------------------------------------------------

class TestSatRecEpoch:

    def test_float32_satrec_rejected_by_jd_propagators(self, sat):
        sat32 = sat._replace(
            jdsatepoch=jnp.float32(sat.jdsatepoch),
            jdsatepochF=jnp.float32(sat.jdsatepochF),
        )
        with pytest.raises(TypeError, match="satrec.jdsatepoch"):
            sgp4jax.propagate_jd(sat32, jnp.float64(2458893.5), jnp.float64(0.7))
        with pytest.raises(TypeError, match="satrec.jdsatepoch"):
            sgp4jax.propagate_gcrf(sat32, jnp.float64(100.0))

    def test_float32_satrec_still_propagates_on_tsince(self, sat):
        """Relative-time propagation stays permissive (see test_float32.py)."""
        from sgp4jax import SatRec
        sat32 = SatRec(*[jnp.float32(f) for f in sat])
        r, v, err = sgp4jax.propagate(sat32, jnp.float32(100.0))
        assert int(err) == 0
        assert jnp.all(jnp.isfinite(r))


# ---------------------------------------------------------------------------
# Covariance entry points
# ---------------------------------------------------------------------------

class TestCovarianceDtype:

    def test_tle_ric_covariance_rejects_float32(self, sat):
        with pytest.raises(TypeError, match="`jd`"):
            sgp4jax.tle_ric_covariance(
                sat, jnp.float32(2458893.5), jnp.float64(0.7))

    def test_elements_jacobian_rejects_float32(self, sat):
        with pytest.raises(TypeError, match="`fr`"):
            sgp4jax.elements_jacobian(
                sat, jnp.float64(2458893.5), jnp.float32(0.7))

    def test_cov_transforms_reject_float32(self, sat):
        cov = jnp.eye(6)
        with pytest.raises(TypeError, match="`jd`"):
            sgp4jax.cov_elements_to_teme(
                cov, sat, jnp.float32(2458893.5), jnp.float64(0.7))


# ---------------------------------------------------------------------------
# IERS
# ---------------------------------------------------------------------------

class TestIersDtype:

    def test_utc_to_ut1_rejects_float32(self):
        from sgp4jax import _iers
        if _iers._mjd is None:
            pytest.skip("IERS table not cached; run update_iers_table()")
        with pytest.raises(TypeError, match="`jd_utc`"):
            sgp4jax.utc_to_ut1(jnp.float32(2458893.5), jnp.float64(0.7))


# ---------------------------------------------------------------------------
# satrec_from_elements
# ---------------------------------------------------------------------------

class TestSatrecFromElements:

    def test_float32_epoch_rejected(self):
        with pytest.raises(TypeError, match="`epoch_jd`"):
            sgp4jax.satrec_from_elements(
                inclo=0.9006, nodeo=1.2217, ecco=0.0004, argpo=4.6194,
                mo=3.6160, no_kozai=0.0672,
                epoch_jd=jnp.float32(2458924.686),
            )

    def test_float64_epoch_accepted(self):
        sat = sgp4jax.satrec_from_elements(
            inclo=0.9006, nodeo=1.2217, ecco=0.0004, argpo=4.6194,
            mo=3.6160, no_kozai=0.0672, epoch_jd=2458924.686,
        )
        assert sat.jdsatepoch.dtype == jnp.float64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
