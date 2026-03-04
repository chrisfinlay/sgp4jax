"""Test float32 precision and JIT compatibility."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgp4jax import tle_to_satrec, propagate, WGS72, SatRec


# ISS TLE
LINE1 = '1 25544U 98067A   20045.18587073  .00000950  00000-0  25302-4 0  9990'
LINE2 = '2 25544  51.6443 242.0161 0004397 264.6060 207.3845 15.49165514212791'


def satrec_to_float32(sat):
    """Cast all SatRec fields from float64 to float32."""
    return SatRec(*[jnp.float32(f) for f in sat])


class TestFloat32Propagation:
    """Float32 propagation should produce physically reasonable results."""

    def test_float32_produces_valid_results(self):
        sat64 = tle_to_satrec(LINE1, LINE2, gravity=WGS72)
        sat32 = satrec_to_float32(sat64)

        r32, v32, err32 = propagate(sat32, jnp.float32(100.0))

        assert int(err32) == 0, f"Float32 propagation error: {err32}"
        assert jnp.all(jnp.isfinite(r32)), "Float32 position should be finite"
        assert jnp.all(jnp.isfinite(v32)), "Float32 velocity should be finite"

    def test_float32_physically_reasonable(self):
        """LEO satellite position should be ~6500-7000 km from Earth center."""
        sat64 = tle_to_satrec(LINE1, LINE2, gravity=WGS72)
        sat32 = satrec_to_float32(sat64)

        r32, v32, err32 = propagate(sat32, jnp.float32(0.0))

        pos_mag = float(jnp.linalg.norm(r32))
        # ISS orbits at ~400 km altitude → ~6778 km from center
        assert 6000.0 < pos_mag < 8000.0, (
            f"Position magnitude {pos_mag:.1f} km not in LEO range")

        vel_mag = float(jnp.linalg.norm(v32))
        # LEO velocity ~7.5 km/s
        assert 5.0 < vel_mag < 10.0, (
            f"Velocity magnitude {vel_mag:.3f} km/s not in LEO range")

    def test_float32_vs_float64_precision(self):
        """Document float32 precision degradation vs float64 reference."""
        sat64 = tle_to_satrec(LINE1, LINE2, gravity=WGS72)
        sat32 = satrec_to_float32(sat64)

        r64, v64, _ = propagate(sat64, jnp.array(100.0))
        r32, v32, _ = propagate(sat32, jnp.float32(100.0))

        pos_err = float(jnp.linalg.norm(
            jnp.float64(r32) - r64))
        vel_err = float(jnp.linalg.norm(
            jnp.float64(v32) - v64))

        # Float32 has ~7 decimal digits of precision.
        # For positions of ~7000 km, we expect ~1 m precision at best.
        # Use relaxed tolerance: 1 km position, 1 m/s velocity
        assert pos_err < 1.0, (
            f"Float32 position error {pos_err:.4f} km exceeds 1 km threshold")
        assert vel_err < 0.01, (
            f"Float32 velocity error {vel_err:.6f} km/s exceeds 0.01 km/s threshold")

    def test_float32_jit(self):
        """JIT should work with float32 inputs."""
        sat64 = tle_to_satrec(LINE1, LINE2, gravity=WGS72)
        sat32 = satrec_to_float32(sat64)

        jitted = jax.jit(propagate)
        r, v, err = jitted(sat32, jnp.float32(100.0))

        assert int(err) == 0
        assert jnp.all(jnp.isfinite(r))

    def test_float32_multiple_times(self):
        """Float32 should work across a range of times."""
        sat64 = tle_to_satrec(LINE1, LINE2, gravity=WGS72)
        sat32 = satrec_to_float32(sat64)

        times = jnp.float32(jnp.array([0.0, 60.0, 360.0, 720.0, 1440.0]))
        batched = jax.vmap(propagate, in_axes=(None, 0))
        r_batch, v_batch, err_batch = batched(sat32, times)

        assert r_batch.shape == (5, 3)
        assert jnp.all(err_batch == 0)
        # All positions should be physically reasonable
        norms = jnp.linalg.norm(r_batch, axis=1)
        assert jnp.all(norms > 6000.0)
        assert jnp.all(norms < 8000.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
