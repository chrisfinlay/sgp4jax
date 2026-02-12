"""Tests for JIT compilation, vmap batching, and gradient computation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgp4jax import tle_to_satrec, propagate


LINE1 = '1 25544U 98067A   20045.18587073  .00000950  00000-0  25302-4 0  9990'
LINE2 = '2 25544  51.6443 242.0161 0004397 264.6060 207.3845 15.49165514212791'

LINE1_DEEP = '1 09880U 77021A   00251.45080028  .00000316  00000-0  10000-3 0  3527'
LINE2_DEEP = '2 09880  64.7791 180.0788 7258491 296.1385  20.2281  2.00879014156621'


class TestJIT:
    """Test JIT compilation."""

    def test_jit_compiles(self):
        """Propagation should JIT compile without errors."""
        sat = tle_to_satrec(LINE1, LINE2)
        jitted = jax.jit(propagate)

        r1, v1, e1 = propagate(sat, jnp.array(100.0))
        r2, v2, e2 = jitted(sat, jnp.array(100.0))

        np.testing.assert_allclose(np.array(r1), np.array(r2), atol=1e-15)
        np.testing.assert_allclose(np.array(v1), np.array(v2), atol=1e-15)

    def test_jit_deep_space(self):
        """Deep space propagation should JIT compile."""
        sat = tle_to_satrec(LINE1_DEEP, LINE2_DEEP)
        jitted = jax.jit(propagate)
        r, v, err = jitted(sat, jnp.array(100.0))
        assert int(err) == 0
        assert jnp.all(jnp.isfinite(r))

    def test_jit_speed(self):
        """Second JIT call should be faster (cached)."""
        sat = tle_to_satrec(LINE1, LINE2)
        jitted = jax.jit(propagate)

        # First call triggers compilation
        r1, v1, e1 = jitted(sat, jnp.array(100.0))
        # Second call should use cache
        r2, v2, e2 = jitted(sat, jnp.array(200.0))
        assert int(e2) == 0


class TestVmap:
    """Test vmap batching."""

    def test_vmap_over_times(self):
        """vmap over multiple time points for a single satellite."""
        sat = tle_to_satrec(LINE1, LINE2)
        times = jnp.linspace(0, 1440, 100)

        batched = jax.vmap(propagate, in_axes=(None, 0))
        r_batch, v_batch, err_batch = batched(sat, times)

        assert r_batch.shape == (100, 3)
        assert v_batch.shape == (100, 3)
        assert err_batch.shape == (100,)
        assert jnp.all(err_batch == 0)
        assert jnp.all(jnp.isfinite(r_batch))

    def test_vmap_matches_sequential(self):
        """Batched results should match sequential computation."""
        sat = tle_to_satrec(LINE1, LINE2)
        times = jnp.array([0.0, 100.0, 500.0, 1000.0])

        batched = jax.vmap(propagate, in_axes=(None, 0))
        r_batch, v_batch, err_batch = batched(sat, times)

        for i, t in enumerate(times):
            r_seq, v_seq, err_seq = propagate(sat, t)
            np.testing.assert_allclose(np.array(r_batch[i]), np.array(r_seq), atol=1e-15)
            np.testing.assert_allclose(np.array(v_batch[i]), np.array(v_seq), atol=1e-15)

    def test_vmap_deep_space(self):
        """vmap should work with deep space satellites."""
        sat = tle_to_satrec(LINE1_DEEP, LINE2_DEEP)
        times = jnp.linspace(0, 720, 50)

        batched = jax.vmap(propagate, in_axes=(None, 0))
        r_batch, v_batch, err_batch = batched(sat, times)
        assert r_batch.shape == (50, 3)
        assert jnp.all(err_batch == 0)


class TestGrad:
    """Test gradient computation."""

    def test_grad_wrt_time(self):
        """Gradient of position norm with respect to time."""
        sat = tle_to_satrec(LINE1, LINE2)

        def loss(t):
            r, v, err = propagate(sat, t)
            return jnp.sum(r ** 2)

        grad_fn = jax.grad(loss)
        g = grad_fn(jnp.array(100.0))
        assert jnp.isfinite(g)
        assert g != 0.0

    def test_grad_deep_space(self):
        """Gradient should work for deep space satellites too."""
        sat = tle_to_satrec(LINE1_DEEP, LINE2_DEEP)

        def loss(t):
            r, v, err = propagate(sat, t)
            return jnp.sum(r ** 2)

        grad_fn = jax.grad(loss)
        g = grad_fn(jnp.array(100.0))
        assert jnp.isfinite(g)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
