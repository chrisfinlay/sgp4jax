"""Test gradient computation and validation against finite differences."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgp4jax import tle_to_satrec, tles_to_satrec, propagate, make_satrec, WGS72


# Near-earth (ISS)
LINE1_NEAR = '1 25544U 98067A   20045.18587073  .00000950  00000-0  25302-4 0  9990'
LINE2_NEAR = '2 25544  51.6443 242.0161 0004397 264.6060 207.3845 15.49165514212791'

# Deep-space (Molniya)
LINE1_DEEP = '1 09880U 77021A   06176.56157475  .00000421  00000-0  10000-3 0  9814'
LINE2_DEEP = '2 09880  64.5968 349.3786 7069051 270.0229  16.3320  2.00813614112380'


def finite_diff_grad(fn, x, eps=1e-6):
    """Compute finite-difference gradient of scalar function fn at x."""
    return (fn(x + eps) - fn(x - eps)) / (2.0 * eps)


class TestGradWrtTime:
    """Gradient of position norm w.r.t. tsince."""

    def test_near_earth_grad_time(self):
        sat = tle_to_satrec(LINE1_NEAR, LINE2_NEAR, gravity=WGS72)

        def loss(t):
            r, v, err = propagate(sat, t)
            return jnp.sum(r ** 2)

        t0 = jnp.array(100.0)
        ad_grad = float(jax.grad(loss)(t0))
        fd_grad = float(finite_diff_grad(loss, t0))

        assert jnp.isfinite(ad_grad)
        np.testing.assert_allclose(ad_grad, fd_grad, rtol=1e-4,
                                   err_msg="Near-earth grad vs finite diff")

    def test_deep_space_grad_time(self):
        sat = tle_to_satrec(LINE1_DEEP, LINE2_DEEP, gravity=WGS72)

        def loss(t):
            r, v, err = propagate(sat, t)
            return jnp.sum(r ** 2)

        t0 = jnp.array(100.0)
        ad_grad = float(jax.grad(loss)(t0))
        fd_grad = float(finite_diff_grad(loss, t0))

        assert jnp.isfinite(ad_grad)
        np.testing.assert_allclose(ad_grad, fd_grad, rtol=1e-4,
                                   err_msg="Deep-space grad vs finite diff")


class TestGradWrtTLEElements:
    """Gradient of position norm w.r.t. TLE elements via custom SatRec."""

    @pytest.fixture
    def base_satrec(self):
        return tle_to_satrec(LINE1_NEAR, LINE2_NEAR, gravity=WGS72)

    @pytest.mark.parametrize("field", [
        "bstar", "ecco", "mo", "no_kozai",
    ])
    def test_grad_wrt_element(self, base_satrec, field):
        """Gradient w.r.t. each TLE element should match finite differences.

        Note: Only fields that don't trigger re-initialization through
        deep-space while_loop are tested here. Fields like inclo, nodeo,
        argpo can change the satellite classification and trigger
        non-differentiable code paths.
        """
        sat = base_satrec
        field_idx = sat._fields.index(field)
        base_val = float(getattr(sat, field))

        def loss_from_field(val):
            fields = list(sat)
            fields[field_idx] = val
            modified_sat = type(sat)(*fields)
            r, v, err = propagate(modified_sat, jnp.array(100.0))
            return jnp.sum(r ** 2)

        x0 = jnp.array(base_val)
        ad_grad = float(jax.grad(loss_from_field)(x0))
        fd_grad = float(finite_diff_grad(loss_from_field, x0, eps=1e-8))

        assert jnp.isfinite(ad_grad), f"Non-finite gradient for {field}"
        if abs(fd_grad) > 1e-10:
            np.testing.assert_allclose(
                ad_grad, fd_grad, rtol=1e-3,
                err_msg=f"Gradient mismatch for {field}")


class TestJacobian:
    """Full Jacobian of (r, v) w.r.t. tsince."""

    def test_jacobian_wrt_time(self):
        sat = tle_to_satrec(LINE1_NEAR, LINE2_NEAR, gravity=WGS72)

        def state(t):
            r, v, err = propagate(sat, t)
            return jnp.concatenate([r, v])

        t0 = jnp.array(100.0)
        jac = jax.jacobian(state)(t0)

        assert jac.shape == (6,), "Jacobian should be 6-element vector (d(r,v)/dt)"
        assert jnp.all(jnp.isfinite(jac)), "Jacobian should be finite"

        # Validate against finite differences
        eps = 1e-6
        fd_jac = (state(t0 + eps) - state(t0 - eps)) / (2 * eps)
        np.testing.assert_allclose(np.array(jac), np.array(fd_jac), rtol=1e-4)


class TestVmapPlusGrad:
    """Combined vmap + grad: per-satellite gradients in a batch."""

    def test_per_satellite_gradients(self):
        tles = [
            (LINE1_NEAR, LINE2_NEAR),
            (LINE1_DEEP, LINE2_DEEP),
        ]
        batch_sat = tles_to_satrec(tles, gravity=WGS72)

        def single_loss(sat, t):
            r, v, err = propagate(sat, t)
            return jnp.sum(r ** 2)

        # Grad w.r.t. time, vmap over satellites
        grad_fn = jax.grad(single_loss, argnums=1)
        batched_grad = jax.vmap(grad_fn, in_axes=(0, None))

        t0 = jnp.array(100.0)
        grads = batched_grad(batch_sat, t0)

        assert grads.shape == (2,)
        assert jnp.all(jnp.isfinite(grads))

        # Each gradient should match the sequential version
        for i, (l1, l2) in enumerate(tles):
            sat = tle_to_satrec(l1, l2, gravity=WGS72)
            expected = float(jax.grad(single_loss, argnums=1)(sat, t0))
            np.testing.assert_allclose(
                float(grads[i]), expected, rtol=1e-10,
                err_msg=f"Batched gradient mismatch for satellite {i}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
