"""Test the orbit fitting example: sgp4init is differentiable and BFGS converges."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.optimize import minimize as jax_minimize

from sgp4jax import WGS72, tle_to_satrec, propagate
from sgp4jax._sgp4init import sgp4init


# ISS TLE for testing
LINE1 = "1 25544U 98067A   24045.51782528  .00016717  00000-0  10270-3 0  9006"
LINE2 = "2 25544  51.6400  10.2827 0003856 197.0300 163.0590 15.49560044439368"


def predict_positions(params, gravity, epoch, jdsatepoch, jdsatepochF, times):
    """Forward model: orbital parameters -> predicted positions."""
    sat = sgp4init(
        gravity, epoch, params[6],
        0.0, 0.0,
        params[2], params[3], params[0], params[4], params[5], params[1],
        jdsatepoch, jdsatepochF,
    )
    r, v, err = jax.vmap(propagate, (None, 0))(sat, times)
    return r


def loss_fn(params, gravity, epoch, jdsatepoch, jdsatepochF, times, r_obs, sigma):
    """Weighted sum of squared residuals."""
    r_pred = predict_positions(params, gravity, epoch, jdsatepoch, jdsatepochF, times)
    residuals = r_pred - r_obs
    return 0.5 * jnp.sum(residuals**2) / sigma**2


class TestSgp4initDifferentiable:
    """sgp4init should be differentiable w.r.t. orbital parameters."""

    def test_gradient_through_sgp4init(self):
        """Gradient of loss w.r.t. params via sgp4init should be finite."""
        sat_true = tle_to_satrec(LINE1, LINE2)
        times = jnp.linspace(0.0, 1440.0, 10)
        r_true, _, _ = jax.vmap(propagate, (None, 0))(sat_true, times)

        true_params = jnp.array([
            sat_true.inclo, sat_true.nodeo, sat_true.ecco,
            sat_true.argpo, sat_true.mo, sat_true.no_kozai, sat_true.bstar,
        ])

        gravity = WGS72
        epoch = float(sat_true.jdsatepoch) + float(sat_true.jdsatepochF) - 2433281.5
        jdsatepoch = float(sat_true.jdsatepoch)
        jdsatepochF = float(sat_true.jdsatepochF)

        grad_fn = jax.grad(loss_fn)
        g = grad_fn(true_params, gravity, epoch, jdsatepoch, jdsatepochF,
                     times, r_true, 1.0)

        assert g.shape == (7,)
        assert jnp.all(jnp.isfinite(g)), f"Non-finite gradients: {g}"

    def test_jacobian_through_sgp4init(self):
        """Jacobian of positions w.r.t. params should be finite."""
        sat_true = tle_to_satrec(LINE1, LINE2)
        times = jnp.linspace(0.0, 1440.0, 5)

        params = jnp.array([
            sat_true.inclo, sat_true.nodeo, sat_true.ecco,
            sat_true.argpo, sat_true.mo, sat_true.no_kozai, sat_true.bstar,
        ])

        gravity = WGS72
        epoch = float(sat_true.jdsatepoch) + float(sat_true.jdsatepochF) - 2433281.5
        jdsatepoch = float(sat_true.jdsatepoch)
        jdsatepochF = float(sat_true.jdsatepochF)

        J = jax.jacobian(predict_positions)(params, gravity, epoch,
                                            jdsatepoch, jdsatepochF, times)
        assert J.shape == (5, 3, 7)
        assert jnp.all(jnp.isfinite(J)), "Non-finite Jacobian entries"


class TestOrbitFitting:
    """End-to-end orbit fitting with BFGS should recover true parameters."""

    @pytest.fixture
    def setup(self):
        sat_true = tle_to_satrec(LINE1, LINE2)
        times = jnp.linspace(0.0, 1440.0, 50)
        r_true, _, _ = jax.vmap(propagate, (None, 0))(sat_true, times)

        sigma = 1.0
        key = jax.random.PRNGKey(42)
        noise = sigma * jax.random.normal(key, shape=r_true.shape)
        r_obs = r_true + noise

        true_params = jnp.array([
            sat_true.inclo, sat_true.nodeo, sat_true.ecco,
            sat_true.argpo, sat_true.mo, sat_true.no_kozai, sat_true.bstar,
        ])

        gravity = WGS72
        epoch = float(sat_true.jdsatepoch) + float(sat_true.jdsatepochF) - 2433281.5
        jdsatepoch = float(sat_true.jdsatepoch)
        jdsatepochF = float(sat_true.jdsatepochF)

        return dict(
            true_params=true_params, times=times, r_obs=r_obs,
            sigma=sigma, gravity=gravity, epoch=epoch,
            jdsatepoch=jdsatepoch, jdsatepochF=jdsatepochF,
        )

    def test_bfgs_converges(self, setup):
        """BFGS optimizer should converge and reduce the loss."""
        d = setup
        perturbation = jnp.array([1e-4, 1e-4, 1e-5, 1e-4, 1e-4, 1e-6, 1e-6])
        key2 = jax.random.PRNGKey(123)
        x0 = d["true_params"] + perturbation * jax.random.normal(key2, shape=(7,))

        param_scale = perturbation

        def scaled_loss(x_scaled, param_scale, gravity, epoch, jdsatepoch,
                        jdsatepochF, times, r_obs, sigma):
            params = x_scaled * param_scale
            return loss_fn(params, gravity, epoch, jdsatepoch, jdsatepochF,
                           times, r_obs, sigma)

        x0_scaled = x0 / param_scale
        result = jax_minimize(
            scaled_loss, x0_scaled,
            args=(param_scale, d["gravity"], d["epoch"], d["jdsatepoch"],
                  d["jdsatepochF"], d["times"], d["r_obs"], d["sigma"]),
            method="BFGS",
        )

        params_fit = result.x * param_scale

        # Loss should be finite
        assert jnp.isfinite(result.fun), f"Loss is not finite: {result.fun}"

        # Fitted positions should have RMS residual below 2x noise level
        r_fit = predict_positions(
            params_fit, d["gravity"], d["epoch"], d["jdsatepoch"],
            d["jdsatepochF"], d["times"])
        residuals = r_fit - d["r_obs"]
        rms = float(jnp.sqrt(jnp.mean(residuals**2)))
        assert rms < 2.0 * d["sigma"], (
            f"RMS residual {rms:.3f} km exceeds 2x noise level"
        )

        # Key orbital parameters should be close to truth
        # inclo and no_kozai are well-constrained
        np.testing.assert_allclose(
            float(params_fit[0]), float(d["true_params"][0]),
            atol=1e-3, err_msg="Inclination not recovered")
        np.testing.assert_allclose(
            float(params_fit[5]), float(d["true_params"][5]),
            atol=1e-5, err_msg="Mean motion not recovered")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
