#!/usr/bin/env python
"""Orbit Fitting with sgp4jax
==============================

This example demonstrates how to use sgp4jax's **differentiable SGP4
propagator** to fit orbital elements to noisy position observations.

Because sgp4jax is built on JAX, we get automatic differentiation for
free — enabling gradient-based optimization and analytic uncertainty
estimation.

Workflow:

1. Generate synthetic observations from a known TLE with Gaussian noise
2. Define a forward model mapping orbital parameters → predicted positions
3. Fit 7 orbital parameters using JAX's built-in BFGS optimizer
4. Estimate parameter uncertainties via Fisher information
"""

# %%
# Imports
# -------

import jax

# sgp4jax requires JAX double precision; enable it before importing.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.scipy.optimize import minimize as jax_minimize

import sgp4jax
from sgp4jax import WGS72, tle_to_satrec, propagate
from sgp4jax._sgp4init import sgp4init


# %%
# 1. Generate Synthetic Observations
# -----------------------------------
#
# We start from a known ISS TLE as ground truth, propagate it to 50 time
# points over one day (1440 minutes), and add Gaussian noise with σ = 1 km
# to the positions.  This noise level is realistic for ground-based radar
# tracking.

line1 = "1 25544U 98067A   24045.51782528  .00016717  00000-0  10270-3 0  9006"
line2 = "2 25544  51.6400  10.2827 0003856 197.0300 163.0590 15.49560044439368"

sat_true = tle_to_satrec(line1, line2)

times = jnp.linspace(0.0, 1440.0, 50)

r_true, v_true, errs = jax.vmap(propagate, (None, 0))(sat_true, times)
print(f"Propagated {len(times)} time steps, positions shape: {r_true.shape}")

sigma = 1.0  # km
key = jax.random.PRNGKey(42)
noise = sigma * jax.random.normal(key, shape=r_true.shape)
r_obs = r_true + noise

print(f"RMS noise: {jnp.sqrt(jnp.mean(noise**2)):.3f} km")


# %%
# 2. The Forward Model
# ---------------------
#
# We define a function that takes 7 orbital parameters, builds a ``SatRec``
# via ``sgp4init``, and propagates to all observation times.
#
# ========  ============  ================================
# Index     Parameter     Description
# ========  ============  ================================
# 0         ``inclo``     Inclination (rad)
# 1         ``nodeo``     Right ascension of ascending node (rad)
# 2         ``ecco``      Eccentricity
# 3         ``argpo``     Argument of perigee (rad)
# 4         ``mo``        Mean anomaly (rad)
# 5         ``no_kozai``  Mean motion (rad/min), Kozai
# 6         ``bstar``     Drag coefficient (B*)
# ========  ============  ================================

def predict_positions(params, gravity, epoch, jdsatepoch, jdsatepochF, times):
    """Forward model: orbital parameters -> predicted positions."""
    inclo = params[0]
    nodeo = params[1]
    ecco = params[2]
    argpo = params[3]
    mo = params[4]
    no_kozai = params[5]
    bstar = params[6]

    sat = sgp4init(
        gravity, epoch, bstar,
        0.0, 0.0,  # ndot, nddot (fixed)
        ecco, argpo, inclo, mo, no_kozai, nodeo,
        jdsatepoch, jdsatepochF,
    )
    r, v, err = jax.vmap(propagate, (None, 0))(sat, times)
    return r  # (n_times, 3)


# %%
# 3. Loss Function
# -----------------
#
# Weighted sum of squared residuals:
#
# .. math::
#
#    \mathcal{L}(\boldsymbol{\theta})
#    = \frac{1}{2\sigma^2}
#      \sum_{i=1}^{N}
#      \|\mathbf{r}_\text{pred}(t_i;\boldsymbol{\theta})
#        - \mathbf{r}_\text{obs}(t_i)\|^2

def loss_fn(params, gravity, epoch, jdsatepoch, jdsatepochF, times, r_obs, sigma):
    """Weighted sum of squared residuals."""
    r_pred = predict_positions(params, gravity, epoch, jdsatepoch, jdsatepochF, times)
    residuals = r_pred - r_obs
    return 0.5 * jnp.sum(residuals**2) / sigma**2


# %%
# 4. Initial Guess & Optimization
# --------------------------------
#
# We start from a slightly perturbed version of the true parameters and
# use JAX's built-in BFGS optimizer (``jax.scipy.optimize.minimize``).
#
# Parameter scaling is critical: we normalize by the expected perturbation
# size so that the BFGS initial Hessian approximation (identity) produces
# conservative step sizes across all parameters.

true_params = jnp.array([
    sat_true.inclo,
    sat_true.nodeo,
    sat_true.ecco,
    sat_true.argpo,
    sat_true.mo,
    sat_true.no_kozai,
    sat_true.bstar,
])

gravity = WGS72
epoch = float(sat_true.jdsatepoch) + float(sat_true.jdsatepochF) - 2433281.5
jdsatepoch = float(sat_true.jdsatepoch)
jdsatepochF = float(sat_true.jdsatepochF)

key2 = jax.random.PRNGKey(123)
perturbation = jnp.array([1e-4, 1e-4, 1e-5, 1e-4, 1e-4, 1e-6, 1e-6])
x0 = true_params + perturbation * jax.random.normal(key2, shape=(7,))

param_names = ["inclo", "nodeo", "ecco", "argpo", "mo", "no_kozai", "bstar"]

# Parameter scaling: normalize by perturbation magnitude so that a
# unit step in the scaled space corresponds to a perturbation-sized
# step in physical space.  This keeps the BFGS line search stable.
param_scale = perturbation


def scaled_loss(x_scaled, param_scale, gravity, epoch, jdsatepoch,
                jdsatepochF, times, r_obs, sigma):
    """Loss in normalized parameter space."""
    params = x_scaled * param_scale
    return loss_fn(params, gravity, epoch, jdsatepoch, jdsatepochF,
                   times, r_obs, sigma)


x0_scaled = x0 / param_scale

result = jax_minimize(
    scaled_loss, x0_scaled,
    args=(param_scale, gravity, epoch, jdsatepoch, jdsatepochF,
          times, r_obs, sigma),
    method="BFGS",
)

params_fit = result.x * param_scale
print(f"\nOptimization finished: fun_val = {float(result.fun):.4f}, "
      f"nit = {int(result.nit)}")
print()
print("Parameter        Initial          Optimized        True")
print("-" * 70)
for name, xi, fi, ti in zip(param_names, x0, params_fit, true_params):
    print(f"{name:12s}  {float(xi):14.8f}  {float(fi):14.8f}  {float(ti):14.8f}")


# %%
# 5. Fisher Information & Parameter Uncertainties
# -------------------------------------------------
#
# With the Jacobian of the forward model we estimate parameter
# uncertainties via the Fisher information matrix:
#
# .. math::
#
#    \mathbf{F} = \frac{1}{\sigma^2}\,\mathbf{J}^\top\mathbf{J},
#    \qquad
#    \mathrm{Cov}(\boldsymbol{\theta}) \approx \mathbf{F}^{-1}
#
# The 1-σ uncertainties are
# :math:`\sqrt{\mathrm{diag}(\mathbf{F}^{-1})}`.

jacobian_fn = jax.jit(jax.jacobian(predict_positions))
J_full = jacobian_fn(params_fit, gravity, epoch, jdsatepoch, jdsatepochF, times)

n_times = times.shape[0]
J = J_full.reshape(n_times * 3, 7)

F = J.T @ J / sigma**2
cov = jnp.linalg.inv(F)
uncertainties = jnp.sqrt(jnp.diag(cov))

print("\nParameter     Best Fit          1-sigma Uncertainty")
print("-" * 55)
for name, fi, ui in zip(param_names, params_fit, uncertainties):
    print(f"{name:12s}  {float(fi):14.8f}  +/- {float(ui):.2e}")


# %%
# 6. Results
# ----------
#
# Visualize the position residuals and the parameter correlation matrix.

try:
    import matplotlib.pyplot as plt

    r_fit = predict_positions(
        params_fit, gravity, epoch, jdsatepoch, jdsatepochF, times)
    residuals = r_fit - r_obs

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Plot 1: Position residuals
    ax = axes[0]
    for i, label in enumerate(["X", "Y", "Z"]):
        ax.plot(times, residuals[:, i], ".", label=label, markersize=4)
    ax.axhline(sigma, color="gray", ls="--", alpha=0.5, label=f"$\\pm${sigma} km")
    ax.axhline(-sigma, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Time since epoch (min)")
    ax.set_ylabel("Residual (km)")
    ax.set_title("Position Residuals (fit - observed)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Correlation matrix
    ax = axes[1]
    std = jnp.sqrt(jnp.diag(cov))
    corr = cov / jnp.outer(std, std)
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(7))
    ax.set_xticklabels(param_names, rotation=45, ha="right")
    ax.set_yticks(range(7))
    ax.set_yticklabels(param_names)
    ax.set_title("Parameter Correlation Matrix")
    fig.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.show()

except ImportError:
    print("(matplotlib not available — skipping plots)")


# %%
# Summary
# -------
#
# This example demonstrated a complete orbit determination workflow:
#
# 1. **Synthetic data** — propagated a known TLE and added 1 km noise
# 2. **Forward model** — ``sgp4init`` + ``propagate`` map orbital
#    parameters to positions
# 3. **BFGS optimization** — JAX automatic differentiation provides
#    exact gradients through the entire SGP4 computation
# 4. **Uncertainty estimation** — the Fisher information matrix from
#    the Jacobian gives 1-σ parameter uncertainties and correlations
#
# The key advantage of sgp4jax is **differentiability**: gradients,
# Jacobians, and Hessians are computed automatically and efficiently,
# enabling gradient-based optimization and rigorous uncertainty
# quantification without finite differences.
