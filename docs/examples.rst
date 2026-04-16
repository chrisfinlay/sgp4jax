Examples
========

Orbit Fitting
-------------

This example demonstrates how to use sgp4jax's **differentiable SGP4
propagator** to fit orbital elements to noisy position observations.

Because sgp4jax is built on JAX, we get automatic differentiation for
free — enabling gradient-based optimization and analytic uncertainty
estimation.

**Workflow:**

1. Generate synthetic observations from a known TLE with Gaussian noise
2. Define a forward model mapping orbital parameters → predicted positions
3. Fit 7 orbital parameters using JAX's built-in BFGS optimizer
4. Estimate parameter uncertainties via Fisher information

.. literalinclude:: examples/orbit_fitting.py
   :language: python
   :linenos:

7×7 Element-Space Prior Covariance
------------------------------------

This example shows how to build a full 7×7 prior covariance matrix over
the SGP4 element vector ``(inclo, nodeo, ecco, argpo, mo, no_kozai, bstar)``
at a target time one day after TLE epoch.

The approach combines three functions:

1. :func:`~sgp4jax.tle_ric_covariance` — empirical 6×6 RIC position/velocity
   covariance based on TLE age and drag coefficient.
2. :func:`~sgp4jax.cov_ric_to_elements` — transforms the RIC covariance to
   6-element space ``(inclo, nodeo, ecco, argpo, mo, no_kozai)`` via the
   Keplerian Jacobian.  This is a square, full-rank transform.
3. :func:`~sgp4jax.tle_bstar_sigma` — empirical 1-σ for bstar based on TLE
   age and atmospheric-density variability, appended as an independent 7th
   block.

bstar is treated as independent of the Keplerian elements in the prior.
This is physically justified: drag is a satellite property independent of
orbital geometry.  Any bstar ↔ (mo, no_kozai) correlations emerge naturally
from the likelihood during fitting and should not be imposed by the prior.

The resulting matrix is a symmetric positive-definite ``(7, 7)`` covariance
suitable as a Gaussian prior ``θ ~ N(θ_tle, Σ)`` in Bayesian TLE fitting.

.. literalinclude:: examples/covariance_prior.py
   :language: python
   :linenos:
