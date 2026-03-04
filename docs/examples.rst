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
