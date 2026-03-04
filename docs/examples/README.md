# sgp4jax Examples

## orbit_fitting.py

Demonstrates gradient-based orbit determination using sgp4jax's differentiable SGP4 propagator.

**What it does:**

1. Generates synthetic position observations from a known ISS TLE with 1 km Gaussian noise
2. Fits 7 orbital parameters (inclination, RAAN, eccentricity, argument of perigee, mean anomaly, mean motion, B*) using BFGS optimization
3. Computes analytic 1-σ parameter uncertainties via the Fisher information matrix (Jacobian computed with `jax.jacobian`)
4. Plots position residuals and the parameter correlation matrix (requires matplotlib)

**Prerequisites:**

```
pip install -r requirements.txt
```

Or install the examples extra directly:

```
pip install "sgp4jax[examples]"
```

**Run:**

```
python orbit_fitting.py
```

**Expected output:**

```
Propagated 50 time steps, positions shape: (50, 3)
RMS noise: 1.000 km

Optimization finished: fun_val = ..., nit = ...

Parameter        Initial          Optimized        True
----------------------------------------------------------------------
inclo         0.90118...       0.90119...       0.90119...
...

Parameter     Best Fit          1-sigma Uncertainty
-------------------------------------------------------
inclo         0.90119...  +/- ...
...
```

If matplotlib is installed, two plots are shown: position residuals over time and the parameter correlation matrix.
