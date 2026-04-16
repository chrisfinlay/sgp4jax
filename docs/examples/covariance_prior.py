"""Build a 7×7 element-space prior covariance from a TLE.

This example shows how to combine :func:`~sgp4jax.tle_ric_covariance`,
:func:`~sgp4jax.cov_ric_to_elements`, and :func:`~sgp4jax.tle_bstar_sigma`
to produce a full 7×7 prior covariance over the SGP4 element vector

    (inclo, nodeo, ecco, argpo, mo, no_kozai, bstar)

at a target time one day after TLE epoch.  This matrix is suitable as a
Gaussian prior in Bayesian TLE fitting.

Workflow
--------
1.  Parse the TLE and define the target time.
2.  Get the 6×6 RIC position/velocity covariance via the empirical
    Vallado-style error-growth model.
3.  Transform to 6-element space (inclo, nodeo, ecco, argpo, mo, no_kozai)
    using the Keplerian Jacobian.  This is a square, full-rank transform.
4.  Append the empirical bstar variance as an independent 7th block.

Why not use ``cov_ric_to_elements7``?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The 7-element Jacobian has shape (6, 7): six state dimensions cannot
uniquely constrain seven parameters.  Its pseudo-inverse produces a
rank-deficient (rank ≤ 6) covariance with numerically artefactual
cross-terms.  For a *prior*, bstar independence from the Keplerian
elements is physically justified — drag is a satellite property
independent of orbital geometry.  The bstar ↔ (mo, no_kozai)
correlations you care about are *posterior* correlations that emerge
naturally from the likelihood; they should not be baked into the prior.
"""

import jax.numpy as jnp

from sgp4jax import (
    tle_to_satrec,
    tle_ric_covariance,
    tle_bstar_sigma,
    cov_ric_to_elements,
)

# ---------------------------------------------------------------------------
# TLE for the International Space Station (example)
# ---------------------------------------------------------------------------
LINE1 = "1 25544U 98067A   24001.50000000  .00003317  00000-0  38117-4 0  9994"
LINE2 = "2 25544  51.6416 247.4627 0006703 130.5360  13.6717 15.50026396432903"

sat = tle_to_satrec(LINE1, LINE2)

# ---------------------------------------------------------------------------
# Target time: 1 day after TLE epoch
# ---------------------------------------------------------------------------
jd = sat.jdsatepoch + 1.0   # whole part of Julian date
fr = sat.jdsatepochF        # fractional part (unchanged)

# ---------------------------------------------------------------------------
# Step 1 — 6×6 RIC position/velocity covariance
#
# Diagonal matrix; units km² (position block) and km²/s² (velocity block).
# In-track (T) growth is scaled by bstar relative to the LEO population
# median.
# ---------------------------------------------------------------------------
cov_ric = tle_ric_covariance(sat, jd, fr)

print("6×6 RIC 1-σ (km, km/s):")
for label, var in zip(["R", "T", "N", "Ṙ", "Ṫ", "Ṅ"], jnp.diag(cov_ric)):
    print(f"  σ_{label} = {float(jnp.sqrt(var)):.4f}")

# ---------------------------------------------------------------------------
# Step 2 — Transform to 6-element space via Keplerian Jacobian
#
# cov_ric_to_elements chains:  RIC → TEME → elements6
# The result is a full-rank (6, 6) covariance.
# ---------------------------------------------------------------------------
cov_el6 = cov_ric_to_elements(cov_ric, sat, jd, fr)

print("\n6-element 1-σ:")
el6_labels = [
    "inclo     (rad)",
    "nodeo     (rad)",
    "ecco          ",
    "argpo     (rad)",
    "mo        (rad)",
    "no_kozai  (rad/min)",
]
for label, var in zip(el6_labels, jnp.diag(cov_el6)):
    print(f"  σ_{label} = {float(jnp.sqrt(var)):.3e}")

# ---------------------------------------------------------------------------
# Step 3 — Empirical bstar uncertainty
#
# tle_bstar_sigma returns a scalar 1-σ based on TLE age and |bstar|.
# At 1 day: σ_bstar ≈ 30% + 10%·1 day = 40% of |bstar|.
# ---------------------------------------------------------------------------
sigma_bstar = tle_bstar_sigma(sat, jd, fr)
print(f"\nbstar = {float(sat.bstar):.3e} km⁻¹")
print(f"σ_bstar at Δt=1 day = {float(sigma_bstar):.3e} km⁻¹  "
      f"({100*float(sigma_bstar)/abs(float(sat.bstar)):.0f}%)")

# ---------------------------------------------------------------------------
# Step 4 — Assemble the 7×7 prior covariance
#
# bstar is appended as an independent block: no cross-terms with the
# Keplerian elements.  The result is symmetric positive-definite.
# ---------------------------------------------------------------------------
cov_prior = jnp.block([
    [cov_el6,            jnp.zeros((6, 1))],
    [jnp.zeros((1, 6)),  jnp.array([[sigma_bstar ** 2]])],
])

print("\n7×7 prior: 1-σ diagonal:")
el7_labels = el6_labels + ["bstar     (km⁻¹)  "]
for label, var in zip(el7_labels, jnp.diag(cov_prior)):
    print(f"  σ_{label} = {float(jnp.sqrt(var)):.3e}")

print("\ncov_prior shape:", cov_prior.shape)
eigenvalues = jnp.linalg.eigvalsh(cov_prior)
print(f"Minimum eigenvalue: {float(eigenvalues.min()):.3e}  (≈ machine-ε × max eigenvalue — numerically PD)")

# ---------------------------------------------------------------------------
# This cov_prior is suitable as a Gaussian prior:
#
#   θ ~ N(θ_tle, cov_prior)
#
# where θ = (inclo, nodeo, ecco, argpo, mo, no_kozai, bstar).
# ---------------------------------------------------------------------------
