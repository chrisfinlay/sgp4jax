API Reference
=============

Core Functions
--------------

.. autofunction:: sgp4jax.tle_to_satrec

.. autofunction:: sgp4jax.tles_to_satrec

.. autofunction:: sgp4jax.satrec_from_elements

.. autofunction:: sgp4jax.propagate

.. autofunction:: sgp4jax.propagate_jd

Specialized Propagators
------------------------

These propagators are optimised for specific orbit families and are
significantly faster than :func:`~sgp4jax.propagate` for homogeneous
batches.  Each eliminates the dead-branch computation present in the
general propagator:

* :func:`~sgp4jax.propagate_leo` — near-earth orbits (``method=0``), removes the
  deep-space integrator entirely.
* :func:`~sgp4jax.propagate_sdp4_nr` — deep-space no-resonance orbits
  (``method=1``, ``irez=0``, e.g. GPS/GLONASS/Galileo MEO), replaces the
  64-step resonance scan with five scalar multiplications.
* :func:`~sgp4jax.propagate_mixed` — heterogeneous batches of any orbit type;
  groups satellites by type internally and dispatches each group to the
  appropriate specialised propagator.

.. autofunction:: sgp4jax.propagate_leo

.. autofunction:: sgp4jax.propagate_jd_leo

.. autofunction:: sgp4jax.propagate_sdp4_nr

.. autofunction:: sgp4jax.propagate_jd_sdp4_nr

.. autofunction:: sgp4jax.propagate_mixed

Frame Transformations
---------------------

.. autofunction:: sgp4jax.teme_to_gcrf

.. autofunction:: sgp4jax.itrf_to_gcrf

.. autofunction:: sgp4jax.gcrf_to_itrf

.. autofunction:: sgp4jax.propagate_gcrf

.. autofunction:: sgp4jax.propagate_jd_gcrf

Covariance Transformations
--------------------------

Transform covariance matrices between the RIC (Radial, In-track, Cross-track)
frame, TEME Cartesian, and orbital element space.  All Jacobians are computed
via :func:`jax.jacobian`, so all transforms are fully differentiable.

Two element models are provided:

* **6-element** — ``(inclo, nodeo, ecco, argpo, mo, no_kozai)`` — Keplerian
  Jacobian, no drag.
* **7-element** — ``(inclo, nodeo, ecco, argpo, mo, no_kozai, bstar)`` — SGP4
  Jacobian through the full ``sgp4init → sgp4`` pipeline; bstar's influence on
  the trajectory is captured.

Empirical Covariance from TLE Age
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: sgp4jax.tle_ric_covariance

.. autofunction:: sgp4jax.tle_bstar_sigma

Frame utilities
~~~~~~~~~~~~~~~

.. autofunction:: sgp4jax.ric_rotation

.. autofunction:: sgp4jax.cov_ric_to_teme

.. autofunction:: sgp4jax.cov_teme_to_ric

6-element model
~~~~~~~~~~~~~~~

.. autofunction:: sgp4jax.elements_jacobian

.. autofunction:: sgp4jax.cov_elements_to_teme

.. autofunction:: sgp4jax.cov_teme_to_elements

.. autofunction:: sgp4jax.cov_ric_to_elements

.. autofunction:: sgp4jax.cov_elements_to_ric

7-element model (with drag)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: sgp4jax.elements7_jacobian

.. autofunction:: sgp4jax.cov_elements7_to_teme

.. autofunction:: sgp4jax.cov_teme_to_elements7

.. autofunction:: sgp4jax.cov_elements7_to_ric

.. autofunction:: sgp4jax.cov_ric_to_elements7

Keplerian Propagation (no perturbations)
-----------------------------------------

Pure two-body Keplerian motion using only the six classical TLE elements
(inclination, RAAN, eccentricity, argument of perigee, mean anomaly, mean
motion).  All perturbations are ignored.  Fully differentiable.

.. note::
   TLE elements are Brouwer *mean* elements.  Because SGP4 adds short-period
   corrections when converting to osculating elements, Keplerian positions will
   differ from SGP4 by O(1–50 km).  This is expected behaviour, not a bug.

.. autofunction:: sgp4jax.kepler_gcrf_positions

.. autofunction:: sgp4jax.kepler_gcrf_positions_multi

Batch GCRF Convenience Functions
---------------------------------

These functions propagate one or more satellites to an array of UTC Julian
dates and return positions and velocities in the GCRF frame.  Choose the
variant that matches the orbit type of your satellite batch for best
performance:

.. autofunction:: sgp4jax.gcrf_positions

.. autofunction:: sgp4jax.gcrf_positions_multi

.. autofunction:: sgp4jax.gcrf_positions_multi_leo

.. autofunction:: sgp4jax.gcrf_positions_multi_sdp4_nr

.. autofunction:: sgp4jax.gcrf_positions_mixed

Earth Orientation (IERS)
------------------------

These functions manage the IERS Bulletin A table used for UTC → UT1 conversion,
which is needed for accurate ITRF ↔ GCRF frame transformations.

.. autofunction:: sgp4jax.update_iers_table

.. autofunction:: sgp4jax.load_iers_table

.. autofunction:: sgp4jax.utc_to_ut1

Data Structures
---------------

.. autoclass:: sgp4jax.SatRec
   :members:
   :undoc-members:

.. autofunction:: sgp4jax.make_satrec

Gravity Constants
-----------------

.. autoclass:: sgp4jax._constants.GravityConstants

.. autodata:: sgp4jax.WGS84

.. data:: sgp4jax.WGS72

   WGS-72 Earth gravity constants (default for SGP4).
   Field names identical to :class:`~sgp4jax._constants.GravityConstants`.

.. data:: sgp4jax.WGS72OLD

   WGS-72 "old" Earth gravity constants (Vallado original).
   Field names identical to :class:`~sgp4jax._constants.GravityConstants`.
