API Reference
=============

Core Functions
--------------

.. autofunction:: sgp4jax.tle_to_satrec

.. autofunction:: sgp4jax.tles_to_satrec

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
   :members:

.. autodata:: sgp4jax.WGS84

.. autodata:: sgp4jax.WGS72

.. autodata:: sgp4jax.WGS72OLD
