API Reference
=============

Core Functions
--------------

.. autofunction:: sgp4jax.tle_to_satrec

.. autofunction:: sgp4jax.tles_to_satrec

.. autofunction:: sgp4jax.propagate

.. autofunction:: sgp4jax.propagate_jd

Frame Transformations
---------------------

.. autofunction:: sgp4jax.teme_to_gcrf

.. autofunction:: sgp4jax.propagate_gcrf

.. autofunction:: sgp4jax.propagate_jd_gcrf

Batch Convenience Functions
---------------------------

.. autofunction:: sgp4jax.gcrf_positions

.. autofunction:: sgp4jax.gcrf_positions_multi

Data Structures
---------------

.. autoclass:: sgp4jax.SatRec
   :members:
   :undoc-members:

.. autofunction:: sgp4jax.make_satrec

Gravity Constants
-----------------

.. autodata:: sgp4jax.WGS84

.. autodata:: sgp4jax.WGS72

.. autodata:: sgp4jax.WGS72OLD
