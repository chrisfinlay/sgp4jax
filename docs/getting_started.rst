Getting Started
===============

Basic propagation
-----------------

Parse a TLE and propagate to a time offset (minutes from epoch):

.. code-block:: python

   import jax.numpy as jnp
   import sgp4jax

   line1 = "1 25544U 98067A   20045.18587073  .00000950  00000-0  25302-4 0  9990"
   line2 = "2 25544  51.6443 242.0161 0004397 264.6060 207.3845 15.49165514212791"

   sat = sgp4jax.tle_to_satrec(line1, line2)
   r, v, error = sgp4jax.propagate(sat, jnp.array(0.0))
   print(f"Position (km): {r}")
   print(f"Velocity (km/s): {v}")

Julian Date propagation
-----------------------

Use :func:`~sgp4jax.propagate_jd` with a split Julian Date:

.. code-block:: python

   import jax.numpy as jnp

   jd = jnp.array(sat.jdsatepoch)
   fr = jnp.array(sat.jdsatepochF + 0.5)  # 12 hours later
   r, v, error = sgp4jax.propagate_jd(sat, jd, fr)

JIT compilation
---------------

The propagation function is already JIT-compiled by default. You can
also explicitly JIT-compile:

.. code-block:: python

   import jax

   jitted_propagate = jax.jit(sgp4jax.propagate)
   r, v, error = jitted_propagate(sat, jnp.array(100.0))

Batch propagation with vmap
----------------------------

Propagate a single satellite over many time steps at once:

.. code-block:: python

   times = jnp.linspace(0, 1440, 1000)  # one day, 1000 steps
   batched = jax.vmap(sgp4jax.propagate, in_axes=(None, 0))
   r_batch, v_batch, err_batch = batched(sat, times)
   # r_batch.shape == (1000, 3)

Gradients
---------

Compute gradients of any scalar function of position/velocity:

.. code-block:: python

   def loss(t):
       r, v, err = sgp4jax.propagate(sat, t)
       return jnp.sum(r ** 2)

   grad_fn = jax.grad(loss)
   g = grad_fn(jnp.array(100.0))

GCRF output
-----------

SGP4 outputs position and velocity in the TEME (True Equator Mean Equinox)
frame. To get GCRF (Geocentric Celestial Reference Frame, ≈ICRS) output,
use the GCRF convenience functions:

.. code-block:: python

   # Propagate directly to GCRF
   r_gcrf, v_gcrf, error = sgp4jax.propagate_gcrf(sat, jnp.array(100.0))
   print(f"Position (GCRF, km): {r_gcrf}")
   print(f"Velocity (GCRF, km/s): {v_gcrf}")

   # Or use Julian Date
   jd = jnp.array(sat.jdsatepoch)
   fr = jnp.array(sat.jdsatepochF + 0.5)
   r_gcrf, v_gcrf, error = sgp4jax.propagate_jd_gcrf(sat, jd, fr)

You can also apply the transform separately with :func:`~sgp4jax.teme_to_gcrf`:

.. code-block:: python

   r_teme, v_teme, error = sgp4jax.propagate(sat, jnp.array(100.0))
   jd = jnp.array(sat.jdsatepoch)
   fr = jnp.array(sat.jdsatepochF + 100.0 / 1440.0)
   r_gcrf, v_gcrf = sgp4jax.teme_to_gcrf(r_teme, v_teme, jd, fr)

Gravity models
--------------

Three gravity models are available:

- ``sgp4jax.WGS72`` (default)
- ``sgp4jax.WGS84``
- ``sgp4jax.WGS72OLD``

Pass a different model to :func:`~sgp4jax.tle_to_satrec`:

.. code-block:: python

   sat = sgp4jax.tle_to_satrec(line1, line2, gravity=sgp4jax.WGS84)

Batch TLE parsing
-----------------

Parse multiple TLEs at once with :func:`~sgp4jax.tles_to_satrec`, which
returns a batched :class:`~sgp4jax.SatRec` ready for ``jax.vmap``:

.. code-block:: python

   import jax
   import jax.numpy as jnp
   import sgp4jax

   tles = [
       ("1 25544U 98067A   20045.18587073  .00000950  00000-0  25302-4 0  9990",
        "2 25544  51.6443 242.0161 0004397 264.6060 207.3845 15.49165514212791"),
       ("1 00005U 58002B   20045.93498537  .00000023  00000-0  24901-3 0  9999",
        "2 00005  34.2513 243.5765 1847090 326.4186  22.2640 10.84386407185708"),
   ]

   sats = sgp4jax.tles_to_satrec(tles)

   # Propagate all satellites at once
   batched = jax.vmap(sgp4jax.propagate, in_axes=(0, None))
   r, v, err = batched(sats, jnp.array(0.0))
   # r.shape == (2, 3)

Convenience batch functions
---------------------------

:func:`~sgp4jax.gcrf_positions` propagates a single satellite to many
Julian dates and returns GCRF positions/velocities:

.. code-block:: python

   sat = sgp4jax.tle_to_satrec(tles[0][0], tles[0][1])
   times_jd = jnp.linspace(2458900.5, 2458901.5, 100)
   r_gcrf, v_gcrf = sgp4jax.gcrf_positions(sat, times_jd)
   # r_gcrf.shape == (100, 3)

:func:`~sgp4jax.gcrf_positions_multi` propagates multiple satellites to
many Julian dates:

.. code-block:: python

   sats = sgp4jax.tles_to_satrec(tles)
   r_gcrf, v_gcrf = sgp4jax.gcrf_positions_multi(sats, times_jd)
   # r_gcrf.shape == (2, 100, 3)

Specialised propagators
-----------------------

For large homogeneous batches, orbit-type-specific propagators eliminate
dead-branch computation and are substantially faster than the general
:func:`~sgp4jax.propagate`:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Function
     - Orbit type
     - When to use
   * - :func:`~sgp4jax.propagate_leo`
     - Near-earth (``method=0``)
     - LEO, MEO, HEO below the deep-space threshold (~225 min period)
   * - :func:`~sgp4jax.propagate_sdp4_nr`
     - Deep-space, no resonance (``irez=0``)
     - GPS, GLONASS, Galileo, BeiDou MEO — outside resonance bands
   * - :func:`~sgp4jax.propagate`
     - Any (general)
     - Mixed or unknown orbit types; GEO (``irez=1``), Molniya (``irez=2``)

Near-earth (LEO) propagator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~sgp4jax.propagate_leo` drops the deep-space integrator entirely.
It is a drop-in replacement for :func:`~sgp4jax.propagate` for near-earth
satellites and is fully JIT/vmap/AD-compatible:

.. code-block:: python

   import jax
   import jax.numpy as jnp
   import sgp4jax

   line1 = "1 25544U 98067A   20045.18587073  .00000950  00000-0  25302-4 0  9990"
   line2 = "2 25544  51.6443 242.0161 0004397 264.6060 207.3845 15.49165514212791"

   sat = sgp4jax.tle_to_satrec(line1, line2)

   # Single point
   r, v, err = sgp4jax.propagate_leo(sat, jnp.array(60.0))

   # Batch over times
   times = jnp.linspace(0.0, 1440.0, 1000)
   r_batch, v_batch, err_batch = jax.vmap(
       sgp4jax.propagate_leo, in_axes=(None, 0)
   )(sat, times)
   # r_batch.shape == (1000, 3)

Use :func:`~sgp4jax.propagate_jd_leo` to supply a Julian Date instead of
minutes since epoch:

.. code-block:: python

   jd = jnp.array(sat.jdsatepoch)
   fr = jnp.array(sat.jdsatepochF + 1.0)  # 1 day after epoch
   r, v, err = sgp4jax.propagate_jd_leo(sat, jd, fr)

Deep-space no-resonance propagator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~sgp4jax.propagate_sdp4_nr` is optimised for deep-space satellites
that fall outside both resonance bands (``irez=0``): GPS, GLONASS, Galileo,
BeiDou MEO, and many GTO/HEO transfer stages.  It replaces the 64-step
resonance scan with five scalar multiplications:

.. code-block:: python

   import jax
   import jax.numpy as jnp
   import sgp4jax

   # GPS NAVSTAR 53
   gps_l1 = "1 28129U 03058A   06175.57071136 -.00000104  00000-0  10000-3 0   459"
   gps_l2 = "2 28129  54.7298 324.8098 0048506 266.2640  93.1663  2.00562768 18443"

   sat = sgp4jax.tle_to_satrec(gps_l1, gps_l2, gravity=sgp4jax.WGS84)

   # Single point
   r, v, err = sgp4jax.propagate_sdp4_nr(sat, jnp.array(720.0))

   # Batch over times
   times = jnp.linspace(0.0, 1440.0, 500)
   r_batch, v_batch, _ = jax.vmap(
       sgp4jax.propagate_sdp4_nr, in_axes=(None, 0)
   )(sat, times)
   # r_batch.shape == (500, 3)

Use :func:`~sgp4jax.propagate_jd_sdp4_nr` for Julian Date input:

.. code-block:: python

   jd = jnp.array(sat.jdsatepoch)
   fr = jnp.array(sat.jdsatepochF + 0.5)  # 12 hours after epoch
   r, v, err = sgp4jax.propagate_jd_sdp4_nr(sat, jd, fr)

Heterogeneous constellation propagation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~sgp4jax.propagate_mixed` handles a batch of satellites of mixed
orbit types.  It groups satellites by type internally and dispatches each
group to the appropriate specialised propagator, then reassembles results in
the original input order:

.. code-block:: python

   import jax.numpy as jnp
   import sgp4jax

   # ISS (near-earth, LEO)
   iss_l1 = "1 25544U 98067A   20045.18587073  .00000950  00000-0  25302-4 0  9990"
   iss_l2 = "2 25544  51.6443 242.0161 0004397 264.6060 207.3845 15.49165514212791"

   # GPS NAVSTAR 53 (deep-space, irez=0)
   gps_l1 = "1 28129U 03058A   06175.57071136 -.00000104  00000-0  10000-3 0   459"
   gps_l2 = "2 28129  54.7298 324.8098 0048506 266.2640  93.1663  2.00562768 18443"

   # ITALSAT 2 (GEO, irez=1)
   geo_l1 = "1 24208U 96044A   06177.04061740 -.00000094  00000-0  10000-3 0  1600"
   geo_l2 = "2 24208   3.8536  80.0121 0026640 311.0977  48.3000  1.00778054 36119"

   tles = [[iss_l1, iss_l2], [gps_l1, gps_l2], [geo_l1, geo_l2]]
   sats = sgp4jax.tles_to_satrec(tles, gravity=sgp4jax.WGS84)

   times = jnp.array([0.0, 60.0, 360.0, 720.0, 1440.0])   # minutes since epoch
   r, v, err = sgp4jax.propagate_mixed(sats, times)
   # r.shape == (3, 5, 3)  — (n_sat, n_times, 3)

.. note::

   :func:`~sgp4jax.propagate_mixed` is not JIT-compilable as a whole and
   does not compose with ``jax.grad`` or ``jax.vmap``.  For JIT / AD / vmap
   compatibility, group satellites by orbit type and call the specialised
   propagators directly.

Batch GCRF — specialised propagators
--------------------------------------

Each specialised propagator has a corresponding batch GCRF function that
propagates a homogeneous satellite batch to an array of UTC Julian dates and
returns positions and velocities in the GCRF frame.

Near-earth satellites (LEO)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import jax.numpy as jnp
   import sgp4jax

   iss_l1 = "1 25544U 98067A   20045.18587073  .00000950  00000-0  25302-4 0  9990"
   iss_l2 = "2 25544  51.6443 242.0161 0004397 264.6060 207.3845 15.49165514212791"
   sen_l1 = "1 39634U 14016A   20045.50000000  .00000023  00000-0  14064-4 0  9994"
   sen_l2 = "2 39634  98.1825 145.6352 0001346  88.3457 271.7897 14.59198523314592"

   leo_sats = sgp4jax.tles_to_satrec(
       [[iss_l1, iss_l2], [sen_l1, sen_l2]], gravity=sgp4jax.WGS84
   )

   # 100 evenly-spaced times over one day, as UTC Julian dates
   jd0 = leo_sats.jdsatepoch[0] + leo_sats.jdsatepochF[0]
   times_jd = jnp.linspace(float(jd0), float(jd0) + 1.0, 100)

   r_gcrf, v_gcrf = sgp4jax.gcrf_positions_multi_leo(leo_sats, times_jd)
   # r_gcrf.shape == (2, 100, 3)

Deep-space no-resonance satellites (GPS/MEO)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import jax.numpy as jnp
   import sgp4jax

   gps1_l1 = "1 28129U 03058A   06175.57071136 -.00000104  00000-0  10000-3 0   459"
   gps1_l2 = "2 28129  54.7298 324.8098 0048506 266.2640  93.1663  2.00562768 18443"
   gps2_l1 = "1 20413U 83020D   05363.79166667  .00000000  00000-0  00000+0 0  7041"
   gps2_l2 = "2 20413  12.3514 187.4253 7864447 196.3027 356.5478  0.24690082  7978"

   gps_sats = sgp4jax.tles_to_satrec(
       [[gps1_l1, gps1_l2], [gps2_l1, gps2_l2]], gravity=sgp4jax.WGS84
   )

   jd0 = gps_sats.jdsatepoch[0] + gps_sats.jdsatepochF[0]
   times_jd = jnp.linspace(float(jd0), float(jd0) + 1.0, 100)

   r_gcrf, v_gcrf = sgp4jax.gcrf_positions_multi_sdp4_nr(gps_sats, times_jd)
   # r_gcrf.shape == (2, 100, 3)

Heterogeneous constellation in GCRF
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~sgp4jax.gcrf_positions_mixed` combines the grouping logic of
:func:`~sgp4jax.propagate_mixed` with TEME→GCRF rotation, accepting
absolute UTC Julian dates so each satellite's epoch is handled correctly:

.. code-block:: python

   import jax.numpy as jnp
   import sgp4jax

   iss_l1 = "1 25544U 98067A   20045.18587073  .00000950  00000-0  25302-4 0  9990"
   iss_l2 = "2 25544  51.6443 242.0161 0004397 264.6060 207.3845 15.49165514212791"
   gps_l1 = "1 28129U 03058A   06175.57071136 -.00000104  00000-0  10000-3 0   459"
   gps_l2 = "2 28129  54.7298 324.8098 0048506 266.2640  93.1663  2.00562768 18443"
   geo_l1 = "1 24208U 96044A   06177.04061740 -.00000094  00000-0  10000-3 0  1600"
   geo_l2 = "2 24208   3.8536  80.0121 0026640 311.0977  48.3000  1.00778054 36119"

   mixed_sats = sgp4jax.tles_to_satrec(
       [[iss_l1, iss_l2], [gps_l1, gps_l2], [geo_l1, geo_l2]],
       gravity=sgp4jax.WGS84,
   )

   # Shared absolute observation times (UTC Julian dates)
   times_jd = jnp.linspace(2453736.5, 2453737.5, 200)

   r_gcrf, v_gcrf = sgp4jax.gcrf_positions_mixed(mixed_sats, times_jd)
   # r_gcrf.shape == (3, 200, 3)  — LEO, GPS, GEO all in GCRF
