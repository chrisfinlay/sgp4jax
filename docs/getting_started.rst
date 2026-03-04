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
