Getting Started
===============

Basic propagation
-----------------

Parse a TLE and propagate to a time offset (minutes from epoch):

.. code-block:: python

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

Gravity models
--------------

Three gravity models are available:

- ``sgp4jax.WGS84`` (default)
- ``sgp4jax.WGS72``
- ``sgp4jax.WGS72OLD``

Pass a different model to :func:`~sgp4jax.tle_to_satrec`:

.. code-block:: python

   sat = sgp4jax.tle_to_satrec(line1, line2, gravity=sgp4jax.WGS72)
