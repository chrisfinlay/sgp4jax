sgp4jax
=======

**JAX-compatible SGP4 satellite propagation.**

sgp4jax is a pure-JAX implementation of the SGP4/SDP4 orbital propagator,
the standard algorithm used to predict satellite positions from
Two-Line Element (TLE) sets. Because the entire propagation path is
written in JAX, you get:

- **JIT compilation** — fast repeated evaluations via ``jax.jit``
- **Automatic vectorization** — propagate thousands of time steps in
  parallel with ``jax.vmap``
- **Automatic differentiation** — compute gradients of position/velocity
  with respect to time (or any input) via ``jax.grad``

.. important::
   sgp4jax requires JAX double precision.  Enable it with
   ``jax.config.update("jax_enable_x64", True)`` before importing, or set
   ``JAX_ENABLE_X64=1`` in the environment — see :ref:`double-precision`.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   getting_started
   examples
   api
