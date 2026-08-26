"""Shared pytest configuration.

sgp4jax no longer enables JAX double precision on import, so the test suite
enables it here — before any test module imports sgp4jax.
"""

import jax

jax.config.update("jax_enable_x64", True)
