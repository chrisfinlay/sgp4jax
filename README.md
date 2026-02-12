# sgp4jax

**JAX-compatible SGP4/SDP4 satellite propagation.**

sgp4jax is a pure-JAX reimplementation of the SGP4/SDP4 orbital propagator — the standard algorithm used worldwide to predict satellite positions from NORAD Two-Line Element (TLE) sets. Because the entire propagation pipeline is written in JAX, you get JIT compilation, automatic vectorization (`vmap`), and automatic differentiation (`grad`) for free.

## Features

- **Accurate** — Matches the reference [python-sgp4](https://github.com/brandon-rhodes/python-sgp4) library to sub-millimetre precision across the full SGP4 verification dataset
- **JIT-compiled** — Propagation is `jax.jit`-compiled for fast repeated evaluation
- **Vectorizable** — Use `jax.vmap` to propagate thousands of time steps in a single call
- **Differentiable** — Compute gradients of position/velocity with respect to time or any other input via `jax.grad`
- **Near-earth and deep-space** — Full SGP4 and SDP4 support, including lunar-solar perturbations and deep-space resonance
- **Multiple gravity models** — WGS84, WGS72, and WGS72OLD

## Installation

```bash
pip install sgp4jax
```

From source:

```bash
git clone https://github.com/chrisfinlay/sgp4jax.git
cd sgp4jax
pip install -e ".[test]"
```

## Quick Start

```python
import jax
import jax.numpy as jnp
import sgp4jax

# Parse a TLE
line1 = "1 25544U 98067A   20045.18587073  .00000950  00000-0  25302-4 0  9990"
line2 = "2 25544  51.6443 242.0161 0004397 264.6060 207.3845 15.49165514212791"

sat = sgp4jax.tle_to_satrec(line1, line2)

# Propagate to 100 minutes after epoch
r, v, error = sgp4jax.propagate(sat, jnp.array(100.0))
print(f"Position (TEME, km): {r}")
print(f"Velocity (TEME, km/s): {v}")
```

### Batch propagation with vmap

```python
times = jnp.linspace(0, 1440, 1000)  # one day, 1000 steps
batched = jax.vmap(sgp4jax.propagate, in_axes=(None, 0))
r_batch, v_batch, err_batch = batched(sat, times)
# r_batch.shape == (1000, 3)
```

### Gradients

```python
def loss(t):
    r, v, err = sgp4jax.propagate(sat, t)
    return jnp.sum(r ** 2)

grad_fn = jax.grad(loss)
g = grad_fn(jnp.array(100.0))
```

### Julian Date propagation

```python
jd = jnp.array(sat.jdsatepoch)
fr = jnp.array(sat.jdsatepochF + 0.5)  # 12 hours after epoch
r, v, error = sgp4jax.propagate_jd(sat, jd, fr)
```

### Gravity models

```python
sat_wgs72 = sgp4jax.tle_to_satrec(line1, line2, gravity=sgp4jax.WGS72)
```

## API

| Function / Object | Description |
|---|---|
| `tle_to_satrec(line1, line2, gravity=WGS84)` | Parse a TLE and initialize a satellite record |
| `propagate(satrec, tsince)` | Propagate to `tsince` minutes from epoch |
| `propagate_jd(satrec, jd, fr)` | Propagate to a split Julian Date |
| `SatRec` | NamedTuple holding all satellite state (JAX arrays) |
| `make_satrec(**kwargs)` | Create a SatRec with defaults of 0.0 for unspecified fields |
| `WGS84`, `WGS72`, `WGS72OLD` | Gravity model constants |

## Documentation

Full documentation is available at [Read the Docs](https://sgp4jax.readthedocs.io/) (or build locally):

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

## Testing

```bash
pip install -e ".[test]"
pytest tests/ -v
```

The test suite validates against the reference [python-sgp4](https://github.com/brandon-rhodes/python-sgp4) library, including the full SGP4-VER.TLE verification dataset (30+ satellites, thousands of data points).

## Acknowledgements

This project is a JAX reimplementation of the SGP4/SDP4 algorithm. The implementation is derived from the [python-sgp4](https://github.com/brandon-rhodes/python-sgp4) library by **Brandon Rhodes**, which is itself a Python translation of the original Fortran code by **David Vallado** and others.

The SGP4 algorithm was originally published in:

> Hoots, F. R., and Roehrich, R. L., "Spacetrack Report No. 3: Models for Propagation of NORAD Element Sets," U.S. Air Force Aerospace Defense Command, Colorado Springs, CO, 1980.

> Vallado, D. A., Crawford, P., Hujsak, R., and Kelso, T. S., "Revisiting Spacetrack Report #3," presented at the AIAA/AAS Astrodynamics Specialist Conference, Keystone, CO, 2006.

## License

This project is licensed under the **GNU General Public License v3.0** — see the [LICENSE](LICENSE) file.

This project contains code derived from [python-sgp4](https://github.com/brandon-rhodes/python-sgp4), which is licensed under the **MIT License** (Copyright (c) 2012-2016 Brandon Rhodes). See [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES) for the full text.
