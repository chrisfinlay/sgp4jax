"""SatRec NamedTuple for JAX-compatible SGP4 propagation."""

from typing import Any, NamedTuple
import jax
import jax.numpy as jnp


class SatRec(NamedTuple):
    """JAX-compatible satellite record.

    All fields are jnp.ndarray scalars (float64).
    String flags are encoded as floats:

    - method: 0.0 = near-earth, 1.0 = deep-space
    - isimp: 0.0 = full perturbations, 1.0 = simplified
    - irez: 0.0 = no resonance, 1.0 = synchronous, 2.0 = half-day
    """
    # --- TLE-derived orbital elements ---
    bstar: jax.Array
    ecco: jax.Array
    argpo: jax.Array
    inclo: jax.Array
    mo: jax.Array
    no_kozai: jax.Array
    nodeo: jax.Array
    ndot: jax.Array
    nddot: jax.Array

    # --- Gravity model constants ---
    j2: jax.Array
    j3: jax.Array
    j4: jax.Array
    j3oj2: jax.Array
    xke: jax.Array
    mu: jax.Array
    radiusearthkm: jax.Array
    tumin: jax.Array

    # --- Epoch ---
    jdsatepoch: jax.Array
    jdsatepochF: jax.Array

    # --- Computed by _initl / sgp4init (near-earth) ---
    no_unkozai: jax.Array
    a: jax.Array
    alta: jax.Array
    altp: jax.Array
    con41: jax.Array
    cc1: jax.Array
    cc4: jax.Array
    cc5: jax.Array
    d2: jax.Array
    d3: jax.Array
    d4: jax.Array
    delmo: jax.Array
    eta: jax.Array
    argpdot: jax.Array
    omgcof: jax.Array
    sinmao: jax.Array
    t2cof: jax.Array
    t3cof: jax.Array
    t4cof: jax.Array
    t5cof: jax.Array
    x1mth2: jax.Array
    x7thm1: jax.Array
    mdot: jax.Array
    nodedot: jax.Array
    xlcof: jax.Array
    xmcof: jax.Array
    nodecf: jax.Array
    aycof: jax.Array
    gsto: jax.Array

    # --- Deep space coefficients ---
    d2201: jax.Array
    d2211: jax.Array
    d3210: jax.Array
    d3222: jax.Array
    d4410: jax.Array
    d4422: jax.Array
    d5220: jax.Array
    d5232: jax.Array
    d5421: jax.Array
    d5433: jax.Array
    dedt: jax.Array
    del1: jax.Array
    del2: jax.Array
    del3: jax.Array
    didt: jax.Array
    dmdt: jax.Array
    dnodt: jax.Array
    domdt: jax.Array
    e3: jax.Array
    ee2: jax.Array
    peo: jax.Array
    pgho: jax.Array
    pho: jax.Array
    pinco: jax.Array
    plo: jax.Array
    se2: jax.Array
    se3: jax.Array
    sgh2: jax.Array
    sgh3: jax.Array
    sgh4: jax.Array
    sh2: jax.Array
    sh3: jax.Array
    si2: jax.Array
    si3: jax.Array
    sl2: jax.Array
    sl3: jax.Array
    sl4: jax.Array
    xfact: jax.Array
    xgh2: jax.Array
    xgh3: jax.Array
    xgh4: jax.Array
    xh2: jax.Array
    xh3: jax.Array
    xi2: jax.Array
    xi3: jax.Array
    xl2: jax.Array
    xl3: jax.Array
    xl4: jax.Array
    xlamo: jax.Array
    xli: jax.Array
    xni: jax.Array
    zmol: jax.Array
    zmos: jax.Array
    atime: jax.Array

    # --- Flags (encoded as floats) ---
    method: jax.Array     # 0.0 = near-earth, 1.0 = deep-space
    isimp: jax.Array      # 0.0 or 1.0
    irez: jax.Array       # 0.0, 1.0, or 2.0


def make_satrec(**kwargs: Any) -> SatRec:
    """Create a SatRec with defaults of 0.0 for unspecified fields."""
    defaults = {field: jnp.array(0.0) for field in SatRec._fields}
    defaults.update(kwargs)
    return SatRec(**defaults)
