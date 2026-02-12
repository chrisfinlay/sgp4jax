"""SatRec NamedTuple for JAX-compatible SGP4 propagation."""

from typing import NamedTuple
import jax.numpy as jnp


class SatRec(NamedTuple):
    """JAX-compatible satellite record.

    All fields are jnp.ndarray scalars (float64).
    String flags are encoded as floats:
      method: 0.0 = near-earth, 1.0 = deep-space
      isimp: 0.0 = full perturbations, 1.0 = simplified
      irez: 0.0 = no resonance, 1.0 = synchronous, 2.0 = half-day
    """
    # --- TLE-derived orbital elements ---
    bstar: jnp.ndarray
    ecco: jnp.ndarray
    argpo: jnp.ndarray
    inclo: jnp.ndarray
    mo: jnp.ndarray
    no_kozai: jnp.ndarray
    nodeo: jnp.ndarray
    ndot: jnp.ndarray
    nddot: jnp.ndarray

    # --- Gravity model constants ---
    j2: jnp.ndarray
    j3: jnp.ndarray
    j4: jnp.ndarray
    j3oj2: jnp.ndarray
    xke: jnp.ndarray
    mu: jnp.ndarray
    radiusearthkm: jnp.ndarray
    tumin: jnp.ndarray

    # --- Epoch ---
    jdsatepoch: jnp.ndarray
    jdsatepochF: jnp.ndarray

    # --- Computed by _initl / sgp4init (near-earth) ---
    no_unkozai: jnp.ndarray
    a: jnp.ndarray
    alta: jnp.ndarray
    altp: jnp.ndarray
    con41: jnp.ndarray
    cc1: jnp.ndarray
    cc4: jnp.ndarray
    cc5: jnp.ndarray
    d2: jnp.ndarray
    d3: jnp.ndarray
    d4: jnp.ndarray
    delmo: jnp.ndarray
    eta: jnp.ndarray
    argpdot: jnp.ndarray
    omgcof: jnp.ndarray
    sinmao: jnp.ndarray
    t2cof: jnp.ndarray
    t3cof: jnp.ndarray
    t4cof: jnp.ndarray
    t5cof: jnp.ndarray
    x1mth2: jnp.ndarray
    x7thm1: jnp.ndarray
    mdot: jnp.ndarray
    nodedot: jnp.ndarray
    xlcof: jnp.ndarray
    xmcof: jnp.ndarray
    nodecf: jnp.ndarray
    aycof: jnp.ndarray
    gsto: jnp.ndarray

    # --- Deep space coefficients ---
    d2201: jnp.ndarray
    d2211: jnp.ndarray
    d3210: jnp.ndarray
    d3222: jnp.ndarray
    d4410: jnp.ndarray
    d4422: jnp.ndarray
    d5220: jnp.ndarray
    d5232: jnp.ndarray
    d5421: jnp.ndarray
    d5433: jnp.ndarray
    dedt: jnp.ndarray
    del1: jnp.ndarray
    del2: jnp.ndarray
    del3: jnp.ndarray
    didt: jnp.ndarray
    dmdt: jnp.ndarray
    dnodt: jnp.ndarray
    domdt: jnp.ndarray
    e3: jnp.ndarray
    ee2: jnp.ndarray
    peo: jnp.ndarray
    pgho: jnp.ndarray
    pho: jnp.ndarray
    pinco: jnp.ndarray
    plo: jnp.ndarray
    se2: jnp.ndarray
    se3: jnp.ndarray
    sgh2: jnp.ndarray
    sgh3: jnp.ndarray
    sgh4: jnp.ndarray
    sh2: jnp.ndarray
    sh3: jnp.ndarray
    si2: jnp.ndarray
    si3: jnp.ndarray
    sl2: jnp.ndarray
    sl3: jnp.ndarray
    sl4: jnp.ndarray
    xfact: jnp.ndarray
    xgh2: jnp.ndarray
    xgh3: jnp.ndarray
    xgh4: jnp.ndarray
    xh2: jnp.ndarray
    xh3: jnp.ndarray
    xi2: jnp.ndarray
    xi3: jnp.ndarray
    xl2: jnp.ndarray
    xl3: jnp.ndarray
    xl4: jnp.ndarray
    xlamo: jnp.ndarray
    xli: jnp.ndarray
    xni: jnp.ndarray
    zmol: jnp.ndarray
    zmos: jnp.ndarray
    atime: jnp.ndarray

    # --- Flags (encoded as floats) ---
    method: jnp.ndarray     # 0.0 = near-earth, 1.0 = deep-space
    isimp: jnp.ndarray      # 0.0 or 1.0
    irez: jnp.ndarray       # 0.0, 1.0, or 2.0


def make_satrec(**kwargs) -> SatRec:
    """Create a SatRec with defaults of 0.0 for unspecified fields."""
    defaults = {field: jnp.array(0.0) for field in SatRec._fields}
    defaults.update(kwargs)
    return SatRec(**defaults)
