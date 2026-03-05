"""propagate_mixed — heterogeneous multi-satellite convenience propagator."""

import functools

import numpy as np
import jax
import jax.numpy as jnp
import jax.typing

from sgp4jax._types import SatRec
from sgp4jax._propagation import sgp4 as _propagate_full
from sgp4jax._propagation_leo import sgp4_leo as _propagate_leo
from sgp4jax._propagation_sdp4_nr import sgp4_sdp4_nr as _propagate_sdp4_nr


def _slice_satrec(satrec: SatRec, indices: np.ndarray) -> SatRec:
    """Return a sub-batch of a stacked SatRec at the given integer indices."""
    return SatRec(*[field[indices] for field in satrec])


@functools.lru_cache(maxsize=8)
def _group_propagator(method: int, irez: int):
    """Return a cached JIT-compiled vmap(vmap(fn)) for the given satellite type.

    At most four distinct compilations occur in practice:
      method=0            → sgp4_leo        (near-earth)
      method=1, irez=0    → sgp4_sdp4_nr    (deep-space, no resonance)
      method=1, irez=1    → sgp4            (synchronous resonance, GEO)
      method=1, irez=2    → sgp4            (half-day resonance, Molniya)
    """
    if method == 0:
        fn = _propagate_leo
    elif irez == 0:
        fn = _propagate_sdp4_nr
    else:
        fn = _propagate_full
    # outer vmap over satellites, inner vmap over times → (N_group, M, 3)
    return jax.jit(jax.vmap(jax.vmap(fn, in_axes=(None, 0)), in_axes=(0, None)))


def propagate_mixed(
    satrec_batch: SatRec,
    times: jax.typing.ArrayLike,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Propagate a heterogeneous batch of satellites over a shared array of times.

    Groups satellites by type (near-earth / deep-space-no-resonance /
    deep-space-resonant) and dispatches each group to the appropriate
    specialized propagator, avoiding dead-branch computation for every group.
    Results are reassembled into the original satellite ordering.

    .. note::
        This function is **not** JIT-compilable as a whole and does not compose
        with ``jax.grad`` or ``jax.vmap``.  For JIT / AD / vmap compatibility,
        group satellites by type and call the specialized propagators directly
        (:func:`propagate_leo`, :func:`propagate_sdp4_nr`, :func:`propagate`).

    Args:
        satrec_batch: Batched SatRec from :func:`tles_to_satrec`, ``N`` satellites.
        times: 1-D array of times since epoch in minutes, shape ``(M,)``.

    Returns:
        r:     Positions in TEME frame, shape ``(N, M, 3)`` in km.
        v:     Velocities in TEME frame, shape ``(N, M, 3)`` in km/s.
        error: Error codes, shape ``(N, M)``  (0 = success).
    """
    times  = jnp.asarray(times)
    n_sats  = int(satrec_batch.method.shape[0])
    n_times = int(times.shape[0])

    # Read satellite types as concrete numpy values for Python-level grouping
    methods = np.asarray(satrec_batch.method)
    irezs   = np.asarray(satrec_batch.irez)

    # Group satellite indices by (method, irez)
    groups: dict[tuple[int, int], list[int]] = {}
    for i in range(n_sats):
        key = (int(methods[i]), int(irezs[i]))
        groups.setdefault(key, []).append(i)

    # Preallocate output arrays; .at[].set() scatter is differentiable
    r_out = jnp.zeros((n_sats, n_times, 3))
    v_out = jnp.zeros((n_sats, n_times, 3))
    e_out = jnp.zeros((n_sats, n_times), dtype=jnp.int32)

    for (method, irez), sat_indices in groups.items():
        idx = np.array(sat_indices)
        sub = _slice_satrec(satrec_batch, idx)
        fn  = _group_propagator(method, irez)
        r_g, v_g, e_g = fn(sub, times)       # (n_group, M, 3) / (n_group, M)
        r_out = r_out.at[idx].set(r_g)
        v_out = v_out.at[idx].set(v_g)
        e_out = e_out.at[idx].set(e_g.astype(jnp.int32))

    return r_out, v_out, e_out
