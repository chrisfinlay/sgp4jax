"""Covariance transformations between RIC, Cartesian (TEME), and orbital element frames.

The RIC (Radial, In-track, Cross-track) frame is defined at a given state (r, v):

* **R** — radial: r̂ = r / |r|
* **I** — in-track: Î = Ĉ × R̂
* **C** — cross-track: Ĉ = (r × v) / |r × v|  (along angular-momentum vector)

Two element models are supported:

**6-element model** — pure Keplerian, element ordering::

    (inclo, nodeo, ecco, argpo, mo, no_kozai)

Jacobian via :func:`jax.jacobian` on the Keplerian forward map.  Fast,
fully JIT-compilable.  Does not include drag.

**7-element model** — SGP4 + drag, element ordering::

    (inclo, nodeo, ecco, argpo, mo, no_kozai, bstar)

Jacobian via :func:`jax.jacobian` through the full ``sgp4init → sgp4``
pipeline, so bstar's influence on the trajectory is captured.

.. note::
   For the 7-element model the Jacobian J has shape ``(6, 7)`` (6 state
   dimensions, 7 element dimensions).  The forward transform
   Σ_rv = J Σ_el7 Jᵀ is always well-defined.  The reverse
   Σ_el7 = J† Σ_rv (J†)ᵀ uses the right pseudo-inverse
   J† = Jᵀ(JJᵀ)⁻¹ and produces a **rank-deficient** (rank ≤ 6) matrix —
   a 6-dimensional state cannot fully constrain 7 element dimensions.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.typing

from sgp4jax._types import SatRec
from sgp4jax._constants import GravityConstants
from sgp4jax._kepler import _kepler_rv_teme
from sgp4jax._sgp4init import sgp4init
from sgp4jax._propagation import sgp4 as _sgp4


# ---------------------------------------------------------------------------
# RIC ↔ TEME rotation
# ---------------------------------------------------------------------------

def ric_rotation(
    r: jax.Array,
    v: jax.Array,
) -> jax.Array:
    """Build the 3×3 rotation matrix from the RIC frame to Cartesian (TEME).

    The columns of the returned matrix are the RIC unit vectors expressed in
    the Cartesian frame::

        T[:, 0] = R̂  (radial)
        T[:, 1] = Î  (in-track)
        T[:, 2] = Ĉ  (cross-track)

    So a position expressed in RIC coordinates transforms to Cartesian as
    ``x_cart = T @ x_ric``.

    Args:
        r: Position vector in TEME, shape ``(3,)``, km.
        v: Velocity vector in TEME, shape ``(3,)``, km/s.

    Returns:
        T: Rotation matrix, shape ``(3, 3)``.
    """
    r_hat = r / jnp.linalg.norm(r)
    h = jnp.cross(r, v)
    c_hat = h / jnp.linalg.norm(h)
    i_hat = jnp.cross(c_hat, r_hat)
    return jnp.stack([r_hat, i_hat, c_hat], axis=1)


def _ric_rotation_6(r: jax.Array, v: jax.Array) -> jax.Array:
    """Block-diagonal 6×6 version of :func:`ric_rotation`."""
    T = ric_rotation(r, v)
    zeros = jnp.zeros((3, 3))
    return jnp.block([[T, zeros], [zeros, T]])


def cov_ric_to_teme(
    cov_ric: jax.typing.ArrayLike,
    r: jax.Array,
    v: jax.Array,
) -> jax.Array:
    """Transform a 6×6 state covariance from the RIC frame to TEME Cartesian.

    Args:
        cov_ric: Covariance matrix in RIC frame, shape ``(6, 6)``.  Block
            structure: ``[[Σ_pos, Σ_pv], [Σ_vp, Σ_vel]]`` where each block
            is ``(3, 3)``.
        r: Position in TEME at the epoch of the covariance, shape ``(3,)``, km.
        v: Velocity in TEME at the epoch of the covariance, shape ``(3,)``, km/s.

    Returns:
        Covariance in TEME frame, shape ``(6, 6)``.
    """
    T6 = _ric_rotation_6(r, v)
    return T6 @ jnp.asarray(cov_ric) @ T6.T


def cov_teme_to_ric(
    cov_teme: jax.typing.ArrayLike,
    r: jax.Array,
    v: jax.Array,
) -> jax.Array:
    """Transform a 6×6 TEME Cartesian covariance to the RIC frame.

    Args:
        cov_teme: Covariance matrix in TEME, shape ``(6, 6)``.
        r: Position in TEME at the epoch of the covariance, shape ``(3,)``, km.
        v: Velocity in TEME at the epoch of the covariance, shape ``(3,)``, km/s.

    Returns:
        Covariance in RIC frame, shape ``(6, 6)``.
    """
    T6 = _ric_rotation_6(r, v)
    return T6.T @ jnp.asarray(cov_teme) @ T6


# ---------------------------------------------------------------------------
# Keplerian Jacobian  ∂(r, v) / ∂(elements)
# ---------------------------------------------------------------------------

def _elements_to_rv_flat(
    elements: jax.Array,
    mu: jax.typing.ArrayLike,
    jdsatepoch: jax.typing.ArrayLike,
    jdsatepochF: jax.typing.ArrayLike,
    jd: jax.typing.ArrayLike,
    fr: jax.typing.ArrayLike,
) -> jax.Array:
    """Map flat element vector → flat (r, v) for use with :func:`jax.jacobian`."""
    inclo, nodeo, ecco, argpo, mo, no_kozai = elements
    r, v = _kepler_rv_teme(
        inclo, nodeo, ecco, argpo, mo, no_kozai, mu,
        jdsatepoch, jdsatepochF, jd, fr,
    )
    return jnp.concatenate([r, v])


def elements_jacobian(
    satrec: SatRec,
    jd: jax.typing.ArrayLike,
    fr: jax.typing.ArrayLike,
) -> jax.Array:
    """Jacobian of the Keplerian state map with respect to orbital elements.

    Computes  J = ∂(r, v) / ∂(inclo, nodeo, ecco, argpo, mo, no_kozai)
    evaluated at the given Julian date using the pure two-body Keplerian map.

    Args:
        satrec: Scalar SatRec from :func:`~sgp4jax.tle_to_satrec`.
        jd: Julian date, whole part (scalar).
        fr: Julian date, fractional part (scalar).

    Returns:
        J: Jacobian matrix, shape ``(6, 6)``.  Rows correspond to
        ``(r_x, r_y, r_z, v_x, v_y, v_z)``; columns to
        ``(inclo, nodeo, ecco, argpo, mo, no_kozai)``.
    """
    elements = jnp.array([
        satrec.inclo, satrec.nodeo, satrec.ecco,
        satrec.argpo, satrec.mo, satrec.no_kozai,
    ])
    return jax.jacobian(_elements_to_rv_flat)(  # type: ignore[no-any-return]
        elements, satrec.mu, satrec.jdsatepoch, satrec.jdsatepochF, jd, fr,
    )


# ---------------------------------------------------------------------------
# TEME ↔ element covariance
# ---------------------------------------------------------------------------

def cov_elements_to_teme(
    cov_elements: jax.typing.ArrayLike,
    satrec: SatRec,
    jd: jax.typing.ArrayLike,
    fr: jax.typing.ArrayLike,
) -> jax.Array:
    """Transform a 6×6 orbital element covariance to TEME Cartesian covariance.

    Element ordering: ``(inclo, nodeo, ecco, argpo, mo, no_kozai)``.

    Uses the linear propagation rule  Σ_rv = J Σ_el J^T  where J is the
    Keplerian Jacobian ∂(r,v)/∂(elements) at the given time.

    Args:
        cov_elements: Covariance in element space, shape ``(6, 6)``.
        satrec: Scalar SatRec from :func:`~sgp4jax.tle_to_satrec`.
        jd: Julian date, whole part (scalar).
        fr: Julian date, fractional part (scalar).

    Returns:
        Covariance in TEME Cartesian frame, shape ``(6, 6)``.
    """
    J = elements_jacobian(satrec, jd, fr)
    return J @ jnp.asarray(cov_elements) @ J.T


def cov_teme_to_elements(
    cov_teme: jax.typing.ArrayLike,
    satrec: SatRec,
    jd: jax.typing.ArrayLike,
    fr: jax.typing.ArrayLike,
) -> jax.Array:
    """Transform a 6×6 TEME Cartesian covariance to orbital element covariance.

    Element ordering: ``(inclo, nodeo, ecco, argpo, mo, no_kozai)``.

    Inverts the Keplerian Jacobian:  Σ_el = J⁻¹ Σ_rv J⁻ᵀ.

    Args:
        cov_teme: Covariance in TEME Cartesian frame, shape ``(6, 6)``.
        satrec: Scalar SatRec from :func:`~sgp4jax.tle_to_satrec`.
        jd: Julian date, whole part (scalar).
        fr: Julian date, fractional part (scalar).

    Returns:
        Covariance in element space, shape ``(6, 6)``.
        Element ordering: ``(inclo, nodeo, ecco, argpo, mo, no_kozai)``.
    """
    J = elements_jacobian(satrec, jd, fr)
    J_inv = jnp.linalg.inv(J)
    return J_inv @ jnp.asarray(cov_teme) @ J_inv.T  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Full chain: RIC → elements
# ---------------------------------------------------------------------------

def cov_ric_to_elements(
    cov_ric: jax.typing.ArrayLike,
    satrec: SatRec,
    jd: jax.typing.ArrayLike,
    fr: jax.typing.ArrayLike,
) -> jax.Array:
    """Transform a 6×6 RIC covariance to orbital element covariance.

    Full chain: RIC → TEME → elements.  The Keplerian state at ``(jd, fr)``
    is used both to define the RIC frame and to evaluate the Jacobian.

    Element ordering: ``(inclo, nodeo, ecco, argpo, mo, no_kozai)``.

    Args:
        cov_ric: Covariance in RIC frame, shape ``(6, 6)``.
        satrec: Scalar SatRec from :func:`~sgp4jax.tle_to_satrec`.
        jd: Julian date, whole part (scalar).
        fr: Julian date, fractional part (scalar).

    Returns:
        Covariance in element space, shape ``(6, 6)``.
    """
    r_teme, v_teme = _kepler_rv_teme(
        satrec.inclo, satrec.nodeo, satrec.ecco,
        satrec.argpo, satrec.mo, satrec.no_kozai,
        satrec.mu, satrec.jdsatepoch, satrec.jdsatepochF, jd, fr,
    )
    cov_teme = cov_ric_to_teme(cov_ric, r_teme, v_teme)
    return cov_teme_to_elements(cov_teme, satrec, jd, fr)


def cov_elements_to_ric(
    cov_elements: jax.typing.ArrayLike,
    satrec: SatRec,
    jd: jax.typing.ArrayLike,
    fr: jax.typing.ArrayLike,
) -> jax.Array:
    """Transform a 6×6 orbital element covariance to the RIC frame.

    Full chain: elements → TEME → RIC.  Inverse of :func:`cov_ric_to_elements`.

    Element ordering: ``(inclo, nodeo, ecco, argpo, mo, no_kozai)``.

    Args:
        cov_elements: Covariance in element space, shape ``(6, 6)``.
        satrec: Scalar SatRec from :func:`~sgp4jax.tle_to_satrec`.
        jd: Julian date, whole part (scalar).
        fr: Julian date, fractional part (scalar).

    Returns:
        Covariance in RIC frame, shape ``(6, 6)``.
    """
    r_teme, v_teme = _kepler_rv_teme(
        satrec.inclo, satrec.nodeo, satrec.ecco,
        satrec.argpo, satrec.mo, satrec.no_kozai,
        satrec.mu, satrec.jdsatepoch, satrec.jdsatepochF, jd, fr,
    )
    cov_teme = cov_elements_to_teme(cov_elements, satrec, jd, fr)
    return cov_teme_to_ric(cov_teme, r_teme, v_teme)


# ---------------------------------------------------------------------------
# 7-element model: (inclo, nodeo, ecco, argpo, mo, no_kozai, bstar)
# ---------------------------------------------------------------------------

def _elements7_to_rv_flat(
    elements7: jax.Array,
    satrec_ref: SatRec,
    jd: jax.typing.ArrayLike,
    fr: jax.typing.ArrayLike,
) -> jax.Array:
    """Map 7-element vector → flat (r, v) via SGP4, for use with jax.jacobian."""
    inclo, nodeo, ecco, argpo, mo, no_kozai, bstar = elements7
    whichconst = GravityConstants(
        satrec_ref.tumin, satrec_ref.mu, satrec_ref.radiusearthkm,  # type: ignore[arg-type]
        satrec_ref.xke, satrec_ref.j2, satrec_ref.j3, satrec_ref.j4, satrec_ref.j3oj2,  # type: ignore[arg-type]
    )
    epoch = satrec_ref.jdsatepoch + satrec_ref.jdsatepochF - 2433281.5
    satrec_new = sgp4init(
        whichconst, epoch,
        bstar, satrec_ref.ndot, satrec_ref.nddot,
        ecco, argpo, inclo, mo, no_kozai, nodeo,
        satrec_ref.jdsatepoch, satrec_ref.jdsatepochF,
    )
    tsince = (jd - satrec_new.jdsatepoch + fr - satrec_new.jdsatepochF) * 1440.0
    r, v, _ = _sgp4(satrec_new, tsince)
    return jnp.concatenate([r, v])


def elements7_jacobian(
    satrec: SatRec,
    jd: jax.typing.ArrayLike,
    fr: jax.typing.ArrayLike,
) -> jax.Array:
    """Jacobian of the SGP4 state map with respect to the 7-element model.

    Computes  J = ∂(r, v) / ∂(inclo, nodeo, ecco, argpo, mo, no_kozai, bstar)
    by differentiating through the full ``sgp4init → sgp4`` pipeline.

    Unlike the 6-element :func:`elements_jacobian`, this captures bstar's
    influence on the trajectory via the drag terms in SGP4.

    Args:
        satrec: Scalar SatRec from :func:`~sgp4jax.tle_to_satrec`.
        jd: Julian date, whole part (scalar).
        fr: Julian date, fractional part (scalar).

    Returns:
        J: Jacobian matrix, shape ``(6, 7)``.  Rows correspond to
        ``(r_x, r_y, r_z, v_x, v_y, v_z)``; columns to
        ``(inclo, nodeo, ecco, argpo, mo, no_kozai, bstar)``.
    """
    elements7 = jnp.array([
        satrec.inclo, satrec.nodeo, satrec.ecco,
        satrec.argpo, satrec.mo, satrec.no_kozai, satrec.bstar,
    ])
    return jax.jacobian(_elements7_to_rv_flat)(  # type: ignore[no-any-return]
        elements7, satrec, jd, fr,
    )


def cov_elements7_to_teme(
    cov_elements7: jax.typing.ArrayLike,
    satrec: SatRec,
    jd: jax.typing.ArrayLike,
    fr: jax.typing.ArrayLike,
) -> jax.Array:
    """Transform a 7×7 element covariance to TEME Cartesian covariance.

    Element ordering: ``(inclo, nodeo, ecco, argpo, mo, no_kozai, bstar)``.

    Uses the linear propagation rule  Σ_rv = J Σ_el7 Jᵀ  where J is the
    SGP4 Jacobian ∂(r,v)/∂(elements7), shape ``(6, 7)``.

    Args:
        cov_elements7: Covariance in 7-element space, shape ``(7, 7)``.
        satrec: Scalar SatRec from :func:`~sgp4jax.tle_to_satrec`.
        jd: Julian date, whole part (scalar).
        fr: Julian date, fractional part (scalar).

    Returns:
        Covariance in TEME Cartesian frame, shape ``(6, 6)``.
    """
    J = elements7_jacobian(satrec, jd, fr)  # (6, 7)
    return J @ jnp.asarray(cov_elements7) @ J.T


def cov_teme_to_elements7(
    cov_teme: jax.typing.ArrayLike,
    satrec: SatRec,
    jd: jax.typing.ArrayLike,
    fr: jax.typing.ArrayLike,
) -> jax.Array:
    """Transform a 6×6 TEME Cartesian covariance to 7-element covariance.

    Element ordering: ``(inclo, nodeo, ecco, argpo, mo, no_kozai, bstar)``.

    Uses the right pseudo-inverse J† = Jᵀ(JJᵀ)⁻¹ to invert the 6×7 Jacobian:
    Σ_el7 = J† Σ_rv (J†)ᵀ.

    .. warning::
        The result is **rank-deficient** (rank ≤ 6).  A 6-dimensional Cartesian
        state cannot uniquely constrain all 7 element dimensions — bstar
        is only weakly observable from a single-epoch state snapshot.
        For a full bstar estimate, propagate and fit over multiple epochs.

    Args:
        cov_teme: Covariance in TEME Cartesian frame, shape ``(6, 6)``.
        satrec: Scalar SatRec from :func:`~sgp4jax.tle_to_satrec`.
        jd: Julian date, whole part (scalar).
        fr: Julian date, fractional part (scalar).

    Returns:
        Covariance in 7-element space, shape ``(7, 7)``.  Rank ≤ 6.
    """
    J = elements7_jacobian(satrec, jd, fr)  # (6, 7)
    # Right pseudo-inverse: J† = Jᵀ (J Jᵀ)⁻¹,  shape (7, 6)
    JJT_inv = jnp.linalg.inv(J @ J.T)
    J_pinv = J.T @ JJT_inv
    return J_pinv @ jnp.asarray(cov_teme) @ J_pinv.T  # type: ignore[no-any-return]


def cov_elements7_to_ric(
    cov_elements7: jax.typing.ArrayLike,
    satrec: SatRec,
    jd: jax.typing.ArrayLike,
    fr: jax.typing.ArrayLike,
) -> jax.Array:
    """Transform a 7×7 element covariance to the RIC frame.

    Full chain: elements7 → TEME → RIC.  The SGP4 state at ``(jd, fr)``
    defines the RIC frame.

    Element ordering: ``(inclo, nodeo, ecco, argpo, mo, no_kozai, bstar)``.

    Args:
        cov_elements7: Covariance in 7-element space, shape ``(7, 7)``.
        satrec: Scalar SatRec from :func:`~sgp4jax.tle_to_satrec`.
        jd: Julian date, whole part (scalar).
        fr: Julian date, fractional part (scalar).

    Returns:
        Covariance in RIC frame, shape ``(6, 6)``.
    """
    r_teme, v_teme, _ = _sgp4(satrec, (jd - satrec.jdsatepoch + fr - satrec.jdsatepochF) * 1440.0)
    cov_teme = cov_elements7_to_teme(cov_elements7, satrec, jd, fr)
    return cov_teme_to_ric(cov_teme, r_teme, v_teme)


def cov_ric_to_elements7(
    cov_ric: jax.typing.ArrayLike,
    satrec: SatRec,
    jd: jax.typing.ArrayLike,
    fr: jax.typing.ArrayLike,
) -> jax.Array:
    """Transform a 6×6 RIC covariance to 7-element covariance.

    Full chain: RIC → TEME → elements7.  The SGP4 state at ``(jd, fr)``
    defines the RIC frame.

    Element ordering: ``(inclo, nodeo, ecco, argpo, mo, no_kozai, bstar)``.

    .. warning::
        The result is **rank-deficient** (rank ≤ 6).  See
        :func:`cov_teme_to_elements7` for details.

    Args:
        cov_ric: Covariance in RIC frame, shape ``(6, 6)``.
        satrec: Scalar SatRec from :func:`~sgp4jax.tle_to_satrec`.
        jd: Julian date, whole part (scalar).
        fr: Julian date, fractional part (scalar).

    Returns:
        Covariance in 7-element space, shape ``(7, 7)``.  Rank ≤ 6.
    """
    r_teme, v_teme, _ = _sgp4(satrec, (jd - satrec.jdsatepoch + fr - satrec.jdsatepochF) * 1440.0)
    cov_teme = cov_ric_to_teme(cov_ric, r_teme, v_teme)
    return cov_teme_to_elements7(cov_teme, satrec, jd, fr)
