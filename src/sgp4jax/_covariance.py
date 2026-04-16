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

    Parameters
    ----------
    r : jax.Array, shape (3,)
        Position vector in TEME, km.
    v : jax.Array, shape (3,)
        Velocity vector in TEME, km/s.

    Returns
    -------
    T : jax.Array, shape (3, 3)
        Rotation matrix.
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

    Parameters
    ----------
    cov_ric : array-like, shape (6, 6)
        Covariance matrix in RIC frame.  Block structure:
        ``[[Σ_pos, Σ_pv], [Σ_vp, Σ_vel]]`` where each block is ``(3, 3)``.
    r : jax.Array, shape (3,)
        Position in TEME at the epoch of the covariance, km.
    v : jax.Array, shape (3,)
        Velocity in TEME at the epoch of the covariance, km/s.

    Returns
    -------
    jax.Array, shape (6, 6)
        Covariance in TEME frame.
    """
    T6 = _ric_rotation_6(r, v)
    return T6 @ jnp.asarray(cov_ric) @ T6.T


def cov_teme_to_ric(
    cov_teme: jax.typing.ArrayLike,
    r: jax.Array,
    v: jax.Array,
) -> jax.Array:
    """Transform a 6×6 TEME Cartesian covariance to the RIC frame.

    Parameters
    ----------
    cov_teme : array-like, shape (6, 6)
        Covariance matrix in TEME.
    r : jax.Array, shape (3,)
        Position in TEME at the epoch of the covariance, km.
    v : jax.Array, shape (3,)
        Velocity in TEME at the epoch of the covariance, km/s.

    Returns
    -------
    jax.Array, shape (6, 6)
        Covariance in RIC frame.
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

    Parameters
    ----------
    satrec : SatRec
        Scalar SatRec from :func:`~sgp4jax.tle_to_satrec`.
    jd : array-like
        Julian date, whole part (scalar).
    fr : array-like
        Julian date, fractional part (scalar).

    Returns
    -------
    J : jax.Array, shape (6, 6)
        Jacobian matrix.  Rows correspond to
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

    Parameters
    ----------
    cov_elements : array-like, shape (6, 6)
        Covariance in element space.
    satrec : SatRec
        Scalar SatRec from :func:`~sgp4jax.tle_to_satrec`.
    jd : array-like
        Julian date, whole part (scalar).
    fr : array-like
        Julian date, fractional part (scalar).

    Returns
    -------
    jax.Array, shape (6, 6)
        Covariance in TEME Cartesian frame.
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

    Parameters
    ----------
    cov_teme : array-like, shape (6, 6)
        Covariance in TEME Cartesian frame.
    satrec : SatRec
        Scalar SatRec from :func:`~sgp4jax.tle_to_satrec`.
    jd : array-like
        Julian date, whole part (scalar).
    fr : array-like
        Julian date, fractional part (scalar).

    Returns
    -------
    jax.Array, shape (6, 6)
        Covariance in element space.
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

    Parameters
    ----------
    cov_ric : array-like, shape (6, 6)
        Covariance in RIC frame.
    satrec : SatRec
        Scalar SatRec from :func:`~sgp4jax.tle_to_satrec`.
    jd : array-like
        Julian date, whole part (scalar).
    fr : array-like
        Julian date, fractional part (scalar).

    Returns
    -------
    jax.Array, shape (6, 6)
        Covariance in element space.
        Element ordering: ``(inclo, nodeo, ecco, argpo, mo, no_kozai)``.
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

    Parameters
    ----------
    cov_elements : array-like, shape (6, 6)
        Covariance in element space.
        Element ordering: ``(inclo, nodeo, ecco, argpo, mo, no_kozai)``.
    satrec : SatRec
        Scalar SatRec from :func:`~sgp4jax.tle_to_satrec`.
    jd : array-like
        Julian date, whole part (scalar).
    fr : array-like
        Julian date, fractional part (scalar).

    Returns
    -------
    jax.Array, shape (6, 6)
        Covariance in RIC frame.
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

    Parameters
    ----------
    satrec : SatRec
        Scalar SatRec from :func:`~sgp4jax.tle_to_satrec`.
    jd : array-like
        Julian date, whole part (scalar).
    fr : array-like
        Julian date, fractional part (scalar).

    Returns
    -------
    J : jax.Array, shape (6, 7)
        Jacobian matrix.  Rows correspond to
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

    Parameters
    ----------
    cov_elements7 : array-like, shape (7, 7)
        Covariance in 7-element space.
    satrec : SatRec
        Scalar SatRec from :func:`~sgp4jax.tle_to_satrec`.
    jd : array-like
        Julian date, whole part (scalar).
    fr : array-like
        Julian date, fractional part (scalar).

    Returns
    -------
    jax.Array, shape (6, 6)
        Covariance in TEME Cartesian frame.
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

    Parameters
    ----------
    cov_teme : array-like, shape (6, 6)
        Covariance in TEME Cartesian frame.
    satrec : SatRec
        Scalar SatRec from :func:`~sgp4jax.tle_to_satrec`.
    jd : array-like
        Julian date, whole part (scalar).
    fr : array-like
        Julian date, fractional part (scalar).

    Returns
    -------
    jax.Array, shape (7, 7)
        Covariance in 7-element space.  Rank ≤ 6.
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

    Parameters
    ----------
    cov_elements7 : array-like, shape (7, 7)
        Covariance in 7-element space.
    satrec : SatRec
        Scalar SatRec from :func:`~sgp4jax.tle_to_satrec`.
    jd : array-like
        Julian date, whole part (scalar).
    fr : array-like
        Julian date, fractional part (scalar).

    Returns
    -------
    jax.Array, shape (6, 6)
        Covariance in RIC frame.
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

    Parameters
    ----------
    cov_ric : array-like, shape (6, 6)
        Covariance in RIC frame.
    satrec : SatRec
        Scalar SatRec from :func:`~sgp4jax.tle_to_satrec`.
    jd : array-like
        Julian date, whole part (scalar).
    fr : array-like
        Julian date, fractional part (scalar).

    Returns
    -------
    jax.Array, shape (7, 7)
        Covariance in 7-element space.  Rank ≤ 6.
    """
    r_teme, v_teme, _ = _sgp4(satrec, (jd - satrec.jdsatepoch + fr - satrec.jdsatepochF) * 1440.0)
    cov_teme = cov_ric_to_teme(cov_ric, r_teme, v_teme)
    return cov_teme_to_elements7(cov_teme, satrec, jd, fr)


# ---------------------------------------------------------------------------
# Empirical RIC covariance from TLE age
# ---------------------------------------------------------------------------

def tle_ric_covariance(
    satrec: SatRec,
    jd: jax.typing.ArrayLike,
    fr: jax.typing.ArrayLike,
    *,
    sigma_r0: float = 0.050,
    sigma_t0: float = 0.300,
    sigma_n0: float = 0.050,
) -> jax.Array:
    """Empirical RIC position-velocity covariance estimate based on TLE age.

    Implements a Vallado-style error-growth model.  At epoch the dominant
    uncertainty is in-track (along-track timing).  Both in-track and
    (to a lesser degree) radial errors grow with time as tracking
    information becomes stale; in-track growth is further scaled by the
    drag coefficient ``bstar`` relative to the LEO population median.

    The velocity block is derived from the position block via the
    Keplerian approximation ``σ_ṽ ≈ n · σ_r``, where *n* is the mean
    motion.  Position-velocity cross-terms are set to zero (diagonal
    model); they are small relative to the diagonal terms over typical
    TLE-age timescales and the uncertainty in the growth-rate parameters
    themselves dominates.

    **Error-growth model (all in km):**

    .. code-block:: text

        σ_R(Δt) = σ_R0 + 0.05  · |Δt|          radial
        σ_T(Δt) = σ_T0 + γ_T   · |Δt|          in-track
        σ_N(Δt) = σ_N0 + 0.05  · |Δt|          cross-track

        γ_T = 0.5 + 2.0 · (|bstar| / 3.6×10⁻⁴)   km/day

    The in-track growth rate ``γ_T`` has two contributions:

    * **Base term** (0.5 km/day) — from mean-motion uncertainty, present
      even at zero drag.
    * **Drag term** (2.0 km/day at median LEO ``bstar``) — from
      atmospheric-density uncertainty amplified by the drag coefficient.
      The reference ``bstar`` of 3.6×10⁻⁴ km⁻¹ is the empirical median
      over 13 901 active LEO objects (March 2026).

    Typical 1-σ in-track error at median ``bstar``:

    * Fresh TLE (Δt = 0):  0.3 km
    * 1 day old:            2.8 km
    * 3 days old:           8.1 km
    * 7 days old:           18 km (treat as effectively unusable)

    .. note::
        This model is calibrated for well-tracked LEO objects (radar
        cross-section ≳ 10 cm, altitude 200–2000 km).  High
        area-to-mass objects (debris, balloons) degrade 5–10× faster.
        When a CDM covariance is available from space-track it should
        be preferred over this estimate.

    Parameters
    ----------
    satrec : SatRec
        Initialized SatRec from :func:`~sgp4jax.tle_to_satrec`.
    jd : array-like
        Target Julian date, whole part (scalar).
    fr : array-like
        Target Julian date, fractional part (scalar).
    sigma_r0 : float, optional
        1-σ radial position uncertainty at TLE epoch, km.
        Default 0.050 (50 m), reflecting typical OD residuals.
    sigma_t0 : float, optional
        1-σ in-track position uncertainty at TLE epoch, km.  Default 0.300.
    sigma_n0 : float, optional
        1-σ cross-track position uncertainty at TLE epoch, km.  Default 0.050.

    Returns
    -------
    cov_ric : jax.Array, shape (6, 6)
        Covariance matrix in the RIC frame, block ordered
        ``[R, T, N, Ṙ, Ṫ, Ṅ]``.  Units: km² (position block),
        km²/s² (velocity block).  Cross-terms are zero.
    """
    # --- Time since TLE epoch (days), symmetric about epoch ---
    dt_days = jnp.abs(
        jd - satrec.jdsatepoch + fr - satrec.jdsatepochF
    )

    # --- Mean motion (rad/s) for velocity uncertainty scaling ---
    n_rad_per_s = satrec.no_unkozai / 60.0

    # --- In-track growth rate (km/day) ---
    # Base: 0.5 km/day from mean-motion uncertainty at all altitudes.
    # Drag: 2.0 km/day * (|bstar| / bstar_ref), empirical LEO median.
    bstar_ref = 3.6e-4   # km⁻¹  (median from 13,901 active LEO TLEs)
    drag_scale = jnp.abs(satrec.bstar) / bstar_ref
    gamma_t = 0.5 + 2.0 * drag_scale   # km/day

    # Radial and cross-track growth rates (km/day) — much smaller,
    # driven by inclination/RAAN estimation uncertainty.
    gamma_r = 0.05
    gamma_n = 0.05

    # --- Position 1-σ (km) ---
    sr = jnp.asarray(sigma_r0) + gamma_r * dt_days
    st = jnp.asarray(sigma_t0) + gamma_t * dt_days
    sn = jnp.asarray(sigma_n0) + gamma_n * dt_days

    # --- Velocity 1-σ (km/s): Keplerian approximation σ_ṽ ≈ n · σ_r ---
    sr_dot = n_rad_per_s * sr
    st_dot = n_rad_per_s * st
    sn_dot = n_rad_per_s * sn

    # --- Build diagonal 6×6 covariance ---
    variances = jnp.array([
        sr ** 2, st ** 2, sn ** 2,
        sr_dot ** 2, st_dot ** 2, sn_dot ** 2,
    ])
    return jnp.diag(variances)


def tle_bstar_sigma(
    satrec: SatRec,
    jd: jax.typing.ArrayLike,
    fr: jax.typing.ArrayLike,
    *,
    bstar_frac0: float = 0.30,
    bstar_floor: float = 1e-5,
    bstar_growth_per_day: float = 0.10,
) -> jax.Array:
    """Empirical 1-σ uncertainty on ``bstar`` based on TLE age.

    ``bstar`` is estimated by fitting TLE residuals to tracking data.  Its
    uncertainty has two components:

    1. **Epoch uncertainty** — from OD fit residuals and short-term
       atmospheric variability (~10 % diurnal cycle).  Modelled as a
       fraction of the fitted |bstar| value with a minimum floor for
       low-drag objects.

    2. **Age growth** — atmospheric density varies with solar activity
       (F10.7 index, 27-day solar-rotation cycle, solar-cycle envelope).
       A TLE's fitted bstar does not track these changes, so uncertainty
       grows at roughly 10 % of |bstar| per day.  By 7 days the bstar
       estimate is effectively unconstrained; by 3 days it has roughly
       doubled from its epoch value.

    **Model (units: km⁻¹):**

    .. code-block:: text

        σ_bstar₀  = max(bstar_frac0 · |bstar|, bstar_floor)
        σ_bstar(Δt) = σ_bstar₀ + bstar_growth_per_day · |bstar| · |Δt|

    Typical values at median LEO ``bstar`` (3.6×10⁻⁴ km⁻¹):

    * Fresh TLE (Δt = 0):  1.1×10⁻⁴ km⁻¹  (≈ 30 %)
    * 3 days old:           2.2×10⁻⁴ km⁻¹  (≈ 60 %)
    * 7 days old:           3.6×10⁻⁴ km⁻¹  (≈ 100 %)

    This σ is suitable as the standard deviation of a Gaussian or
    half-normal prior on bstar in Bayesian TLE fitting.  For a
    log-normal prior on |bstar|, use this value divided by |bstar| as
    the log-scale σ.

    Parameters
    ----------
    satrec : SatRec
        Initialized SatRec from :func:`~sgp4jax.tle_to_satrec`.
    jd : array-like
        Target Julian date, whole part (scalar).
    fr : array-like
        Target Julian date, fractional part (scalar).
    bstar_frac0 : float, optional
        Fractional uncertainty in bstar at TLE epoch.  Default 0.30 (30 %).
    bstar_floor : float, optional
        Minimum 1-σ at epoch, km⁻¹.  Prevents near-zero bstar from
        producing an unrealistically tight prior.  Default 1×10⁻⁵ km⁻¹.
    bstar_growth_per_day : float, optional
        Fractional growth in bstar uncertainty per day, relative to
        ``|bstar|``.  Default 0.10 (10 %/day), reflecting typical
        atmospheric-density variability from solar activity.

    Returns
    -------
    sigma_bstar : jax.Array
        Scalar 1-σ uncertainty on ``bstar``, km⁻¹.
    """
    dt_days = jnp.abs(
        jd - satrec.jdsatepoch + fr - satrec.jdsatepochF
    )
    abs_bstar = jnp.abs(satrec.bstar)

    # Epoch uncertainty: fraction of |bstar|, floored for low-drag sats
    sigma0 = jnp.maximum(bstar_frac0 * abs_bstar, jnp.asarray(bstar_floor))

    # Age-driven growth: atmospheric density variability
    return sigma0 + bstar_growth_per_day * abs_bstar * dt_days
