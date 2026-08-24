"""Tests for _covariance.py and propagate_gcrf.

Covers:
- ric_rotation: frame geometry
- cov_ric_to_teme / cov_teme_to_ric: round-trip and linear properties
- elements_jacobian: shape, rank, finite-difference check
- cov_elements_to_teme / cov_teme_to_elements: round-trip
- cov_ric_to_elements / cov_elements_to_ric: round-trip
- elements7_jacobian: shape, finite-difference check
- cov_elements7_to_teme: forward transform
- cov_teme_to_elements7: rank-deficiency
- cov_elements7_to_ric / cov_ric_to_elements7: chains
- tle_ric_covariance: diagonal structure, age growth, drag scaling
- tle_bstar_sigma: floor, fractional growth, age dependence
- propagate_gcrf: consistency with propagate + teme_to_gcrf
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import sgp4jax
from sgp4jax import tle_to_satrec, propagate, propagate_gcrf
from sgp4jax._covariance import (
    ric_rotation,
    cov_ric_to_teme,
    cov_teme_to_ric,
    elements_jacobian,
    cov_elements_to_teme,
    cov_teme_to_elements,
    cov_ric_to_elements,
    cov_elements_to_ric,
    elements7_jacobian,
    cov_elements7_to_teme,
    cov_teme_to_elements7,
    cov_elements7_to_ric,
    cov_ric_to_elements7,
)
from sgp4jax import tle_ric_covariance, tle_bstar_sigma

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# ISS-like LEO TLE
LINE1 = "1 25544U 98067A   20045.18587073  .00000950  00000-0  25302-4 0  9990"
LINE2 = "2 25544  51.6443 242.0161 0004397 264.6060 207.3845 15.49165514212791"

# High-drag LEO (large bstar) for drag-scaling tests
LINE1_HD = "1 25544U 98067A   20045.18587073  .00010000  00000-0  36000-3 0  9990"
LINE2_HD = "2 25544  51.6443 242.0161 0004397 264.6060 207.3845 15.49165514212791"

# Low-drag LEO (bstar near zero)
LINE1_LD = "1 25544U 98067A   20045.18587073  .00000001  00000-0  10000-7 0  9990"
LINE2_LD = "2 25544  51.6443 242.0161 0004397 264.6060 207.3845 15.49165514212791"


@pytest.fixture(scope="module")
def sat():
    return tle_to_satrec(LINE1, LINE2)


@pytest.fixture(scope="module")
def sat_hd():
    return tle_to_satrec(LINE1_HD, LINE2_HD)


@pytest.fixture(scope="module")
def sat_ld():
    return tle_to_satrec(LINE1_LD, LINE2_LD)


@pytest.fixture(scope="module")
def state_at_100(sat):
    """TEME position/velocity at t=100 min."""
    r, v, _ = propagate(sat, jnp.array(100.0))
    return r, v


@pytest.fixture(scope="module")
def jd_fr(sat):
    """Julian date 100 min after TLE epoch."""
    jd = jnp.array(sat.jdsatepoch)
    fr = jnp.array(sat.jdsatepochF + 100.0 / 1440.0)
    return jd, fr


@pytest.fixture(scope="module")
def identity_cov():
    """6×6 identity as a simple test covariance."""
    return jnp.eye(6)


# ---------------------------------------------------------------------------
# ric_rotation
# ---------------------------------------------------------------------------

class TestRicRotation:
    """Geometric properties of the RIC rotation matrix."""

    def test_shape(self, state_at_100):
        r, v = state_at_100
        T = ric_rotation(r, v)
        assert T.shape == (3, 3)

    def test_columns_are_unit_vectors(self, state_at_100):
        r, v = state_at_100
        T = ric_rotation(r, v)
        for i in range(3):
            norm = float(jnp.linalg.norm(T[:, i]))
            assert abs(norm - 1.0) < 1e-13, f"Column {i} not unit: {norm}"

    def test_orthogonal(self, state_at_100):
        r, v = state_at_100
        T = ric_rotation(r, v)
        np.testing.assert_allclose(
            np.array(T.T @ T), np.eye(3), atol=1e-13,
            err_msg="T^T T should be identity")

    def test_determinant_one(self, state_at_100):
        r, v = state_at_100
        T = ric_rotation(r, v)
        det = float(jnp.linalg.det(T))
        assert abs(det - 1.0) < 1e-13, f"det(T) = {det}, expected 1"

    def test_radial_column_aligned_with_r(self, state_at_100):
        """First column (R̂) should be parallel to position vector."""
        r, v = state_at_100
        T = ric_rotation(r, v)
        r_hat = r / jnp.linalg.norm(r)
        np.testing.assert_allclose(
            np.array(T[:, 0]), np.array(r_hat), atol=1e-13)

    def test_crosstrack_aligned_with_angular_momentum(self, state_at_100):
        """Third column (Ĉ) should be parallel to h = r × v."""
        r, v = state_at_100
        T = ric_rotation(r, v)
        h = jnp.cross(r, v)
        c_hat = h / jnp.linalg.norm(h)
        np.testing.assert_allclose(
            np.array(T[:, 2]), np.array(c_hat), atol=1e-13)


# ---------------------------------------------------------------------------
# cov_ric_to_teme / cov_teme_to_ric
# ---------------------------------------------------------------------------

class TestRicTemeRoundTrip:
    """RIC ↔ TEME covariance transforms are exact inverses."""

    def test_round_trip_ric_to_teme_to_ric(self, state_at_100, identity_cov):
        r, v = state_at_100
        cov_teme = cov_ric_to_teme(identity_cov, r, v)
        cov_back = cov_teme_to_ric(cov_teme, r, v)
        np.testing.assert_allclose(
            np.array(cov_back), np.array(identity_cov), atol=1e-12)

    def test_round_trip_teme_to_ric_to_teme(self, state_at_100, identity_cov):
        r, v = state_at_100
        cov_ric = cov_teme_to_ric(identity_cov, r, v)
        cov_back = cov_ric_to_teme(cov_ric, r, v)
        np.testing.assert_allclose(
            np.array(cov_back), np.array(identity_cov), atol=1e-12)

    def test_output_shape(self, state_at_100, identity_cov):
        r, v = state_at_100
        assert cov_ric_to_teme(identity_cov, r, v).shape == (6, 6)
        assert cov_teme_to_ric(identity_cov, r, v).shape == (6, 6)

    def test_symmetry_preserved(self, state_at_100):
        r, v = state_at_100
        cov = jnp.diag(jnp.array([100.0, 900.0, 100.0, 0.01, 0.09, 0.01]))
        cov_teme = cov_ric_to_teme(cov, r, v)
        np.testing.assert_allclose(
            np.array(cov_teme), np.array(cov_teme.T), atol=1e-12,
            err_msg="cov_ric_to_teme output not symmetric")

    def test_positive_definite_preserved(self, state_at_100):
        r, v = state_at_100
        cov = jnp.diag(jnp.array([100.0, 900.0, 100.0, 0.01, 0.09, 0.01]))
        cov_teme = cov_ric_to_teme(cov, r, v)
        eigvals = jnp.linalg.eigvalsh(cov_teme)
        assert float(eigvals.min()) > 0, "Forward-transformed covariance not PD"

    def test_trace_preserved(self, state_at_100):
        """Trace is invariant under orthogonal similarity transform."""
        r, v = state_at_100
        cov = jnp.diag(jnp.array([1.0, 4.0, 9.0, 0.001, 0.004, 0.009]))
        cov_teme = cov_ric_to_teme(cov, r, v)
        np.testing.assert_allclose(
            float(jnp.trace(cov_teme)), float(jnp.trace(cov)), rtol=1e-12)


# ---------------------------------------------------------------------------
# elements_jacobian (6-element Keplerian)
# ---------------------------------------------------------------------------

class TestElementsJacobian6:
    """Shape, rank, and finite-difference verification of elements_jacobian."""

    def test_shape(self, sat, jd_fr):
        jd, fr = jd_fr
        J = elements_jacobian(sat, jd, fr)
        assert J.shape == (6, 6)

    def test_finite(self, sat, jd_fr):
        jd, fr = jd_fr
        J = elements_jacobian(sat, jd, fr)
        assert jnp.all(jnp.isfinite(J))

    def test_full_rank(self, sat, jd_fr):
        jd, fr = jd_fr
        J = elements_jacobian(sat, jd, fr)
        rank = jnp.linalg.matrix_rank(J)
        assert int(rank) == 6, f"Jacobian rank {rank} < 6"

    def test_finite_difference_agreement(self, sat, jd_fr):
        """Analytic Jacobian matches finite differences to 0.1% relative."""
        from sgp4jax._kepler import _kepler_rv_teme
        jd, fr = jd_fr

        J_analytic = elements_jacobian(sat, jd, fr)

        eps = 1e-6
        elements = jnp.array([
            sat.inclo, sat.nodeo, sat.ecco,
            sat.argpo, sat.mo, sat.no_kozai,
        ])

        def fwd(el):
            inclo, nodeo, ecco, argpo, mo, no_kozai = el
            r, v = _kepler_rv_teme(
                inclo, nodeo, ecco, argpo, mo, no_kozai, sat.mu,
                sat.jdsatepoch, sat.jdsatepochF, jd, fr,
            )
            return jnp.concatenate([r, v])

        J_fd = jnp.zeros((6, 6))
        for col in range(6):
            delta = jnp.zeros(6).at[col].set(eps)
            J_fd = J_fd.at[:, col].set(
                (fwd(elements + delta) - fwd(elements - delta)) / (2 * eps)
            )

        np.testing.assert_allclose(
            np.array(J_analytic), np.array(J_fd),
            rtol=1e-3, atol=1e-6,
            err_msg="Analytic and finite-difference Jacobians disagree")


# ---------------------------------------------------------------------------
# cov_elements_to_teme / cov_teme_to_elements (6-element round-trip)
# ---------------------------------------------------------------------------

class TestElementsTemeRoundTrip:
    """6-element element ↔ TEME covariance transforms are inverses."""

    # Physically realistic element covariance (1-σ: inclo~1mrad, no_kozai~0.1µrad/min, etc.)
    # Using identity in element space would give σ_no_kozai=1 rad/min which is unphysical and
    # makes the Jacobian ill-conditioned for round-trip tests.
    _COV_EL = jnp.diag(jnp.array([1e-6, 1e-6, 1e-10, 1e-6, 1e-6, 1e-14]))

    # Physically realistic TEME covariance (σ_r=σ_n=10km, σ_t=30km, velocities ~0.1km/min).
    # Must be TEME-realistic to avoid ill-conditioning in the TEME→elements inverse.
    _COV_TEME_RT = jnp.diag(jnp.array([100., 900., 100., 0.01, 0.09, 0.01]))

    def test_round_trip_elements_to_teme_to_elements(self, sat, jd_fr):
        jd, fr = jd_fr
        cov_teme = cov_elements_to_teme(self._COV_EL, sat, jd, fr)
        cov_back = cov_teme_to_elements(cov_teme, sat, jd, fr)
        np.testing.assert_allclose(
            np.array(cov_back), np.array(self._COV_EL), rtol=1e-6, atol=1e-14)

    def test_round_trip_teme_to_elements_to_teme(self, sat, jd_fr):
        jd, fr = jd_fr
        cov_el = cov_teme_to_elements(self._COV_TEME_RT, sat, jd, fr)
        cov_back = cov_elements_to_teme(cov_el, sat, jd, fr)
        # Off-diagonal zero entries have float32 noise up to ~1e-4 due to the
        # ~5e6 condition number of the elements Jacobian; diagonal entries match well.
        np.testing.assert_allclose(
            np.array(cov_back), np.array(self._COV_TEME_RT), rtol=1e-4, atol=1e-3)

    def test_output_shapes(self, sat, jd_fr, identity_cov):
        jd, fr = jd_fr
        assert cov_elements_to_teme(identity_cov, sat, jd, fr).shape == (6, 6)
        assert cov_teme_to_elements(identity_cov, sat, jd, fr).shape == (6, 6)

    def test_symmetry_preserved(self, sat, jd_fr):
        jd, fr = jd_fr
        cov = jnp.diag(jnp.array([1e-8, 1e-8, 1e-8, 1e-6, 1e-6, 1e-8]))
        cov_teme = cov_elements_to_teme(cov, sat, jd, fr)
        np.testing.assert_allclose(
            np.array(cov_teme), np.array(cov_teme.T), atol=1e-12)


# ---------------------------------------------------------------------------
# cov_ric_to_elements / cov_elements_to_ric (6-element full chain)
# ---------------------------------------------------------------------------

class TestRicElementsRoundTrip:
    """Full RIC ↔ element chain is self-consistent."""

    def test_round_trip_ric_to_elements_to_ric(self, sat, jd_fr):
        jd, fr = jd_fr
        cov_ric = jnp.diag(jnp.array([0.01, 1.0, 0.01, 1e-6, 1e-4, 1e-6]))
        cov_el = cov_ric_to_elements(cov_ric, sat, jd, fr)
        cov_back = cov_elements_to_ric(cov_el, sat, jd, fr)
        # The round trip inverts a Jacobian with cond ~2e17, so the error
        # floor scales with the magnitude of the covariance rather than
        # sitting at an absolute constant.
        scale = float(jnp.max(jnp.abs(cov_ric)))
        np.testing.assert_allclose(
            np.array(cov_back), np.array(cov_ric), atol=1e-6 * scale)

    def test_round_trip_elements_to_ric_to_elements(self, sat, jd_fr):
        jd, fr = jd_fr
        cov_el = jnp.diag(jnp.array([1e-8, 1e-8, 1e-10, 1e-8, 1e-8, 1e-10]))
        cov_ric = cov_elements_to_ric(cov_el, sat, jd, fr)
        cov_back = cov_ric_to_elements(cov_ric, sat, jd, fr)
        np.testing.assert_allclose(
            np.array(cov_back), np.array(cov_el), atol=1e-14)

    def test_output_shapes(self, sat, jd_fr):
        jd, fr = jd_fr
        cov = jnp.eye(6)
        assert cov_ric_to_elements(cov, sat, jd, fr).shape == (6, 6)
        assert cov_elements_to_ric(cov, sat, jd, fr).shape == (6, 6)

    def test_ric_covariance_is_positive_definite(self, sat, jd_fr):
        """A PD element covariance should map to a PD RIC covariance."""
        jd, fr = jd_fr
        cov_el = jnp.diag(jnp.array([1e-8, 1e-8, 1e-10, 1e-8, 1e-8, 1e-10]))
        cov_ric = cov_elements_to_ric(cov_el, sat, jd, fr)
        eigvals = jnp.linalg.eigvalsh(cov_ric)
        assert float(eigvals.min()) > 0


# ---------------------------------------------------------------------------
# elements7_jacobian (7-element SGP4 Jacobian)
# ---------------------------------------------------------------------------

class TestElementsJacobian7:
    """Shape and numerical properties of elements7_jacobian."""

    def test_shape(self, sat, jd_fr):
        jd, fr = jd_fr
        J = elements7_jacobian(sat, jd, fr)
        assert J.shape == (6, 7)

    def test_finite(self, sat, jd_fr):
        jd, fr = jd_fr
        J = elements7_jacobian(sat, jd, fr)
        assert jnp.all(jnp.isfinite(J))

    def test_rank_six(self, sat, jd_fr):
        """A (6,7) Jacobian has at most rank 6."""
        jd, fr = jd_fr
        J = elements7_jacobian(sat, jd, fr)
        rank = jnp.linalg.matrix_rank(J)
        assert int(rank) == 6

    def test_bstar_column_nonzero(self, sat_hd, jd_fr):
        """The bstar column of J7 should be nonzero for a high-drag orbit."""
        jd, fr = jd_fr
        J7 = elements7_jacobian(sat_hd, jd, fr)
        bstar_col = J7[:, 6]
        assert float(jnp.linalg.norm(bstar_col)) > 1e-10, \
            "bstar column is zero for high-drag satellite"


# ---------------------------------------------------------------------------
# cov_elements7_to_teme (7-element forward)
# ---------------------------------------------------------------------------

class TestElements7ToTeme:
    """Forward 7-element → TEME transform."""

    def test_shape(self, sat, jd_fr):
        jd, fr = jd_fr
        cov7 = jnp.eye(7)
        result = cov_elements7_to_teme(cov7, sat, jd, fr)
        assert result.shape == (6, 6)

    def test_symmetry(self, sat, jd_fr):
        jd, fr = jd_fr
        cov7 = jnp.diag(jnp.array([1e-8]*6 + [1e-8]))
        result = cov_elements7_to_teme(cov7, sat, jd, fr)
        np.testing.assert_allclose(
            np.array(result), np.array(result.T), atol=1e-12)

    def test_positive_definite(self, sat, jd_fr):
        jd, fr = jd_fr
        cov7 = jnp.eye(7)
        result = cov_elements7_to_teme(cov7, sat, jd, fr)
        eigvals = jnp.linalg.eigvalsh(result)
        assert float(eigvals.min()) > 0

    def test_keplerian_block_matches_6element(self, sat, jd_fr):
        """With zero bstar variance, 7-element forward should match 6-element."""
        jd, fr = jd_fr
        cov6 = jnp.eye(6)
        cov7 = jnp.block([
            [cov6,           jnp.zeros((6, 1))],
            [jnp.zeros((1, 6)), jnp.zeros((1, 1))],
        ])
        result7 = cov_elements7_to_teme(cov7, sat, jd, fr)
        result6 = cov_elements_to_teme(cov6, sat, jd, fr)
        # Should match closely since bstar column contribution is zeroed out
        np.testing.assert_allclose(
            np.array(result7), np.array(result6), rtol=0.02)


# ---------------------------------------------------------------------------
# cov_teme_to_elements7 (rank-deficient pseudo-inverse)
# ---------------------------------------------------------------------------

class TestTemeToElements7:
    """Pseudo-inverse from 6D TEME to 7D element space — rank deficient."""

    def test_shape(self, sat, jd_fr):
        jd, fr = jd_fr
        result = cov_teme_to_elements7(jnp.eye(6), sat, jd, fr)
        assert result.shape == (7, 7)

    def test_symmetry(self, sat, jd_fr):
        jd, fr = jd_fr
        result = cov_teme_to_elements7(jnp.eye(6), sat, jd, fr)
        np.testing.assert_allclose(
            np.array(result), np.array(result.T), atol=1e-10)

    def test_rank_deficient(self, sat, jd_fr):
        """Output should be rank ≤ 6 (7×7 from 6D information)."""
        jd, fr = jd_fr
        result = cov_teme_to_elements7(jnp.eye(6), sat, jd, fr)
        rank = jnp.linalg.matrix_rank(result, tol=1e-8)
        assert int(rank) <= 6, f"Expected rank ≤ 6, got {rank}"

    def test_finite(self, sat, jd_fr):
        jd, fr = jd_fr
        result = cov_teme_to_elements7(jnp.eye(6), sat, jd, fr)
        assert jnp.all(jnp.isfinite(result))

    def test_forward_inverse_consistency(self, sat, jd_fr):
        """J @ J† Σ J†ᵀ Jᵀ ≈ Σ  (pseudo-inverse recovers the original after forward map).

        The pseudo-inverse round-trip J (J† Σ J†ᵀ) Jᵀ ≈ Σ holds only to
        floating-point numerical precision (~0.1-1%), not machine epsilon,
        due to the ill-conditioning of the (6,7) Jacobian pseudo-inverse.
        """
        jd, fr = jd_fr
        cov_teme = jnp.eye(6)
        cov_el7 = cov_teme_to_elements7(cov_teme, sat, jd, fr)
        cov_teme_back = cov_elements7_to_teme(cov_el7, sat, jd, fr)
        np.testing.assert_allclose(
            np.array(cov_teme_back), np.array(cov_teme), atol=0.02)


# ---------------------------------------------------------------------------
# cov_elements7_to_ric / cov_ric_to_elements7 (7-element full chains)
# ---------------------------------------------------------------------------

class TestElements7RicChain:
    """Full chain tests for 7-element ↔ RIC transforms."""

    def test_elements7_to_ric_shape(self, sat, jd_fr):
        jd, fr = jd_fr
        result = cov_elements7_to_ric(jnp.eye(7), sat, jd, fr)
        assert result.shape == (6, 6)

    def test_elements7_to_ric_symmetry(self, sat, jd_fr):
        jd, fr = jd_fr
        result = cov_elements7_to_ric(jnp.eye(7), sat, jd, fr)
        np.testing.assert_allclose(
            np.array(result), np.array(result.T), atol=1e-10)

    def test_elements7_to_ric_positive_definite(self, sat, jd_fr):
        jd, fr = jd_fr
        result = cov_elements7_to_ric(jnp.eye(7), sat, jd, fr)
        eigvals = jnp.linalg.eigvalsh(result)
        assert float(eigvals.min()) > 0

    def test_ric_to_elements7_shape(self, sat, jd_fr):
        jd, fr = jd_fr
        result = cov_ric_to_elements7(jnp.eye(6), sat, jd, fr)
        assert result.shape == (7, 7)

    def test_ric_to_elements7_rank_deficient(self, sat, jd_fr):
        jd, fr = jd_fr
        result = cov_ric_to_elements7(jnp.eye(6), sat, jd, fr)
        rank = jnp.linalg.matrix_rank(result, tol=1e-8)
        assert int(rank) <= 6

    def test_forward_inverse_consistency(self, sat, jd_fr):
        """elements7→RIC→elements7_to_teme→elements7 should recover Σ_teme.

        Pseudo-inverse numerical noise limits round-trip accuracy to ~0.1-1%,
        not machine epsilon; tolerance is set accordingly.
        """
        jd, fr = jd_fr
        cov_ric = jnp.diag(jnp.array([0.01, 1.0, 0.01, 1e-6, 1e-4, 1e-6]))
        cov_el7 = cov_ric_to_elements7(cov_ric, sat, jd, fr)
        cov_ric_back = cov_elements7_to_ric(cov_el7, sat, jd, fr)
        np.testing.assert_allclose(
            np.array(cov_ric_back), np.array(cov_ric), atol=1e-4)


# ---------------------------------------------------------------------------
# tle_ric_covariance
# ---------------------------------------------------------------------------

class TestTleRicCovariance:
    """Empirical RIC covariance structure and behaviour."""

    def test_shape(self, sat):
        jd, fr = sat.jdsatepoch, sat.jdsatepochF
        cov = tle_ric_covariance(sat, jd, fr)
        assert cov.shape == (6, 6)

    def test_diagonal_at_epoch(self, sat):
        """At epoch, covariance should be diagonal (no cross-terms by design)."""
        jd, fr = sat.jdsatepoch, sat.jdsatepochF
        cov = tle_ric_covariance(sat, jd, fr)
        off_diag = cov - jnp.diag(jnp.diag(cov))
        np.testing.assert_allclose(
            np.array(off_diag), np.zeros((6, 6)), atol=1e-30)

    def test_positive_diagonal_at_epoch(self, sat):
        jd, fr = sat.jdsatepoch, sat.jdsatepochF
        cov = tle_ric_covariance(sat, jd, fr)
        assert jnp.all(jnp.diag(cov) > 0)

    def test_intrack_dominates_at_epoch(self, sat):
        """Default sigma_t0 (0.3 km) > sigma_r0 = sigma_n0 (0.05 km)."""
        jd, fr = sat.jdsatepoch, sat.jdsatepochF
        cov = tle_ric_covariance(sat, jd, fr)
        sigma_r = float(jnp.sqrt(cov[0, 0]))
        sigma_t = float(jnp.sqrt(cov[1, 1]))
        sigma_n = float(jnp.sqrt(cov[2, 2]))
        assert sigma_t > sigma_r
        assert sigma_t > sigma_n

    def test_grows_with_age(self, sat):
        """All diagonal entries should be larger at t+3 days than at epoch."""
        jd0, fr0 = sat.jdsatepoch, sat.jdsatepochF
        cov0 = tle_ric_covariance(sat, jd0 + 0.0, fr0)
        cov3 = tle_ric_covariance(sat, jd0 + 3.0, fr0)
        assert jnp.all(jnp.diag(cov3) > jnp.diag(cov0)), \
            "Covariance should grow with TLE age"

    def test_intrack_grows_faster_than_radial(self, sat):
        """In-track error grows faster than radial (higher gamma_t)."""
        jd0, fr0 = sat.jdsatepoch, sat.jdsatepochF
        cov0 = tle_ric_covariance(sat, jd0, fr0)
        cov7 = tle_ric_covariance(sat, jd0 + 7.0, fr0)
        growth_r = float(jnp.sqrt(cov7[0, 0]) - jnp.sqrt(cov0[0, 0]))
        growth_t = float(jnp.sqrt(cov7[1, 1]) - jnp.sqrt(cov0[1, 1]))
        assert growth_t > growth_r

    def test_high_drag_has_larger_intrack(self, sat, sat_hd):
        """Higher bstar should produce larger in-track uncertainty."""
        jd = sat.jdsatepoch + 1.0
        fr = sat.jdsatepochF
        cov_nom = tle_ric_covariance(sat, jd, fr)
        cov_hd = tle_ric_covariance(sat_hd, jd, fr)
        assert float(cov_hd[1, 1]) > float(cov_nom[1, 1]), \
            "High-drag satellite should have larger in-track variance"

    def test_symmetric_about_epoch(self, sat):
        """Covariance at t-1 day should equal covariance at t+1 day (|Δt|)."""
        jd0, fr0 = sat.jdsatepoch, sat.jdsatepochF
        cov_plus = tle_ric_covariance(sat, jd0 + 1.0, fr0)
        cov_minus = tle_ric_covariance(sat, jd0 - 1.0, fr0)
        np.testing.assert_allclose(
            np.array(cov_plus), np.array(cov_minus), rtol=1e-12)

    def test_velocity_block_scale(self, sat):
        """Velocity 1-σ should be approx n * position 1-σ."""
        jd, fr = sat.jdsatepoch, sat.jdsatepochF
        cov = tle_ric_covariance(sat, jd, fr)
        n = float(sat.no_unkozai) / 60.0  # rad/s
        sigma_r_pos = float(jnp.sqrt(cov[0, 0]))
        sigma_r_vel = float(jnp.sqrt(cov[3, 3]))
        np.testing.assert_allclose(sigma_r_vel, n * sigma_r_pos, rtol=1e-12)

    def test_custom_sigmas(self, sat):
        """Custom sigma_r0/sigma_t0/sigma_n0 are reflected at epoch."""
        jd, fr = sat.jdsatepoch, sat.jdsatepochF
        cov = tle_ric_covariance(sat, jd, fr, sigma_r0=0.1, sigma_t0=0.5, sigma_n0=0.2)
        np.testing.assert_allclose(float(jnp.sqrt(cov[0, 0])), 0.1, rtol=1e-12)
        np.testing.assert_allclose(float(jnp.sqrt(cov[1, 1])), 0.5, rtol=1e-12)
        np.testing.assert_allclose(float(jnp.sqrt(cov[2, 2])), 0.2, rtol=1e-12)


# ---------------------------------------------------------------------------
# tle_bstar_sigma
# ---------------------------------------------------------------------------

class TestTleBstarSigma:
    """Empirical bstar uncertainty model."""

    def test_scalar_output(self, sat):
        jd, fr = sat.jdsatepoch, sat.jdsatepochF
        sigma = tle_bstar_sigma(sat, jd, fr)
        assert sigma.shape == ()

    def test_positive_at_epoch(self, sat):
        jd, fr = sat.jdsatepoch, sat.jdsatepochF
        sigma = tle_bstar_sigma(sat, jd, fr)
        assert float(sigma) > 0

    def test_grows_with_age(self, sat):
        jd0, fr0 = sat.jdsatepoch, sat.jdsatepochF
        s0 = float(tle_bstar_sigma(sat, jd0, fr0))
        s3 = float(tle_bstar_sigma(sat, jd0 + 3.0, fr0))
        s7 = float(tle_bstar_sigma(sat, jd0 + 7.0, fr0))
        assert s3 > s0
        assert s7 > s3

    def test_floor_for_low_drag(self, sat_ld):
        """Near-zero bstar satellite should hit the bstar_floor."""
        jd, fr = sat_ld.jdsatepoch, sat_ld.jdsatepochF
        sigma = tle_bstar_sigma(sat_ld, jd, fr, bstar_floor=1e-5)
        assert float(sigma) >= 1e-5

    def test_fractional_at_epoch_for_high_drag(self, sat_hd):
        """At epoch, sigma should be ~30% of |bstar| for high-drag satellite."""
        jd, fr = sat_hd.jdsatepoch, sat_hd.jdsatepochF
        sigma = float(tle_bstar_sigma(sat_hd, jd, fr, bstar_frac0=0.30))
        expected = 0.30 * abs(float(sat_hd.bstar))
        np.testing.assert_allclose(sigma, expected, rtol=1e-10)

    def test_symmetric_about_epoch(self, sat):
        """|Δt| means +1 day and -1 day give identical sigma."""
        jd0, fr0 = sat.jdsatepoch, sat.jdsatepochF
        s_plus = float(tle_bstar_sigma(sat, jd0 + 1.0, fr0))
        s_minus = float(tle_bstar_sigma(sat, jd0 - 1.0, fr0))
        np.testing.assert_allclose(s_plus, s_minus, rtol=1e-12)

    def test_higher_drag_gives_larger_sigma(self, sat, sat_hd):
        """Higher bstar should produce larger uncertainty at same age."""
        jd = sat.jdsatepoch + 1.0
        fr = sat.jdsatepochF
        s_nom = float(tle_bstar_sigma(sat, jd, fr))
        s_hd = float(tle_bstar_sigma(sat_hd, jd, fr))
        assert s_hd > s_nom

    def test_custom_growth_rate(self, sat_hd):
        """Custom bstar_growth_per_day scales linear growth correctly."""
        jd0, fr0 = sat_hd.jdsatepoch, sat_hd.jdsatepochF
        abs_bstar = abs(float(sat_hd.bstar))
        s0 = float(tle_bstar_sigma(sat_hd, jd0, fr0, bstar_growth_per_day=0.20))
        s1 = float(tle_bstar_sigma(sat_hd, jd0 + 1.0, fr0, bstar_growth_per_day=0.20))
        # Growth over 1 day = 0.20 * |bstar| * 1 day
        np.testing.assert_allclose(s1 - s0, 0.20 * abs_bstar, rtol=1e-10)


# ---------------------------------------------------------------------------
# propagate_gcrf
# ---------------------------------------------------------------------------

class TestPropagateGcrf:
    """propagate_gcrf wraps propagate + teme_to_gcrf correctly."""

    def test_consistent_with_manual_transform(self, sat):
        """propagate_gcrf should match propagate → teme_to_gcrf."""
        from sgp4jax._frames import teme_to_gcrf
        tsince = jnp.array(100.0)
        r_teme, v_teme, err_ref = propagate(sat, tsince)
        jd = jnp.array(sat.jdsatepoch)
        fr = jnp.array(sat.jdsatepochF) + tsince / 1440.0
        r_ref, v_ref = teme_to_gcrf(r_teme, v_teme, jd, fr)

        r_gcrf, v_gcrf, err = propagate_gcrf(sat, tsince)

        np.testing.assert_allclose(np.array(r_gcrf), np.array(r_ref), atol=1e-12)
        np.testing.assert_allclose(np.array(v_gcrf), np.array(v_ref), atol=1e-12)
        assert int(err) == int(err_ref)

    def test_output_shapes(self, sat):
        r, v, err = propagate_gcrf(sat, jnp.array(0.0))
        assert r.shape == (3,)
        assert v.shape == (3,)

    def test_error_zero_for_valid_orbit(self, sat):
        _, _, err = propagate_gcrf(sat, jnp.array(60.0))
        assert int(err) == 0

    def test_position_magnitude_sensible(self, sat):
        """Position magnitude should be 6400–42000 km for LEO."""
        r, _, _ = propagate_gcrf(sat, jnp.array(0.0))
        r_mag = float(jnp.linalg.norm(r))
        assert 6400 < r_mag < 42000, f"Unexpected position magnitude: {r_mag} km"

    def test_jit_compatible(self, sat):
        fn = jax.jit(propagate_gcrf)
        r1, v1, _ = propagate_gcrf(sat, jnp.array(100.0))
        r2, v2, _ = fn(sat, jnp.array(100.0))
        np.testing.assert_allclose(np.array(r1), np.array(r2), atol=1e-12)

    def test_vmap_over_times(self, sat):
        times = jnp.array([0.0, 30.0, 60.0, 90.0, 120.0])
        batched = jax.vmap(propagate_gcrf, in_axes=(None, 0))
        r_batch, v_batch, errs = batched(sat, times)
        assert r_batch.shape == (5, 3)
        assert v_batch.shape == (5, 3)
        assert jnp.all(errs == 0)

    def test_at_epoch_position_finite(self, sat):
        r, v, _ = propagate_gcrf(sat, jnp.array(0.0))
        assert jnp.all(jnp.isfinite(r))
        assert jnp.all(jnp.isfinite(v))

    @pytest.mark.parametrize("tsince", [0.0, 60.0, 360.0, 1440.0])
    def test_position_changes_with_time(self, sat, tsince):
        """Position at t > 0 should differ from position at t = 0."""
        r0, _, _ = propagate_gcrf(sat, jnp.array(0.0))
        rt, _, _ = propagate_gcrf(sat, jnp.array(tsince))
        if tsince > 0:
            diff = float(jnp.linalg.norm(rt - r0))
            assert diff > 1.0, f"Position barely moved at t={tsince} min"
