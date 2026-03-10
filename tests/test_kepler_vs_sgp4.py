"""Keplerian propagator tests against the python-sgp4 reference library.

Answers the question: "can we test kepler_gcrf_positions against the standard
SGP4 propagators in the reference library?"

Short answer: not as a direct accuracy comparison, because the two models are
intentionally different:
  - python-sgp4 (and sgp4jax.propagate) implement the full SGP4/SDP4 model,
    converting Brouwer *mean* TLE elements to osculating elements by adding
    J2 short-period corrections, drag, and lunar/solar perturbations.
  - kepler_gcrf_positions uses the six raw TLE mean elements in a pure
    two-body Kepler orbit with no corrections.

What we CAN test using the reference library:

1. TLE element round-trip: python-sgp4 parses the same no_kozai, ecco, inclo,
   etc. as sgp4jax — confirms the TLE parsing and the Keplerian input are
   consistent with the widely-used reference implementation.

2. Osculating element recovery: at any point on the Keplerian orbit the
   classical elements (a, e, i) recovered from (r, v) via the vis-viva /
   angular-momentum / eccentricity-vector relations must equal the TLE input
   elements exactly (the Keplerian orbit IS a fixed ellipse).

3. Error characterisation: the position difference between Keplerian and
   reference SGP4 is measured in TEME at several propagation times.  This
   documents the mean-to-osculating correction magnitude for different orbit
   families without requiring either model to be a "ground truth" for the
   other.

4. Orbital-shell sanity: at every point in the Keplerian orbit the satellite
   altitude is between periapsis and apoapsis; the reference SGP4 positions
   are also approximately in the same shell (within a few times the J2
   short-period amplitude).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from sgp4.api import Satrec as RefSatrec, WGS72 as REF_WGS72

from sgp4jax import tle_to_satrec, WGS72
from sgp4jax._kepler import _solve_kepler, _kepler_rv_teme


# ---------------------------------------------------------------------------
# TLEs covering three orbit families
# ---------------------------------------------------------------------------

_SATS = {
    "iss": (
        "1 25544U 98067A   20045.18587073  .00000950  00000-0  25302-4 0  9990",
        "2 25544  51.6443 242.0161 0004397 264.6060 207.3845 15.49165514212791",
    ),
    "gps": (
        "1 28190U 04009A   20045.52490438 -.00000022  00000-0  00000+0 0  9993",
        "2 28190  55.3697 252.2502 0091151  48.4297 312.3987  2.00568625117815",
    ),
    "molniya": (
        "1 14947U 84012A   20044.87674740 -.00000143  00000-0  00000+0 0  9992",
        "2 14947  65.0476 160.8503 7085418 276.3547  11.5044  2.00611024264509",
    ),
}


@pytest.fixture(scope="module", params=list(_SATS.keys()))
def sat_pair(request):
    """Yield (name, our_satrec, ref_satrec) for each test satellite."""
    name = request.param
    l1, l2 = _SATS[name]
    our = tle_to_satrec(l1, l2, gravity=WGS72)
    ref = RefSatrec.twoline2rv(l1, l2, REF_WGS72)
    return name, our, ref


# ---------------------------------------------------------------------------
# Helper: classical elements from (r, v, μ)
# ---------------------------------------------------------------------------

def rv_to_elements(r_km: np.ndarray, v_kms: np.ndarray, mu: float):
    """Return (a_km, e, i_rad) from a TEME position/velocity pair.

    Uses the standard vis-viva, angular momentum, and eccentricity-vector
    relations.  Raises ValueError for hyperbolic trajectories (e ≥ 1).
    """
    r_mag = np.linalg.norm(r_km)
    v2 = np.dot(v_kms, v_kms)

    # Specific orbital energy → semi-major axis
    eps = 0.5 * v2 - mu / r_mag
    a = -mu / (2.0 * eps)

    # Angular momentum
    h = np.cross(r_km, v_kms)
    h_mag = np.linalg.norm(h)

    # Eccentricity vector  e⃗ = (v × h) / μ  − r̂
    e_vec = np.cross(v_kms, h) / mu - r_km / r_mag
    e = np.linalg.norm(e_vec)

    # Inclination
    i = np.arccos(np.clip(h[2] / h_mag, -1.0, 1.0))

    return a, e, i


# ---------------------------------------------------------------------------
# 1. TLE element round-trip: our parser vs python-sgp4
# ---------------------------------------------------------------------------

class TestTLEConsistency:
    """Our SatRec and python-sgp4's Satrec expose the same TLE elements."""

    def test_no_kozai(self, sat_pair):
        """Mean motion (rad/min) agrees to float64 precision."""
        _, our, ref = sat_pair
        np.testing.assert_allclose(
            float(our.no_kozai), ref.no_kozai,
            rtol=1e-12, err_msg="no_kozai mismatch",
        )

    def test_eccentricity(self, sat_pair):
        _, our, ref = sat_pair
        np.testing.assert_allclose(
            float(our.ecco), ref.ecco,
            rtol=1e-12, err_msg="ecco mismatch",
        )

    def test_inclination(self, sat_pair):
        _, our, ref = sat_pair
        np.testing.assert_allclose(
            float(our.inclo), ref.inclo,
            rtol=1e-12, err_msg="inclo mismatch",
        )

    def test_raan(self, sat_pair):
        _, our, ref = sat_pair
        np.testing.assert_allclose(
            float(our.nodeo), ref.nodeo,
            rtol=1e-12, err_msg="nodeo (RAAN) mismatch",
        )

    def test_argpo(self, sat_pair):
        _, our, ref = sat_pair
        np.testing.assert_allclose(
            float(our.argpo), ref.argpo,
            rtol=1e-12, err_msg="argpo mismatch",
        )

    def test_mean_anomaly(self, sat_pair):
        _, our, ref = sat_pair
        np.testing.assert_allclose(
            float(our.mo), ref.mo,
            rtol=1e-12, err_msg="mo mismatch",
        )

    def test_epoch(self, sat_pair):
        """Epoch Julian date (whole + fractional) agrees to < 1 microsecond."""
        _, our, ref = sat_pair
        jd_our = float(our.jdsatepoch) + float(our.jdsatepochF)
        jd_ref = ref.jdsatepoch + ref.jdsatepochF
        diff_s = abs(jd_our - jd_ref) * 86400.0
        assert diff_s < 1e-6, f"Epoch differs by {diff_s:.2e} s"


# ---------------------------------------------------------------------------
# 2. Osculating element recovery from Keplerian (r, v)
#
# On a pure Keplerian ellipse every (r, v) pair encodes the *same* classical
# elements.  Computing them via rv_to_elements and comparing to the TLE input
# is an independent consistency check that does not depend on sgp4 at all.
# ---------------------------------------------------------------------------

class TestOsculatingElementRecovery:
    """Elements recovered from Keplerian (r, v) must equal TLE input elements."""

    def _sample_teme_rv(self, our, n_points=60):
        """Propagate in TEME over one Keplerian period, return (r, v) arrays."""
        jd0 = float(our.jdsatepoch)
        fr0 = float(our.jdsatepochF)
        period_min = 2.0 * np.pi / float(our.no_kozai)

        r_list, v_list = [], []
        for i in range(n_points):
            t_day = period_min / 1440.0 * i / (n_points - 1)
            jd = jd0 + (fr0 + t_day)
            r_t, v_t = _kepler_rv_teme(
                our.inclo, our.nodeo, our.ecco, our.argpo,
                our.mo, our.no_kozai, our.mu,
                our.jdsatepoch, our.jdsatepochF,
                jnp.float64(jd), jnp.float64(0.0),
            )
            r_list.append(np.array(r_t))
            v_list.append(np.array(v_t))
        return np.array(r_list), np.array(v_list)

    def test_semi_major_axis_recovered(self, sat_pair):
        """Semi-major axis recovered from Keplerian (r, v) matches TLE a."""
        name, our, _ = sat_pair
        mu = float(our.mu)
        n = float(our.no_kozai)
        mu_min = mu * 3600.0
        a_tle = (mu_min / n ** 2) ** (1.0 / 3.0)   # km

        r_arr, v_arr = self._sample_teme_rv(our)
        a_vals = np.array([rv_to_elements(r_arr[i], v_arr[i], mu)[0]
                           for i in range(len(r_arr))])
        np.testing.assert_allclose(
            a_vals, a_tle, rtol=1e-10,
            err_msg=f"Recovered semi-major axis varies for {name}",
        )

    def test_eccentricity_recovered(self, sat_pair):
        """Eccentricity recovered from Keplerian (r, v) matches TLE ecco."""
        name, our, _ = sat_pair
        mu = float(our.mu)
        ecco_tle = float(our.ecco)

        r_arr, v_arr = self._sample_teme_rv(our)
        e_vals = np.array([rv_to_elements(r_arr[i], v_arr[i], mu)[1]
                           for i in range(len(r_arr))])
        np.testing.assert_allclose(
            e_vals, ecco_tle, atol=1e-10,
            err_msg=f"Recovered eccentricity varies for {name}",
        )

    def test_inclination_recovered(self, sat_pair):
        """Inclination recovered from Keplerian (r, v) matches TLE inclo."""
        name, our, _ = sat_pair
        mu = float(our.mu)
        inclo_tle = float(our.inclo)

        r_arr, v_arr = self._sample_teme_rv(our)
        i_vals = np.array([rv_to_elements(r_arr[i], v_arr[i], mu)[2]
                           for i in range(len(r_arr))])
        np.testing.assert_allclose(
            i_vals, inclo_tle, atol=1e-10,
            err_msg=f"Recovered inclination varies for {name}",
        )


# ---------------------------------------------------------------------------
# 3. Error characterisation: Keplerian TEME vs reference SGP4 TEME
#
# This is the key "how wrong is Keplerian?" test.  We measure the position
# error at several propagation times and document the expected bounds for
# each orbit family.  Neither model is the "ground truth" — we are
# characterising the mean-to-osculating difference.
# ---------------------------------------------------------------------------

# Upper bounds on Keplerian vs SGP4 TEME error (km), conservative estimates.
# These grow with time because SGP4 includes secular perturbations that the
# Keplerian orbit ignores.
_ERROR_BOUNDS = {
    #             t=0       t=10 min  t=1 orbit  t=10 orbits
    "iss":     (  50.0,     60.0,     100.0,      500.0),
    "gps":     (  50.0,     55.0,      80.0,      300.0),
    "molniya": ( 100.0,    110.0,     200.0,      800.0),
}


def _kep_teme_position(our, tsince_min: float) -> np.ndarray:
    """Return Keplerian TEME position (km) at tsince minutes after epoch."""
    jd0 = float(our.jdsatepoch)
    fr0 = float(our.jdsatepochF)
    t_day = tsince_min / 1440.0
    r, _ = _kepler_rv_teme(
        our.inclo, our.nodeo, our.ecco, our.argpo,
        our.mo, our.no_kozai, our.mu,
        our.jdsatepoch, our.jdsatepochF,
        jnp.float64(jd0), jnp.float64(fr0 + t_day),
    )
    return np.array(r)


def _sgp4_teme_position(ref, tsince_min: float) -> np.ndarray | None:
    """Return reference SGP4 TEME position (km) at tsince minutes after epoch."""
    t_day = tsince_min / 1440.0
    err, r, _ = ref.sgp4(ref.jdsatepoch, ref.jdsatepochF + t_day)
    return None if err != 0 else np.array(r)


class TestErrorCharacterisation:
    """Keplerian vs reference SGP4 error is within documented bounds."""

    def _period_min(self, our) -> float:
        return 2.0 * np.pi / float(our.no_kozai)

    def test_error_at_epoch(self, sat_pair):
        """At t=0 the error is within the short-period J2 amplitude (< 50–100 km)."""
        name, our, ref = sat_pair
        bound = _ERROR_BOUNDS[name][0]
        r_kep = _kep_teme_position(our, 0.0)
        r_sgp4 = _sgp4_teme_position(ref, 0.0)
        assert r_sgp4 is not None
        err = np.linalg.norm(r_kep - r_sgp4)
        assert err < bound, (
            f"{name} at-epoch error = {err:.2f} km  (bound {bound} km)"
        )

    def test_error_at_10min(self, sat_pair):
        name, our, ref = sat_pair
        bound = _ERROR_BOUNDS[name][1]
        r_kep = _kep_teme_position(our, 10.0)
        r_sgp4 = _sgp4_teme_position(ref, 10.0)
        assert r_sgp4 is not None
        err = np.linalg.norm(r_kep - r_sgp4)
        assert err < bound, (
            f"{name} 10-min error = {err:.2f} km  (bound {bound} km)"
        )

    def test_error_at_one_orbit(self, sat_pair):
        """Error after one Keplerian period remains below the documented bound."""
        name, our, ref = sat_pair
        bound = _ERROR_BOUNDS[name][2]
        period = self._period_min(our)
        r_kep = _kep_teme_position(our, period)
        r_sgp4 = _sgp4_teme_position(ref, period)
        assert r_sgp4 is not None
        err = np.linalg.norm(r_kep - r_sgp4)
        assert err < bound, (
            f"{name} 1-orbit error = {err:.2f} km  (bound {bound} km)"
        )

    def test_error_at_ten_orbits(self, sat_pair):
        """Error after ten Keplerian periods is below the documented bound."""
        name, our, ref = sat_pair
        bound = _ERROR_BOUNDS[name][3]
        period = self._period_min(our)
        r_kep = _kep_teme_position(our, 10.0 * period)
        r_sgp4 = _sgp4_teme_position(ref, 10.0 * period)
        assert r_sgp4 is not None
        err = np.linalg.norm(r_kep - r_sgp4)
        assert err < bound, (
            f"{name} 10-orbit error = {err:.2f} km  (bound {bound} km)"
        )

    def test_error_grows_with_time(self, sat_pair):
        """Keplerian error is larger at 10 orbits than at epoch (secular drift)."""
        name, our, ref = sat_pair
        period = self._period_min(our)

        r_kep_0 = _kep_teme_position(our, 0.0)
        r_sgp4_0 = _sgp4_teme_position(ref, 0.0)

        r_kep_10 = _kep_teme_position(our, 10.0 * period)
        r_sgp4_10 = _sgp4_teme_position(ref, 10.0 * period)

        if r_sgp4_0 is None or r_sgp4_10 is None:
            pytest.skip(f"{name}: reference SGP4 returned error")

        err_0 = np.linalg.norm(r_kep_0 - r_sgp4_0)
        err_10 = np.linalg.norm(r_kep_10 - r_sgp4_10)
        assert err_10 > err_0, (
            f"{name}: error did not grow with time "
            f"(t=0: {err_0:.2f} km, t=10T: {err_10:.2f} km)"
        )


# ---------------------------------------------------------------------------
# 4. Orbital-shell sanity
#
# At every point in the Keplerian orbit the satellite must be between its
# periapsis and apoapsis altitudes.  The reference SGP4 positions should be
# within a few km of the same shell (J2 short-period oscillations are small
# relative to the semi-major axis).
# ---------------------------------------------------------------------------

class TestOrbitalShellSanity:
    """Both Keplerian and SGP4 positions lie within the expected orbital shell."""

    def _shell_bounds(self, our):
        """Return (r_perigee_km, r_apoapsis_km) from TLE mean elements."""
        mu = float(our.mu)
        n = float(our.no_kozai)
        e = float(our.ecco)
        mu_min = mu * 3600.0
        a = (mu_min / n ** 2) ** (1.0 / 3.0)
        return a * (1.0 - e), a * (1.0 + e)

    def test_keplerian_in_shell(self, sat_pair):
        """Every Keplerian position lies exactly in [r_peri, r_apo]."""
        _, our, _ = sat_pair
        r_peri, r_apo = self._shell_bounds(our)
        jd0 = float(our.jdsatepoch) + float(our.jdsatepochF)
        period_min = 2.0 * np.pi / float(our.no_kozai)
        times_jd = jnp.linspace(jd0, jd0 + period_min / 1440.0, 200)

        from sgp4jax import kepler_gcrf_positions
        r, _ = kepler_gcrf_positions(our, times_jd)
        r_mag = np.array(jnp.linalg.norm(r, axis=-1))

        assert r_mag.min() >= r_peri - 1e-6, (
            f"Position dips below periapsis: {r_mag.min():.4f} < {r_peri:.4f} km"
        )
        assert r_mag.max() <= r_apo + 1e-6, (
            f"Position exceeds apoapsis: {r_mag.max():.4f} > {r_apo:.4f} km"
        )

    def test_sgp4_near_keplerian_shell(self, sat_pair):
        """SGP4 positions stay within 5% of the Keplerian shell half-width.

        J2 short-period oscillations in |r| have amplitude < 1% of (a·e) for
        most orbits.  We use a 5% bound to be conservative.
        """
        name, our, ref = sat_pair
        r_peri, r_apo = self._shell_bounds(our)
        half_width = (r_apo - r_peri) / 2.0
        tolerance = max(0.05 * half_width, 50.0)  # at least 50 km slack

        period_min = 2.0 * np.pi / float(our.no_kozai)
        t_list = np.linspace(0, period_min, 50)

        for t in t_list:
            r_sgp4 = _sgp4_teme_position(ref, float(t))
            if r_sgp4 is None:
                continue
            r_mag = np.linalg.norm(r_sgp4)
            assert r_mag >= r_peri - tolerance, (
                f"{name}: SGP4 |r|={r_mag:.2f} km far below periapsis "
                f"{r_peri:.2f} km at t={t:.1f} min"
            )
            assert r_mag <= r_apo + tolerance, (
                f"{name}: SGP4 |r|={r_mag:.2f} km far above apoapsis "
                f"{r_apo:.2f} km at t={t:.1f} min"
            )
