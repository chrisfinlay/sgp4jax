"""Test gravity model comparison: WGS72OLD, WGS72, WGS84."""

import jax.numpy as jnp
import numpy as np
import pytest
from sgp4.api import Satrec as RefSatrec
from sgp4.api import WGS72 as REF_WGS72, WGS84 as REF_WGS84, WGS72OLD as REF_WGS72OLD

from sgp4jax import tle_to_satrec, propagate, WGS72, WGS72OLD, WGS84


# ISS TLE for testing
LINE1 = '1 25544U 98067A   20045.18587073  .00000950  00000-0  25302-4 0  9990'
LINE2 = '2 25544  51.6443 242.0161 0004397 264.6060 207.3845 15.49165514212791'


class TestGravityModelResults:
    """All three gravity models should produce valid but different results."""

    def test_all_models_produce_valid_results(self):
        for grav in [WGS72OLD, WGS72, WGS84]:
            sat = tle_to_satrec(LINE1, LINE2, gravity=grav)
            r, v, err = propagate(sat, jnp.array(100.0))
            assert int(err) == 0, f"Error with gravity model {grav}"
            assert jnp.all(jnp.isfinite(r)), f"Non-finite position with {grav}"
            assert jnp.all(jnp.isfinite(v)), f"Non-finite velocity with {grav}"

    def test_models_give_different_results(self):
        results = {}
        for name, grav in [('WGS72OLD', WGS72OLD), ('WGS72', WGS72), ('WGS84', WGS84)]:
            sat = tle_to_satrec(LINE1, LINE2, gravity=grav)
            r, v, err = propagate(sat, jnp.array(100.0))
            results[name] = np.array(r)

        # All three models produce slightly different results.
        # At t=100 min differences are small (~0.01 km) but nonzero.
        for a, b in [('WGS72OLD', 'WGS72'), ('WGS72', 'WGS84'), ('WGS72OLD', 'WGS84')]:
            diff = np.linalg.norm(results[a] - results[b])
            assert diff > 0.0, f"{a} and {b} should not be identical"


class TestMatchPythonSGP4:
    """Results should match python-sgp4 for each gravity model."""

    @pytest.mark.parametrize("tsince", [0.0, 100.0, 720.0, 1440.0])
    def test_wgs72_matches_reference(self, tsince):
        sat = tle_to_satrec(LINE1, LINE2, gravity=WGS72)
        ref = RefSatrec.twoline2rv(LINE1, LINE2, REF_WGS72)
        r, v, err = propagate(sat, jnp.array(tsince))
        e_ref, r_ref, v_ref = ref.sgp4(
            ref.jdsatepoch, ref.jdsatepochF + tsince / 1440.0)
        assert int(err) == 0
        assert e_ref == 0
        np.testing.assert_allclose(np.array(r), np.array(r_ref), atol=1e-6)
        np.testing.assert_allclose(np.array(v), np.array(v_ref), atol=1e-7)

    @pytest.mark.parametrize("tsince", [0.0, 100.0, 720.0, 1440.0])
    def test_wgs84_matches_reference(self, tsince):
        sat = tle_to_satrec(LINE1, LINE2, gravity=WGS84)
        ref = RefSatrec.twoline2rv(LINE1, LINE2, REF_WGS84)
        r, v, err = propagate(sat, jnp.array(tsince))
        e_ref, r_ref, v_ref = ref.sgp4(
            ref.jdsatepoch, ref.jdsatepochF + tsince / 1440.0)
        assert int(err) == 0
        assert e_ref == 0
        np.testing.assert_allclose(np.array(r), np.array(r_ref), atol=1e-6)
        np.testing.assert_allclose(np.array(v), np.array(v_ref), atol=1e-7)

    @pytest.mark.parametrize("tsince", [0.0, 100.0, 720.0, 1440.0])
    def test_wgs72old_matches_reference(self, tsince):
        sat = tle_to_satrec(LINE1, LINE2, gravity=WGS72OLD)
        ref = RefSatrec.twoline2rv(LINE1, LINE2, REF_WGS72OLD)
        r, v, err = propagate(sat, jnp.array(tsince))
        e_ref, r_ref, v_ref = ref.sgp4(
            ref.jdsatepoch, ref.jdsatepochF + tsince / 1440.0)
        assert int(err) == 0
        assert e_ref == 0
        np.testing.assert_allclose(np.array(r), np.array(r_ref), atol=1e-6)
        np.testing.assert_allclose(np.array(v), np.array(v_ref), atol=1e-7)


class TestGravityConstants:
    """Verify gravity constants match reference values."""

    def test_wgs72_constants(self):
        assert WGS72.mu == pytest.approx(398600.8, rel=1e-10)
        assert WGS72.radiusearthkm == pytest.approx(6378.135, rel=1e-10)
        assert WGS72.j2 == pytest.approx(0.001082616, rel=1e-10)
        assert WGS72.j3 == pytest.approx(-0.00000253881, rel=1e-10)
        assert WGS72.j4 == pytest.approx(-0.00000165597, rel=1e-10)

    def test_wgs84_constants(self):
        assert WGS84.mu == pytest.approx(398600.5, rel=1e-10)
        assert WGS84.radiusearthkm == pytest.approx(6378.137, rel=1e-10)
        assert WGS84.j2 == pytest.approx(0.00108262998905, rel=1e-10)
        assert WGS84.j3 == pytest.approx(-0.00000253215306, rel=1e-10)
        assert WGS84.j4 == pytest.approx(-0.00000161098761, rel=1e-10)

    def test_wgs72old_constants(self):
        assert WGS72OLD.mu == pytest.approx(398600.79964, rel=1e-10)
        assert WGS72OLD.radiusearthkm == pytest.approx(6378.135, rel=1e-10)
        assert WGS72OLD.xke == pytest.approx(0.0743669161, rel=1e-10)

    def test_j3oj2_derived(self):
        """j3oj2 should equal j3/j2 for all models."""
        for grav in [WGS72OLD, WGS72, WGS84]:
            np.testing.assert_allclose(
                grav.j3oj2, grav.j3 / grav.j2, rtol=1e-12,
                err_msg=f"j3oj2 mismatch for {grav}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
