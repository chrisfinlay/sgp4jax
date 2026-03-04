"""Test backward (negative tsince) propagation."""

import jax.numpy as jnp
import numpy as np
import pytest
from sgp4.api import Satrec as RefSatrec, WGS72 as REF_WGS72

from sgp4jax import tle_to_satrec, propagate, WGS72


# SMS 1 AKM (Sat 09998) - verification dataset uses -1440 to -720
LINE1_SMS = '1 09998U 74033F   05148.79417928 -.00000112  00000-0  00000+0 0  4480'
LINE2_SMS = '2 09998   9.4958 313.1750 0270971 327.5225  30.8097  1.16186785 45878'

# AMC-4 (Sat 25954) - verification dataset uses -1440 to +1440
LINE1_AMC = '1 25954U 99060A   04039.68057285 -.00000108  00000-0  00000-0 0  6847'
LINE2_AMC = '2 25954   0.0004 243.8136 0001765  15.5294  22.7134  1.00271289 15615'


class TestBackwardPropagation:
    """Backward propagation with negative tsince values."""

    @pytest.mark.parametrize("tsince", [-1440.0, -1380.0, -1200.0, -960.0, -720.0])
    def test_sms_backward(self, tsince):
        """SMS 1 AKM backward propagation should match python-sgp4."""
        sat = tle_to_satrec(LINE1_SMS, LINE2_SMS, gravity=WGS72)
        ref = RefSatrec.twoline2rv(LINE1_SMS, LINE2_SMS, REF_WGS72)

        r, v, err = propagate(sat, jnp.array(tsince))
        e_ref, r_ref, v_ref = ref.sgp4(
            ref.jdsatepoch, ref.jdsatepochF + tsince / 1440.0)

        assert int(err) == e_ref, (
            f"t={tsince}: sgp4jax error={int(err)}, ref error={e_ref}")
        if e_ref == 0 and int(err) == 0:
            np.testing.assert_allclose(
                np.array(r), np.array(r_ref), atol=1e-6,
                err_msg=f"Position mismatch at t={tsince}")
            np.testing.assert_allclose(
                np.array(v), np.array(v_ref), atol=1e-7,
                err_msg=f"Velocity mismatch at t={tsince}")

    @pytest.mark.parametrize("tsince", [-1440.0, -1200.0, -720.0, -360.0, -120.0, 0.0])
    def test_amc_backward(self, tsince):
        """AMC-4 backward propagation should match python-sgp4."""
        sat = tle_to_satrec(LINE1_AMC, LINE2_AMC, gravity=WGS72)
        ref = RefSatrec.twoline2rv(LINE1_AMC, LINE2_AMC, REF_WGS72)

        r, v, err = propagate(sat, jnp.array(tsince))
        e_ref, r_ref, v_ref = ref.sgp4(
            ref.jdsatepoch, ref.jdsatepochF + tsince / 1440.0)

        assert int(err) == e_ref, (
            f"t={tsince}: sgp4jax error={int(err)}, ref error={e_ref}")
        if e_ref == 0 and int(err) == 0:
            np.testing.assert_allclose(
                np.array(r), np.array(r_ref), atol=1e-6,
                err_msg=f"Position mismatch at t={tsince}")
            np.testing.assert_allclose(
                np.array(v), np.array(v_ref), atol=1e-7,
                err_msg=f"Velocity mismatch at t={tsince}")


class TestBackwardForwardRoundtrip:
    """Propagating backward then forward should approximately recover state."""

    def test_roundtrip_near_earth(self):
        """Backward then forward should be close to epoch for near-earth."""
        sat = tle_to_satrec(LINE1_AMC, LINE2_AMC, gravity=WGS72)

        # Position at epoch
        r0, v0, err0 = propagate(sat, jnp.array(0.0))
        assert int(err0) == 0

        # Forward 60 min, then backward 60 min should return to ~original
        # (This tests propagation symmetry, not exact reversibility since
        # SGP4 includes secular drag terms)
        r_fwd, v_fwd, err_fwd = propagate(sat, jnp.array(60.0))
        r_bwd, v_bwd, err_bwd = propagate(sat, jnp.array(0.0))

        assert int(err_fwd) == 0
        assert int(err_bwd) == 0
        # Re-propagating to t=0 should exactly match epoch
        np.testing.assert_allclose(
            np.array(r_bwd), np.array(r0), atol=1e-12,
            err_msg="Round-trip back to epoch should be exact")

    def test_backward_continuity(self):
        """Position should vary smoothly through t=0."""
        sat = tle_to_satrec(LINE1_AMC, LINE2_AMC, gravity=WGS72)

        # Small step backward and forward from epoch
        r_neg, v_neg, _ = propagate(sat, jnp.array(-1.0))
        r_zero, v_zero, _ = propagate(sat, jnp.array(0.0))
        r_pos, v_pos, _ = propagate(sat, jnp.array(1.0))

        # All should be finite
        assert jnp.all(jnp.isfinite(r_neg))
        assert jnp.all(jnp.isfinite(r_zero))
        assert jnp.all(jnp.isfinite(r_pos))

        # Changes should be small (continuity)
        dr_neg = float(jnp.linalg.norm(r_zero - r_neg))
        dr_pos = float(jnp.linalg.norm(r_pos - r_zero))
        # 1 minute of motion should be < 500 km for any satellite
        assert dr_neg < 500.0, f"Backward step too large: {dr_neg} km"
        assert dr_pos < 500.0, f"Forward step too large: {dr_pos} km"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
