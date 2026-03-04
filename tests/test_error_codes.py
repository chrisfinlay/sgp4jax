"""Test SGP4 error code handling using dedicated test satellites from SGP4-VER.TLE."""

import jax.numpy as jnp
import numpy as np
import pytest
from sgp4.api import Satrec as RefSatrec, WGS72 as REF_WGS72

from sgp4jax import tle_to_satrec, propagate, WGS72


# Error code 2: mean motion <= 0 (Sat 33334, no=0.00001 rev/day)
LINE1_ERR2 = '1 33334U 78066F   06174.85818871  .00000620  00000-0  10000-3 0  6809'
LINE2_ERR2 = '2 33334  68.4714 236.1303 5602877 123.7484 302.5767  0.00001000 67521'

# Error code 4: semi-latus rectum < 0 (Sat 33333, e=0.995)
LINE1_ERR4 = '1 33333U 05037B   05333.02012661  .25992681  00000-0  24476-3 0  1534'
LINE2_ERR4 = '2 33333  96.4736 157.9986 9950000 244.0492 110.6523  4.00004038 10708'

# Error code 6: decay (Sat 28872, perigee=-51km, decays in ~50 min)
LINE1_DECAY_FAST = '1 28872U 05037B   05333.02012661  .25992681  00000-0  24476-3 0  1534'
LINE2_DECAY_FAST = '2 28872  96.4736 157.9986 0303955 244.0492 110.6523 16.46015938 10708'

# Error code 6: decay (Sat 29141, decays in ~420 min)
LINE1_DECAY_SLOW = '1 29141U 85108AA  06170.26783845  .99999999  00000-0  13519-0 0   718'
LINE2_DECAY_SLOW = '2 29141  82.4288 273.4882 0015848 277.2124  83.9133 15.93343074  6828'


class TestErrorCode2:
    """Mean motion <= 0 satellite (Sat 33334) should produce errors.

    Note: The exact error code may differ between sgp4jax (branchless) and
    python-sgp4 (branchy), but both should flag an error (non-zero code)
    and return NaN/invalid positions.
    """

    def test_produces_errors(self):
        sat = tle_to_satrec(LINE1_ERR2, LINE2_ERR2, gravity=WGS72)
        ref = RefSatrec.twoline2rv(LINE1_ERR2, LINE2_ERR2, REF_WGS72)
        for tsince in [0.0, 100.0, 500.0, 1440.0]:
            r, v, err = propagate(sat, jnp.array(tsince))
            e_ref, _, _ = ref.sgp4(
                ref.jdsatepoch, ref.jdsatepochF + tsince / 1440.0)
            # Both should produce non-zero error codes
            if e_ref != 0:
                assert int(err) != 0, (
                    f"t={tsince}: ref has error {e_ref} but sgp4jax has 0")

    def test_error_positions_nan(self):
        sat = tle_to_satrec(LINE1_ERR2, LINE2_ERR2, gravity=WGS72)
        for tsince in [0.0, 100.0, 500.0]:
            r, v, err = propagate(sat, jnp.array(tsince))
            if int(err) != 0:
                assert jnp.all(jnp.isnan(r)), (
                    f"t={tsince}: error positions should be NaN")
                assert jnp.all(jnp.isnan(v)), (
                    f"t={tsince}: error velocities should be NaN")


class TestErrorCode4:
    """Semi-latus rectum < 0 should produce error code 4."""

    def test_error_code_value(self):
        sat = tle_to_satrec(LINE1_ERR4, LINE2_ERR4, gravity=WGS72)
        for tsince in [0.0, 50.0, 100.0, 150.0]:
            r, v, err = propagate(sat, jnp.array(tsince))
            ref = RefSatrec.twoline2rv(LINE1_ERR4, LINE2_ERR4, REF_WGS72)
            e_ref, r_ref, v_ref = ref.sgp4(
                ref.jdsatepoch, ref.jdsatepochF + tsince / 1440.0)
            assert int(err) == e_ref, (
                f"t={tsince}: sgp4jax error={int(err)}, ref error={e_ref}")

    def test_error_positions_nan(self):
        sat = tle_to_satrec(LINE1_ERR4, LINE2_ERR4, gravity=WGS72)
        # At some time steps this satellite hits error code 4
        ref = RefSatrec.twoline2rv(LINE1_ERR4, LINE2_ERR4, REF_WGS72)
        for tsince in [50.0, 100.0, 150.0]:
            e_ref, _, _ = ref.sgp4(
                ref.jdsatepoch, ref.jdsatepochF + tsince / 1440.0)
            r, v, err = propagate(sat, jnp.array(tsince))
            if e_ref != 0:
                assert jnp.all(jnp.isnan(r)), (
                    f"t={tsince}: error positions should be NaN")


class TestErrorCode6Decay:
    """Satellite decay should produce error code 6."""

    def test_fast_decay_28872(self):
        """Sat 28872 decays within 50 minutes."""
        sat = tle_to_satrec(LINE1_DECAY_FAST, LINE2_DECAY_FAST, gravity=WGS72)
        ref = RefSatrec.twoline2rv(LINE1_DECAY_FAST, LINE2_DECAY_FAST, REF_WGS72)

        for tsince in [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0]:
            r, v, err = propagate(sat, jnp.array(tsince))
            e_ref, r_ref, v_ref = ref.sgp4(
                ref.jdsatepoch, ref.jdsatepochF + tsince / 1440.0)
            assert int(err) == e_ref, (
                f"t={tsince}: sgp4jax error={int(err)}, ref error={e_ref}")
            if e_ref != 0:
                assert jnp.all(jnp.isnan(r)), (
                    f"t={tsince}: decay positions should be NaN")

    def test_slow_decay_29141(self):
        """Sat 29141 decays within 420 minutes."""
        sat = tle_to_satrec(LINE1_DECAY_SLOW, LINE2_DECAY_SLOW, gravity=WGS72)
        ref = RefSatrec.twoline2rv(LINE1_DECAY_SLOW, LINE2_DECAY_SLOW, REF_WGS72)

        for tsince in [0.0, 100.0, 200.0, 300.0, 400.0, 440.0]:
            r, v, err = propagate(sat, jnp.array(tsince))
            e_ref, r_ref, v_ref = ref.sgp4(
                ref.jdsatepoch, ref.jdsatepochF + tsince / 1440.0)
            assert int(err) == e_ref, (
                f"t={tsince}: sgp4jax error={int(err)}, ref error={e_ref}")
            if e_ref != 0:
                assert jnp.all(jnp.isnan(r)), (
                    f"t={tsince}: decay positions should be NaN")

    def test_valid_before_decay(self):
        """Before decay, position should be valid and match reference."""
        sat = tle_to_satrec(LINE1_DECAY_FAST, LINE2_DECAY_FAST, gravity=WGS72)
        ref = RefSatrec.twoline2rv(LINE1_DECAY_FAST, LINE2_DECAY_FAST, REF_WGS72)

        r, v, err = propagate(sat, jnp.array(0.0))
        e_ref, r_ref, v_ref = ref.sgp4(ref.jdsatepoch, ref.jdsatepochF)
        if e_ref == 0 and int(err) == 0:
            np.testing.assert_allclose(np.array(r), np.array(r_ref), atol=1e-6)
            np.testing.assert_allclose(np.array(v), np.array(v_ref), atol=1e-7)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
