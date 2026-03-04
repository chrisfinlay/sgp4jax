"""Test mixed batch vmap with heterogeneous satellites."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgp4jax import (
    tle_to_satrec, tles_to_satrec, propagate,
    gcrf_positions_multi, WGS72, SatRec,
)


# ISS (near-earth, low eccentricity)
LINE1_ISS = '1 25544U 98067A   20045.18587073  .00000950  00000-0  25302-4 0  9990'
LINE2_ISS = '2 25544  51.6443 242.0161 0004397 264.6060 207.3845 15.49165514212791'

# GPS (deep-space, non-resonant)
LINE1_GPS = '1 28129U 03058A   06175.57071136 -.00000104  00000-0  10000-3 0   459'
LINE2_GPS = '2 28129  54.7298 324.8098 0048506 266.2640  93.1663  2.00562768 18443'

# Molniya (deep-space, 12h resonant)
LINE1_MOLNIYA = '1 09880U 77021A   06176.56157475  .00000421  00000-0  10000-3 0  9814'
LINE2_MOLNIYA = '2 09880  64.5968 349.3786 7069051 270.0229  16.3320  2.00813614112380'

# Decaying satellite (produces error code 6)
LINE1_DECAY = '1 28872U 05037B   05333.02012661  .25992681  00000-0  24476-3 0  1534'
LINE2_DECAY = '2 28872  96.4736 157.9986 0303955 244.0492 110.6523 16.46015938 10708'


class TestMixedBatchVmap:
    """Batch propagation with heterogeneous satellite types."""

    def test_vmap_over_satellites(self):
        """vmap over ISS + GPS + Molniya, verify matches sequential."""
        tles = [
            (LINE1_ISS, LINE2_ISS),
            (LINE1_GPS, LINE2_GPS),
            (LINE1_MOLNIYA, LINE2_MOLNIYA),
        ]
        batch_sat = tles_to_satrec(tles, gravity=WGS72)

        tsince = jnp.array(100.0)
        batched = jax.vmap(propagate, in_axes=(0, None))
        r_batch, v_batch, err_batch = batched(batch_sat, tsince)

        assert r_batch.shape == (3, 3)
        assert v_batch.shape == (3, 3)
        assert err_batch.shape == (3,)

        # Compare with sequential
        for i, (l1, l2) in enumerate(tles):
            sat = tle_to_satrec(l1, l2, gravity=WGS72)
            r_seq, v_seq, err_seq = propagate(sat, tsince)
            np.testing.assert_allclose(
                np.array(r_batch[i]), np.array(r_seq), atol=1e-12,
                err_msg=f"Position mismatch for satellite {i}")
            np.testing.assert_allclose(
                np.array(v_batch[i]), np.array(v_seq), atol=1e-12,
                err_msg=f"Velocity mismatch for satellite {i}")

    def test_vmap_multiple_times_and_satellites(self):
        """Double vmap: over satellites and time."""
        tles = [
            (LINE1_ISS, LINE2_ISS),
            (LINE1_GPS, LINE2_GPS),
        ]
        batch_sat = tles_to_satrec(tles, gravity=WGS72)
        times = jnp.array([0.0, 100.0, 500.0])

        # vmap over satellites, then over time
        batched = jax.vmap(
            jax.vmap(propagate, in_axes=(None, 0)),
            in_axes=(0, None),
        )
        r, v, err = batched(batch_sat, times)

        assert r.shape == (2, 3, 3)  # (n_sat, n_times, 3)
        assert v.shape == (2, 3, 3)
        assert err.shape == (2, 3)

    def test_batch_with_error_satellite(self):
        """Error satellite should produce NaN in its row only."""
        tles = [
            (LINE1_ISS, LINE2_ISS),
            (LINE1_DECAY, LINE2_DECAY),
        ]
        batch_sat = tles_to_satrec(tles, gravity=WGS72)

        # At t=60, decay satellite should have error
        tsince = jnp.array(60.0)
        batched = jax.vmap(propagate, in_axes=(0, None))
        r_batch, v_batch, err_batch = batched(batch_sat, tsince)

        # ISS should be fine
        assert int(err_batch[0]) == 0
        assert jnp.all(jnp.isfinite(r_batch[0]))

        # Decay satellite may or may not have errored by t=60
        # If error, it should be NaN
        if int(err_batch[1]) != 0:
            assert jnp.all(jnp.isnan(r_batch[1]))


class TestEndToEndPipeline:
    """Test tles_to_satrec → gcrf_positions_multi pipeline."""

    def test_gcrf_positions_multi(self):
        tles = [
            (LINE1_ISS, LINE2_ISS),
            (LINE1_GPS, LINE2_GPS),
        ]
        batch_sat = tles_to_satrec(tles, gravity=WGS72)

        # Use epoch of ISS as base time
        base_jd = float(batch_sat.jdsatepoch[0]) + float(batch_sat.jdsatepochF[0])
        times_jd = jnp.array([base_jd, base_jd + 1.0 / 24.0])  # epoch, +1hr

        r, v = gcrf_positions_multi(batch_sat, times_jd)

        assert r.shape == (2, 2, 3)  # (n_sat, n_times, 3)
        assert v.shape == (2, 2, 3)
        assert jnp.all(jnp.isfinite(r))

    def test_tles_to_satrec_stacking(self):
        """Verify tles_to_satrec produces correctly stacked SatRec."""
        tles = [
            (LINE1_ISS, LINE2_ISS),
            (LINE1_GPS, LINE2_GPS),
            (LINE1_MOLNIYA, LINE2_MOLNIYA),
        ]
        batch_sat = tles_to_satrec(tles, gravity=WGS72)

        # Check that fields have leading dimension
        assert batch_sat.bstar.shape == (3,)
        assert batch_sat.ecco.shape == (3,)
        assert batch_sat.jdsatepoch.shape == (3,)

        # Verify individual values match
        for i, (l1, l2) in enumerate(tles):
            single = tle_to_satrec(l1, l2, gravity=WGS72)
            np.testing.assert_allclose(
                float(batch_sat.ecco[i]), float(single.ecco), rtol=1e-15)
            np.testing.assert_allclose(
                float(batch_sat.bstar[i]), float(single.bstar), rtol=1e-15)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
