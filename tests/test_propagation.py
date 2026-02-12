"""Test SGP4-JAX against the reference sgp4 library."""

import os
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from sgp4.api import Satrec as RefSatrec, WGS84 as REF_WGS84

from sgp4jax import tle_to_satrec, propagate, propagate_jd, WGS84


# Near-earth test satellite (ISS-like)
LINE1_NEAR = '1 25544U 98067A   20045.18587073  .00000950  00000-0  25302-4 0  9990'
LINE2_NEAR = '2 25544  51.6443 242.0161 0004397 264.6060 207.3845 15.49165514212791'

# Deep-space test satellite (GPS-like, NOAA 14)
LINE1_DEEP = '1 23455U 94089A   97320.90946019  .00000140  00000-0  10191-3 0  2621'
LINE2_DEEP = '2 23455  99.0090 272.6745 0008546 223.1686 136.8816 14.11711747148495'

# Vanguard (from SGP4 verification set)
LINE1_VANGUARD = '1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753'
LINE2_VANGUARD = '2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667'

# High eccentricity deep space (Molniya-like from verification set)
LINE1_MOLNIYA = '1 09880U 77021A   00251.45080028  .00000316  00000-0  10000-3 0  3527'
LINE2_MOLNIYA = '2 09880  64.7791 180.0788 7258491 296.1385  20.2281  2.00879014156621'

# Path to SGP4-VER.TLE verification dataset
SGP4_VER_TLE = os.path.join(
    os.path.dirname(__file__), '..', '..', 'python-sgp4', 'sgp4', 'SGP4-VER.TLE')


def get_ref_satrec(line1, line2):
    """Get reference Satrec using WGS84."""
    return RefSatrec.twoline2rv(line1, line2, REF_WGS84)


class TestNearEarth:
    """Tests for near-earth satellites."""

    def test_epoch_propagation(self):
        """Propagation at epoch (t=0) should give valid results."""
        sat = tle_to_satrec(LINE1_NEAR, LINE2_NEAR)
        ref = get_ref_satrec(LINE1_NEAR, LINE2_NEAR)

        r, v, err = propagate(sat, jnp.array(0.0))
        e_ref, r_ref, v_ref = ref.sgp4(ref.jdsatepoch, ref.jdsatepochF)

        assert int(err) == 0, f"Error code: {err}"
        np.testing.assert_allclose(np.array(r), np.array(r_ref), atol=1e-6,
                                   err_msg="Position mismatch at epoch")
        np.testing.assert_allclose(np.array(v), np.array(v_ref), atol=1e-7,
                                   err_msg="Velocity mismatch at epoch")

    def test_propagation_multiple_times(self):
        """Test at multiple time points."""
        sat = tle_to_satrec(LINE1_NEAR, LINE2_NEAR)
        ref = get_ref_satrec(LINE1_NEAR, LINE2_NEAR)

        for tsince in [0.0, 10.0, 100.0, 360.0, 720.0, 1440.0]:
            r, v, err = propagate(sat, jnp.array(tsince))
            # Compute reference using tsince
            jd_offset = tsince / 1440.0
            e_ref, r_ref, v_ref = ref.sgp4(ref.jdsatepoch, ref.jdsatepochF + jd_offset)
            if e_ref != 0:
                continue  # skip error cases

            assert int(err) == 0, f"Error at t={tsince}: {err}"
            np.testing.assert_allclose(
                np.array(r), np.array(r_ref), atol=1e-6,
                err_msg=f"Position mismatch at t={tsince}")
            np.testing.assert_allclose(
                np.array(v), np.array(v_ref), atol=1e-7,
                err_msg=f"Velocity mismatch at t={tsince}")


class TestVanguard:
    """Tests for Vanguard satellite (high eccentricity near-earth)."""

    def test_propagation(self):
        sat = tle_to_satrec(LINE1_VANGUARD, LINE2_VANGUARD)
        ref = get_ref_satrec(LINE1_VANGUARD, LINE2_VANGUARD)

        for tsince in [0.0, 120.0, 360.0, 720.0, 1440.0]:
            r, v, err = propagate(sat, jnp.array(tsince))
            jd_offset = tsince / 1440.0
            e_ref, r_ref, v_ref = ref.sgp4(ref.jdsatepoch, ref.jdsatepochF + jd_offset)
            if e_ref != 0:
                continue

            assert int(err) == 0, f"Error at t={tsince}: {err}"
            np.testing.assert_allclose(
                np.array(r), np.array(r_ref), atol=1e-6,
                err_msg=f"Position mismatch at t={tsince}")
            np.testing.assert_allclose(
                np.array(v), np.array(v_ref), atol=1e-7,
                err_msg=f"Velocity mismatch at t={tsince}")


class TestDeepSpace:
    """Tests for deep-space satellites."""

    def test_deep_space_propagation(self):
        sat = tle_to_satrec(LINE1_DEEP, LINE2_DEEP)
        ref = get_ref_satrec(LINE1_DEEP, LINE2_DEEP)

        for tsince in [0.0, 120.0, 360.0, 720.0, 1440.0]:
            r, v, err = propagate(sat, jnp.array(tsince))
            jd_offset = tsince / 1440.0
            e_ref, r_ref, v_ref = ref.sgp4(ref.jdsatepoch, ref.jdsatepochF + jd_offset)
            if e_ref != 0:
                continue

            assert int(err) == 0, f"Error at t={tsince}: {err}"
            np.testing.assert_allclose(
                np.array(r), np.array(r_ref), atol=1e-6,
                err_msg=f"Position mismatch at t={tsince}")
            np.testing.assert_allclose(
                np.array(v), np.array(v_ref), atol=1e-7,
                err_msg=f"Velocity mismatch at t={tsince}")

    def test_molniya_propagation(self):
        """Molniya orbit - high eccentricity deep space with resonance."""
        sat = tle_to_satrec(LINE1_MOLNIYA, LINE2_MOLNIYA)
        ref = get_ref_satrec(LINE1_MOLNIYA, LINE2_MOLNIYA)

        for tsince in [0.0, 120.0, 360.0, 720.0]:
            r, v, err = propagate(sat, jnp.array(tsince))
            jd_offset = tsince / 1440.0
            e_ref, r_ref, v_ref = ref.sgp4(ref.jdsatepoch, ref.jdsatepochF + jd_offset)
            if e_ref != 0:
                continue

            assert int(err) == 0, f"Error at t={tsince}: {err}"
            np.testing.assert_allclose(
                np.array(r), np.array(r_ref), atol=1e-6,
                err_msg=f"Position mismatch at t={tsince}")
            np.testing.assert_allclose(
                np.array(v), np.array(v_ref), atol=1e-7,
                err_msg=f"Velocity mismatch at t={tsince}")


class TestJulianDate:
    """Test the Julian Date convenience function."""

    def test_propagate_jd(self):
        sat = tle_to_satrec(LINE1_NEAR, LINE2_NEAR)
        ref = get_ref_satrec(LINE1_NEAR, LINE2_NEAR)

        jd = jnp.array(ref.jdsatepoch)
        fr = jnp.array(ref.jdsatepochF + 0.5)  # 12 hours after epoch

        r, v, err = propagate_jd(sat, jd, fr)
        e_ref, r_ref, v_ref = ref.sgp4(float(jd), float(fr))

        assert int(err) == 0
        np.testing.assert_allclose(np.array(r), np.array(r_ref), atol=1e-6)
        np.testing.assert_allclose(np.array(v), np.array(v_ref), atol=1e-7)


class TestInitFieldComparison:
    """Compare initialized SatRec fields against reference."""

    def test_near_earth_fields(self):
        sat = tle_to_satrec(LINE1_NEAR, LINE2_NEAR)
        # Use pure Python Satrec for field access
        from sgp4.model import Satrec as PySatrec
        from sgp4.earth_gravity import wgs84
        ref = PySatrec()
        from sgp4.io import twoline2rv
        twoline2rv(LINE1_NEAR, LINE2_NEAR, wgs84, 'i', ref)

        fields_to_check = [
            'bstar', 'ecco', 'argpo', 'inclo', 'mo', 'no_kozai', 'nodeo',
            'a', 'alta', 'altp',
            'con41', 'cc1', 'cc4', 'cc5',
            'd2', 'd3', 'd4', 'delmo', 'eta',
            'argpdot', 'omgcof', 'sinmao',
            't2cof', 't3cof', 't4cof', 't5cof',
            'x1mth2', 'x7thm1', 'mdot', 'nodedot',
            'xlcof', 'xmcof', 'nodecf', 'aycof', 'gsto',
        ]

        for field in fields_to_check:
            jax_val = float(getattr(sat, field))
            ref_val = float(getattr(ref, field))
            np.testing.assert_allclose(
                jax_val, ref_val, rtol=1e-10, atol=1e-15,
                err_msg=f"Field '{field}' mismatch: JAX={jax_val}, ref={ref_val}")


def parse_ver_tle(filepath):
    """Parse SGP4-VER.TLE file into list of (line1, line2, startmfe, stopmfe, deltamin)."""
    satellites = []
    with open(filepath) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if line.startswith('1 ') and len(line) >= 69:
            line1 = line[:69]
            i += 1
            if i < len(lines):
                line2_full = lines[i].rstrip()
                line2 = line2_full[:69]
                # Extra fields after column 69: startmfe stopmfe deltamin
                extra = line2_full[69:].split()
                if len(extra) >= 3:
                    startmfe = float(extra[0])
                    stopmfe = float(extra[1])
                    deltamin = float(extra[2])
                    satellites.append((line1, line2, startmfe, stopmfe, deltamin))
        i += 1
    return satellites


class TestVerificationDataset:
    """Comprehensive test against SGP4-VER.TLE verification dataset."""

    @pytest.fixture(autouse=True)
    def _check_verfile(self):
        if not os.path.exists(SGP4_VER_TLE):
            pytest.skip("SGP4-VER.TLE not found")

    def test_all_satellites(self):
        """All verification satellites should match reference to floating-point precision."""
        satellites = parse_ver_tle(SGP4_VER_TLE)
        assert len(satellites) >= 30, f"Expected >=30 satellites, got {len(satellites)}"

        max_pos_err = 0.0
        max_vel_err = 0.0
        failures = []

        for line1, line2, startmfe, stopmfe, deltamin in satellites:
            satnum = line1[2:7].strip()
            try:
                sat = tle_to_satrec(line1, line2)
            except Exception as e:
                failures.append(f"sat={satnum} init failed: {e}")
                continue

            # Generate time steps
            if deltamin <= 0:
                continue
            tsince = startmfe
            while tsince <= stopmfe + deltamin * 0.5:
                # Fresh reference for each propagation to avoid mutable state issues
                ref = RefSatrec.twoline2rv(line1, line2, REF_WGS84)
                jd_offset = tsince / 1440.0
                e_ref, r_ref, v_ref = ref.sgp4(
                    ref.jdsatepoch, ref.jdsatepochF + jd_offset)

                r_jax, v_jax, err_jax = propagate(sat, jnp.array(tsince))

                if e_ref != 0:
                    # Reference had an error - JAX should too (or NaN)
                    tsince += deltamin
                    continue

                if int(err_jax) != 0:
                    # JAX error but reference succeeded
                    failures.append(
                        f"sat={satnum} t={tsince}: JAX error={int(err_jax)} but ref OK")
                    tsince += deltamin
                    continue

                pos_err = float(jnp.linalg.norm(
                    jnp.array(r_ref) - r_jax))
                vel_err = float(jnp.linalg.norm(
                    jnp.array(v_ref) - v_jax))

                max_pos_err = max(max_pos_err, pos_err)
                max_vel_err = max(max_vel_err, vel_err)

                # 1e-6 km = 1mm tolerance (generous for floating-point match)
                if pos_err > 1e-6:
                    failures.append(
                        f"sat={satnum} t={tsince:.1f}: pos_err={pos_err:.2e} km")

                tsince += deltamin

        assert len(failures) == 0, (
            f"Verification failures ({len(failures)}):\n" +
            "\n".join(failures[:20]) +
            f"\n\nMax pos err: {max_pos_err:.2e} km, max vel err: {max_vel_err:.2e} km/s")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
