"""Test direct verification against tcppver.out reference vectors."""

import os
import re

import jax.numpy as jnp
import numpy as np
import pytest

from sgp4jax import tle_to_satrec, propagate, WGS72


# Paths to verification files
_BASEDIR = os.path.join(os.path.dirname(__file__), '..', '..', 'python-sgp4', 'sgp4')
SGP4_VER_TLE = os.path.join(_BASEDIR, 'SGP4-VER.TLE')
TCPPVER_OUT = os.path.join(_BASEDIR, 'tcppver.out')


def parse_ver_tle(filepath):
    """Parse SGP4-VER.TLE into dict keyed by satnum."""
    satellites = {}
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
                extra = line2_full[69:].split()
                if len(extra) >= 3:
                    satnum = int(line1[2:7])
                    satellites[satnum] = (line1, line2)
        i += 1
    return satellites


def parse_tcppver_out(filepath):
    """Parse tcppver.out into dict: satnum -> list of (tsince, r, v)."""
    results = {}
    current_sat = None

    with open(filepath) as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue

            # Satellite header: "NNNNN xx"
            m = re.match(r'^(\d+)\s+xx\s*$', line)
            if m:
                current_sat = int(m.group(1))
                if current_sat not in results:
                    results[current_sat] = []
                continue

            if current_sat is None:
                continue

            # Data line: tsince  x  y  z  vx  vy  vz  [optional keplerian elements]
            parts = line.split()
            if len(parts) >= 7:
                try:
                    tsince = float(parts[0])
                    r = (float(parts[1]), float(parts[2]), float(parts[3]))
                    v = (float(parts[4]), float(parts[5]), float(parts[6]))
                    results[current_sat].append((tsince, r, v))
                except ValueError:
                    continue

    return results


class TestTcppverOut:
    """Direct verification against tcppver.out C++ reference output."""

    @pytest.fixture(autouse=True)
    def _check_files(self):
        if not os.path.exists(SGP4_VER_TLE):
            pytest.skip("SGP4-VER.TLE not found")
        if not os.path.exists(TCPPVER_OUT):
            pytest.skip("tcppver.out not found")

    def test_all_satellites_against_tcppver(self):
        """Compare sgp4jax output against tcppver.out reference vectors."""
        tle_data = parse_ver_tle(SGP4_VER_TLE)
        ref_data = parse_tcppver_out(TCPPVER_OUT)

        pos_tol = 1e-6   # 1 mm
        vel_tol = 1e-7    # 0.1 mm/s

        failures = []
        max_pos_err = 0.0
        max_vel_err = 0.0
        total_points = 0
        sat_max_errors = {}

        for satnum, ref_points in ref_data.items():
            if satnum not in tle_data:
                continue

            line1, line2 = tle_data[satnum]
            try:
                sat = tle_to_satrec(line1, line2, gravity=WGS72)
            except Exception as e:
                failures.append(f"sat={satnum} init failed: {e}")
                continue

            sat_max_pos = 0.0
            for tsince, r_ref, v_ref in ref_points:
                r_jax, v_jax, err_jax = propagate(sat, jnp.array(tsince))

                if int(err_jax) != 0:
                    # Skip error cases (reference may also have errors)
                    continue

                total_points += 1
                r_jax_np = np.array(r_jax)
                v_jax_np = np.array(v_jax)
                r_ref_np = np.array(r_ref)
                v_ref_np = np.array(v_ref)

                pos_err = float(np.linalg.norm(r_jax_np - r_ref_np))
                vel_err = float(np.linalg.norm(v_jax_np - v_ref_np))

                max_pos_err = max(max_pos_err, pos_err)
                max_vel_err = max(max_vel_err, vel_err)
                sat_max_pos = max(sat_max_pos, pos_err)

                if pos_err > pos_tol:
                    failures.append(
                        f"sat={satnum} t={tsince:.1f}: "
                        f"pos_err={pos_err:.2e} km > {pos_tol}")
                if vel_err > vel_tol:
                    failures.append(
                        f"sat={satnum} t={tsince:.1f}: "
                        f"vel_err={vel_err:.2e} km/s > {vel_tol}")

            sat_max_errors[satnum] = sat_max_pos

        assert total_points > 0, "No valid comparison points found"

        # Report per-satellite max errors for diagnostics
        report = (
            f"\nTotal points compared: {total_points}\n"
            f"Max position error: {max_pos_err:.2e} km\n"
            f"Max velocity error: {max_vel_err:.2e} km/s\n"
            f"Per-satellite max pos errors:\n"
        )
        for satnum in sorted(sat_max_errors):
            report += f"  {satnum:5d}: {sat_max_errors[satnum]:.2e} km\n"

        assert len(failures) == 0, (
            f"tcppver.out verification failures ({len(failures)}):\n"
            + "\n".join(failures[:30])
            + report)

    def test_tcppver_point_count(self):
        """Verify we parse a reasonable number of reference points."""
        ref_data = parse_tcppver_out(TCPPVER_OUT)
        total = sum(len(pts) for pts in ref_data.values())
        assert total >= 100, f"Expected >= 100 reference points, got {total}"
        assert len(ref_data) >= 20, f"Expected >= 20 satellites, got {len(ref_data)}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
