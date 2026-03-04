"""Test TLE parsing edge cases."""

import jax.numpy as jnp
import numpy as np
import pytest
from math import pi
from sgp4.api import Satrec as RefSatrec, WGS72 as REF_WGS72

from sgp4jax import tle_to_satrec, propagate, WGS72
from sgp4jax._tle import parse_tle


# Standard ISS TLE for baseline comparison
LINE1_ISS = '1 25544U 98067A   20045.18587073  .00000950  00000-0  25302-4 0  9990'
LINE2_ISS = '2 25544  51.6443 242.0161 0004397 264.6060 207.3845 15.49165514212791'

# Negative bstar (Sat 21897, Molniya 1-83)
LINE1_NEG_BSTAR = '1 21897U 92011A   06176.02341244 -.00001273  00000-0 -13525-3 0  3044'
LINE2_NEG_BSTAR = '2 21897  62.1749 198.0096 7421690 253.0462  20.1561  2.01269994104880'

# Zero bstar/nddot
LINE1_ZERO_BSTAR = '1 09998U 74033F   05148.79417928 -.00000112  00000-0  00000+0 0  4480'
LINE2_ZERO_BSTAR = '2 09998   9.4958 313.1750 0270971 327.5225  30.8097  1.16186785 45878'

# Very low eccentricity (Sat 28057, CBERS 2, ecc=0.0000884)
LINE1_LOW_ECC = '1 28057U 03049A   06177.78615833  .00000060  00000-0  35940-4 0  1836'
LINE2_LOW_ECC = '2 28057  98.4283 247.6961 0000884  88.1964 271.9322 14.35478080140550'


class TestEpochYearBoundary:
    """Test epoch year interpretation: 00-56 → 2000-2056, 57-99 → 1957-1999."""

    def test_year_2020(self):
        """Year 20 should map to 2020."""
        params = parse_tle(LINE1_ISS, LINE2_ISS)
        # ISS epoch is 20045.xxx → year 2020
        jd = params['jdsatepoch'] + params['jdsatepochF']
        # JD for 2020-02-14 should be around 2458893
        assert 2458890 < jd < 2458900, f"JD {jd} not in expected range for 2020"

    def test_year_2006(self):
        """Year 06 should map to 2006."""
        params = parse_tle(LINE1_NEG_BSTAR, LINE2_NEG_BSTAR)
        jd = params['jdsatepoch'] + params['jdsatepochF']
        # JD for 2006-06-25 should be around 2453911
        assert 2453900 < jd < 2453920, f"JD {jd} not in expected range for 2006"

    def test_year_2005(self):
        """Year 05 should map to 2005."""
        params = parse_tle(LINE1_ZERO_BSTAR, LINE2_ZERO_BSTAR)
        jd = params['jdsatepoch'] + params['jdsatepochF']
        # JD for 2005-05-28 should be around 2453518
        assert 2453510 < jd < 2453530, f"JD {jd} not in expected range for 2005"

    def test_year_boundary_56(self):
        """Year 56 should map to 2056 (last year in 2000s range)."""
        # Modify ISS TLE to have epoch year 56
        line1_56 = LINE1_ISS[:18] + '56' + LINE1_ISS[20:]
        params = parse_tle(line1_56, LINE2_ISS)
        jd = params['jdsatepoch'] + params['jdsatepochF']
        # 2056 JD should be much larger than 2020
        assert jd > 2470000, f"Year 56 → 2056: JD {jd} too small"

    def test_year_boundary_57(self):
        """Year 57 should map to 1957 (first year in 1900s range)."""
        line1_57 = LINE1_ISS[:18] + '57' + LINE1_ISS[20:]
        params = parse_tle(line1_57, LINE2_ISS)
        jd = params['jdsatepoch'] + params['jdsatepochF']
        # 1957 JD should be around 2435xxx
        assert jd < 2440000, f"Year 57 → 1957: JD {jd} too large"


class TestBstarParsing:
    """Test bstar field parsing including negative and zero values."""

    def test_negative_bstar(self):
        """Sat 21897 has negative bstar (-0.13525e-3)."""
        params = parse_tle(LINE1_NEG_BSTAR, LINE2_NEG_BSTAR)
        assert params['bstar'] < 0, f"bstar should be negative: {params['bstar']}"

        # Verify against python-sgp4
        ref = RefSatrec.twoline2rv(LINE1_NEG_BSTAR, LINE2_NEG_BSTAR, REF_WGS72)
        np.testing.assert_allclose(
            params['bstar'], ref.bstar, rtol=1e-10,
            err_msg="Negative bstar mismatch")

    def test_zero_bstar(self):
        """Sat 09998 has zero bstar (00000+0)."""
        params = parse_tle(LINE1_ZERO_BSTAR, LINE2_ZERO_BSTAR)
        assert params['bstar'] == 0.0, f"bstar should be zero: {params['bstar']}"

    def test_positive_bstar(self):
        """ISS has positive bstar."""
        params = parse_tle(LINE1_ISS, LINE2_ISS)
        assert params['bstar'] > 0, f"bstar should be positive: {params['bstar']}"


class TestEccentricityParsing:
    """Test eccentricity field parsing (implied leading decimal point)."""

    def test_low_eccentricity(self):
        """CBERS 2 has eccentricity 0.0000884."""
        params = parse_tle(LINE1_LOW_ECC, LINE2_LOW_ECC)
        ref = RefSatrec.twoline2rv(LINE1_LOW_ECC, LINE2_LOW_ECC, REF_WGS72)
        np.testing.assert_allclose(
            params['ecco'], ref.ecco, rtol=1e-10,
            err_msg="Low eccentricity mismatch")

    def test_high_eccentricity(self):
        """Molniya has high eccentricity 0.7421690."""
        params = parse_tle(LINE1_NEG_BSTAR, LINE2_NEG_BSTAR)
        ref = RefSatrec.twoline2rv(LINE1_NEG_BSTAR, LINE2_NEG_BSTAR, REF_WGS72)
        np.testing.assert_allclose(
            params['ecco'], ref.ecco, rtol=1e-10,
            err_msg="High eccentricity mismatch")


class TestRoundTrip:
    """Parse TLE and verify parsed values match python-sgp4."""

    @pytest.mark.parametrize("line1,line2", [
        (LINE1_ISS, LINE2_ISS),
        (LINE1_NEG_BSTAR, LINE2_NEG_BSTAR),
        (LINE1_ZERO_BSTAR, LINE2_ZERO_BSTAR),
        (LINE1_LOW_ECC, LINE2_LOW_ECC),
    ])
    def test_parsed_fields_match_reference(self, line1, line2):
        """All parsed orbital elements should match python-sgp4."""
        params = parse_tle(line1, line2)
        ref = RefSatrec.twoline2rv(line1, line2, REF_WGS72)

        np.testing.assert_allclose(params['bstar'], ref.bstar, rtol=1e-10,
                                   err_msg="bstar mismatch")
        np.testing.assert_allclose(params['ecco'], ref.ecco, rtol=1e-10,
                                   err_msg="ecco mismatch")
        np.testing.assert_allclose(params['inclo'], ref.inclo, rtol=1e-10,
                                   err_msg="inclo mismatch")
        np.testing.assert_allclose(params['nodeo'], ref.nodeo, rtol=1e-10,
                                   err_msg="nodeo mismatch")
        np.testing.assert_allclose(params['argpo'], ref.argpo, rtol=1e-10,
                                   err_msg="argpo mismatch")
        np.testing.assert_allclose(params['mo'], ref.mo, rtol=1e-10,
                                   err_msg="mo mismatch")
        np.testing.assert_allclose(params['no_kozai'], ref.no_kozai, rtol=1e-10,
                                   err_msg="no_kozai mismatch")

    @pytest.mark.parametrize("line1,line2", [
        (LINE1_ISS, LINE2_ISS),
        (LINE1_NEG_BSTAR, LINE2_NEG_BSTAR),
        (LINE1_LOW_ECC, LINE2_LOW_ECC),
    ])
    def test_propagation_after_parsing(self, line1, line2):
        """Parsed and initialized satellite should propagate correctly."""
        sat = tle_to_satrec(line1, line2, gravity=WGS72)
        ref = RefSatrec.twoline2rv(line1, line2, REF_WGS72)

        r, v, err = propagate(sat, jnp.array(0.0))
        e_ref, r_ref, v_ref = ref.sgp4(ref.jdsatepoch, ref.jdsatepochF)

        assert int(err) == e_ref
        if e_ref == 0:
            np.testing.assert_allclose(np.array(r), np.array(r_ref), atol=1e-6)
            np.testing.assert_allclose(np.array(v), np.array(v_ref), atol=1e-7)


class TestAngleConversions:
    """Verify angle fields are correctly converted to radians."""

    def test_angles_in_radians(self):
        params = parse_tle(LINE1_ISS, LINE2_ISS)
        # ISS inclination is 51.6443 degrees → should be ~0.901 radians
        assert 0.8 < params['inclo'] < 1.0, (
            f"inclo={params['inclo']} not in expected radian range")
        # All angles should be in [0, 2*pi] range (approximately)
        assert 0.0 <= params['inclo'] <= pi
        assert 0.0 <= params['argpo'] <= 2 * pi
        assert 0.0 <= params['mo'] <= 2 * pi


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
