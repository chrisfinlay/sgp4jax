"""Optional performance benchmarks.

Run with::

    pytest tests/test_performance.py -m perf -v

These tests are excluded from the default suite (``pytest tests/``).
Each test measures a specific operation and asserts that it completes
within a generous wall-clock budget, providing a safety net against
serious regressions.  The budgets are intentionally loose so that
slow CI runners do not produce false failures.

Timings are printed to stdout via ``capsys``/``print`` so they are
visible when running with ``-s`` or when a test fails.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import sgp4jax
from sgp4jax import (
    tle_to_satrec,
    tles_to_satrec,
    propagate,
    propagate_leo,
    propagate_sdp4_nr,
    propagate_mixed,
    kepler_gcrf_positions,
    kepler_gcrf_positions_multi,
    gcrf_positions,
    gcrf_positions_multi,
    gcrf_positions_multi_leo,
    gcrf_positions_multi_sdp4_nr,
    gcrf_positions_mixed,
    elements_jacobian,
    elements7_jacobian,
)


# ---------------------------------------------------------------------------
# Reference TLEs
# ---------------------------------------------------------------------------

# ISS — LEO
_ISS_L1 = "1 25544U 98067A   20045.18587073  .00000950  00000-0  25302-4 0  9990"
_ISS_L2 = "2 25544  51.6443 242.0161 0004397 264.6060 207.3845 15.49165514212791"

# GPS IIR-M — MEO (deep-space, no-resonance, SDP4-NR candidate)
_GPS_L1 = "1 28474U 04045A   20045.50000000 -.00000023  00000-0  00000+0 0  9993"
_GPS_L2 = "2 28474  55.4408  47.7022 0095788 316.9611  42.2705  2.00563847112720"

# Molniya — deep-space resonance
_MOL_L1 = "1 09880U 77021A   00251.45080028  .00000316  00000-0  10000-3 0  3527"
_MOL_L2 = "2 09880  64.7791 180.0788 7258491 296.1385  20.2281  2.00879014156621"

# A reference JD for GCRF tests
_JD = jnp.array(2458900.5)


def _warmup(fn, *args):
    """Call fn(*args) once to trigger JIT compilation, return the result."""
    return fn(*args)


def _time_n(fn, *args, n: int = 20) -> float:
    """Return median wall-clock seconds for n repeated calls (post-JIT)."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_leo_batch(n: int):
    """Return a batched SatRec of n copies of the ISS."""
    return tles_to_satrec([[_ISS_L1, _ISS_L2]] * n)


def _make_mixed_batch(n: int):
    """Return a batched SatRec mixing LEO, MEO, and Molniya."""
    tles = (
        [[_ISS_L1, _ISS_L2]] * (n // 3)
        + [[_GPS_L1, _GPS_L2]] * (n // 3)
        + [[_MOL_L1, _MOL_L2]] * (n - 2 * (n // 3))
    )
    return tles_to_satrec(tles)


# ---------------------------------------------------------------------------
# JIT compile-time tests
# ---------------------------------------------------------------------------

@pytest.mark.perf
class TestJITCompileTime:
    """Measure first-call (compile + run) latency for each propagator."""

    def test_compile_propagate(self):
        sat = tle_to_satrec(_ISS_L1, _ISS_L2)
        jitted = jax.jit(propagate)
        t0 = time.perf_counter()
        jax.block_until_ready(jitted(sat, jnp.array(100.0)))
        elapsed = time.perf_counter() - t0
        print(f"\n  propagate JIT compile: {elapsed:.2f}s")
        assert elapsed < 60.0, f"JIT compile took {elapsed:.1f}s (budget 60s)"

    def test_compile_propagate_leo(self):
        sat = tle_to_satrec(_ISS_L1, _ISS_L2)
        jitted = jax.jit(propagate_leo)
        t0 = time.perf_counter()
        jax.block_until_ready(jitted(sat, jnp.array(100.0)))
        elapsed = time.perf_counter() - t0
        print(f"\n  propagate_leo JIT compile: {elapsed:.2f}s")
        assert elapsed < 60.0

    def test_compile_propagate_sdp4_nr(self):
        sat = tle_to_satrec(_GPS_L1, _GPS_L2)
        jitted = jax.jit(propagate_sdp4_nr)
        t0 = time.perf_counter()
        jax.block_until_ready(jitted(sat, jnp.array(100.0)))
        elapsed = time.perf_counter() - t0
        print(f"\n  propagate_sdp4_nr JIT compile: {elapsed:.2f}s")
        assert elapsed < 60.0

    def test_compile_kepler_gcrf_positions(self):
        sat = tle_to_satrec(_ISS_L1, _ISS_L2)
        times = jnp.linspace(_JD, _JD + 1.0, 100)
        jitted = jax.jit(kepler_gcrf_positions)
        t0 = time.perf_counter()
        jax.block_until_ready(jitted(sat, times))
        elapsed = time.perf_counter() - t0
        print(f"\n  kepler_gcrf_positions JIT compile (100 times): {elapsed:.2f}s")
        assert elapsed < 60.0

    def test_compile_elements_jacobian(self):
        sat = tle_to_satrec(_ISS_L1, _ISS_L2)
        jitted = jax.jit(elements_jacobian)
        t0 = time.perf_counter()
        jax.block_until_ready(jitted(sat, _JD, jnp.array(0.0)))
        elapsed = time.perf_counter() - t0
        print(f"\n  elements_jacobian JIT compile: {elapsed:.2f}s")
        assert elapsed < 120.0

    def test_compile_elements7_jacobian(self):
        sat = tle_to_satrec(_ISS_L1, _ISS_L2)
        jitted = jax.jit(elements7_jacobian)
        t0 = time.perf_counter()
        jax.block_until_ready(jitted(sat, _JD, jnp.array(0.0)))
        elapsed = time.perf_counter() - t0
        print(f"\n  elements7_jacobian JIT compile: {elapsed:.2f}s")
        assert elapsed < 120.0


# ---------------------------------------------------------------------------
# Throughput tests (post-JIT)
# ---------------------------------------------------------------------------

@pytest.mark.perf
class TestThroughput:
    """Measure median wall-clock time for repeated post-JIT calls."""

    # -- scalar propagation --------------------------------------------------

    def test_throughput_propagate_scalar(self):
        sat = tle_to_satrec(_ISS_L1, _ISS_L2)
        jitted = jax.jit(propagate)
        _warmup(jitted, sat, jnp.array(100.0))
        t = _time_n(jitted, sat, jnp.array(100.0))
        print(f"\n  propagate scalar median: {t*1e3:.3f} ms")
        assert t < 0.5, f"propagate scalar took {t*1e3:.2f} ms (budget 500 µs)"

    def test_throughput_propagate_leo_scalar(self):
        sat = tle_to_satrec(_ISS_L1, _ISS_L2)
        jitted = jax.jit(propagate_leo)
        _warmup(jitted, sat, jnp.array(100.0))
        t = _time_n(jitted, sat, jnp.array(100.0))
        print(f"\n  propagate_leo scalar median: {t*1e3:.3f} ms")
        assert t < 0.5

    def test_throughput_propagate_sdp4_nr_scalar(self):
        sat = tle_to_satrec(_GPS_L1, _GPS_L2)
        jitted = jax.jit(propagate_sdp4_nr)
        _warmup(jitted, sat, jnp.array(100.0))
        t = _time_n(jitted, sat, jnp.array(100.0))
        print(f"\n  propagate_sdp4_nr scalar median: {t*1e3:.3f} ms")
        assert t < 0.5

    # -- batched over times --------------------------------------------------

    @pytest.mark.parametrize("n_times", [100, 1_000, 10_000])
    def test_throughput_gcrf_positions(self, n_times):
        sat = tle_to_satrec(_ISS_L1, _ISS_L2)
        times = jnp.linspace(_JD, _JD + 1.0, n_times)
        jitted = jax.jit(gcrf_positions)
        _warmup(jitted, sat, times)
        t = _time_n(jitted, sat, times, n=10)
        rate = n_times / t
        print(f"\n  gcrf_positions n={n_times}: {t*1e3:.1f} ms  ({rate:.0f} prop/s)")
        assert t < 10.0, f"gcrf_positions n={n_times} took {t:.2f}s"

    # -- batched over satellites ---------------------------------------------

    @pytest.mark.parametrize("n_sats", [10, 100, 1_000])
    def test_throughput_gcrf_positions_multi_leo(self, n_sats):
        satrecs = _make_leo_batch(n_sats)
        times = jnp.linspace(_JD, _JD + 1.0, 100)
        jitted = jax.jit(gcrf_positions_multi_leo)
        _warmup(jitted, satrecs, times)
        t = _time_n(jitted, satrecs, times, n=5)
        total = n_sats * 100
        rate = total / t
        print(f"\n  gcrf_positions_multi_leo {n_sats}×100: {t*1e3:.1f} ms  ({rate:.0f} prop/s)")
        assert t < 30.0

    @pytest.mark.parametrize("n_sats", [10, 100])
    def test_throughput_gcrf_positions_mixed(self, n_sats):
        # gcrf_positions_mixed is not JIT-compilable (dispatches by orbit type
        # at Python level), so we time the raw call including its internal JIT.
        satrecs = _make_mixed_batch(n_sats)
        times = jnp.linspace(_JD, _JD + 1.0, 100)
        _warmup(gcrf_positions_mixed, satrecs, times)
        t = _time_n(gcrf_positions_mixed, satrecs, times, n=5)
        total = n_sats * 100
        rate = total / t
        print(f"\n  gcrf_positions_mixed {n_sats}×100: {t*1e3:.1f} ms  ({rate:.0f} prop/s)")
        assert t < 30.0

    # -- Kepler propagator ---------------------------------------------------

    @pytest.mark.parametrize("n_times", [100, 10_000])
    def test_throughput_kepler_gcrf_positions(self, n_times):
        sat = tle_to_satrec(_ISS_L1, _ISS_L2)
        times = jnp.linspace(_JD, _JD + 1.0, n_times)
        jitted = jax.jit(kepler_gcrf_positions)
        _warmup(jitted, sat, times)
        t = _time_n(jitted, sat, times, n=10)
        rate = n_times / t
        print(f"\n  kepler_gcrf_positions n={n_times}: {t*1e3:.1f} ms  ({rate:.0f} prop/s)")
        assert t < 10.0

    @pytest.mark.parametrize("n_sats", [10, 100])
    def test_throughput_kepler_gcrf_positions_multi(self, n_sats):
        satrecs = _make_leo_batch(n_sats)
        times = jnp.linspace(_JD, _JD + 1.0, 100)
        jitted = jax.jit(kepler_gcrf_positions_multi)
        _warmup(jitted, satrecs, times)
        t = _time_n(jitted, satrecs, times, n=5)
        rate = n_sats * 100 / t
        print(f"\n  kepler_gcrf_positions_multi {n_sats}×100: {t*1e3:.1f} ms  ({rate:.0f} prop/s)")
        assert t < 10.0

    # -- covariance / Jacobians ----------------------------------------------

    def test_throughput_elements_jacobian(self):
        sat = tle_to_satrec(_ISS_L1, _ISS_L2)
        jitted = jax.jit(elements_jacobian)
        _warmup(jitted, sat, _JD, jnp.array(0.0))
        t = _time_n(jitted, sat, _JD, jnp.array(0.0))
        print(f"\n  elements_jacobian median: {t*1e3:.3f} ms")
        assert t < 1.0, f"elements_jacobian took {t*1e3:.2f} ms (budget 1000 µs)"

    def test_throughput_elements7_jacobian(self):
        sat = tle_to_satrec(_ISS_L1, _ISS_L2)
        jitted = jax.jit(elements7_jacobian)
        _warmup(jitted, sat, _JD, jnp.array(0.0))
        t = _time_n(jitted, sat, _JD, jnp.array(0.0))
        print(f"\n  elements7_jacobian median: {t*1e3:.3f} ms")
        assert t < 2.0


# ---------------------------------------------------------------------------
# Relative speed comparisons
# ---------------------------------------------------------------------------

@pytest.mark.perf
class TestRelativeSpeed:
    """Sanity checks: specialised propagators should be faster than generic."""

    def test_leo_faster_than_generic(self):
        sat = tle_to_satrec(_ISS_L1, _ISS_L2)
        gen = jax.jit(propagate)
        leo = jax.jit(propagate_leo)
        t_arg = jnp.array(100.0)

        _warmup(gen, sat, t_arg)
        _warmup(leo, sat, t_arg)

        t_gen = _time_n(gen, sat, t_arg)
        t_leo = _time_n(leo, sat, t_arg)
        print(f"\n  propagate: {t_gen*1e3:.3f} ms  propagate_leo: {t_leo*1e3:.3f} ms")
        # LEO should not be more than 3× slower than generic (usually faster)
        assert t_leo < t_gen * 3.0, (
            f"propagate_leo ({t_leo*1e3:.2f} ms) unexpectedly ≫ propagate ({t_gen*1e3:.2f} ms)"
        )

    def test_sdp4_nr_faster_than_generic(self):
        sat = tle_to_satrec(_GPS_L1, _GPS_L2)
        gen = jax.jit(propagate)
        nr = jax.jit(propagate_sdp4_nr)
        t_arg = jnp.array(100.0)

        _warmup(gen, sat, t_arg)
        _warmup(nr, sat, t_arg)

        t_gen = _time_n(gen, sat, t_arg)
        t_nr = _time_n(nr, sat, t_arg)
        print(f"\n  propagate: {t_gen*1e3:.3f} ms  propagate_sdp4_nr: {t_nr*1e3:.3f} ms")
        assert t_nr < t_gen * 3.0

    def test_kepler_faster_than_sgp4(self):
        sat = tle_to_satrec(_ISS_L1, _ISS_L2)
        times = jnp.linspace(_JD, _JD + 1.0, 1000)

        kep = jax.jit(kepler_gcrf_positions)
        sgp = jax.jit(gcrf_positions)

        _warmup(kep, sat, times)
        _warmup(sgp, sat, times)

        t_kep = _time_n(kep, sat, times, n=10)
        t_sgp = _time_n(sgp, sat, times, n=10)
        print(f"\n  kepler 1000×: {t_kep*1e3:.1f} ms  gcrf_positions 1000×: {t_sgp*1e3:.1f} ms")
        # Keplerian should not be slower than SGP4 (it's strictly simpler)
        assert t_kep < t_sgp * 3.0
