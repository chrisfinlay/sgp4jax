"""Propagator performance benchmarks.

Compares sgp4 (full), sgp4_leo, and sgp4_sdp4_nr against the reference
sgp4 C library across four scenarios:

  1. Single satellite, single time      — raw dispatch latency
  2. Single satellite, N times          — temporal batch (vmap over tsince)
  3. N satellites, single time          — constellation batch (vmap over satrec)
  4. N satellites × M times (primary)  — full constellation × time grid

Usage::

    python benchmarks/bench_propagators.py
    python benchmarks/bench_propagators.py --scenario nm --sat-counts 10,100 --time-counts 100,1000
    python benchmarks/bench_propagators.py --repeats 20

"""

import argparse
import time
from functools import partial

import jax

# sgp4jax requires JAX double precision; enable it before importing.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from sgp4.api import Satrec as RefSatrec, SatrecArray, WGS84 as REF_WGS84

import sgp4jax
from sgp4jax import tle_to_satrec, tles_to_satrec, WGS84
from sgp4jax import propagate, propagate_leo, propagate_sdp4_nr


# ---------------------------------------------------------------------------
# TLEs
# ---------------------------------------------------------------------------

# LEO — ISS
_LEO_L1 = '1 25544U 98067A   20045.18587073  .00000950  00000-0  25302-4 0  9990'
_LEO_L2 = '2 25544  51.6443 242.0161 0004397 264.6060 207.3845 15.49165514212791'

# Deep-space irez=0 — NAVSTAR 53 GPS
_GPS_L1 = '1 28129U 03058A   06175.57071136 -.00000104  00000-0  10000-3 0   459'
_GPS_L2 = '2 28129  54.7298 324.8098 0048506 266.2640  93.1663  2.00562768 18443'

# Reference objects
_ref_leo = RefSatrec.twoline2rv(_LEO_L1, _LEO_L2, REF_WGS84)
_ref_gps = RefSatrec.twoline2rv(_GPS_L1, _GPS_L2, REF_WGS84)


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _bench(fn, warmup: int = 3, repeats: int = 10) -> float:
    """Return median wall-clock time (seconds) for one call to fn()."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def _bench_jax(fn, warmup: int = 3, repeats: int = 10) -> float:
    """Like _bench but calls jax.block_until_ready on the return value."""
    def wrapped():
        result = fn()
        # block_until_ready works on arrays or nested structures
        jax.block_until_ready(result)

    return _bench(wrapped, warmup=warmup, repeats=repeats)


def _fmt_time(seconds: float) -> str:
    if seconds >= 1.0:
        return f"{seconds * 1e3:8.2f} ms"
    elif seconds >= 1e-3:
        return f"{seconds * 1e3:8.3f} ms"
    elif seconds >= 1e-6:
        return f"{seconds * 1e6:8.2f} µs"
    else:
        return f"{seconds * 1e9:8.1f} ns"


def _throughput(n: int, seconds: float) -> str:
    rate = n / seconds
    if rate >= 1e6:
        return f"{rate / 1e6:.2f} M/s"
    elif rate >= 1e3:
        return f"{rate / 1e3:.1f} k/s"
    else:
        return f"{rate:.0f} /s"


def _row(label: str, t: float, n: int = 1, ref_t: float | None = None) -> str:
    speedup = f"  {ref_t / t:5.1f}x faster" if ref_t is not None and t < ref_t else (
              f"  {t / ref_t:5.1f}x slower" if ref_t is not None else "")
    return f"  {label:<28s} {_fmt_time(t)}   {_throughput(n, t)}{speedup}"


def _header(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    print(f"  {'Implementation':<28s} {'Time':>12}   {'Throughput'}")
    print(f"  {'-' * 60}")


# ---------------------------------------------------------------------------
# Scenario 1: single satellite, single time
# ---------------------------------------------------------------------------

def bench_single(repeats: int) -> None:
    _header("Scenario 1 — Single satellite, single time")

    sat_leo = tle_to_satrec(_LEO_L1, _LEO_L2, gravity=WGS84)
    sat_gps = tle_to_satrec(_GPS_L1, _GPS_L2, gravity=WGS84)
    t = jnp.array(360.0)

    # Pre-compile JAX functions
    _ = jax.block_until_ready(propagate(sat_leo, t))
    _ = jax.block_until_ready(propagate_leo(sat_leo, t))
    _ = jax.block_until_ready(propagate(sat_gps, t))
    _ = jax.block_until_ready(propagate_sdp4_nr(sat_gps, t))

    print("\n  [LEO satellite — ISS]")
    ref_t = _bench(lambda: _ref_leo.sgp4(_ref_leo.jdsatepoch,
                                         _ref_leo.jdsatepochF + 0.25), repeats=repeats)
    t_full = _bench_jax(lambda: propagate(sat_leo, t), repeats=repeats)
    t_leo  = _bench_jax(lambda: propagate_leo(sat_leo, t), repeats=repeats)

    print(_row("sgp4 (reference C)",  ref_t))
    print(_row("sgp4 (JAX full)",     t_full, ref_t=ref_t))
    print(_row("sgp4_leo (JAX)",      t_leo,  ref_t=ref_t))

    print("\n  [Deep-space irez=0 — GPS NAVSTAR 53]")
    ref_t2 = _bench(lambda: _ref_gps.sgp4(_ref_gps.jdsatepoch,
                                           _ref_gps.jdsatepochF + 0.25), repeats=repeats)
    t_full2 = _bench_jax(lambda: propagate(sat_gps, t), repeats=repeats)
    t_nr    = _bench_jax(lambda: propagate_sdp4_nr(sat_gps, t), repeats=repeats)

    print(_row("sgp4 (reference C)",      ref_t2))
    print(_row("sgp4 (JAX full)",         t_full2, ref_t=ref_t2))
    print(_row("sgp4_sdp4_nr (JAX)",      t_nr,    ref_t=ref_t2))


# ---------------------------------------------------------------------------
# Scenario 2: single satellite, N times (temporal batch)
# ---------------------------------------------------------------------------

def bench_temporal_batch(batch_sizes: list[int], repeats: int) -> None:
    _header("Scenario 2 — Single satellite, N times (vmap over tsince)")

    sat_leo = tle_to_satrec(_LEO_L1, _LEO_L2, gravity=WGS84)
    sat_gps = tle_to_satrec(_GPS_L1, _GPS_L2, gravity=WGS84)

    vmap_leo_full = jax.jit(jax.vmap(propagate,     in_axes=(None, 0)))
    vmap_leo_spec = jax.jit(jax.vmap(propagate_leo, in_axes=(None, 0)))
    vmap_gps_full = jax.jit(jax.vmap(propagate,        in_axes=(None, 0)))
    vmap_gps_spec = jax.jit(jax.vmap(propagate_sdp4_nr, in_axes=(None, 0)))

    for n in batch_sizes:
        times_jnp = jnp.linspace(0.0, 1440.0, n)
        times_np  = np.linspace(0.0, 1440.0, n)

        # Warm up JAX vmaps
        _ = jax.block_until_ready(vmap_leo_full(sat_leo, times_jnp))
        _ = jax.block_until_ready(vmap_leo_spec(sat_leo, times_jnp))
        _ = jax.block_until_ready(vmap_gps_full(sat_gps, times_jnp))
        _ = jax.block_until_ready(vmap_gps_spec(sat_gps, times_jnp))

        jd_leo = np.full(n, _ref_leo.jdsatepoch)
        fr_leo = _ref_leo.jdsatepochF + times_np / 1440.0
        jd_gps = np.full(n, _ref_gps.jdsatepoch)
        fr_gps = _ref_gps.jdsatepochF + times_np / 1440.0

        print(f"\n  N = {n:,}")
        print(f"\n  [LEO — ISS]")
        ref_t = _bench(lambda: _ref_leo.sgp4_array(jd_leo, fr_leo), repeats=repeats)
        t_full = _bench_jax(lambda: vmap_leo_full(sat_leo, times_jnp), repeats=repeats)
        t_leo  = _bench_jax(lambda: vmap_leo_spec(sat_leo, times_jnp), repeats=repeats)

        print(_row("sgp4_array (reference C)",  ref_t,  n))
        print(_row("sgp4 vmap (JAX full)",       t_full, n, ref_t=ref_t))
        print(_row("sgp4_leo vmap (JAX)",        t_leo,  n, ref_t=ref_t))

        print(f"\n  [Deep-space irez=0 — GPS]")
        ref_t2 = _bench(lambda: _ref_gps.sgp4_array(jd_gps, fr_gps), repeats=repeats)
        t_full2 = _bench_jax(lambda: vmap_gps_full(sat_gps, times_jnp), repeats=repeats)
        t_nr    = _bench_jax(lambda: vmap_gps_spec(sat_gps, times_jnp), repeats=repeats)

        print(_row("sgp4_array (reference C)",      ref_t2,  n))
        print(_row("sgp4 vmap (JAX full)",           t_full2, n, ref_t=ref_t2))
        print(_row("sgp4_sdp4_nr vmap (JAX)",        t_nr,    n, ref_t=ref_t2))


# ---------------------------------------------------------------------------
# Scenario 3: N satellites, single time (constellation batch)
# ---------------------------------------------------------------------------

def bench_constellation_batch(batch_sizes: list[int], repeats: int) -> None:
    _header("Scenario 3 — N satellites, single time (vmap over satrec)")

    t_jnp = jnp.array(360.0)

    vmap_full = jax.jit(jax.vmap(propagate,        in_axes=(0, None)))
    vmap_leo  = jax.jit(jax.vmap(propagate_leo,    in_axes=(0, None)))
    vmap_nr   = jax.jit(jax.vmap(propagate_sdp4_nr, in_axes=(0, None)))

    for n in batch_sizes:
        leo_tles = [[_LEO_L1, _LEO_L2]] * n
        gps_tles = [[_GPS_L1, _GPS_L2]] * n

        sat_leo_batch = tles_to_satrec(leo_tles, gravity=WGS84)
        sat_gps_batch = tles_to_satrec(gps_tles, gravity=WGS84)

        ref_leo_arr = SatrecArray([RefSatrec.twoline2rv(_LEO_L1, _LEO_L2, REF_WGS84)] * n)
        ref_gps_arr = SatrecArray([RefSatrec.twoline2rv(_GPS_L1, _GPS_L2, REF_WGS84)] * n)
        jd_leo = np.full(n, _ref_leo.jdsatepoch)
        fr_leo = np.full(n, _ref_leo.jdsatepochF + 0.25)
        jd_gps = np.full(n, _ref_gps.jdsatepoch)
        fr_gps = np.full(n, _ref_gps.jdsatepochF + 0.25)

        # Warm up JAX vmaps
        _ = jax.block_until_ready(vmap_full(sat_leo_batch, t_jnp))
        _ = jax.block_until_ready(vmap_leo(sat_leo_batch, t_jnp))
        _ = jax.block_until_ready(vmap_full(sat_gps_batch, t_jnp))
        _ = jax.block_until_ready(vmap_nr(sat_gps_batch, t_jnp))

        print(f"\n  N = {n:,}")
        print(f"\n  [LEO — ISS ×{n}]")
        ref_t = _bench(lambda: ref_leo_arr.sgp4(jd_leo, fr_leo), repeats=repeats)
        t_full = _bench_jax(lambda: vmap_full(sat_leo_batch, t_jnp), repeats=repeats)
        t_leo  = _bench_jax(lambda: vmap_leo(sat_leo_batch, t_jnp), repeats=repeats)

        print(_row("SatrecArray.sgp4 (ref C)",  ref_t,  n))
        print(_row("sgp4 vmap (JAX full)",       t_full, n, ref_t=ref_t))
        print(_row("sgp4_leo vmap (JAX)",        t_leo,  n, ref_t=ref_t))

        print(f"\n  [Deep-space irez=0 — GPS ×{n}]")
        ref_t2 = _bench(lambda: ref_gps_arr.sgp4(jd_gps, fr_gps), repeats=repeats)
        t_full2 = _bench_jax(lambda: vmap_full(sat_gps_batch, t_jnp), repeats=repeats)
        t_nr    = _bench_jax(lambda: vmap_nr(sat_gps_batch, t_jnp), repeats=repeats)

        print(_row("SatrecArray.sgp4 (ref C)",      ref_t2,  n))
        print(_row("sgp4 vmap (JAX full)",           t_full2, n, ref_t=ref_t2))
        print(_row("sgp4_sdp4_nr vmap (JAX)",        t_nr,    n, ref_t=ref_t2))


# ---------------------------------------------------------------------------
# Scenario 4: N satellites × M times (the primary use case)
# ---------------------------------------------------------------------------

def bench_nm_batch(sat_counts: list[int], time_counts: list[int], repeats: int) -> None:
    _header("Scenario 4 — N satellites × M times (primary use case)")
    print(f"  Reference: SatrecArray.sgp4(jd[M], fr[M]) → (N, M, 3)")
    print(f"  JAX:       vmap(vmap(propagate, (None,0)), (0,None))(satrec[N], tsince[M])")

    # JAX: outer vmap over satellites, inner vmap over times
    vmap_full = jax.jit(jax.vmap(
        jax.vmap(propagate,        in_axes=(None, 0)), in_axes=(0, None)))
    vmap_leo  = jax.jit(jax.vmap(
        jax.vmap(propagate_leo,    in_axes=(None, 0)), in_axes=(0, None)))
    vmap_nr   = jax.jit(jax.vmap(
        jax.vmap(propagate_sdp4_nr, in_axes=(None, 0)), in_axes=(0, None)))

    for n_sats in sat_counts:
        leo_tles = [[_LEO_L1, _LEO_L2]] * n_sats
        gps_tles = [[_GPS_L1, _GPS_L2]] * n_sats
        sat_leo_batch = tles_to_satrec(leo_tles, gravity=WGS84)
        sat_gps_batch = tles_to_satrec(gps_tles, gravity=WGS84)
        ref_leo_arr = SatrecArray([RefSatrec.twoline2rv(_LEO_L1, _LEO_L2, REF_WGS84)] * n_sats)
        ref_gps_arr = SatrecArray([RefSatrec.twoline2rv(_GPS_L1, _GPS_L2, REF_WGS84)] * n_sats)

        for n_times in time_counts:
            n_total = n_sats * n_times
            times_jnp = jnp.linspace(0.0, 1440.0, n_times)
            times_np  = np.linspace(0.0, 1440.0, n_times)
            jd_leo = np.full(n_times, _ref_leo.jdsatepoch)
            fr_leo = _ref_leo.jdsatepochF + times_np / 1440.0
            jd_gps = np.full(n_times, _ref_gps.jdsatepoch)
            fr_gps = _ref_gps.jdsatepochF + times_np / 1440.0

            # Warm up JAX
            _ = jax.block_until_ready(vmap_full(sat_leo_batch, times_jnp))
            _ = jax.block_until_ready(vmap_leo(sat_leo_batch, times_jnp))
            _ = jax.block_until_ready(vmap_full(sat_gps_batch, times_jnp))
            _ = jax.block_until_ready(vmap_nr(sat_gps_batch, times_jnp))

            print(f"\n  N = {n_sats:,} sats × M = {n_times:,} times  ({n_total:,} propagations)")

            print(f"\n  [LEO — ISS]")
            ref_t  = _bench(lambda: ref_leo_arr.sgp4(jd_leo, fr_leo), repeats=repeats)
            t_full = _bench_jax(lambda: vmap_full(sat_leo_batch, times_jnp), repeats=repeats)
            t_leo  = _bench_jax(lambda: vmap_leo(sat_leo_batch, times_jnp), repeats=repeats)
            print(_row("SatrecArray.sgp4 (ref C)",  ref_t,  n_total))
            print(_row("sgp4 vmap×vmap (JAX full)", t_full, n_total, ref_t=ref_t))
            print(_row("sgp4_leo vmap×vmap (JAX)",  t_leo,  n_total, ref_t=ref_t))

            print(f"\n  [Deep-space irez=0 — GPS]")
            ref_t2  = _bench(lambda: ref_gps_arr.sgp4(jd_gps, fr_gps), repeats=repeats)
            t_full2 = _bench_jax(lambda: vmap_full(sat_gps_batch, times_jnp), repeats=repeats)
            t_nr    = _bench_jax(lambda: vmap_nr(sat_gps_batch, times_jnp), repeats=repeats)
            print(_row("SatrecArray.sgp4 (ref C)",       ref_t2,  n_total))
            print(_row("sgp4 vmap×vmap (JAX full)",      t_full2, n_total, ref_t=ref_t2))
            print(_row("sgp4_sdp4_nr vmap×vmap (JAX)",   t_nr,    n_total, ref_t=ref_t2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-sizes", default="1,10,100,1000",
                        help="Comma-separated list of N values for batch scenarios "
                             "(default: 1,10,100,1000)")
    parser.add_argument("--sat-counts", default="10,100",
                        help="Comma-separated satellite counts for scenario 4 "
                             "(default: 10,100)")
    parser.add_argument("--time-counts", default="100,1000",
                        help="Comma-separated time-point counts for scenario 4 "
                             "(default: 100,1000)")
    parser.add_argument("--repeats", type=int, default=10,
                        help="Timing repetitions per measurement (default: 10)")
    parser.add_argument("--scenario",
                        choices=["single", "temporal", "constellation", "nm", "all"],
                        default="all", help="Which scenario(s) to run (default: all)")
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    sat_counts  = [int(x) for x in args.sat_counts.split(",")]
    time_counts = [int(x) for x in args.time_counts.split(",")]

    print(f"\nsgp4jax propagator benchmarks")
    print(f"  JAX backend : {jax.default_backend()}")
    print(f"  JAX devices : {jax.devices()}")
    print(f"  Repeats     : {args.repeats}")
    print(f"  Batch sizes : {batch_sizes}")

    if args.scenario in ("single", "all"):
        bench_single(args.repeats)

    if args.scenario in ("temporal", "all"):
        bench_temporal_batch(batch_sizes, args.repeats)

    if args.scenario in ("constellation", "all"):
        bench_constellation_batch(batch_sizes, args.repeats)

    if args.scenario in ("nm", "all"):
        bench_nm_batch(sat_counts, time_counts, args.repeats)

    print()


if __name__ == "__main__":
    main()
