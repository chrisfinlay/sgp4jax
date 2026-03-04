"""_dspace - deep space secular effects (JIT-compatible)."""

import jax
import jax.numpy as jnp


def dspace(irez, d2201, d2211, d3210, d3222, d4410, d4422,
           d5220, d5232, d5421, d5433,
           dedt, del1, del2, del3, didt, dmdt, dnodt, domdt,
           argpo, argpdot, t, tc, gsto, xfact, xlamo, no,
           atime, em, argpm, inclm, xli, mm, xni, nodem, nm):
    """Deep space contributions to mean elements.

    All inputs/outputs are JAX arrays. Fully JIT-compatible.
    """
    fasx2 = 0.13130908
    fasx4 = 2.8843198
    fasx6 = 0.37448087
    g22 = 5.7686396
    g32 = 0.95240898
    g44 = 1.8014998
    g52 = 1.0508330
    g54 = 4.4108898
    rptim = 4.37526908801129966e-3
    stepp = 720.0
    stepn = -720.0
    step2 = 259200.0
    twopi = 2.0 * jnp.pi

    # Calculate deep space resonance effects
    dndt = jnp.array(0.0)
    theta = (gsto + tc * rptim) % twopi
    em = em + dedt * t
    inclm = inclm + didt * t
    argpm = argpm + domdt * t
    nodem = nodem + dnodt * t
    mm = mm + dmdt * t

    # Resonance integration
    ft = jnp.array(0.0)

    # Only do resonance if irez != 0
    # Reset conditions
    should_reset = (atime == 0.0) | (t * atime <= 0.0) | (jnp.abs(t) < jnp.abs(atime))
    atime = jnp.where((irez != 0) & should_reset, 0.0, atime)
    xni = jnp.where((irez != 0) & should_reset, no, xni)
    xli = jnp.where((irez != 0) & should_reset, xlamo, xli)

    delt = jnp.where(t > 0.0, stepp, stepn)

    def _compute_xndt_xnddt(xli, xni, atime, irez, d2201, d2211, d3210, d3222,
                              d4410, d4422, d5220, d5232, d5421, d5433,
                              del1, del2, del3, xfact, argpo, argpdot):
        """Compute xndt, xldot, xnddt for both resonance types."""
        # Near-synchronous (irez != 2)
        xndt_sync = (del1 * jnp.sin(xli - fasx2) +
                     del2 * jnp.sin(2.0 * (xli - fasx4)) +
                     del3 * jnp.sin(3.0 * (xli - fasx6)))
        xldot_sync = xni + xfact
        xnddt_sync = (del1 * jnp.cos(xli - fasx2) +
                      2.0 * del2 * jnp.cos(2.0 * (xli - fasx4)) +
                      3.0 * del3 * jnp.cos(3.0 * (xli - fasx6)))
        xnddt_sync = xnddt_sync * xldot_sync

        # Half-day resonance (irez == 2)
        xomi = argpo + argpdot * atime
        x2omi = xomi + xomi
        x2li = xli + xli
        xndt_half = (d2201 * jnp.sin(x2omi + xli - g22) +
                     d2211 * jnp.sin(xli - g22) +
                     d3210 * jnp.sin(xomi + xli - g32) +
                     d3222 * jnp.sin(-xomi + xli - g32) +
                     d4410 * jnp.sin(x2omi + x2li - g44) +
                     d4422 * jnp.sin(x2li - g44) +
                     d5220 * jnp.sin(xomi + xli - g52) +
                     d5232 * jnp.sin(-xomi + xli - g52) +
                     d5421 * jnp.sin(xomi + x2li - g54) +
                     d5433 * jnp.sin(-xomi + x2li - g54))
        xldot_half = xni + xfact
        xnddt_half = (d2201 * jnp.cos(x2omi + xli - g22) +
                      d2211 * jnp.cos(xli - g22) +
                      d3210 * jnp.cos(xomi + xli - g32) +
                      d3222 * jnp.cos(-xomi + xli - g32) +
                      d5220 * jnp.cos(xomi + xli - g52) +
                      d5232 * jnp.cos(-xomi + xli - g52) +
                      2.0 * (d4410 * jnp.cos(x2omi + x2li - g44) +
                             d4422 * jnp.cos(x2li - g44) +
                             d5421 * jnp.cos(xomi + x2li - g54) +
                             d5433 * jnp.cos(-xomi + x2li - g54)))
        xnddt_half = xnddt_half * xldot_half

        is_half = (irez == 2)
        xndt = jnp.where(is_half, xndt_half, xndt_sync)
        xldot = jnp.where(is_half, xldot_half, xldot_sync)
        xnddt = jnp.where(is_half, xnddt_half, xnddt_sync)

        return xndt, xldot, xnddt

    # Use lax.scan with fixed iterations instead of while_loop
    # (while_loop doesn't support reverse-mode AD).
    # Max iterations: ceil(max_propagation_time / 720) + 1.
    # For propagation up to ~30 days (43200 min), 64 is more than enough.
    _MAX_ITERS = 64

    def scan_body(carry, _):
        atime_c, xni_c, xli_c, active = carry

        xndt, xldot, xnddt = _compute_xndt_xnddt(
            xli_c, xni_c, atime_c, irez, d2201, d2211, d3210, d3222,
            d4410, d4422, d5220, d5232, d5421, d5433,
            del1, del2, del3, xfact, argpo, argpdot)

        should_step = jnp.abs(t - atime_c) >= stepp
        do_step = active & should_step

        xli_new = xli_c + xldot * delt + xndt * step2
        xni_new = xni_c + xndt * delt + xnddt * step2
        atime_new = atime_c + delt

        xli_c = jnp.where(do_step, xli_new, xli_c)
        xni_c = jnp.where(do_step, xni_new, xni_c)
        atime_c = jnp.where(do_step, atime_new, atime_c)
        # Once we stop stepping, stay inactive
        active = active & should_step

        return (atime_c, xni_c, xli_c, active), None

    init_active = (irez != 0)
    init_carry = (atime, xni, xli, init_active)
    (atime, xni, xli, _), _ = jax.lax.scan(scan_body, init_carry, None, length=_MAX_ITERS)

    # Final interpolation
    ft = jnp.where(irez != 0, t - atime, 0.0)

    xndt, xldot, xnddt = _compute_xndt_xnddt(
        xli, xni, atime, irez, d2201, d2211, d3210, d3222,
        d4410, d4422, d5220, d5232, d5421, d5433,
        del1, del2, del3, xfact, argpo, argpdot)

    nm_new = xni + xndt * ft + xnddt * ft * ft * 0.5
    xl = xli + xldot * ft + xndt * ft * ft * 0.5

    # mm depends on irez
    mm_irez_not1 = xl - 2.0 * nodem + 2.0 * theta
    mm_irez_1 = xl - nodem - argpm + theta
    mm_new = jnp.where(irez != 1, mm_irez_not1, mm_irez_1)
    dndt_new = nm_new - no
    nm_new = no + dndt_new

    # Only apply resonance results if irez != 0
    nm = jnp.where(irez != 0, nm_new, nm)
    mm = jnp.where(irez != 0, mm_new, mm)
    dndt = jnp.where(irez != 0, dndt_new, dndt)

    return atime, em, argpm, inclm, xli, mm, xni, nodem, dndt, nm
