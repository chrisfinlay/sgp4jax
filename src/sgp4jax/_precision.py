"""Double-precision requirements and input validation.

sgp4jax works with absolute Julian dates and Earth-centred coordinates, both
of which need the ~16 significant digits of ``float64``:

* a Julian date is ~2.45e6 days, which ``float32`` resolves to only 0.25 day;
  even in the split ``(jd, fr)`` form the fraction resolves to ~5 ms, i.e.
  ~40 m of along-track motion for a satellite in low Earth orbit;
* a geocentric position is ~7e3 km, which ``float32`` resolves to ~0.5 mm,
  and the frame rotations that build on it degrade accordingly.

JAX defaults to single precision, and sgp4jax deliberately does **not** change
that global setting on the user's behalf.  Instead the entry points validate
their inputs and raise here, with instructions, when double precision is
missing.  Single-precision support is planned for a future release.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = [
    "x64_enabled",
    "require_x64",
    "check_float64",
    "check_jd_fr",
    "check_satrec_epoch",
]


_ENABLE_X64 = """\
Enable double precision before the first JAX operation in your program:

    import jax
    jax.config.update("jax_enable_x64", True)
    import sgp4jax

or set the environment variable before starting Python:

    export JAX_ENABLE_X64=1
"""

_WHY_FLOAT64 = """\
Julian dates and coordinates need float64.  A float32 Julian date (~2.45e6)
is resolved to only 0.25 day, and even a split (jd, fr) pair loses the
fraction below ~5 ms, i.e. ~40 m along-track for a LEO satellite.\
"""


def x64_enabled() -> bool:
    """Return ``True`` when JAX is configured for double precision."""
    return bool(jax.config.jax_enable_x64)  # type: ignore[attr-defined]


def require_x64(context: str | None = None) -> None:
    """Raise :exc:`RuntimeError` unless JAX double precision is enabled.

    Parameters
    ----------
    context : str, optional
        Name of the calling function, used in the error message.
    """
    if x64_enabled():
        return
    where = f"{context}() requires" if context else "sgp4jax requires"
    raise RuntimeError(
        f"{where} JAX double precision (float64), which is currently "
        f"disabled.\n\n{_ENABLE_X64}\n{_WHY_FLOAT64}\n"
        "sgp4jax refuses to run in single precision rather than return "
        "silently wrong positions.  Single-precision support is planned for "
        "a future release."
    )


def check_float64(
    value: jax.typing.ArrayLike,
    name: str,
    *,
    context: str | None = None,
) -> jax.Array:
    """Return *value* as a float64 array, raising if it is not double precision.

    Integer inputs are promoted to float64.  Any other non-float64 dtype
    raises :exc:`TypeError`, because casting it here would not recover the
    digits already lost.

    Parameters
    ----------
    value : array-like
        The time or coordinate to validate.
    name : str
        Argument name, used in the error message.
    context : str, optional
        Name of the calling function, used in the error message.

    Returns
    -------
    jax.Array
        *value* as a float64 array.
    """
    # Fast path: an existing float64 array (or tracer) needs no work, and its
    # dtype already proves x64 is enabled.  Keeps the per-call cost of
    # validation negligible next to JAX dispatch.
    if isinstance(value, jax.Array) and value.dtype == jnp.float64:
        return value

    require_x64(context)
    arr = jnp.asarray(value)
    if arr.dtype == jnp.float64:
        return arr
    if jnp.issubdtype(arr.dtype, jnp.integer):
        return arr.astype(jnp.float64)
    where = f" passed to {context}()" if context else ""
    raise TypeError(
        f"sgp4jax: `{name}`{where} must be a float64 array, got "
        f"{arr.dtype}.\n\n{_WHY_FLOAT64}\n\n"
        f"Build the value in double precision, e.g.\n\n"
        f"    {name} = jnp.asarray(..., dtype=jnp.float64)\n\n"
        f"Casting an array that was already computed in float32 does not "
        f"recover the lost digits."
    )


def check_jd_fr(
    jd: jax.typing.ArrayLike,
    fr: jax.typing.ArrayLike,
    *,
    context: str | None = None,
    names: tuple[str, str] = ("jd", "fr"),
) -> tuple[jax.Array, jax.Array]:
    """Validate a split Julian date.  See :func:`check_float64`."""
    return (
        check_float64(jd, names[0], context=context),
        check_float64(fr, names[1], context=context),
    )


def check_satrec_epoch(satrec: object, *, context: str | None = None) -> None:
    """Raise unless a SatRec carries its epoch in float64.

    Absolute times are computed relative to ``jdsatepoch``/``jdsatepochF``, so
    a single-precision epoch corrupts every propagation to a Julian date even
    when the requested times are float64.
    """
    for name in ("jdsatepoch", "jdsatepochF"):
        field = getattr(satrec, name)
        dtype = field.dtype if isinstance(field, jax.Array) else jnp.asarray(field).dtype
        if dtype != jnp.float64:
            require_x64(context)
            where = f" passed to {context}()" if context else ""
            raise TypeError(
                f"sgp4jax: `satrec.{name}`{where} must be float64, got "
                f"{dtype}.\n\n{_WHY_FLOAT64}\n\n"
                "Rebuild the SatRec with tle_to_satrec() / tles_to_satrec() "
                "while double precision is enabled."
            )
