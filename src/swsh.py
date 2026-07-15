import math
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def swsh(s, l, m, theta, phi):
    """
    Spin-weighted spherical harmonic -s_Y_{l,m}(theta, phi).

    Follows Kidder (arXiv:0710.0614), Eqs. (4, 5).
    Passing s=2 gives _{-2}Y_{l,m}, appropriate for GW polarizations.

    Parameters
    ----------
    s, l, m : int
        Python ints — static from JAX's perspective. The k-summation loop
        is unrolled at trace time, so this is fully JIT-compatible.
    theta : JAX array, polar angle in [0, pi]
    phi   : JAX array, azimuthal angle in [0, 2*pi]

    Returns
    -------
    JAX complex128 array, same shape as theta / phi.

    Usage inside waveform model
    ---------------------------
    def my_swsh_fn(l, m, iota):
        swsh_p = swsh(2, l,  m, iota, 0.0)   # _{-2}Y_{l, m}
        swsh_m = swsh(2, l, -m, iota, 0.0)   # _{-2}Y_{l,-m}
        return swsh_p, swsh_m
    """
    theta = jnp.asarray(theta, dtype=jnp.float64)
    phi   = jnp.asarray(phi,   dtype=jnp.float64)

    # Prefactor: pure Python — evaluated once at trace time, becomes a constant.
    prefactor = (
        (-1) ** s
        * math.sqrt((2 * l + 1) / (4.0 * math.pi))
        * math.sqrt(
            math.factorial(l + m) * math.factorial(l - m)
            * math.factorial(l + s) * math.factorial(l - s)
        )
    )

    ki = max(0, m - s)
    kf = min(l + m, l - s)

    # Sum over k.  The Python for-loop is unrolled at JAX trace time;
    # each k contributes one real-valued term (phi dependence is factored out).
    total = jnp.zeros_like(theta)
    for k in range(ki, kf + 1):
        e1 = 2 * k + s - m           # exponent on sin(theta/2)
        e2 = 2 * l + m - s - 2 * k   # exponent on cos(theta/2)
        coeff = (-1) ** k / (
            math.factorial(k)
            * math.factorial(l + m - k)
            * math.factorial(l - s - k)
            * math.factorial(s - m + k)
        )
        total = total + coeff * jnp.sin(theta / 2) ** e1 * jnp.cos(theta / 2) ** e2

    return prefactor * total.astype(jnp.complex128) * jnp.exp(1j * m * phi)
