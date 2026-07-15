"""
General ringdown waveform model for arbitrary QNM modes (l, m, n).

Public API
----------
load_qnm_table(data_dir, modes)          -> QNMTable
h_lm(t, f, gamma, A, fp, fc, swsh_p, swsh_m, l)
ln_likelihood_general(...)               (also _jit variant)
ln_likelihood_general_full(...)          (also _jit variant)
cov_resp(srate, T, psdH, psdL, factor)  -> L_H, L_L, resp_H, resp_L
set_detectors(resp_H, resp_L)
set_detector_locations(rH, rL)
initialize_det_locs()                    -> rH, rL

swsh_fn interface expected by ln_likelihood_general_full:
    swsh_fn(l: int, m: int, iota: jax scalar) -> complex jax scalar
    Example using src/swsh.py:
        from src.swsh import swsh
        def my_swsh(l, m, iota): return swsh(2, l, m, iota, 0.0)
"""

import os
import numpy as onp
import jax, jax.numpy as jnp
import lal
from scipy.linalg import toeplitz

jax.config.update("jax_enable_x64", True)

# ─── Constants ────────────────────────────────────────────────────────────────

reference_amplitude = jnp.asarray(1e-20, dtype=jnp.float64)
MTSUN = jnp.float64(lal.MTSUN_SI)
C_SI  = jnp.float64(lal.C_SI)

# ─── Detector state ───────────────────────────────────────────────────────────
# Call set_detectors() and set_detector_locations() once at notebook startup.

_det_H = None
_det_L = None
_rH    = None
_rL    = None


def set_detectors(resp_H, resp_L):
    global _det_H, _det_L
    _det_H = resp_H
    _det_L = resp_L


def set_detector_locations(rH, rL):
    global _rH, _rL
    _rH = rH
    _rL = rL


def initialize_det_locs():
    H1 = lal.CachedDetectors[lal.LALDetectorIndexLHODIFF]
    L1 = lal.CachedDetectors[lal.LALDetectorIndexLLODIFF]
    rH = jnp.array([H1.location[0], H1.location[1], H1.location[2]], dtype=jnp.float64)
    rL = jnp.array([L1.location[0], L1.location[1], L1.location[2]], dtype=jnp.float64)
    return rH, rL


# ─── Utilities ────────────────────────────────────────────────────────────────

def Ntime(srate, T):
    N = int(round(T * srate))
    return N + (N % 2)


def interp1d_jax(xp, fp):
    xp = jnp.asarray(xp, dtype=jnp.float64)
    fp = jnp.asarray(fp, dtype=jnp.float64)
    def _interp(x):
        x   = jnp.asarray(x, dtype=jnp.float64)
        idx = jnp.clip(jnp.searchsorted(xp, x, side="right"), 1, xp.size - 1)
        x0  = xp[idx - 1]; x1 = xp[idx]
        y0  = fp[idx - 1]; y1 = fp[idx]
        w   = (x - x0) / (x1 - x0)
        return y0 * (1.0 - w) + y1 * w
    return _interp


def ACFs(srate, T, psdH, psdL, factor=10):
    dt  = 1.0 / srate
    NN  = Ntime(srate, factor * T)
    N   = Ntime(srate, T)
    freqs = jnp.fft.rfftfreq(NN, d=dt)
    df  = freqs[1] - freqs[0]
    rhoH = 0.5 * jnp.fft.irfft(psdH(freqs)).real * df * NN
    rhoL = 0.5 * jnp.fft.irfft(psdL(freqs)).real * df * NN
    return rhoH[:N], rhoL[:N]


def cov_resp(srate, T, psdH, psdL, factor):
    """
    Build Cholesky factors and detector response tensors.

    Returns
    -------
    L_H, L_L   : Cholesky factors for whitening
    resp_H, resp_L : (3,3) detector response tensors
    """
    rhoH, rhoL = ACFs(srate=srate, T=T, psdH=psdH, psdL=psdL, factor=factor)
    L_H = jnp.linalg.cholesky(toeplitz(rhoH))
    L_L = jnp.linalg.cholesky(toeplitz(rhoL))
    resp_H = jnp.asarray(lal.CachedDetectors[lal.LALDetectorIndexLHODIFF].response, dtype=jnp.float64)
    resp_L = jnp.asarray(lal.CachedDetectors[lal.LALDetectorIndexLLODIFF].response, dtype=jnp.float64)
    return L_H, L_L, resp_H, resp_L


# ─── Geometry ─────────────────────────────────────────────────────────────────

def _src_unit_ecef(ra, dec, gmst):
    H    = ra - gmst
    cosd = jnp.cos(dec)
    return jnp.stack([cosd * jnp.cos(H), cosd * jnp.sin(H), jnp.sin(dec)], axis=-1)


def dt_L1_minus_H1(ra, dec, gmst, rL_ecef, rH_ecef):
    s_hat    = _src_unit_ecef(ra, dec, gmst)
    baseline = rL_ecef - rH_ecef
    return -jnp.sum(baseline * s_hat, axis=-1) / C_SI


def antenna_pattern_jax(ra, dec, psi, gmst, resp):
    gha    = gmst - ra
    cosgha = jnp.cos(gha);  singha = jnp.sin(gha)
    cosdec = jnp.cos(dec);  sindec = jnp.sin(dec)
    cospsi = jnp.cos(psi);  sinpsi = jnp.sin(psi)

    x = jnp.stack([-cospsi*singha - sinpsi*cosgha*sindec,
                   -cospsi*cosgha + sinpsi*singha*sindec,
                    sinpsi*cosdec], axis=-1)
    y = jnp.stack([ sinpsi*singha - cospsi*cosgha*sindec,
                    sinpsi*cosgha + cospsi*singha*sindec,
                    cospsi*cosdec], axis=-1)

    dx = jnp.einsum("ij,...j->...i", resp, x)
    dy = jnp.einsum("ij,...j->...i", resp, y)
    return jnp.sum(x*dx - y*dy, axis=-1), jnp.sum(x*dy + y*dx, axis=-1)


def patterns_for_params(ra, dec, psi, gmst):
    FpH, FcH = antenna_pattern_jax(ra, dec, psi, gmst, _det_H)
    FpL, FcL = antenna_pattern_jax(ra, dec, psi, gmst, _det_L)
    return FpH, FcH, FpL, FcL


# ─── QNM data ─────────────────────────────────────────────────────────────────

def qnm_path(data_dir, l, m, n):
    """
    Construct file path for a QNM data file.

    n : int, overtone index in physics convention (0 = fundamental).
        Maps to Berti's file convention: n_file = n + 1.
    m : int, azimuthal number; negative m uses the 'mm{|m|}' filename suffix.
    """
    m_str = f"mm{abs(m)}" if m < 0 else f"m{m}"
    return os.path.join(data_dir, f"l{l}", f"n{n+1}l{l}{m_str}.dat")


class QNMTable:
    """
    Hashable dict-like container for QNM interpolators.

    load_qnm_table() returns this type.  Pass the same instance across
    repeated jit calls — identity-based hashing avoids unnecessary retracing.

    Usage
    -----
    qnm = load_qnm_table(data_dir, modes)
    omega_r_fn, omega_i_fn = qnm[(l, m, n)]
    """
    def __init__(self, table: dict):
        self._table = table

    def __getitem__(self, key):
        return self._table[key]

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    def __repr__(self):
        return f"QNMTable({list(self._table.keys())})"


def load_qnm(data_dir, l, m, n):
    """Load interpolators for a single mode. Returns (omega_r_fn, omega_i_fn)."""
    data    = onp.genfromtxt(qnm_path(data_dir, l, m, n))
    chi     = jnp.asarray(data[:, 0], dtype=jnp.float64)
    omega_r = interp1d_jax(chi, jnp.asarray(data[:, 1], dtype=jnp.float64))
    omega_i = interp1d_jax(chi, jnp.asarray(data[:, 2], dtype=jnp.float64))
    return omega_r, omega_i


def load_qnm_table(data_dir, modes):
    """
    Load QNM interpolators for a collection of modes.

    Parameters
    ----------
    data_dir : str
        Root data directory containing l2/, l3/, ... subdirectories.
    modes : iterable of (l, m, n) tuples
        Any combination with 2 <= l <= 7, |m| <= l, n >= 0.
        n=0 is the fundamental mode (first overtone n=1, etc.).

    Returns
    -------
    QNMTable
        Hashable mapping (l, m, n) -> (omega_r_fn, omega_i_fn).
        Designed for use as a jit static_argname.
    """
    return QNMTable({(l, m, n): load_qnm(data_dir, l, m, n) for l, m, n in modes})


# ─── Waveform ─────────────────────────────────────────────────────────────────

def h_lm(t, f, gamma, A, fp, fc, swsh_p, swsh_m, l):
    """
    Single-mode ringdown waveform projected onto one detector.

    The formula is the standard QNM decomposition:

        h = exp(-gamma*t) * [ sY_{l,m}  * A       * exp( i 2pi f t)
                            + (-1)^l
                            * sY_{l,-m} * conj(A)  * exp(-i 2pi f t) ]
        h_detector = fp * Re(h) + fc * Im(h)

    Parameters
    ----------
    t      : (N,) float64 array, time samples [s]
    f      : float64 scalar, oscillation frequency [Hz]
    gamma  : float64 scalar, damping rate [1/s], positive (signal decays)
    A      : complex128 scalar, amplitude*exp(i*phase)*reference_amplitude
    fp, fc : float64 scalars, detector + and x antenna pattern values
    swsh_p : complex128, spin-weighted spherical harmonic sY_{l, m}(iota, 0)
    swsh_m : complex128, spin-weighted spherical harmonic sY_{l,-m}(iota, 0)
    l      : int (Python) — (-1)^l is a compile-time constant in jitted code
    """
    sign  = (-1) ** l
    h_tmp = jnp.exp(-gamma * t) * (
          swsh_p * A               * jnp.exp( 1j * 2 * jnp.pi * f * t)
        + sign   * swsh_m * jnp.conj(A) * jnp.exp(-1j * 2 * jnp.pi * f * t)
    )
    return fp * jnp.real(h_tmp) + fc * jnp.imag(h_tmp)


# ─── Likelihoods ──────────────────────────────────────────────────────────────

def simulate_ringdown(params, qnm_table, modes,
                       Fp1, Fc1, Fp2, Fc2, swsh_p, swsh_m,
                       T, srate, t0, t2_minus_t1, ref_det='H1'):
    """
    Generate noiseless ringdown waveforms for both detectors.

    Same parameter layout and static-arg conventions as ln_likelihood_general.
    Use for injections, forward-model checks, and SNR calculations.

    Parameters
    ----------
    params : array [M, chi, A_0, P_0, ..., A_{K-1}, P_{K-1}]  (2 + 2K)
    (all other args identical to ln_likelihood_general)

    Returns
    -------
    waveform1, waveform2 : (N,) float64, time-domain strain at det1 and det2
    """
    params = jnp.asarray(params, dtype=jnp.float64)
    M, chi = params[0], params[1]
    prefactor_r = 1.0 / (2.0 * jnp.pi * M * MTSUN)
    prefactor_i = 1.0 / (M * MTSUN)

    N  = Ntime(srate, T)
    dt = jnp.asarray(1.0 / srate, dtype=jnp.float64)
    t  = jnp.arange(N, dtype=jnp.float64) * dt

    waveform1 = jnp.zeros(N, dtype=jnp.float64)
    waveform2 = jnp.zeros(N, dtype=jnp.float64)

    for j, (l, m, n) in enumerate(modes):
        omega_r_fn, omega_i_fn = qnm_table[(l, m, n)]
        f     = prefactor_r * omega_r_fn(chi)
        gamma = -prefactor_i * omega_i_fn(chi)
        A     = params[2 + 2*j] * jnp.exp(1j * params[3 + 2*j]) * reference_amplitude

        if ref_det == 'H1':
            waveform1 += h_lm(t + t0,               f, gamma, A, Fp1, Fc1, swsh_p[j], swsh_m[j], l)
            waveform2 += h_lm(t + t0 - t2_minus_t1, f, gamma, A, Fp2, Fc2, swsh_p[j], swsh_m[j], l)
        else:
            waveform1 += h_lm(t + t0 + t2_minus_t1, f, gamma, A, Fp1, Fc1, swsh_p[j], swsh_m[j], l)
            waveform2 += h_lm(t + t0,               f, gamma, A, Fp2, Fc2, swsh_p[j], swsh_m[j], l)

    return waveform1, waveform2


simulate_ringdown_jit = jax.jit(
    simulate_ringdown,
    static_argnames=('qnm_table', 'modes', 'T', 'srate', 'ref_det'),
)


def ln_likelihood_general(data1, data2, params, L1, L2,
                           qnm_table, modes,
                           Fp1, Fc1, Fp2, Fc2,
                           swsh_p, swsh_m,
                           T, srate, t0, t2_minus_t1, ref_det):
    """
    Log-likelihood for an arbitrary set of QNM modes.

    Parameters
    ----------
    data1, data2 : (N,) float64
        Time-domain strain from detector 1 (H1) and 2 (L1).
    params : array of length 2 + 2*K
        [M [M_sun], chi, A_0, P_0 [rad], A_1, P_1, ..., A_{K-1}, P_{K-1}]
        Amplitudes are dimensionless; internally scaled by reference_amplitude.
    L1, L2 : (N,N) lower-triangular Cholesky factors from cov_resp().
    qnm_table : QNMTable  **static for jit**
    modes : tuple of (l, m, n) tuples  **static for jit**
        Order must match the (A_j, P_j) pairs in params.
    Fp1, Fc1, Fp2, Fc2 : float64 scalars, antenna pattern values.
    swsh_p : (K,) complex128, sY_{l, m}(iota, 0) for each mode.
    swsh_m : (K,) complex128, sY_{l,-m}(iota, 0) for each mode.
    T, srate : float  **static for jit**
    t0 : float, analysis window start time relative to trigger [s].
    t2_minus_t1 : float, t_L1 - t_H1 [s].
    ref_det : 'H1' or 'L1'  **static for jit**

    Returns
    -------
    float, log-likelihood = -0.5*(||y1||^2 + ||y2||^2) after whitening.
    """
    params = jnp.asarray(params, dtype=jnp.float64)
    M, chi = params[0], params[1]
    prefactor_r = 1.0 / (2.0 * jnp.pi * M * MTSUN)
    prefactor_i = 1.0 / (M * MTSUN)

    N  = Ntime(srate, T)
    dt = jnp.asarray(1.0 / srate, dtype=jnp.float64)
    t  = jnp.arange(N, dtype=jnp.float64) * dt

    waveform1 = jnp.zeros(N, dtype=jnp.float64)
    waveform2 = jnp.zeros(N, dtype=jnp.float64)

    for j, (l, m, n) in enumerate(modes):
        omega_r_fn, omega_i_fn = qnm_table[(l, m, n)]
        f     = prefactor_r * omega_r_fn(chi)
        gamma = -prefactor_i * omega_i_fn(chi)          # positive: omega_i < 0 in data files
        A     = params[2 + 2*j] * jnp.exp(1j * params[3 + 2*j]) * reference_amplitude

        if ref_det == 'H1':
            waveform1 += h_lm(t + t0,               f, gamma, A, Fp1, Fc1, swsh_p[j], swsh_m[j], l)
            waveform2 += h_lm(t + t0 - t2_minus_t1, f, gamma, A, Fp2, Fc2, swsh_p[j], swsh_m[j], l)
        else:
            waveform1 += h_lm(t + t0 + t2_minus_t1, f, gamma, A, Fp1, Fc1, swsh_p[j], swsh_m[j], l)
            waveform2 += h_lm(t + t0,               f, gamma, A, Fp2, Fc2, swsh_p[j], swsh_m[j], l)

    y1 = jax.scipy.linalg.solve_triangular(L1, jnp.asarray(waveform1 - data1), lower=True)
    y2 = jax.scipy.linalg.solve_triangular(L2, jnp.asarray(waveform2 - data2), lower=True)

    # Amplitude hierarchy across (l,m) groups, per overtone level n.
    # Within each n, consecutive modes (differing in l,m) must satisfy
    # A[j] < A[j_prev]. Overtones of the same (l,m) are unconstrained.
    # Modes must be listed in decreasing dominance order within each n-level.
    from collections import defaultdict
    _n_groups = defaultdict(list)
    for _j, (_l, _m, _n) in enumerate(modes):
        _n_groups[_n].append(_j)
    log_constraint = jnp.asarray(0.0, dtype=jnp.float64)
    for _indices in _n_groups.values():
        for _k in range(1, len(_indices)):
            _j      = _indices[_k]
            _j_prev = _indices[_k - 1]
            log_constraint = log_constraint + jnp.where(
                params[2 + 2*_j] < params[2 + 2*_j_prev], 0.0, -jnp.inf
            )

    return log_constraint + (-0.5 * (jnp.vdot(y1, y1).real + jnp.vdot(y2, y2).real))


def ln_likelihood_general_full(dataH, dataL, params, gmst,
                                L_H, L_L, qnm_table, modes, swsh_fn,
                                T, srate, t0, ref_det='H1'):
    """
    Full-sky log-likelihood: sky location is sampled as part of params.

    Parameters
    ----------
    params : array of length 2 + 2*K + 4
        [M, chi, A_0, P_0, ..., A_{K-1}, P_{K-1}, cosiota, ra, sindec, psi]
    gmst : float  **static for jit**, Greenwich mean sidereal time [rad].
    qnm_table : QNMTable  **static for jit**.
    modes : tuple of (l, m, n) tuples  **static for jit**.
    swsh_fn : callable  **static for jit**
        Signature: swsh_fn(l: int, m: int, iota: jax scalar) -> complex scalar.
        Must be JAX-traceable in iota; l and m are Python ints.
        Canonical usage with src/swsh.py:
            from src.swsh import swsh
            def my_swsh(l, m, iota): return swsh(2, l, m, iota, 0.0)
    T, srate : float  **static for jit**.
    ref_det : 'H1' or 'L1'  **static for jit**.

    Returns
    -------
    float, log-likelihood value.
    """
    n_modes = len(modes)
    p0      = params[:2 + 2*n_modes]
    offset  = 2 + 2*n_modes
    cosiota, ra, sindec, psi = params[offset], params[offset+1], params[offset+2], params[offset+3]

    iota = jnp.arccos(cosiota)
    dec  = jnp.arcsin(sindec)

    FpH, FcH, FpL, FcL = patterns_for_params(ra, dec, psi, gmst)
    t2_minus_t1         = dt_L1_minus_H1(ra, dec, gmst, _rL, _rH)

    # Evaluate spherical harmonics at the current (traced) inclination angle.
    # l and m are Python ints from the static modes tuple, so swsh_fn can
    # branch on them at trace time; iota is a traced JAX value.
    swsh_p = jnp.stack([swsh_fn(l,  m, iota) for l, m, n in modes])
    swsh_m = jnp.stack([swsh_fn(l, -m, iota) for l, m, n in modes])

    return ln_likelihood_general(
        dataH, dataL, p0, L_H, L_L,
        qnm_table, modes,
        FpH, FcH, FpL, FcL,
        swsh_p, swsh_m,
        T, srate, t0, t2_minus_t1, ref_det,
    )


def ln_likelihood_general_fixed_sky(dataH, dataL, params, gmst,
                                     L_H, L_L, qnm_table, modes, swsh_fn,
                                     ra, dec,
                                     T, srate, t0, ref_det='H1'):
    """
    Fixed-sky log-likelihood: ra and dec are constants, not sampled parameters.

    Use this when the sky location is pinned to a known value (e.g. from a
    prior IMR analysis).  Compared to the full-sky version, the parameter
    space shrinks by 2 and t2_minus_t1 is constant-folded by XLA at compile
    time, so each likelihood evaluation is slightly cheaper.

    Parameters
    ----------
    params : array of length 2 + 2*K + 2
        [M, chi, A_0, P_0, ..., A_{K-1}, P_{K-1}, cosiota, psi]
        ra and dec are NOT in params — pass them as keyword arguments below.
    gmst : float  **static for jit**, Greenwich mean sidereal time [rad].
    qnm_table : QNMTable  **static for jit**.
    modes : tuple of (l, m, n) tuples  **static for jit**.
    swsh_fn : callable  **static for jit**
        Signature: swsh_fn(l: int, m: int, iota: jax scalar) -> complex scalar.
    ra : float  **static for jit**, right ascension [rad].
    dec : float  **static for jit**, declination [rad].
    T, srate : float  **static for jit**.
    ref_det : 'H1' or 'L1'  **static for jit**.

    Returns
    -------
    float, log-likelihood value.
    """
    n_modes = len(modes)
    p0      = params[:2 + 2*n_modes]
    cosiota, psi = params[2 + 2*n_modes], params[3 + 2*n_modes]

    iota = jnp.arccos(cosiota)

    FpH, FcH, FpL, FcL = patterns_for_params(ra, dec, psi, gmst)
    t2_minus_t1         = dt_L1_minus_H1(ra, dec, gmst, _rL, _rH)

    swsh_p = jnp.stack([swsh_fn(l,  m, iota) for l, m, n in modes])
    swsh_m = jnp.stack([swsh_fn(l, -m, iota) for l, m, n in modes])

    return ln_likelihood_general(
        dataH, dataL, p0, L_H, L_L,
        qnm_table, modes,
        FpH, FcH, FpL, FcL,
        swsh_p, swsh_m,
        T, srate, t0, t2_minus_t1, ref_det,
    )


# ─── JIT-compiled versions ────────────────────────────────────────────────────

ln_likelihood_general_jit = jax.jit(
    ln_likelihood_general,
    static_argnames=('qnm_table', 'modes', 'T', 'srate', 'ref_det'),
)

ln_likelihood_general_full_jit = jax.jit(
    ln_likelihood_general_full,
    static_argnames=('gmst', 'qnm_table', 'modes', 'swsh_fn', 'T', 'srate', 'ref_det'),
)

ln_likelihood_general_fixed_sky_jit = jax.jit(
    ln_likelihood_general_fixed_sky,
    static_argnames=('gmst', 'ra', 'dec', 'qnm_table', 'modes', 'swsh_fn', 'T', 'srate', 'ref_det'),
)
