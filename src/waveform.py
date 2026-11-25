"""
General ringdown waveform model for arbitrary QNM modes (l, m, n).

Public API
----------
load_qnm_table(data_dir, modes)          -> QNMTable
h_lm(t, f, gamma, A, fp, fc, swsh_p, swsh_m, l)
ln_likelihood_general_full(...)          Full-sky (also _jit variant)
ln_likelihood_general_fixed_sky_prealigned(...)   Fixed-sky (also _jit variant)
set_detectors(resp_H, resp_L)
set_detector_locations(rH, rL)
initialize_det_locs()                    -> rH, rL

Optional detector-frame sky coordinates (see the section of that name below) --
sample (zenith kappa, azimuth epsilon) about a detector baseline instead of
(ra, sin dec); the likelihoods are unchanged, convert before calling them:
    build_baseline_frame(r_ref, r_other)               -> b_hat, e1_hat, e2_hat
    kappa_eps_to_radec(cos_kappa, eps, gmst, *frame)   -> ra, dec
    detframe_theta_to_equatorial(theta, gmst, frame, modes)

swsh_fn interface expected by the likelihoods:
    swsh_fn(l: int, m: int, iota: jax scalar) -> complex jax scalar
    Example using src/swsh.py:
        from src.swsh import swsh
        def my_swsh(l, m, iota): return swsh(2, l, m, iota, 0.0)
"""

import os
import numpy as onp
import jax, jax.numpy as jnp
import lal

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


# ─── Geometry ─────────────────────────────────────────────────────────────────

def _src_unit_ecef(ra, dec, gmst):
    H    = ra - gmst
    cosd = jnp.cos(dec)
    return jnp.stack([cosd * jnp.cos(H), cosd * jnp.sin(H), jnp.sin(dec)], axis=-1)


def dt_between(ra, dec, gmst, r_ecef, r_ref_ecef):
    """t_det - t_ref, for any pair of detector locations (ECEF, meters)."""
    s_hat    = _src_unit_ecef(ra, dec, gmst)
    baseline = r_ecef - r_ref_ecef
    return -jnp.sum(baseline * s_hat, axis=-1) / C_SI


def dt_L1_minus_H1(ra, dec, gmst, rL_ecef, rH_ecef):
    return dt_between(ra, dec, gmst, rL_ecef, rH_ecef)


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


# ─── Detector-frame sky coordinates ───────────────────────────────────────────
#
# Optional reparametrization of sky position, for the full-sky search. Instead of
# (ra, sin dec), sample (zenith kappa, azimuth epsilon) relative to a detector
# baseline: kappa is the angle from the baseline vector, epsilon the rotation
# around it. The inter-detector arrival-time difference then depends on kappa
# ALONE and is exactly independent of epsilon, so the
# light-travel-time degeneracy ring -- curved and diagonal in (ra, dec) -- becomes
# a flat strip: narrow in kappa, free in epsilon.
#
# Because the map is a rigid rotation of the sphere it preserves the sphere's
# invariant measure, so epsilon ~ U(0, 2pi) with cos kappa ~ U(-1, 1) is isotropic
# exactly as ra ~ U(0, 2pi) with sin dec ~ U(-1, 1) is, and NO Jacobian correction
# is needed -- the prior box is reused unchanged, only reinterpreted.
#
# These are helpers for building a sampled theta; the likelihoods themselves are
# untouched and still take (ra, sindec). Convert first, then call them as usual.

def build_baseline_frame(r_ref, r_other):
    """
    Right-handed orthonormal basis {e1_hat, e2_hat, b_hat} with b_hat along the
    r_ref -> r_other baseline. Pass detector locations from initialize_det_locs()
    (ECEF, meters), e.g. build_baseline_frame(rH, rL) for the H1->L1 baseline.

    The choice of e1_hat/e2_hat only fixes where epsilon is measured from; any
    consistent choice is a valid coordinate system for the sphere.

    Returns
    -------
    b_hat, e1_hat, e2_hat : (3,) float64 unit vectors, e1_hat x e2_hat = b_hat.
    """
    baseline = r_other - r_ref
    b_hat    = baseline / jnp.linalg.norm(baseline)

    z_axis = jnp.array([0., 0., 1.], dtype=jnp.float64)
    x_axis = jnp.array([1., 0., 0.], dtype=jnp.float64)
    # Gram-Schmidt needs a reference axis not near-parallel to b_hat
    ref_axis = jnp.where(jnp.abs(jnp.dot(b_hat, z_axis)) > 0.9, x_axis, z_axis)

    e1     = ref_axis - jnp.dot(ref_axis, b_hat) * b_hat
    e1_hat = e1 / jnp.linalg.norm(e1)
    e2_hat = jnp.cross(b_hat, e1_hat)
    return b_hat, e1_hat, e2_hat


def kappa_eps_to_radec(cos_kappa, eps, gmst, b_hat, e1_hat, e2_hat):
    """
    (cos kappa, epsilon) -> (ra, dec).

    Builds the ECEF source unit vector
        n_hat = sin(kappa)cos(eps) e1_hat + sin(kappa)sin(eps) e2_hat + cos(kappa) b_hat
    and inverts _src_unit_ecef's convention (H = ra - gmst), so the result feeds
    straight into the existing dt_between()/patterns_for_params() code paths.
    """
    cos_kappa = jnp.clip(cos_kappa, -1., 1.)
    sin_kappa = jnp.sqrt(jnp.clip(1. - cos_kappa**2, 0., 1.))
    n_hat = (sin_kappa * jnp.cos(eps) * e1_hat
             + sin_kappa * jnp.sin(eps) * e2_hat
             + cos_kappa * b_hat)
    n_hat = n_hat / jnp.linalg.norm(n_hat)   # basis is orthonormal only to float precision

    dec = jnp.arcsin(jnp.clip(n_hat[2], -1., 1.))
    ra  = jnp.mod(gmst + jnp.arctan2(n_hat[1], n_hat[0]), 2 * jnp.pi)
    return ra, dec


def detframe_theta_to_equatorial(theta, gmst, frame, modes):
    """
    Rotate a detector-frame parameter vector into the equatorial one the
    likelihoods expect.

    Reads slots [offset+1], [offset+2] as (epsilon, cos kappa) and returns a copy
    with them replaced by (ra, sin dec), where offset = 2 + 2*len(modes) -- i.e.
    slots 7,8 for a 2-mode fit and 5,6 for a 1-mode fit. Every other entry is
    untouched, so the result is exactly the layout
    ln_likelihood_general_full() documents:
        [M, chi, A_0, P_0, ..., cosiota, ra, sindec, psi]

    Parameters
    ----------
    theta : array of length 2 + 2*K + 4, with (epsilon, cos kappa) in the sky slots.
    frame : (b_hat, e1_hat, e2_hat) from build_baseline_frame().
    modes : tuple of (l, m, n) tuples -- only its length is used.
    """
    b_hat, e1_hat, e2_hat = frame
    offset = 2 + 2 * len(modes)
    eps, cos_kappa = theta[offset + 1], theta[offset + 2]
    ra, dec = kappa_eps_to_radec(cos_kappa, eps, gmst, b_hat, e1_hat, e2_hat)
    return theta.at[offset + 1].set(ra).at[offset + 2].set(jnp.sin(dec))


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


def _amplitude_hierarchy_log_constraint(params, modes):
    """
    Amplitude hierarchy across (l,m) groups, per overtone level n.

    Within each n, consecutive modes (differing in l,m) must satisfy
    A[j] < A[j_prev]. Overtones of the same (l,m) are unconstrained.
    Modes must be listed in decreasing dominance order within each n-level.
    Detector-independent -- shared by all ln_likelihood_general* variants.
    """
    from collections import defaultdict
    _n_groups = defaultdict(list)
    for _j, (_, _, _n) in enumerate(modes):
        _n_groups[_n].append(_j)
    log_constraint = jnp.asarray(0.0, dtype=jnp.float64)
    for _indices in _n_groups.values():
        for _k in range(1, len(_indices)):
            _j      = _indices[_k]
            _j_prev = _indices[_k - 1]
            log_constraint = log_constraint + jnp.where(
                params[2 + 2*_j] <= params[2 + 2*_j_prev], 0.0, -jnp.inf
            )
    return log_constraint


def ln_likelihood_general(data1, data2, params, L1, L2,
                           qnm_table, modes,
                           Fp1, Fc1, Fp2, Fc2,
                           swsh_p, swsh_m,
                           T, srate, t0, t2_minus_t1, ref_det,
                           skip_hierarchy=False):
    """
    Log-likelihood for an arbitrary set of QNM modes.

    Parameters
    ----------
    data1, data2 : (N,) float64
        Time-domain strain from detector 1 (H1) and 2 (L1).
    params : array of length 2 + 2*K
        [M [M_sun], chi, A_0, P_0 [rad], A_1, P_1, ..., A_{K-1}, P_{K-1}]
        Amplitudes are dimensionless; internally scaled by reference_amplitude.
    L1, L2 : (N,N) lower-triangular Cholesky factors of the noise covariance matrices.
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

    # skip_hierarchy=True disables the constraint for trans-dimensional binary-mask
    # sampling, where inactive modes are zeroed and the chain would otherwise
    # break (A_active <= A_inactive=0 -> -inf).
    if not skip_hierarchy:
        return _amplitude_hierarchy_log_constraint(params, modes) + (
            -0.5 * (jnp.vdot(y1, y1).real + jnp.vdot(y2, y2).real))

    return -0.5 * (jnp.vdot(y1, y1).real + jnp.vdot(y2, y2).real)


def ln_likelihood_general_full(dataH, dataL, params, gmst,
                                L_H, L_L, qnm_table, modes, swsh_fn,
                                T, srate, t0, ref_det='H1',
                                skip_hierarchy=False):
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
        skip_hierarchy=skip_hierarchy,
    )


# ─── Pre-aligned likelihoods (no inter-detector template shift) ───────────────
#
# ln_likelihood_general() and ln_likelihood_general_full() above (Full-sky) assume
# data1/data2 were sliced from one SHARED reference window (anchored to ref_det's
# own arrival time), and they compensate by shifting the non-ref detector's
# template by the analytic inter-detector delay (t2_minus_t1). The functions
# below (Fixed-sky) are for the other, equally valid convention: data1/data2
# were each ALREADY sliced starting at that detector's own individually-computed
# true arrival time -- possible whenever the sky location is known, since then
# each detector's arrival time is exactly computable up front. In that case the
# delay is already absorbed by the per-detector windowing, so no template shift
# should be applied -- applying both conventions at once double-counts the delay.
# Use the shared-window likelihoods with data cut at a shared tref/ref_det, or the
# _prealigned ones with data cut per-detector at its own arrival time -- never mix
# the two.

def ln_likelihood_general_prealigned(data1, data2, params, L1, L2,
                                      qnm_table, modes,
                                      Fp1, Fc1, Fp2, Fc2,
                                      swsh_p, swsh_m,
                                      T, srate, t0,
                                      skip_hierarchy=False):
    """
    Log-likelihood for pre-aligned detector data (see module note above):
    data1/data2 were each sliced starting at that detector's OWN true
    arrival time, so both templates share the identical local time grid
    t+t0 -- there is no ref_det/t2_minus_t1 shift here.

    Parameters
    ----------
    data1, data2 : (N,) float64
        Time-domain strain from detector 1 and 2, each pre-sliced starting at
        its own true arrival time.
    params : array of length 2 + 2*K
        [M [M_sun], chi, A_0, P_0 [rad], A_1, P_1, ..., A_{K-1}, P_{K-1}]
    L1, L2 : (N,N) lower-triangular Cholesky factors of the noise covariance matrices.
    qnm_table : QNMTable  **static for jit**
    modes : tuple of (l, m, n) tuples  **static for jit**
    Fp1, Fc1, Fp2, Fc2 : float64 scalars, antenna pattern values.
    swsh_p, swsh_m : (K,) complex128, spin-weighted spherical harmonics.
    T, srate : float  **static for jit**
    t0 : float, analysis window start time relative to each detector's own
        trigger [s].

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

        waveform1 += h_lm(t + t0, f, gamma, A, Fp1, Fc1, swsh_p[j], swsh_m[j], l)
        waveform2 += h_lm(t + t0, f, gamma, A, Fp2, Fc2, swsh_p[j], swsh_m[j], l)

    y1 = jax.scipy.linalg.solve_triangular(L1, jnp.asarray(waveform1 - data1), lower=True)
    y2 = jax.scipy.linalg.solve_triangular(L2, jnp.asarray(waveform2 - data2), lower=True)

    if not skip_hierarchy:
        return _amplitude_hierarchy_log_constraint(params, modes) + (
            -0.5 * (jnp.vdot(y1, y1).real + jnp.vdot(y2, y2).real))

    return -0.5 * (jnp.vdot(y1, y1).real + jnp.vdot(y2, y2).real)


def ln_likelihood_general_fixed_sky_prealigned(dataH, dataL, params, gmst,
                                                L_H, L_L, qnm_table, modes, swsh_fn,
                                                ra, dec,
                                                T, srate, t0,
                                                skip_hierarchy=False):
    """
    Fixed-sky log-likelihood for PRE-ALIGNED data (see module note above):
    use when dataH/dataL were each sliced starting at their own
    individually-computed true arrival time (valid since ra, dec are known),
    rather than from one shared reference window. There is no ref_det or t2_minus_t1
    here -- the inter-detector delay is already absorbed by the per-detector
    windowing. ra, dec are still used for the antenna-pattern (Fp, Fc)
    projection, which is independent of the windowing/timing convention.

    Parameters
    ----------
    params : array of length 2 + 2*K + 2
        [M, chi, A_0, P_0, ..., A_{K-1}, P_{K-1}, cosiota, psi]
        ra and dec are NOT in params -- pass them as keyword arguments below.
    gmst : float  **static for jit**, Greenwich mean sidereal time [rad].
    qnm_table : QNMTable  **static for jit**.
    modes : tuple of (l, m, n) tuples  **static for jit**.
    swsh_fn : callable  **static for jit**
        Signature: swsh_fn(l: int, m: int, iota: jax scalar) -> complex scalar.
    ra : float  **static for jit**, right ascension [rad].
    dec : float  **static for jit**, declination [rad].
    T, srate : float  **static for jit**.
    t0 : float, analysis window start time relative to each detector's own
        trigger [s].

    Returns
    -------
    float, log-likelihood value.
    """
    n_modes = len(modes)
    p0      = params[:2 + 2*n_modes]
    cosiota, psi = params[2 + 2*n_modes], params[3 + 2*n_modes]

    iota = jnp.arccos(cosiota)

    FpH, FcH, FpL, FcL = patterns_for_params(ra, dec, psi, gmst)

    swsh_p = jnp.stack([swsh_fn(l,  m, iota) for l, m, n in modes])
    swsh_m = jnp.stack([swsh_fn(l, -m, iota) for l, m, n in modes])

    return ln_likelihood_general_prealigned(
        dataH, dataL, p0, L_H, L_L,
        qnm_table, modes,
        FpH, FcH, FpL, FcL,
        swsh_p, swsh_m,
        T, srate, t0,
        skip_hierarchy=skip_hierarchy,
    )


# ─── JIT-compiled versions ────────────────────────────────────────────────────

ln_likelihood_general_full_jit = jax.jit(
    ln_likelihood_general_full,
    static_argnames=('gmst', 'qnm_table', 'modes', 'swsh_fn', 'T', 'srate', 'ref_det', 'skip_hierarchy'),
)

ln_likelihood_general_fixed_sky_prealigned_jit = jax.jit(
    ln_likelihood_general_fixed_sky_prealigned,
    static_argnames=('gmst', 'ra', 'dec', 'qnm_table', 'modes', 'swsh_fn', 'T', 'srate', 'skip_hierarchy'),
)

