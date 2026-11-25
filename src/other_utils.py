import jax, jax.numpy as jnp
import numpy as onp
import scipy.signal as sig
from scipy.linalg import cholesky, toeplitz

jax.config.update("jax_enable_x64", True) #double precision


# bandpass_ds adapted from ringdown:
# https://github.com/maxisi/ringdown
#
# Original authors:
# Maximiliano Isi <max.isi@ligo.org>
# Will M. Farr <will.farr@stonybrook.edu>
#
# Original code licensed under the MIT License.
def bandpass_ds(dat, t0: float | None = None,
                ds: int | None = None,
                f_min: float | None = None,
                f_max: float | None = None,
                trim: float = 0.25,
                digital_filter: bool = True,
                remove_mean: bool = True,
                decimate_kws: dict | None = None,
                slice_left: float | None = None,
                slice_right: float | None = None):
    """Condition data.

    Arguments
    ---------
    t0 : float
        target time to be preserved after downsampling.
    ds : int
        decimation factor for downsampling.
    f_min : float
        lower frequency for high passing.
    f_max : float
        higher frequency for low passing.
    trim : float
        fraction of data to trim from edges after conditioning, to avoid
        spectral issues if filtering (default 0.25).
    digital_filter : bool
        apply digital antialiasing filter by discarding Fourier components
        higher than Nyquist; otherwise, filter through
        :func:`scipy.signal.decimate`.(default True).
    remove_mean : bool
        explicitly remove mean from time series after conditioning (default
        True).
    decimate_kws : dict
        options for decimation function.
    slice_left : float
        number of seconds before t0 to slice the strain data, e.g. to avoid
        NaNs
    slice_right : float
        number of seconds after t0 to slice the strain data, e.g. to avoid
        NaNs

    Returns
    -------
    cond_data : Data
        conditioned data object.
    """

    raw_data = dat.value
    raw_time = dat.times.value
    delta_t = raw_time[1] - raw_time[0]

    decimate_kws = decimate_kws or {}

    ds = int(ds or 1)
    if t0 is not None:
        if t0 < raw_time[0] or t0 > raw_time[-1]:
            raise ValueError(f"t0 must be within the time series: {t0} "
                                f"not in [{raw_time[0]}, {raw_time[-1]}]")
        i = onp.argmin(abs(raw_time - t0))
        raw_time = onp.roll(raw_time, -(i % ds))
        raw_data = onp.roll(raw_data, -(i % ds))

    fny = 0.5/(raw_time[1] - raw_time[0])
    # Filter
    if f_min and not f_max:
        b, a = sig.butter(4, f_min/fny, btype='highpass', output='ba')
    elif f_max and not f_min:
        b, a = sig.butter(4, f_max/fny, btype='lowpass', output='ba')
    elif f_min and f_max:
        b, a = sig.butter(4, (f_min/fny, f_max/fny), btype='bandpass',
                            output='ba')

    if f_max == fny:
        print("f_max is at Nyquist frequency but filter will "
                        "be applied anyway; to prevent this, set f_max to"
                        " None (default)")

    if f_min or f_max:
        cond_data = sig.filtfilt(b, a, raw_data)
    else:
        cond_data = raw_data

    # Decimate
    if ds and ds > 1:
        if digital_filter:
            # fft data
            w = sig.windows.tukey(len(cond_data), trim)
            cond_data_fd = onp.fft.rfft(cond_data*w)
            freq = onp.fft.rfftfreq(len(cond_data), delta_t)
            # throw away frequencies
            cond_data_fd[freq > fny/ds] = 0
            # ifft and downsample
            cond_data = onp.fft.irfft(cond_data_fd)
            cond_data = cond_data[::ds]
        else:
            cond_data = sig.decimate(cond_data, ds, zero_phase=True,
                                        **decimate_kws)
        if raw_time is not None:
            cond_time = raw_time[::ds]
    elif raw_time is not None:
        cond_time = raw_time

    N = len(cond_data)
    istart = int(round(trim*N))
    iend = int(round((1-trim)*N))

    cond_time = cond_time[istart:iend]
    cond_data = cond_data[istart:iend]

    if remove_mean:
        cond_data -= onp.mean(cond_data)

    return cond_time, cond_data

def analysis_data(data, times, tstart, n_analyze):
    """
    n_analyze samples starting at the sample CLOSEST to tstart.

    bandpass_ds already shifts the decimated grid so that one sample falls on
    tstart to within a fraction of a sample. Taking the first sample >= tstart
    instead (the previous behaviour) discarded that sample whenever it fell a
    few microseconds before tstart, i.e. about half of the time, and the
    segment then started one full sample (244 us at 4096 Hz, up to ~2 M for
    light remnants) after the time the template assumes as t = 0.
    """
    times = onp.asarray(times)
    tstart_idx = int(onp.argmin(onp.abs(times - tstart)))
    if tstart_idx + n_analyze > len(times):
        raise ValueError(f"analysis window of {n_analyze} samples from index {tstart_idx} exceeds the data ({len(times)} samples)")
    return times[tstart_idx : tstart_idx + n_analyze], data[tstart_idx : tstart_idx + n_analyze]


# ──────────────────────────────────────────────────────────────────────────────
# Noise covariance from the time-domain ACF of a long conditioned stretch
# (td_acf, cholesky_from_acf)
#
# The covariance of an analysis segment of N samples only needs the ACF at
# lags < N dt (0.2 s). Estimating it directly in the time domain from a long
# off-source stretch conditioned exactly like the analysis segment (same
# bandpass_ds) uses every pair of samples, needs no window, no PSD grid and no
# interpolation, and includes the conditioning filter automatically. The route
# Welch PSD -> interpolation -> irfft -> truncation is biased where the PSD
# has narrow lines or steep slopes: with 1-s Hann segments the whitened power
# of off-source data is well below 1 under ~100 Hz. The edges of the stretch
# are tapered, otherwise the discontinuity at the ends leaks across the whole
# spectrum.
#
# The stretch must not contain the event. No low-frequency floor and no
# filtered template are needed: with the high-pass applied before the cut, the
# unfiltered template matches the data at the truth as long as
# f_min <~ f_ringdown / 3; without a high-pass (f_min = None) the covariance is
# still well conditioned in float64 (cond ~ 1e11).
# ──────────────────────────────────────────────────────────────────────────────
def td_acf(x, n_lags, taper=0.1):
    """
    Biased time-domain estimate of the autocorrelation of `x` for lags
    0 .. n_lags-1 (mean removed, zero-padded FFT, Tukey taper of fraction
    `taper` at the edges). The biased estimator guarantees a positive
    semi-definite Toeplitz matrix.
    """
    x = onp.asarray(x, dtype=float)
    x = x - onp.mean(x)
    m = len(x)
    if n_lags > m:
        raise ValueError(f"n_lags={n_lags} larger than the stretch ({m} samples)")
    w = sig.windows.tukey(m, taper) if taper else onp.ones(m)
    X = onp.fft.rfft(x * w, 2 * m)
    return onp.fft.irfft(onp.abs(X) ** 2)[:n_lags] / onp.sum(w ** 2)


def cholesky_from_acf(rho):
    """lower Cholesky factor of the Toeplitz covariance built from the ACF `rho`"""
    return jnp.asarray(cholesky(toeplitz(onp.asarray(rho, dtype=float)), lower=True))


# ──────────────────────────────────────────────────────────────────────────────
# jaxns termination reason
# ──────────────────────────────────────────────────────────────────────────────
JAXNS_TERMINATION_BITS = [        # bit i of NestedSamplerResults.termination_reason (jaxns 2.6.x, utils._print_termination_reason)
    "reached max samples",
    "evidence uncertainty low enough",
    "small remaining evidence",
    "reached ESS",
    "used max num likelihood evaluations",
    "likelihood contour reached",
    "sampler efficiency too low",
    "all live points on a single plateau (possible precision error)",
    "relative spread of live points < rtol",
    "absolute spread of live points < atol",
    "no seed points left",
    "XL < max(XL) * peak_XL_frac",
]
JAXNS_CONVERGED_BITS = {1, 2, 3}   # the only reasons that mean the run finished on its own criterion


def decode_termination(reason):
    """
    (reasons, converged) for a jaxns termination_reason bitmask: the list of
    the set bits' meanings, and whether the run stopped on a convergence
    criterion (evidence uncertainty / remaining evidence / ESS) rather than on
    a cap (max samples, max likelihood evaluations), an efficiency failure or
    a plateau. A run that is not converged has an unreliable log Z.
    """
    reason = int(reason)
    bits = [i for i in range(len(JAXNS_TERMINATION_BITS)) if reason & (1 << i)]
    reasons = [JAXNS_TERMINATION_BITS[i] for i in bits]
    converged = bool(bits) and all(i in JAXNS_CONVERGED_BITS for i in bits)
    return reasons, converged
