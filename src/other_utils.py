import jax, jax.numpy as jnp
import numpy as onp
import scipy.signal as sig
from scipy.linalg import solve_triangular, cholesky, toeplitz

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

def interp1d_jax(xp, fp):
    xp = jnp.asarray(xp,dtype=jnp.float64); fp = jnp.asarray(fp,dtype=jnp.float64)
    def _interp(x):
        x = jnp.asarray(x,dtype=jnp.float64)
        idx = jnp.searchsorted(xp, x, side="right")
        idx = jnp.clip(idx, 1, xp.size - 1)  # extrapolate using end segments
        x0 = xp[idx - 1]; x1 = xp[idx]
        y0 = fp[idx - 1]; y1 = fp[idx]
        w = (x - x0) / (x1 - x0)
        return y0 * (1.0 - w) + y1 * w
    return _interp

def load_tables(qnm1_path, qnm2_path):
    # ligo_psdH  = onp.genfromtxt(psd_H_path, delimiter='')
    # ligo_psdL  = onp.genfromtxt(psd_L_path, delimiter='')
    # psdH = interp1d_jax(jnp.asarray(ligo_psdH[:,0],dtype=jnp.float64), jnp.asarray(ligo_psdH[:,1],dtype=jnp.float64))
    # psdL = interp1d_jax(jnp.asarray(ligo_psdL[:,0],dtype=jnp.float64), jnp.asarray(ligo_psdL[:,1],dtype=jnp.float64))

    berti_fit_data = onp.genfromtxt(qnm1_path)
    omega_r = interp1d_jax(jnp.asarray(berti_fit_data[:,0],dtype=jnp.float64), jnp.asarray(berti_fit_data[:,1],dtype=jnp.float64))
    omega_i = interp1d_jax(jnp.asarray(berti_fit_data[:,0],dtype=jnp.float64), jnp.asarray(berti_fit_data[:,2],dtype=jnp.float64))

    berti_fit_data_OT = onp.genfromtxt(qnm2_path)
    omega_OT_r = interp1d_jax(jnp.asarray(berti_fit_data_OT[:,0],dtype=jnp.float64), jnp.asarray(berti_fit_data_OT[:,1],dtype=jnp.float64))
    omega_OT_i = interp1d_jax(jnp.asarray(berti_fit_data_OT[:,0],dtype=jnp.float64), jnp.asarray(berti_fit_data_OT[:,2],dtype=jnp.float64))

    return omega_r, omega_i, omega_OT_r, omega_OT_i


def calculate_SNR(h_t_, L_):
    y_ = solve_triangular(L_, h_t_, lower=True)
    snr_ = onp.sqrt(onp.dot(y_, y_))
    return snr_


# ──────────────────────────────────────────────────────────────────────────────
# Noise covariance from the time-domain ACF of a long conditioned stretch
# (td_acf, cholesky_from_acf, offsource_whitening_check, format_whitening_check:
# from the psd-filter-correction branch)
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


def offsource_whitening_check(times, data, L, t_exclude, n_analyze, pad_pre=0.5, pad_post=0.1,
                              side="post", stride=None, band_edges=None, max_segments=1000):
    """
    Whiten off-source segments of the conditioned data with the Cholesky
    factor L (as the likelihood does) and compare with white unit-variance noise.

    If L describes the noise of `data`, the whitened samples have unit variance
    and their periodogram |rfft(y)|^2 / N has expectation 1 in every bin. The
    total variance is insensitive to a mismatch confined to a few low-frequency
    bins (they are ~1% of the modes), so the periodogram is also averaged over
    frequency bands (`band_edges`, Hz; default [0, 20, 40, 100, 300, 1000, Nyquist]).

    Segments of n_analyze samples are taken, with a step of `stride` samples
    (default n_analyze // 2, i.e. 50% overlap), from the data AFTER the analysis
    window (side="post") and/or BEFORE it (side="pre"; close to the merger it
    may contain the inspiral). The analysis window [t_exclude, t_exclude +
    n_analyze*dt] is widened by pad_pre before and pad_post after. Uniform
    sampling is assumed (dt from the first two samples).

    Returns dict(n_segments, var_mean, var_scatter, band_edges, band_power, acf1):
    var_scatter is the standard deviation of the per-segment variance (its
    expected value for white noise is sqrt(2/n_analyze) ~ 0.05 for 820 samples),
    not a standard error; band_power is the mean whitened periodogram per band.
    """
    times = onp.asarray(times, dtype=float); data = onp.asarray(data, dtype=float)
    if len(times) != len(data):
        raise ValueError(f"times and data have different lengths ({len(times)} vs {len(data)})")
    L = onp.asarray(L, dtype=float)
    n = int(n_analyze)
    stride = int(stride) if stride else max(n // 2, 1)
    dt = times[1] - times[0]
    nyq = 0.5 / dt
    T_an = n * dt
    lo, hi = t_exclude - pad_pre, t_exclude + T_an + pad_post
    if band_edges is None:
        band_edges = [0.0, 20.0, 40.0, 100.0, 300.0, 1000.0, nyq]
    band_edges = [float(b) for b in band_edges]
    freqs = onp.fft.rfftfreq(n, d=dt)

    var, acf1, pgrams = [], [], []
    for i0 in range(0, len(data) - n + 1, stride):
        t_a, t_b = times[i0], times[i0 + n - 1]
        pre = t_b < lo
        post = t_a > hi
        if not ((side in ("pre", "both") and pre) or (side in ("post", "both") and post)):
            continue
        y = solve_triangular(L, data[i0:i0 + n], lower=True)
        var.append(onp.mean(y ** 2))
        y0 = y - y.mean()
        acf1.append(onp.dot(y0[:-1], y0[1:]) / onp.dot(y0, y0))
        pgrams.append(onp.abs(onp.fft.rfft(y)) ** 2 / n)
        if len(var) >= max_segments:
            break
    nseg = len(var)
    if nseg == 0:
        return dict(n_segments=0, var_mean=onp.nan, var_scatter=onp.nan, band_edges=band_edges,
                    band_power=[onp.nan] * (len(band_edges) - 1), acf1=onp.nan)
    P = onp.mean(onp.asarray(pgrams), axis=0)
    band_power = []
    for a, b in zip(band_edges[:-1], band_edges[1:]):
        m = (freqs >= a) & (freqs < b) if b < nyq else (freqs >= a) & (freqs <= b)
        band_power.append(float(onp.mean(P[m])) if onp.any(m) else onp.nan)
    return dict(n_segments=nseg, var_mean=float(onp.mean(var)), var_scatter=float(onp.std(var)),
                band_edges=band_edges, band_power=band_power, acf1=float(onp.mean(acf1)))


def format_whitening_check(chk):
    """one-line summary of offsource_whitening_check() for the log"""
    if chk["n_segments"] == 0:
        return "no off-source segment available"
    edges = chk["band_edges"]
    bands = "|".join(f"{a:.0f}-{b:.0f}" for a, b in zip(edges[:-1], edges[1:]))
    power = " ".join(f"{p:.2f}" for p in chk["band_power"])
    return (f"{chk['n_segments']} segments, var = {chk['var_mean']:.3f} (scatter {chk['var_scatter']:.3f}), "
            f"acf1 = {chk['acf1']:+.3f}, whitened power per band [{bands} Hz] = {power} (expected 1)")


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
