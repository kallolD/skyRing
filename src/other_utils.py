import jax, jax.numpy as jnp
import numpy as onp
import scipy.signal as sig
from scipy.linalg import solve_triangular

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
    tstart_idx = onp.argwhere(times>=tstart)[0][0]
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