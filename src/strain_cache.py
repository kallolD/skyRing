"""
On-disk cache for GWOSC strain downloads.

fetch_or_cache_strain(ifo, t_start, t_end, sample_rate, cache_dir) -> gwpy TimeSeries

A request is identified by its download parameters (ifo, t_start, t_end, sample_rate). The first call
fetches from GWOSC and writes one .npz per request; later calls with the same parameters load it.

Time labels: GWOSC snaps the returned segment to its own sampling grid (gwpy crops with a floor), so the
series does NOT start at the requested t_start -- for the GW250114 8 s segment it starts 39.1 us earlier.
An earlier cache in skyring_ptmcmc saved only the sample values and rebuilt the series with t0 = t_start,
which shifted every sample time by that amount and changed the analysis windows. Here the series is rebuilt
from GWOSC's own start time, stored as a float (exact, since the grid is a multiple of 2^-14 s; an
integer-nanosecond copy is kept for reference).

Checks, on fetch and on load: t0 lies on the 1/sample_rate grid, at or before t_start and less than one
sample earlier; the length matches the request to within one sample; all values are finite. A cache file
that fails a check, belongs to another request or cannot be read is ignored and the data are fetched again.
"""

import os
import tempfile
import time

import numpy as onp
from gwpy.timeseries import TimeSeries

CACHE_FORMAT = 2


def cache_path(ifo, t_start, t_end, sample_rate, cache_dir):
    fname = f"{ifo}_{t_start:.6f}_{t_end:.6f}_{int(sample_rate)}.npz"
    return os.path.join(cache_dir, fname)


def _gps_ns(t):
    """GPS time (float seconds) -> integer nanoseconds (informational copy of t0)."""
    sec = int(onp.floor(t))
    return sec * 1_000_000_000 + int(round((t - sec) * 1e9))


def _check(value, t0, t_start, t_end, sample_rate):
    """Return None if the series is consistent with the request, else a description of the problem."""
    frac_samples = (t0 - onp.floor(t0)) * sample_rate
    if abs(frac_samples - round(frac_samples)) > 1e-6:
        return f"t0 {t0!r} is not on the 1/{sample_rate} s grid"
    # gwpy floors the start onto the grid: t0 <= t_start < t0 + 1/fs (tolerance covers the float64 ulp of GPS times)
    if not (-5e-7 < t_start - t0 < 1.0 / sample_rate):
        return f"t0 {t0!r} is not within one sample at or before the requested start {t_start!r}"
    n_expected = (t_end - t_start) * sample_rate
    if abs(len(value) - n_expected) > 1:
        return f"{len(value)} samples, expected {n_expected:.0f}"
    if not onp.all(onp.isfinite(value)):
        return f"{int(onp.sum(~onp.isfinite(value)))} non-finite samples (data gap?)"
    return None


def fetch_or_cache_strain(ifo, t_start, t_end, sample_rate, cache_dir, log=print):
    """
    Load strain for [t_start, t_end) at sample_rate from the cache, or fetch it from GWOSC and cache it.

    The returned TimeSeries always carries GWOSC's true start time, whether it was fetched or loaded.
    """
    os.makedirs(cache_dir, exist_ok=True)
    path = cache_path(ifo, t_start, t_end, sample_rate, cache_dir)

    if os.path.exists(path):
        problem = None
        try:
            with onp.load(path) as d:
                if (int(d["format"]) != CACHE_FORMAT or str(d["ifo"]) != ifo
                        or float(d["sample_rate"]) != float(sample_rate)
                        or abs(float(d["requested_t_start"]) - t_start) > 1e-6
                        or abs(float(d["requested_t_end"]) - t_end) > 1e-6):
                    problem = "it belongs to a different request or cache format"
                else:
                    value, t0 = onp.array(d["value"]), float(d["t0"])
                    problem = _check(value, t0, t_start, t_end, sample_rate)
        except Exception as e:                                    # truncated / foreign file
            problem = f"it cannot be read ({e!r})"
        if problem is None:
            log(f"Loading cached {ifo} strain: {path} (t0 = {t0:.6f}, "
                f"{(t0 - t_start)*1e6:+.1f} us from the requested start)")
            return TimeSeries(value, t0=t0, sample_rate=sample_rate)
        log(f"Ignoring cache file {path}: {problem}; re-fetching.")

    log(f"Fetching {ifo} strain from GWOSC: [{t_start:.6f}, {t_end:.6f}) at {sample_rate} Hz")
    ts = TimeSeries.fetch_open_data(ifo, t_start, t_end, sample_rate=sample_rate, verbose=False)
    t0 = float(ts.t0.value)                                  # GWOSC's start, not the requested t_start
    if float(ts.sample_rate.value) != float(sample_rate):
        raise RuntimeError(f"GWOSC returned {ts.sample_rate} for {ifo}, requested {sample_rate} Hz")
    problem = _check(onp.asarray(ts.value), t0, t_start, t_end, sample_rate)
    if problem:
        raise RuntimeError(f"GWOSC data for {ifo} [{t_start}, {t_end}): {problem}")

    fd, tmp = tempfile.mkstemp(dir=cache_dir, suffix=".tmp.npz")
    os.close(fd)
    try:
        onp.savez(tmp, value=onp.asarray(ts.value), t0=t0, t0_ns=_gps_ns(t0),
                  sample_rate=float(sample_rate), ifo=ifo,
                  requested_t_start=float(t_start), requested_t_end=float(t_end),
                  name=str(ts.name), channel=str(ts.channel),
                  fetched_unix_time=time.time(), format=CACHE_FORMAT)
        os.replace(tmp, path)                                # atomic: a crash never leaves a half-written file
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    log(f"Cached {ifo} strain to {path} (t0 = {t0:.6f}, {(t0 - t_start)*1e6:+.1f} us from the requested start)")
    return ts
