"""
Config-free setup of the ringdown analyses of the paper (Full-sky, Fixed-sky and Informed-sky), step by step:

    GWOSC strain -> conditioning (no high-pass, downsampled to 4096 Hz) -> 0.2 s analysis windows
    -> noise covariance from the time-domain autocorrelation of two 64 s off-source stretches
    -> log-likelihood (waveform.py) -> nested sampling (jaxns through numpyro).

Typical use (set the XLA device count before the first `import jax`; see examples/analysis.ipynb):

    import analysis as an
    setup = an.setup_event("GW250114", sky="full")               # or sky="fixed"
    modes = ((2, 2, 0), (2, 2, 1))
    names, low, high = an.prior_box(setup, modes)
    loglik = an.make_loglik(setup, modes)
    result = an.run_nested(loglik, low, high, nlive=10_000, seed=31567)

Informed-sky: build the Full-sky setup and marginalize its likelihood over IMR sky samples,

    nodes, logw = an.imr_sky_nodes(ra_samples, dec_samples)       # K = 100 equal-weight nodes
    loglik = an.make_informed_loglik(setup, modes, nodes, logw)
    names, low, high = an.prior_box(setup, modes, informed=True)   # no (ra, sin dec)

The event settings in EVENTS are those of the paper's runs.
"""

import os

import numpy as onp
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import lal
import numpyro
import numpyro.distributions as dist

import waveform as wf
from swsh import swsh
from other_utils import bandpass_ds, analysis_data, td_acf, cholesky_from_acf, decode_termination
from strain_cache import fetch_or_cache_strain

jax.config.update("jax_enable_x64", True)
numpyro.enable_x64()

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")   # QNM tables
DET_INDEX = {"H1": lal.LALDetectorIndexLHODIFF, "L1": lal.LALDetectorIndexLLODIFF}

# Settings of the paper's runs. Start time: t_peak + start_tM * tM (GW250114) or t_start (GW190521), geocentric.
# acf_centers: centres of the two 64 s off-source stretches used for the noise covariance (fixed for all start times).
EVENTS = {
    "GW250114": dict(t_peak=1420878141.235932, tM=0.337e-3, start_tM=6, ra=2.33, dec=0.19,
                     acf_centers=(1420878141.235932 - 34, 1420878141.235932 + 34),
                     mass=(55., 95.), chi=(0., 0.99), amp={}, amp_default=(0., 4.),
                     modes=((2, 2, 0), (2, 2, 1))),
    "GW190521": dict(t_start=1242442967.445 + 0.006, tM=325 * lal.MTSUN_SI, ra=3.5, dec=0.73,
                     acf_centers=(1242442967.445 + 0.006 - 34, 1242442967.445 + 0.006 + 34),
                     mass=(150., 500.), chi=(0., 0.99), amp={(2, 2, 0): (0., 4.5), (3, 3, 0): (0., 0.6)},
                     amp_default=(0., 4.), modes=((2, 2, 0), (3, 3, 0))),
}


def start_time(event, shift=0):
    """Geocentric ringdown start of the paper's runs; `shift` = number of t_M steps after it (Figs. 4b, 5b)."""
    ev = EVENTS[event]
    if "t_peak" in ev:
        return ev["t_peak"] + (ev["start_tM"] + shift) * ev["tM"]
    return ev["t_start"] + shift * 0.0016007845579834117      # 325 Msun in s, as in the start-time runs


def setup_event(event, sky="full", shift=0, cache_dir="strain_cache", detectors=("H1", "L1"),
                seglen=8, fs=16384, srate=4096, T=0.2, t0=0.0, trim=0.25, acf_seglen=64.0, taper=0.1, log=print):
    """
    Data, analysis windows and noise covariance for one event.

    sky = "full": one window per detector, all starting at the arrival time in the last detector (t_ref).
    sky = "fixed": each detector's window starts at its own arrival time for the fixed (ra, dec) of EVENTS.
    Strain is read from (or written to) `cache_dir` via strain_cache.fetch_or_cache_strain.
    """
    ev = EVENTS[event]
    ra, dec = ev["ra"], ev["dec"]
    tgps = start_time(event, shift)
    raw = {ifo: fetch_or_cache_strain(ifo, tgps - seglen/2, tgps + seglen/2, fs, cache_dir, log=log)
           for ifo in detectors}

    # arrival times at the fixed sky position; the last detector is the reference
    tgps_lal = lal.LIGOTimeGPS(tgps)
    gmst = float(lal.GreenwichMeanSiderealTime(tgps_lal))
    t_arrival = {ifo: tgps + lal.TimeDelayFromEarthCenter(lal.CachedDetectors[DET_INDEX[ifo]].location,
                                                          ra, dec, tgps_lal) for ifo in detectors}
    ref_det = max(t_arrival, key=t_arrival.get)
    tref = t_arrival[ref_det]

    # conditioning (no high-pass; FFT anti-aliasing, downsampling, edge trim) and analysis windows
    ds = fs // srate
    n = wf.Ntime(srate, T)
    anchor = {ifo: (t_arrival[ifo] if sky == "fixed" else tref) for ifo in detectors}
    h_t = {}
    for ifo in detectors:
        cond_t, cond_d = bandpass_ds(raw[ifo], t0=anchor[ifo], ds=ds, trim=trim, f_min=None)
        h_t[ifo] = jnp.asarray(analysis_data(cond_d, cond_t, anchor[ifo], n)[1], dtype=jnp.float64)

    # noise covariance: average of the time-domain ACFs of the off-source stretches, conditioned like the data
    acfs = []
    for c in ev["acf_centers"]:
        a, b = c - acf_seglen/2, c + acf_seglen/2
        assert not (a < tgps + T + 1.0 and b > tgps - 1.0), f"off-source stretch [{a}, {b}] overlaps the event"
        cond = {ifo: bandpass_ds(fetch_or_cache_strain(ifo, a, b, fs, cache_dir, log=log), t0=c, ds=ds, trim=trim,
                                 f_min=None) for ifo in detectors}
        acfs.append({ifo: td_acf(cond[ifo][1], n, taper=taper) for ifo in detectors})
    L = {ifo: cholesky_from_acf(onp.mean([a[ifo] for a in acfs], axis=0)) for ifo in detectors}

    # detector responses and locations used by the likelihoods
    resp = {ifo: jnp.asarray(lal.CachedDetectors[DET_INDEX[ifo]].response, dtype=jnp.float64) for ifo in detectors}
    rH, rL = wf.initialize_det_locs()
    wf.set_detector_locations(rH, rL)
    wf.set_detectors(resp["H1"], resp["L1"])
    log(f"{event} ({sky}-sky, shift {shift}): start {tgps:.6f}, ref detector {ref_det}, {n} samples at {srate} Hz")
    return dict(event=event, sky=sky, shift=shift, tgps=tgps, gmst=gmst, ra=ra, dec=dec, t_arrival=t_arrival,
                ref_det=ref_det, tref=tref, raw=raw, h_t=h_t, L=L, T=T, srate=srate, t0=t0, rH=rH, rL=rL)


def _swsh(l, m, iota):
    return swsh(2, l, m, iota, 0.0)


def prior_box(setup, modes, mass=None, informed=False, sky_frame="equatorial"):
    """
    Uniform prior box, in the parameter order the likelihoods expect:
    M, chi, (A, phi) per mode, cos(iota), [ra, sin(dec)  or  epsilon, cos(kappa) for Full-sky], psi.
    Amplitudes are in units of 1e-20; `mass` overrides the event's mass range (the start-time runs use 30 Msun).
    """
    ev = EVENTS[setup["event"]]
    m_lo, m_hi = mass or ev["mass"]
    names, low, high = ["M", "chi"], [m_lo, ev["chi"][0]], [m_hi, ev["chi"][1]]
    for mode in modes:
        a = ev["amp"].get(tuple(mode), ev["amp_default"])
        lab = "".join(map(str, mode))
        names += [f"A{lab}", f"phi{lab}"]; low += [a[0], 0.]; high += [a[1], 2*onp.pi]
    names.append("cosiota"); low.append(-1.); high.append(1.)
    if setup["sky"] == "full" and not informed:
        names += ["epsilon", "coskappa"] if sky_frame == "detector" else ["ra", "sindec"]
        low += [0., -1.]; high += [2*onp.pi, 1.]
    names.append("psi"); low.append(0.); high.append(onp.pi)
    return names, jnp.array(low, dtype=jnp.float64), jnp.array(high, dtype=jnp.float64)


def make_loglik(setup, modes, sky_frame="equatorial", data_dir=DATA_DIR):
    """
    Log-likelihood of a parameter vector (prior_box order), without the parameter-independent normalization.
    Full-sky: sky sampled, Livingston template shifted by the sky-dependent delay. Fixed-sky: sky fixed at
    EVENTS[event] (ra, dec), windows pre-aligned to each detector's arrival time. Modes with the same overtone
    number are ordered by amplitude in the listed order (e.g. A330 <= A220).
    """
    modes = tuple(tuple(m) for m in modes)
    table = wf.load_qnm_table(data_dir, modes)
    H, Lv, LH, LL = setup["h_t"]["H1"], setup["h_t"]["L1"], setup["L"]["H1"], setup["L"]["L1"]
    gmst, T, srate, t0 = setup["gmst"], setup["T"], setup["srate"], setup["t0"]
    if setup["sky"] == "fixed":
        ra, dec = setup["ra"], setup["dec"]
        return lambda th: wf.ln_likelihood_general_fixed_sky_prealigned_jit(
            H, Lv, th, gmst, LH, LL, table, modes, _swsh, ra, dec, T, srate, t0)
    ref = setup["ref_det"]
    loglik = lambda th: wf.ln_likelihood_general_full_jit(H, Lv, th, gmst, LH, LL, table, modes, _swsh, T, srate, t0, ref)
    if sky_frame == "detector":                       # sample (epsilon, cos kappa) about the H1-L1 baseline
        frame = wf.build_baseline_frame(setup["rH"], setup["rL"])
        return lambda th: loglik(wf.detframe_theta_to_equatorial(th, gmst, frame, modes))
    return loglik


def imr_sky_nodes(ra, dec, K=100, seed=31567):
    """K IMR sky samples, (ra, sin dec), drawn with numpy's default_rng(seed), with equal weights (Informed-sky)."""
    ext = onp.array([ra, onp.sin(dec)]).T
    idxs = onp.random.default_rng(seed).integers(0, len(ext), K)
    return jnp.asarray(ext[idxs]), jnp.full(K, -onp.log(K))


def make_informed_loglik(setup, modes, nodes, logw, data_dir=DATA_DIR):
    """Full-sky likelihood marginalized over the sky with the quadrature nodes and log-weights of imr_sky_nodes."""
    assert setup["sky"] == "full", "Informed-sky uses the Full-sky windows"
    loglik = make_loglik(setup, modes, data_dir=data_dir)
    i_ra = 3 + 2 * len(modes)                         # index of ra in the Full-sky parameter vector
    K = nodes.shape[0]

    def loglike(theta):
        th = jnp.broadcast_to(theta, (K, theta.shape[0]))
        return logsumexp(jax.vmap(loglik)(jnp.concatenate([th[:, :i_ra], nodes, th[:, i_ra:]], axis=1)) + logw)
    return loglike


def run_nested(loglik, low, high, nlive=10_000, seed=31567, max_samples=None, dlogZ=0.01, num_posterior=100_000,
               verbose=True, weighted=False):
    """
    Nested sampling with uniform priors on [low, high]; stops when ln(Z_dead + Z_live) - ln(Z_dead) < dlogZ.
    Returns the equal-weight posterior `theta`, log_Z and its error, the ESS and the termination check
    (and the weighted samples if `weighted`).
    """
    from numpyro.contrib.nested_sampling import NestedSampler

    def model():
        theta = numpyro.sample("theta", dist.Uniform(low, high))
        numpyro.factor("loglike", loglik(theta))

    max_samples = int(max_samples or max(500_000, 10 * nlive * len(low)))
    rng_run, rng_post = jax.random.split(jax.random.PRNGKey(int(seed)))
    ns = NestedSampler(model, constructor_kwargs=dict(num_live_points=int(nlive), max_samples=max_samples,
                                                      verbose=verbose),
                       termination_kwargs=dict(dlogZ=dlogZ))
    ns.run(rng_run)
    r = ns._results
    reasons, converged = decode_termination(int(r.termination_reason))
    out = dict(theta=onp.asarray(ns.get_samples(rng_post, num_samples=num_posterior)["theta"]),
               log_Z=float(r.log_Z_mean), log_Z_uncert=float(r.log_Z_uncert), ESS=float(r.ESS),
               converged=converged, termination=reasons)
    if weighted:
        w, log_w = ns.get_weighted_samples()
        out.update(w_theta=onp.asarray(w["theta"]), log_w=onp.asarray(log_w))
    return out
