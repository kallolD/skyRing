import numpy as onp
import jax, jax.numpy as jnp
import lal
from jax import random, vmap
from scipy.linalg import toeplitz

jax.config.update("jax_enable_x64", True) #double precision
reference_amplitude = jnp.asarray(1e-20, dtype=jnp.float64)

MTSUN  = jnp.float64(lal.MTSUN_SI)

# module-level variables, initially None
_det_H = None
_det_L = None

def set_detectors(resp_H, resp_L):
    """
    Called from main.py to initialize the detector responses.
    resp_H, resp_L: (3,3) JAX arrays
    """
    global _det_H, _det_L
    _det_H = resp_H
    _det_L = resp_L

def antenna_pattern_jax(ra, dec, psi, gmst, resp):
    """
    JAX version of the antenna pattern functions

    Parameters
    ----------
    ra   : float or array
        Right ascension [rad]
    dec  : float or array
        Declination [rad]
    psi  : float or array
        Polarization angle [rad]
    gmst : float or array
        Greenwich mean sidereal time [rad]
    resp : jnp.ndarray, shape (3,3)
        Detector response tensor in Earth-fixed coordinates
        (e.g. lal.CachedDetectors[...].response)

    Returns
    -------
    Fp, Fc : arrays with same shape as inputs (broadcasted)
    """
    # Greenwich hour angle of the source
    gha = gmst - ra

    cosgha = jnp.cos(gha)
    singha = jnp.sin(gha)
    cosdec = jnp.cos(dec)
    sindec = jnp.sin(dec)
    cospsi = jnp.cos(psi)
    sinpsi = jnp.sin(psi)

    # Basis vectors x,y as in pycbc.detector.ground.Detector.antenna_pattern
    x0 = -cospsi * singha - sinpsi * cosgha * sindec
    x1 = -cospsi * cosgha + sinpsi * singha * sindec
    x2 =  sinpsi * cosdec
    x  = jnp.stack([x0, x1, x2], axis=-1)   # shape (..., 3)

    y0 =  sinpsi * singha - cospsi * cosgha * sindec
    y1 =  sinpsi * cosgha + cospsi * singha * sindec
    y2 =  cospsi * cosdec
    y  = jnp.stack([y0, y1, y2], axis=-1)   # shape (..., 3)

    # Apply detector tensor: dx_i = d_{ij} x_j, same for dy
    dx = jnp.einsum("ij,...j->...i", resp, x)
    dy = jnp.einsum("ij,...j->...i", resp, y)

    # Pattern functions
    Fp = jnp.sum(x * dx - y * dy, axis=-1)
    Fc = jnp.sum(x * dy + y * dx, axis=-1)

    return Fp, Fc

def patterns_for_params(ra, dec, psi, gmst):
    # use module-level arrays
    FpH, FcH = antenna_pattern_jax(ra, dec, psi, gmst, _det_H)
    FpL, FcL = antenna_pattern_jax(ra, dec, psi, gmst, _det_L)
    return FpH, FcH, FpL, FcL

def sY_2_2(theta, phi):
    pref = jnp.sqrt(5.0 / (64.0 * jnp.pi))
    return pref * (1.0 + jnp.cos(theta))**2 * jnp.exp(2j * phi)

def sY_2_m2(theta, phi):
    pref = jnp.sqrt(5.0 / (64.0 * jnp.pi))
    return pref * (1.0 - jnp.cos(theta))**2 * jnp.exp(-2j * phi)

def sY_3_3(theta, phi):
    pref = jnp.sqrt(7.0 / (512.0 * jnp.pi))
    return pref * (1.0 + jnp.cos(theta))**3 * jnp.exp(3j * phi)

def sY_3_m3(theta, phi):
    pref = jnp.sqrt(7.0 / (512.0 * jnp.pi))
    return pref * (1.0 - jnp.cos(theta))**3 * jnp.exp(-3j * phi)

def initialize_det_resp():

    resp_H_py = lal.CachedDetectors[lal.LALDetectorIndexLHODIFF].response
    resp_L_py = lal.CachedDetectors[lal.LALDetectorIndexLLODIFF].response

    resp_H = jnp.asarray(resp_H_py, dtype=jnp.float64)
    resp_L = jnp.asarray(resp_L_py, dtype=jnp.float64)

    det_H = Detector(resp_H)
    det_L = Detector(resp_L)

    return det_H, det_L

def Ntime(srate, T):

    Ntime = int(round(T * srate))
    if Ntime % 2 != 0:
        Ntime += 1  # make it even
    return Ntime

def ACFs(srate, T, psdH, psdL, factor=10):

    dt =  1.0 / srate
    TT=factor*T
    NN=Ntime(srate, TT)
    N=Ntime(srate, T)
    freqs = jnp.fft.rfftfreq(NN, d=dt)

    #compute ACT from Sn, cov and cholesky
    df=freqs[1]-freqs[0]
    SnH = psdH(freqs)
    SnL = psdL(freqs)
    rhoH= 0.5*jnp.fft.irfft(SnH).real * df * NN
    rhoL= 0.5*jnp.fft.irfft(SnL).real * df * NN

    return rhoH[:N], rhoL[:N] #truncated ACFs

def importance_sample(theta, w, M=None):
    """
    theta: (N, d) proposals drawn from q
    w:     (N,)  unnormalized importance weights proportional to p/q
    M:     number of resampled draws (default: N)
    """
    if M is None: M = len(theta)
    
    rng = onp.random.default_rng()

    W = w / w.sum()  # normalized weights
    idx = rng.choice(len(theta), size=M, replace=True, p=W) #replace=True allows for drawing the same sample multiple times

    neff=(w.sum())**2/((w**2).sum())

    return theta[idx], neff, neff/len(theta)

def rejection_sample(theta, w):
    
    rng = onp.random.default_rng()

    C = w.max()
    u = rng.uniform(size=len(theta))
    accept = (u < (w / C))
    return theta[accept], len(theta[accept])

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

def load_tables(psd_H_path, psd_L_path, qnm1_path, qnm2_path):
    ligo_psdH  = onp.genfromtxt(psd_H_path, delimiter='')
    ligo_psdL  = onp.genfromtxt(psd_L_path, delimiter='')
    psdH = interp1d_jax(jnp.asarray(ligo_psdH[:,0],dtype=jnp.float64), jnp.asarray(ligo_psdH[:,1],dtype=jnp.float64))
    psdL = interp1d_jax(jnp.asarray(ligo_psdL[:,0],dtype=jnp.float64), jnp.asarray(ligo_psdL[:,1],dtype=jnp.float64))

    berti_fit_data = onp.genfromtxt(qnm1_path)
    omega_r = interp1d_jax(jnp.asarray(berti_fit_data[:,0],dtype=jnp.float64), jnp.asarray(berti_fit_data[:,1],dtype=jnp.float64))
    omega_i = interp1d_jax(jnp.asarray(berti_fit_data[:,0],dtype=jnp.float64), jnp.asarray(berti_fit_data[:,2],dtype=jnp.float64))

    berti_fit_data_OT = onp.genfromtxt(qnm2_path)
    omega_OT_r = interp1d_jax(jnp.asarray(berti_fit_data_OT[:,0],dtype=jnp.float64), jnp.asarray(berti_fit_data_OT[:,1],dtype=jnp.float64))
    omega_OT_i = interp1d_jax(jnp.asarray(berti_fit_data_OT[:,0],dtype=jnp.float64), jnp.asarray(berti_fit_data_OT[:,2],dtype=jnp.float64))

    return psdH, psdL, omega_r, omega_i, omega_OT_r, omega_OT_i

def cov_resp(srate, T, psdH, psdL, factor):
    rhoH, rhoL = ACFs(srate=srate, T=T, psdH=psdH, psdL=psdL, factor=factor)
    covH=toeplitz(rhoH)
    covL=toeplitz(rhoL)
    L_H=jnp.linalg.cholesky(covH)
    L_L=jnp.linalg.cholesky(covL)

    resp_H_py = lal.CachedDetectors[lal.LALDetectorIndexLHODIFF].response
    resp_L_py = lal.CachedDetectors[lal.LALDetectorIndexLLODIFF].response
    resp_H = jnp.asarray(resp_H_py, dtype=jnp.float64)
    resp_L = jnp.asarray(resp_L_py, dtype=jnp.float64)
    return L_H, L_L, resp_H, resp_L

def h22(t, f, gamma, A, fp, fc, swshlmp, swshlmn):
    amp_1, amp_2 = A, jnp.conj(A)
    #swshlmp = lal.SpinWeightedSphericalHarmonic(float(inclination), 0, -2, 2, 2) #float neeeded because inclination will be a jax array
    #swshlmn = lal.SpinWeightedSphericalHarmonic(float(inclination), 0, -2, 2, -2)
    h_tmp = jnp.exp(-gamma*t) * (swshlmp*amp_1*jnp.exp(1j*2*jnp.pi*f*t) + (-1)**2 * swshlmn*amp_2*jnp.exp(-1j*2*jnp.pi*f*t))


    h_p = jnp.real(h_tmp)
    h_c = jnp.imag(h_tmp)
    return fp*h_p + fc*h_c

def h33(t, f, gamma, A, fp, fc, swshlmp, swshlmn):

    amp_1, amp_2 = A, jnp.conj(A)
    #swshlmp = lal.SpinWeightedSphericalHarmonic(float(inclination), 0, -2, 3, 3)
    #swshlmn = lal.SpinWeightedSphericalHarmonic(float(inclination), 0, -2, 3, -3)

    h_tmp = jnp.exp(-gamma*t) * (swshlmp*amp_1*jnp.exp(1j*2*jnp.pi*f*t) + (-1)**3 * swshlmn*amp_2*jnp.exp(-1j*2*jnp.pi*f*t))

    h_p = jnp.real(h_tmp)
    h_c = jnp.imag(h_tmp)
    return fp*h_p + fc*h_c

def noise_gen(key, srate, T, fmin, psd_fn=None):

    N = Ntime(srate, T)
    dt = 1.0/srate

    freqs = jnp.fft.rfftfreq(N, d=dt)
    df = freqs[1] - freqs[0]
    Sn = psd_fn(freqs) if psd_fn is not None else jnp.full_like(freqs, reference_amplitude**2) #if no psd, produce white noise
    sigma = 0.5*jnp.sqrt(Sn/df) 

    sigma = sigma.at[0].multiply(jnp.sqrt(2.0))
    if N % 2 == 0:
        sigma = sigma.at[-1].multiply(jnp.sqrt(2.0))
    #DC and Nyquist multiplied by sqrt(2) because the PSD there needs to be multiplied by 2, 
    #as the first and last bin have no mirrored negative frequency

    # print(type(fmin), type(df))
    # print(fmin.dtype, df.dtype)
    # kmin = int(fmin/df)
    kmin = jnp.trunc(fmin/df).astype(jnp.int64)

    k1, k2 = random.split(key)
    re = random.normal(k1, freqs.shape, dtype=jnp.float64) * sigma
    im = random.normal(k2, freqs.shape, dtype=jnp.float64) * sigma

    idx = jnp.arange(re.shape[0])
    mask = idx<kmin

    re = jnp.where(mask, 0.0, re)
    im = jnp.where(mask, 0.0, im)

    # re = re.at[0:kmin].set(0 + 0*1j)
    # im = im.at[0:kmin].set(0 + 0*1j)
    im=im.at[0].set(0.0)
    if N % 2 == 0:
        im=im.at[-1].set(0.0)
    #im part set to 0 at first and last frequency to ensure td signal is real 
    #[h*(f)=h(-f), which becomes H[k]=H*[N-k] for the DFT, so Nyquist and DC components must be real]
    
    Hf = re + 1j*im
    h = jnp.fft.irfft(Hf).real * df * N
    return h

def noise(key, srate, T, fmin, psd_fn=None, factor=10.):
    n_long=noise_gen(key=key, srate=srate, T=T*factor, fmin=fmin, psd_fn=psd_fn)
    N = Ntime(srate, T)
    return n_long[:N]

def topk_per_row(Hf, k):
    assert 0 < k <= Hf.shape[1], "k must be within [1, F]"
    absH = jnp.abs(Hf)
    idx = jnp.argsort(absH, axis=1)[:, -k:]          # shape (N, k)
    top_vals = jnp.take_along_axis(Hf, idx, axis=1)  # shape (N, k)
    return idx, top_vals


def topk_1d(Hf, k):
    assert 0 < k <= Hf.shape[0], "k must be within [1, F]"
    absH = jnp.abs(Hf)
    idx = jnp.argsort(absH)[-k:]         # (k,)
    top_vals = Hf[idx]                   # (k,)
    return idx, top_vals


def simulate_data(params, psd1, psd2, L1, L2, omega_r, omega_i, omega_OT_r, omega_OT_i,
                     Fp1, Fc1, Fp2, Fc2, key, swsh22p, swsh22m, swsh33p, swsh33m, T, srate, t0, i0, i1, freqs, fmin, components=None, factor=10.):
    #freqs must be unsliced
    M, chi, A220, P220, A221, P221 = jnp.asarray(params,dtype=jnp.float64)
    prefactor_omega_r = (1./(2.*jnp.pi*M*MTSUN))
    prefactor_omega_i = (1./(M*MTSUN))

    A220 = A220*jnp.exp(1j*P220)*reference_amplitude
    A221 = A221*jnp.exp(1j*P221)*reference_amplitude
    A = jnp.array([A220, A221],dtype=jnp.complex128)

    f = jnp.array([prefactor_omega_r*omega_r(chi), prefactor_omega_r*omega_OT_r(chi)],dtype=jnp.float64)
    gamma = jnp.array([-prefactor_omega_i*omega_i(chi), -prefactor_omega_i*omega_OT_i(chi)],dtype=jnp.float64)

    dt = jnp.asarray(1.0/srate, dtype=jnp.float64)
    N = Ntime(srate=srate, T=T)
    t  = jnp.arange(N, dtype=jnp.float64) * dt

    waveform1 = jnp.zeros(N, dtype=jnp.float64)
    waveform2 = jnp.zeros(N, dtype=jnp.float64)
    for j in range(A.shape[0]):
        waveform1 = waveform1 + h22(t+t0, f[j], gamma[j], A[j], Fp1, Fc1, swsh22p, swsh22m)
        waveform2 = waveform2 + h22(t+t0, f[j], gamma[j], A[j], Fp2, Fc2, swsh22p, swsh22m)

    key1, key2 = random.split(key)

    n1 =  noise(key1, srate, T, fmin, psd_fn=psd1, factor=factor)
    n2 =  noise(key2, srate, T, fmin, psd_fn=psd2, factor=factor)

    Ht1 = waveform1 #+ n1
    Ht2 = waveform2 #+ n2

    y1 = jax.scipy.linalg.solve_triangular(L1, jnp.asarray(Ht1), lower=True)
    y2 = jax.scipy.linalg.solve_triangular(L2, jnp.asarray(Ht2), lower=True)

    Hf1_whitened = jnp.fft.rfft(y1)/jnp.sqrt(N)
    Hf2_whitened = jnp.fft.rfft(y2)/jnp.sqrt(N)
    Hf1 = dt * jnp.fft.rfft(Ht1)
    Hf2 = dt * jnp.fft.rfft(Ht2)

    Hf1_whitened = Hf1_whitened[i0:i1]
    Hf2_whitened = Hf2_whitened[i0:i1]
    
    components = int(components) if components is not None else Hf1_whitened.shape[0]

    idx1, Hf1_top = topk_1d(Hf1_whitened, components)
    idx2, Hf2_top = topk_1d(Hf2_whitened, components)

    # Normalize indices
    idx1_norm = idx1.astype(jnp.float32) / Hf1_whitened.shape[0]
    idx2_norm = idx2.astype(jnp.float32) / Hf2_whitened.shape[0]

    Hout = jnp.concatenate(
            [
                idx1_norm,
                jnp.abs(Hf1_top),
                Hf1_top.real,
                idx2_norm,
                jnp.abs(Hf2_top),
                Hf2_top.real
            ],
            axis=0
    ).astype(jnp.float32)

    Hout_t = jnp.concatenate([y1, y2],axis=0).astype(jnp.float32) 

    return Hout, Hout_t, Hf1, Hf2, Ht1, Ht2  

def simulate_data_full(params, gmst, psdH, psdL, L_H, L_L, omega_r, omega_i, omega_OT_r, omega_OT_i,
        key, T, srate, t0, i0, i1, freqs, fmin, components=None, factor=10.):

    p0 = params[:6]
    cosiota, ra, cosdec, psi = params[6:]
    iota = jnp.arccos(cosiota)
    dec = jnp.arccos(cosdec)


    FpH, FcH, FpL, FcL = patterns_for_params(ra, dec, psi, gmst)
    swsh22p = sY_2_2(iota, 0.)
    swsh22m  = sY_2_m2(iota, 0.)
    swsh33p = sY_3_3(iota, 0.)
    swsh33m = sY_3_m3(iota, 0.)

    return simulate_data(p0, psdH, psdL, L_H, L_L, 
            omega_r, omega_i, omega_OT_r, omega_OT_i,
            FpH, FcH, FpL, FcL, key, 
            swsh22p, swsh22m, swsh33p, swsh33m, 
            T, srate, t0, i0, i1, freqs, fmin, components, factor)

def ln_likelihood(data1, data2, params, L1, L2, omega_r, omega_i, omega_OT_r, omega_OT_i,
                     Fp1, Fc1, Fp2, Fc2, swsh22p, swsh22m, swsh33p, swsh33m, T, srate, t0):

    M, chi, A220, P220, A221, P221 = jnp.asarray(params,dtype=jnp.float64)
    prefactor_omega_r = (1./(2.*jnp.pi*M*MTSUN))
    prefactor_omega_i = (1./(M*MTSUN))

    A220 = A220*jnp.exp(1j*P220)*reference_amplitude
    A221 = A221*jnp.exp(1j*P221)*reference_amplitude
    A = jnp.array([A220, A221],dtype=jnp.complex128)

    f = jnp.array([prefactor_omega_r*omega_r(chi), prefactor_omega_r*omega_OT_r(chi)],dtype=jnp.float64)
    gamma = jnp.array([-prefactor_omega_i*omega_i(chi), -prefactor_omega_i*omega_OT_i(chi)],dtype=jnp.float64)

    dt = jnp.asarray(1.0/srate, dtype=jnp.float64)
    N = Ntime(srate=srate, T=T)
    t  = jnp.arange(N, dtype=jnp.float64) * dt

    waveform1 = jnp.zeros(N, dtype=jnp.float64)
    waveform2 = jnp.zeros(N, dtype=jnp.float64)
    for j in range(A.shape[0]):
        waveform1 = waveform1 + h22(t+t0, f[j], gamma[j], A[j], Fp1, Fc1, swsh22p, swsh22m)
        waveform2 = waveform2 + h22(t+t0, f[j], gamma[j], A[j], Fp2, Fc2, swsh22p, swsh22m)

    diff1 = waveform1 - data1
    diff2 = waveform2 - data2

    y1 = jax.scipy.linalg.solve_triangular(L1, jnp.asarray(diff1), lower=True)
    y2 = jax.scipy.linalg.solve_triangular(L2, jnp.asarray(diff2), lower=True)

    return -0.5*(jnp.vdot(y1, y1).real+jnp.vdot(y2, y2).real)


def ln_likelihood_full(dataH, dataL, params, gmst, L_H, L_L, omega_r, omega_i, omega_OT_r, omega_OT_i, T, srate, t0): 

    p0 = params[:6]
    cosiota, ra, sindec, psi = params[6:]
    iota = jnp.arccos(cosiota)
    dec = jnp.arcsin(sindec)

    FpH, FcH, FpL, FcL = patterns_for_params(ra, dec, psi, gmst)
    swsh22p = sY_2_2(iota, 0.)
    swsh22m  = sY_2_m2(iota, 0.)
    swsh33p = sY_3_3(iota, 0.)
    swsh33m = sY_3_m3(iota, 0.)

    return ln_likelihood(dataH, dataL, p0, L_H, L_L, omega_r, omega_i, omega_OT_r, omega_OT_i,
            FpH, FcH, FpL, FcL, swsh22p, swsh22m, swsh33p, swsh33m, T, srate, t0)


simulate_data_jit = jax.jit(
    simulate_data,
    static_argnames=(
        'psd1','psd2','T', 'srate',
        'omega_r','omega_i','omega_OT_r','omega_OT_i',
        'i0', 'i1', 'components', 'factor'
    )
)

simulate_data_full_jit = jax.jit(
    simulate_data_full,
    static_argnames=(
        'gmst', 'psdH','psdL','T', 'srate',
        'omega_r','omega_i','omega_OT_r','omega_OT_i',
        'i0', 'i1', 'components', 'factor'
    )
)

ln_likelihood_jit = jax.jit(
    ln_likelihood,
    static_argnames=(
        'T', 'srate',
        'omega_r','omega_i','omega_OT_r','omega_OT_i'
    )
) 


ln_likelihood_full_jit = jax.jit(
    ln_likelihood_full,
    static_argnames=(
        'gmst', 'T', 'srate',
        'omega_r','omega_i','omega_OT_r','omega_OT_i'
    )
)