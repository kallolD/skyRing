# skyRing

Code for ringdown analyses that estimate the sky location of the source together with all other parameters, directly
from the ringdown signal, instead of fixing it to point estimates from inspiral–merger–ringdown (IMR) analyses.
See *Impact of sky localization uncertainty on ringdown inference* (K. Dey, E. Barausse, M. Crisostomi, R. Trotta)
for details.

Three approaches are implemented:
1. **Full-sky:** the sky position is sampled with an isotropic prior. All detectors are analysed from the arrival
   time at the last detector, and the other detectors' templates are shifted by the sky-dependent delay.
2. **Fixed-sky:** the sky position is fixed to an IMR point estimate, and each detector is analysed from its own
   arrival time.
3. **Informed-sky:** the Full-sky likelihood is averaged over samples of the IMR sky posterior, which act as the sky
   prior.

This `dev` branch holds the code of the revised version of the paper. It uses a noise covariance from the
time-domain autocorrelation of off-source data, applies no high-pass filter, and gives Informed-sky equal-weight IMR
sky samples. The `main` branch keeps the version of the first submission.

## Contents
- `src/analysis.py`: the setup of the paper's runs, without configuration files:
  - GWOSC strain, conditioning and analysis windows, for H1-L1 or H1-L1-V1;
  - noise covariance;
  - Full-sky, Fixed-sky and Informed-sky likelihoods and their prior boxes, in equatorial or detector-frame sky
    coordinates;
  - nested sampling;
  - the noiseless ringdown-only injections of the appendix, with their optimal SNRs.
- `src/waveform.py`: quasinormal-mode templates for any set of (l, m, n) modes, antenna patterns, inter-detector
  delays, two- and three-detector likelihoods, injection templates, and detector-frame sky coordinates.
- `src/other_utils.py`: data conditioning, time-domain autocorrelation and Cholesky factor, optimal SNR.
- `src/swsh.py`: spin-weighted spherical harmonics. `src/strain_cache.py`: GWOSC download with a local cache.
- `src/plot_utils.py`: the paper's plots.
- `examples/plots.ipynb`: reproduces the figures of the paper, and the Bayes factors and posterior-volume ratios it
  quotes, from the data on Zenodo.
- `examples/analysis.ipynb`: how to run Full-sky, Fixed-sky and Informed-sky analyses, with Virgo, at later start
  times, in detector-frame coordinates, and on a ringdown-only injection.
- `examples/GW190521-step-by-step.ipynb`: the GW190521 Full-sky and Fixed-sky analysis written out step by step
  with the library modules only (without `analysis.py`): data, windows, noise covariance, likelihoods, sampling, ln B.
- `tests/smoke_test.py`: checks the likelihoods against the paper's runs (including H1-L1-V1), the injection SNRs,
  the detector-frame transforms, a short sampler run, and `plots.ipynb`.
- `data/`: QNM frequency tables (l = 2, 3) and the PSDs of the ringdown-only injections.

## Installation
```bash
pip install -r requirements.txt
```
Use a fresh Python 3.12 environment. The plots render text with LaTeX, so a LaTeX installation is needed.

## Reproducing the figures of the paper
1. Download `posterior_files.zip` from Zenodo, [10.5281/zenodo.20089761](https://doi.org/10.5281/zenodo.20089761).
   Use version 3 or later; earlier versions hold the data of the first submission.
2. Unzip it, and point `SKYRING_DATA` to the `posterior_files` folder, or edit `DATA` in the notebook.
3. Run `examples/plots.ipynb`. The figures are written to `examples/plots/`.

## Running an analysis
```python
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=12"   # CPU devices, before importing jax
import analysis as an                                                     # with src/ on sys.path

setup = an.setup_event("GW250114", sky="full")             # downloads and caches the GWOSC strain
modes = ((2, 2, 0), (2, 2, 1))
names, low, high = an.prior_box(setup, modes)
result = an.run_nested(an.make_loglik(setup, modes), low, high, nlive=10_000)
```
See `examples/analysis.ipynb` for Fixed-sky, Informed-sky, GW190521, Virgo (`detectors=("H1", "L1", "V1")`),
later start times, detector-frame sky coordinates (`sky_frame="detector"`) and ringdown-only injections. With 10^4 live points, a Full-sky run takes about 35 minutes on a laptop (Apple M4).

## Tests
```bash
SKYRING_DATA=/path/to/posterior_files pytest -q tests/smoke_test.py
```
