# Installation
Install dependencies using 
```bash
pip install -r requirements.txt
``` 
(preferably in a fresh python environment)

# skyRing
This repository contains code for estimating the sky location of sources, in addition to all other relevant parameters, directly from the ringdown gravitational-wave signal, without fixing the location to point estimates obtained from inspiral–merger–ringdown (IMR) analyses. We present three approaches: the standard Fixed-sky method, along with two alternatives that avoid relying on fixed point estimates — Full-sky and Informed-sky. See (arXiv: [link when available]) for more details.

# To reproduce plots from the paper
Samples from PE runs for GW250114 and GW190521 are already available at zenodo ([10.5281/zenodo.20089761](https://doi.org/10.5281/zenodo.20089761))
To reproduce the plots from the paper, download the zenodo data and use the `examples/plots.ipynb` jupyter notebook to plot the results.

# Using the code
Examples are provided in the examples directory in the form of jupyter notebooks.
We provide three approaches to study ringdown
1. Full-sky: Sky location is sampled from a uniform distritbution.
2. Fixed-sky: Sky location is fixed to a `best estimate' from a prior inspiral-merger-ringdown analysis.
3. Informed-sky: Sky location is sampled using posteriors from a previous inspiral-merger-ringdown analysis as the prior.

The analyses using the three approaches are presented in separate directories.

We analyze GW250114 and GW190521; jupyter notebooks for the same can be found in the respective directories.
For instance, the full-sky analysis for GW250114 is presented in `examples/full-sky/GW250114.ipynb`.

These notebooks were tested on macOS. You can tune CPU multi-threading performance by modifying the environment flags at the start of each notebook.

```python
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=12"
```

