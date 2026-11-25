"""
Smoke test: the library reproduces the likelihoods of the paper's runs, a short nested-sampling run completes, and
examples/plots.ipynb reproduces the paper's figures and quoted numbers from the Zenodo data.

    SKYRING_DATA=/path/to/posterior_files pytest -q tests/smoke_test.py

SKYRING_DATA is the unzipped Zenodo folder; its strain/ subfolder holds the GWOSC strain used here. Without it, the
strain is downloaded from GWOSC into a temporary folder and the Informed-sky and plot tests are skipped.
"""

import os
import sys
import json
import contextlib
import io

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as onp
import pytest
import jax.numpy as jnp

import analysis as an

DATA = os.environ.get("SKYRING_DATA")
needs_data = pytest.mark.skipif(not DATA, reason="set SKYRING_DATA to the unzipped Zenodo posterior_files folder")
REFERENCE = json.load(open(os.path.join(ROOT, "tests", "reference.json")))

# numbers quoted in the paper
PAPER_LNB = {"GW250114 full": 7.42, "GW250114 fixed": 9.26, "GW250114 informed": 7.66,
             "GW190521 full": 2.44, "GW190521 fixed": 0.82}
PAPER_VOLUMES = {"GW250114 fixed": 0.16, "GW250114 informed": 0.32, "GW190521 fixed": 0.12,
                 "NRSur fixed": 0.05, "NRSur informed": 0.17}
FIGURES = ["GW250114.pdf", "GW250114_amp_ratio.pdf", "GW250114_amp_ratio_tshifts_with_logB.pdf",
           "GW190521.pdf", "GW190521_amp_ratio.pdf", "GW190521_amp_ratio_tshifts_with_logB.pdf",
           "noiseless-sim-221-220.pdf", "noiseless-sim-221-220-amp-ratio.pdf", "noiseless-sim-330-220.pdf",
           "noiseless-sim-330-220-amp-ratio.pdf", "NRSur_injection.pdf", "NRSur_injection_amp_ratio.pdf",
           "GW250114_analysis_windows.pdf"]


@pytest.fixture(scope="module")
def strain_dir(tmp_path_factory):
    return os.path.join(DATA, "strain") if DATA else str(tmp_path_factory.mktemp("strain_cache"))


@pytest.fixture(scope="module")
def setups(strain_dir):
    quiet = lambda *a, **k: None
    return {(ev, sky): an.setup_event(ev, sky=sky, cache_dir=strain_dir, log=quiet)
            for ev in ("GW250114", "GW190521") for sky in ("full", "fixed")}


@pytest.mark.parametrize("case", [c for c in REFERENCE["cases"] if c["sky"] != "informed"],
                         ids=lambda c: f"{c['event']}-{c['sky']}-{len(c['modes'])}modes")
def test_likelihood_matches_paper_runs(case, setups):
    setup = setups[(case["event"], case["sky"])]
    names, _, _ = an.prior_box(setup, case["modes"])
    assert names == case["names"]
    loglik = an.make_loglik(setup, case["modes"])
    for theta, expected in zip(case["theta"], case["lnL"]):
        assert float(loglik(jnp.asarray(theta))) == pytest.approx(expected, rel=1e-9)


@needs_data
def test_informed_likelihood_matches_paper_run(setups):
    case = next(c for c in REFERENCE["cases"] if c["sky"] == "informed")
    sky = onp.load(os.path.join(DATA, "imr_sky", "GW250114_C00_NRSur7dq4_sky.npz"))
    nodes, logw = an.imr_sky_nodes(sky["ra"], sky["dec"])
    saved = onp.load(os.path.join(DATA, "informed-sky", "posterior.GW250114.informed-sky.npz"))
    assert onp.array_equal(onp.asarray(nodes), saved["nodes"])          # the nodes of the paper's run
    loglik = an.make_informed_loglik(setups[("GW250114", "full")], case["modes"], nodes, logw)
    for theta, expected in zip(case["theta"], case["lnL"]):
        assert float(loglik(jnp.asarray(theta))) == pytest.approx(expected, rel=1e-9)


def test_short_nested_sampling_run(setups):
    setup = setups[("GW250114", "full")]
    modes = ((2, 2, 0),)
    _, low, high = an.prior_box(setup, modes)
    res = an.run_nested(an.make_loglik(setup, modes), low, high, nlive=200, num_posterior=1000, verbose=False)
    assert res["converged"] and onp.isfinite(res["log_Z"]) and res["theta"].shape == (1000, len(low))


@needs_data
def test_plots_notebook_reproduces_paper(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    monkeypatch.setenv("SKYRING_PLOTS", str(tmp_path))
    monkeypatch.chdir(os.path.join(ROOT, "examples"))
    nb = json.load(open("plots.ipynb"))
    ns = {"__name__": "__main__"}
    with contextlib.redirect_stdout(io.StringIO()):
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                exec(compile("".join(cell["source"]), cell.get("id", "cell"), "exec"), ns)
    missing = [f for f in FIGURES if not (tmp_path / f).exists()]
    assert not missing, f"figures not written: {missing}"

    def lnb(folder, stem):
        m, c = (onp.load(os.path.join(DATA, folder, f"{stem}{s}.npz")) for s in ("", ".220"))
        return float(m["log_Z"] - c["log_Z"])
    got = {"GW250114 full": lnb("full-sky", "posterior.GW250114.full-sky"),
           "GW250114 fixed": lnb("fixed-sky", "posterior.GW250114.fixed-sky"),
           "GW250114 informed": lnb("informed-sky", "posterior.GW250114.informed-sky"),
           "GW190521 full": lnb("full-sky", "posterior.GW190521.full-sky"),
           "GW190521 fixed": lnb("fixed-sky", "posterior.GW190521.fixed-sky")}
    assert {k: round(v, 2) for k, v in got.items()} == PAPER_LNB

    th = lambda *p: onp.load(os.path.join(DATA, *p))["theta"]
    drop_sky = [0, 1, 2, 3, 4, 5, 6, 9]
    b250114 = (onp.array([55., 0., 0., 0., 0., 0.]), onp.array([95., 0.99, 4., 0., 4., 0.]))
    b190521 = (onp.array([150., 0., 0., 0., 0., 0.]), onp.array([500., 0.99, 4.5, 0., 0.6, 0.]))
    full_250114 = th("full-sky", "posterior.GW250114.full-sky.npz")[:, drop_sky]
    full_nrsur = th("simulations_NRSur7dq4", "posterior.NRSur-injection.full-sky.npz")[:, drop_sky]
    vol = {"GW250114 fixed": ns["vol_ratio"](th("fixed-sky", "posterior.GW250114.fixed-sky.npz"), full_250114, *b250114),
           "GW250114 informed": ns["vol_ratio"](th("informed-sky", "posterior.GW250114.informed-sky.npz"), full_250114, *b250114),
           "GW190521 fixed": ns["vol_ratio"](th("fixed-sky", "posterior.GW190521.fixed-sky.npz"),
                                             th("full-sky", "posterior.GW190521.full-sky.npz")[:, drop_sky], *b190521),
           "NRSur fixed": ns["vol_ratio"](th("simulations_NRSur7dq4", "posterior.NRSur-injection.fixed-sky.npz"), full_nrsur, *b250114),
           "NRSur informed": ns["vol_ratio"](th("simulations_NRSur7dq4", "posterior.NRSur-injection.informed-sky.npz"), full_nrsur, *b250114)}
    assert {k: round(v, 2) for k, v in vol.items()} == PAPER_VOLUMES
