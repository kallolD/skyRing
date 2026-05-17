import numpy as onp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import to_rgba
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FuncFormatter
import corner
import seaborn as sns
from chainconsumer import ChainConsumer, Chain, ChainConfig, PlotConfig

def default_bound_for_param(p):
    if p == r'$M$':
        return (50.0, 90.0)
    if p == r'$\chi$':
        return (0.0, 0.99)
    if p.startswith(r'$A_{'):
        return (0.0, 4.0)
    if p.startswith(r'$\phi_{'):
        return (0.0, 2 * onp.pi)
    if p == r'$\cos\iota$':
        return (-1.0, 1.0)
    if p == r'$\alpha$':
        return (0.0, 6.28)
    if p == r'$\sin \delta$':
        return (-1.0, 1.0)
    if p == r'$\psi$':
        return (0.0, 3.14)

    raise ValueError(f"No default bounds known for parameter: {p}")

def make_corner_plot(
    *,
    datasets,
    full_sky_params,
    bounds=None,
    truths=None,
    fixed_sky_location=None,
    figsize=(16, 16),
    bins=60,
    levels=(0.683, 0.954),
    tick_overrides=None,
    savepath=None,
):
    """
    Parameters
    ----------
    datasets : list of dict

        Each dict must contain:

        {
            "samples": ndarray,
            "params": list[str],
            "label": str,
            "color": str,
        }

        Optional:
            "diag": bool
            "contour": bool
            "alpha_outer": float
            "alpha_inner": float

    truths : ndarray or None

    fixed_sky_location : dict or None

        {
            "alpha": float,
            "sin_delta": float,
            "color": str (optional),
            "label": str (optional),
        }
    """

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 10,
        "axes.grid": False,
        "axes.linewidth": 1,
    })

    ndim = len(full_sky_params)

    default_bounds = {p: default_bound_for_param(p) for p in full_sky_params}

    if bounds is None:
        bounds = {}

    bounds = {**default_bounds, **bounds}
    ranges = [bounds[p] for p in full_sky_params]

    base = datasets[0]

    fig = corner.corner(
        base["samples"],
        labels=full_sky_params,
        range=ranges,
        color=base["color"],
        bins=bins,
        smooth=None,
        plot_datapoints=False,
        plot_density=False,
        plot_contours=True,
        fill_contours=False,
        levels=levels,
        label_kwargs={"fontsize": 20},
        hist_kwargs={
            "linewidth": 1.,
            "histtype": "step",
            "density": True,
        },
        contour_kwargs={"linewidths": 1.},
        show_titles=False,
    )

    fig.set_size_inches(*figsize)

    axes = onp.array(fig.axes).reshape((ndim, ndim))

    def density_hist(x, param):
        return onp.histogram(
            x,
            bins=bins,
            range=bounds[param],
            density=True,
        )

    def get_levels_from_probs(H, probs=(0.954, 0.683)):
        Hflat = H.flatten()
        Hflat = Hflat[onp.isfinite(Hflat)]

        if Hflat.size == 0 or Hflat.max() <= 0:
            return None

        Hsort = onp.sort(Hflat)[::-1]

        cumsum = onp.cumsum(Hsort)
        cumsum /= cumsum[-1]

        levels_out = []

        for p in probs:
            levels_out.append(Hsort[onp.searchsorted(cumsum, p)])

        return onp.sort(levels_out)

    def filled_contours(
        samples,
        color,
        param_list,
        inds,
        alpha_outer=0.18,
        alpha_inner=0.65,
    ):
        for a, i in enumerate(inds):
            for b, j in enumerate(inds[:a]):

                x = samples[:, b]
                y = samples[:, a]

                xb = bounds[param_list[b]]
                yb = bounds[param_list[a]]

                H, xedges, yedges = onp.histogram2d(
                    x,
                    y,
                    bins=bins,
                    range=[xb, yb],
                    density=True,
                )

                levels_local = get_levels_from_probs(H)

                if levels_local is None:
                    continue

                vmax = onp.nanmax(H)

                if (
                    not onp.isfinite(vmax)
                    or vmax <= levels_local[-1]
                ):
                    continue

                xcenters = 0.5 * (
                    xedges[1:] + xedges[:-1]
                )

                ycenters = 0.5 * (
                    yedges[1:] + yedges[:-1]
                )

                ax = axes[i, j]

                ax.contourf(
                    xcenters,
                    ycenters,
                    H.T,
                    levels=[
                        levels_local[0],
                        levels_local[1],
                        vmax,
                    ],
                    colors=[
                        to_rgba(color, alpha_outer),
                        to_rgba(color, alpha_inner),
                    ],
                    zorder=1,
                )

                ax.contour(
                    xcenters,
                    ycenters,
                    H.T,
                    levels=levels_local,
                    colors=[color],
                    linewidths=1.,
                    alpha=0.9,
                    zorder=2,
                )

    # -------------------------------------------------------
    # Diagonal histograms
    # -------------------------------------------------------

    diag_ymax = {}

    for i, p in enumerate(full_sky_params):

        ax = axes[i, i]
        ax.cla()

        ymax = 0.0

        for ds in datasets:

            if p not in ds["params"]:
                continue

            if not ds.get("diag", True):
                continue

            col = ds["params"].index(p)

            h, edges = density_hist(
                ds["samples"][:, col],
                p,
            )

            ymax = max(ymax, onp.nanmax(h))

            ax.stairs(
                h,
                edges,
                color=ds["color"],
                linewidth=1.,
            )

        ax.set_xlim(bounds[p])
        ax.set_ylim(0.0, 1.15 * ymax)
        ax.set_yticks([])

        diag_ymax[p] = 1.15 * ymax

    # -------------------------------------------------------
    # Filled contours
    # -------------------------------------------------------

    for ds in datasets:

        if not ds.get("contour", True):
            continue

        inds = [
            full_sky_params.index(p)
            for p in ds["params"]
        ]

        filled_contours(
            ds["samples"],
            ds["color"],
            ds["params"],
            inds,
            alpha_outer=ds.get("alpha_outer", 0.18),
            alpha_inner=ds.get("alpha_inner", 0.65),
        )

    # -------------------------------------------------------
    # Truth markers
    # -------------------------------------------------------

    if truths is not None:

        for i in range(ndim):

            axes[i, i].axvline(
                truths[i],
                color="black",
                linewidth=1.,
                linestyle="--",
                zorder=100,
            )

            for j in range(i):

                ax = axes[i, j]

                ax.axvline(
                    truths[j],
                    color="black",
                    linewidth=1.,
                    linestyle="--",
                    zorder=100,
                )

                ax.axhline(
                    truths[i],
                    color="black",
                    linewidth=1.,
                    linestyle="--",
                    zorder=100,
                )

                ax.plot(
                    truths[j],
                    truths[i],
                    marker="o",
                    color="black",
                    markersize=4,
                    zorder=101,
                )

    # -------------------------------------------------------
    # Fixed sky location
    # -------------------------------------------------------

    if fixed_sky_location is not None:

        fixed_sky_color = fixed_sky_location.get(
            "color",
            "#d62728",
        )

        fixed_sky_label = fixed_sky_location.get(
            "label",
            "Fixed sky location",
        )

        i_alpha = full_sky_params.index(r'$\alpha$')
        i_sindelta = full_sky_params.index(r'$\sin \delta$')

        alpha_value = fixed_sky_location["alpha"]
        sin_delta_value = fixed_sky_location["sin_delta"]

        fixed_sky_values = {
            i_alpha: alpha_value,
            i_sindelta: sin_delta_value,
        }

        for i in range(ndim):
            for j in range(i + 1):

                ax = axes[i, j]

                if j in fixed_sky_values:
                    ax.axvline(
                        fixed_sky_values[j],
                        color=fixed_sky_color,
                        linestyle="--",
                        linewidth=2.0,
                        zorder=101,
                    )

                if i != j and i in fixed_sky_values:
                    ax.axhline(
                        fixed_sky_values[i],
                        color=fixed_sky_color,
                        linestyle="--",
                        linewidth=2.0,
                        zorder=101,
                    )

        axes[i_sindelta, i_alpha].plot(
            alpha_value,
            sin_delta_value,
            marker="X",
            color=fixed_sky_color,
            markersize=5,
            linestyle="None",
            zorder=102,
        )

    else:
        fixed_sky_color = None
        fixed_sky_label = None

    # -------------------------------------------------------
    # Formatting
    # -------------------------------------------------------

    for i in range(ndim):
        for j in range(ndim):

            ax = axes[i, j]

            if i < j:
                ax.set_visible(False)
                continue

            ax.set_xlim(bounds[full_sky_params[j]])

            if i == j:
                ax.set_ylim(
                    0.0,
                    diag_ymax[full_sky_params[i]],
                )
                ax.set_yticks([])

            else:
                ax.set_ylim(bounds[full_sky_params[i]])

            if i != ndim - 1:
                ax.tick_params(
                    axis="x",
                    labelbottom=False,
                )
                ax.set_xlabel("")

            if j != 0:
                ax.tick_params(
                    axis="y",
                    labelleft=False,
                )
                ax.set_ylabel("")

            ax.xaxis.set_major_locator(
                MaxNLocator(3)
            )

            ax.yaxis.set_major_locator(
                MaxNLocator(3)
            )

            ax.tick_params(labelsize=15)
            ax.grid(False)

    axes[-1, -1].set_xlabel(
        r'$\psi$',
        fontsize=20,
        labelpad=24,
    )

    axes[-1, -1].xaxis.set_label_coords(
        0.5,
        -0.42,
    )

    for tick in axes[-1, -1].get_xticklabels():
        tick.set_rotation(45)

    axes[0, 0].set_yticks([])
    axes[0, 0].set_ylabel("")

    axes[0, 0].tick_params(
        axis="y",
        left=False,
        labelleft=False,
    )

    # -------------------------------------------------------
    # Manual tick overrides
    # -------------------------------------------------------

    if tick_overrides is not None:

        for (i, j), cfg in tick_overrides.items():

            ax = axes[i, j]

            if "x" in cfg:
                ax.set_xticks(cfg["x"])

            if "y" in cfg:
                ax.set_yticks(cfg["y"])

    # -------------------------------------------------------
    # Legend
    # -------------------------------------------------------

    legend_handles = []

    for ds in datasets:

        legend_handles.append(
            mlines.Line2D(
                [],
                [],
                color=ds["color"],
                label=ds["label"],
                linewidth=2,
            )
        )

    if truths is not None:

        legend_handles.append(
            mlines.Line2D(
                [],
                [],
                color="black",
                label="Injected values",
                linewidth=2,
                linestyle="--",
            )
        )

    if fixed_sky_location is not None:

        legend_handles.append(
            mlines.Line2D(
                [],
                [],
                color=fixed_sky_color,
                label=fixed_sky_label,
                linewidth=2,
                linestyle="--",
            )
        )

    fig.legend(
        handles=legend_handles,
        labelcolor="linecolor",
        fontsize=25,
        loc="upper right",
        frameon=False,
    )

    if savepath is not None:

        plt.savefig(
            savepath,
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()

    return None


def make_amplitude_ratio_plot(
    *,
    datasets,
    numerator_index,
    denominator_index,
    ratio_label,
    bins=80,
    smooth=0,
    sigmas=[0, 1, 2],
    label_font_size=20,
    tick_font_size=15,
    contour_label_font_size=24,
    legend_font_size=15,
    legend_loc="upper right",
    savepath=None,
):
    """
    Parameters
    ----------
    datasets : list of dict

        Each dict must contain:
        {
            "samples": ndarray,
            "label": str,
            "color": str,
        }

    numerator_index : int
        Column index of numerator amplitude.

    denominator_index : int
        Column index of denominator amplitude.

    ratio_label : str
        Example:
            r'$A_{330}/A_{220}$'
    """

    c = ChainConsumer()

    for ds in datasets:
        ratio_samples = (ds["samples"][:, numerator_index]/ds["samples"][:, denominator_index])
        df = pd.DataFrame( ratio_samples, columns=[ratio_label])
        c.add_chain(Chain( samples=df, color=ds["color"], name=ds["label"]))

    c.set_override(ChainConfig(sigmas=sigmas, bins=bins, smooth=smooth))

    c.set_plot_config(
        PlotConfig(
            usetex=True,
            serif=True,
            label_font_size=label_font_size,
            tick_font_size=tick_font_size,
            contour_label_font_size=contour_label_font_size,
            legend_kwargs={
                "fontsize": legend_font_size,
                "loc": legend_loc,
            },
        )
    )

    fig = c.plotter.plot()

    if savepath is not None:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")

    plt.show()

    return None





def make_amp_ratio_tshifts_plot(
    *,
    event_id,
    full_dir,
    fixed_dir,
    full_220_dir,
    fixed_220_dir,
    ratio_label,
    logb_label,
    num_idx=4,
    den_idx=2,
    bins=None,
    trim=None,
    ylim=None,
    bf_yticks=None,
    savepath=None,
):
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 22,
        "axes.linewidth": 1,
    })

    palette = {
        "Full-sky": "#5187ae",
        "Fixed-sky": "#c06161",
    }

    t_shifts = onp.loadtxt(f"{full_dir}/t_shifts.txt")
    n = len(t_shifts)

    logb_full = onp.loadtxt(f"{full_dir}/logZs.txt") - onp.loadtxt(f"{full_220_dir}/logZs_220.txt")
    logb_fixed = onp.loadtxt(f"{fixed_dir}/logZs.txt") - onp.loadtxt(f"{fixed_220_dir}/logZs_220.txt")

    rows = []

    for i in range(n):
        samples = {
            "Full-sky": onp.load(f"{full_dir}/posterior.{event_id}.{i}.npy"),
            "Fixed-sky": onp.load(f"{fixed_dir}/posterior.{event_id}.{i}.npy"),
        }

        for name, arr in samples.items():
            vals = arr[:, num_idx] / arr[:, den_idx]

            if trim is not None:
                lo, hi = onp.quantile(vals, trim)
                vals = vals[(vals >= lo) & (vals <= hi)]

            rows.extend(
                {"index": f"{i}", "value": v, "type": name}
                for v in vals
            )

    df = pd.DataFrame(rows)

    fig, (ax, ax_bf) = plt.subplots(2, 1, figsize=(15, 5), sharex=True, 
                                    gridspec_kw={"height_ratios": [3.5, 1.5], "hspace": 0.05},)

    sns.violinplot(data=df, x="index", y="value", hue="type", split=True, inner=None,
                   cut=0, palette=palette, saturation=1, ax=ax)

    ax.legend(title=None, frameon=True, loc="upper left", fontsize=20)
    ax.set_ylabel(ratio_label)
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: rf"${y:g}$"))

    if ylim is not None:
        ax.set_ylim(*ylim)

    x = onp.arange(n)

    ax_bf.plot(x, logb_full, marker="o", linewidth=2, label="Full-sky", color=palette["Full-sky"])
    ax_bf.plot(x, logb_fixed, marker="o", linewidth=2, label="Fixed-sky", color=palette["Fixed-sky"])
    ax_bf.axhline(0, color="k", linewidth=1, alpha=0.5)

    ax_bf.set_ylabel(logb_label)
    ax_bf.set_xlabel(r"$t - t_{\mathrm{ref}}\ [t_{M}]$", labelpad=10)
    ax_bf.legend(title=None, frameon=True, loc="best", fontsize=18)
    ax_bf.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: rf"${y:g}$"))

    if bf_yticks is not None:
        ax_bf.set_yticks(bf_yticks)

    secax = ax.secondary_xaxis("top")
    secax.set_xticks(ax.get_xticks())
    secax.set_xticklabels([rf"${onp.round(t * 1e3, 2)}$" for t in t_shifts])
    secax.set_xlabel(r"$t - t_{\mathrm{ref}}\ \mathrm{[ms]}$", labelpad=10)

    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")

    plt.show()

    return None