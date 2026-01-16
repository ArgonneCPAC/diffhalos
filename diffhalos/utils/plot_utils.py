# flake8: noqa

import math

import numpy as np

__all__ = (
    "plot_awesome",
    "plot_triangle_GetDist",
)


def plot_awesome(dpi=175, fontsize=9):
    """Function to beautify plots with matplotlib"""
    import matplotlib.pyplot as plt

    # set some parameters to make plots prettier
    plt.rc("savefig", dpi=dpi)
    plt.rc("figure", dpi=dpi)
    plt.rc("text", usetex=True)
    plt.rc("font", size=fontsize)
    plt.rc("xtick", direction="in")
    plt.rc("ytick", direction="in")
    plt.rc("xtick.major", pad=5)
    plt.rc("xtick.minor", pad=5)
    plt.rc("ytick.major", pad=5)
    plt.rc("ytick.minor", pad=5)
    plt.rc("lines", dotted_pattern=[0.5, 1.1])
    return


def plot_triangle_GetDist(
    chains,
    names=None,
    weights=None,
    labels=None,
    label=None,
    settings=None,
    analysis_settings=None,
    subplot_size_inch=2,
    fig_width_inch=None,
    shaded=False,
    param_3d=None,
    fontsize=9,
    lab_fontsize=11,
    legend_fontsize=14,
    filled=None,
    colors=None,
    legend_labels=None,
    legend_loc=None,
    line_args=None,
    markers=None,
    contour_colors=None,
    plot_meanlikes=False,
    param_limits={},
    alpha_filled_add=0.8,
    axis_marker_color="k",
    axis_marker_lw=2.5,
    legend_frame=True,
    tight_layout=True,
    alpha_factor_contour_lines=0.5,
    num_plot_contours=2,
    linewidth_contour=1.0,
    ignore_rows=0.5,
    smooth_scale_1D=0.5,
    smooth_scale_2D=0.5,
    contour_lws=None,
    contour_ls=None,
    axes_fontsize=9,
    legend_rect_border=False,
    figure_legend_ncol=1,
    tick_width=1.0,
    axis_width=1.0,
    tick_size_major=4.5,
    tick_size_minor=2.0,
    legend_linewidth=1.0,
    legend_edgecolor="k",
    add_vlines=None,
    ls_vlines=None,
    lc_vlines=None,
    boundary_correction_order=1,
    mult_bias_correction_order=1,
    scaling_factor=1.5,
    scaling=True,
    figname=None,
    dpi=500,
):
    """
    Plot triangle for a set of chains using GetDist
    """
    import getdist.plots as gplots
    from getdist import MCSamples

    plot_awesome(dpi=dpi, fontsize=fontsize)

    samps = []
    for _chain in chains:
        samps.append(
            MCSamples(
                samples=_chain,
                weights=weights,
                names=names,
                labels=labels,
                label=label,
            )
        )

    # plotting settings
    if settings is None:
        settings = gplots.GetDistPlotSettings(
            subplot_size_inch=subplot_size_inch, fig_width_inch=fig_width_inch
        )
        settings.axis_marker_color = axis_marker_color
        settings.axis_marker_lw = axis_marker_lw
        settings.num_plot_contours = num_plot_contours
        settings.linewidth_contour = linewidth_contour
        settings.tight_layout = tight_layout
        settings.alpha_factor_contour_lines = alpha_factor_contour_lines
        settings.lab_fontsize = lab_fontsize
        settings.axes_fontsize = axes_fontsize
        settings.alpha_filled_add = alpha_filled_add
        settings.figure_legend_frame = legend_frame
        settings.legend_loc = legend_loc
        settings.legend_fontsize = legend_fontsize
        settings.legend_rect_border = legend_rect_border
        settings.figure_legend_ncol = figure_legend_ncol
        settings.scaling_factor = scaling_factor
        settings.scaling = scaling

    if analysis_settings is None:
        analysis_settings = {
            "ignore_rows": ignore_rows,
            "smooth_scale_1D": smooth_scale_1D,
            "smooth_scale_2D": smooth_scale_2D,
        }

    # make triangle plot
    g = gplots.getPlotter(settings=settings, analysis_settings=analysis_settings)
    g.triangle_plot(
        samps,
        filled=filled,
        colors=colors,
        plot_3d_with_param=param_3d,
        shaded=shaded,
        legend_labels=legend_labels,
        line_args=line_args,
        contour_colors=contour_colors,
        plot_meanlikes=plot_meanlikes,
        param_limits=param_limits,
        contour_lws=contour_lws,
        contour_ls=contour_ls,
        legend_loc=legend_loc,
        boundary_correction_order=boundary_correction_order,
        mult_bias_correction_order=mult_bias_correction_order,
        markers=markers,
    )

    # legend box settings
    g.legend.get_frame().set_linewidth(legend_linewidth)
    g.legend.get_frame().set_edgecolor(legend_edgecolor)

    # add vertical lines
    if add_vlines is not None and len(add_vlines) > 0:
        if ls_vlines is None:
            ls_vlines = ["--"] * len(chains)
        if lc_vlines is None:
            lc_vlines = ["k"] * len(chains)
        for i in range(len(names)):
            if add_vlines[i] is not None:
                ax = g.subplots[i, i]
                ax.axvline(
                    add_vlines[i],
                    color=lc_vlines[i],
                    ls=ls_vlines[i],
                )

    # axis settings
    for i in range(len(names)):
        for j in range(len(names)):
            if j <= i:
                ax = g.subplots[i, j]
                for axis in ["top", "bottom", "left", "right"]:
                    ax.spines[axis].set_linewidth(axis_width)
                    ax.xaxis.set_tick_params(which="major", width=tick_width)
                    ax.yaxis.set_tick_params(which="major", width=tick_width)
                    ax.xaxis.set_tick_params(which="minor", width=tick_width)
                    ax.yaxis.set_tick_params(which="minor", width=tick_width)
                    ax.xaxis.set_tick_params(which="major", size=tick_size_major)
                    ax.yaxis.set_tick_params(which="major", size=tick_size_major)
                    ax.xaxis.set_tick_params(which="minor", size=tick_size_minor)
                    ax.yaxis.set_tick_params(which="minor", size=tick_size_minor)
                    ax.tick_params(top=False, right=False)
                if j == 0:
                    ax.tick_params(axis="y", which="major", direction="out")
                    ax.tick_params(axis="y", which="minor", direction="out")
                if i == len(names) - 1:
                    ax.tick_params(axis="x", which="major", direction="out")
                    ax.tick_params(axis="x", which="minor", direction="out")
                else:
                    ax.tick_params(axis="x", which="major", direction="in")
                    ax.tick_params(axis="x", which="minor", direction="in")

    # save plot and exit
    if figname is not None:
        g.export(fname=figname, dpi=dpi)

    return


# ******************************************************************************
# Definition of some pretty color maps
# ------------------------------------------------------------------------------

color_maps = dict()

# ------------------------------------------------------------------------------

color_maps["the_gold_standard"] = {
    0: (203.0 / 255.0, 15.0 / 255.0, 40.0 / 255.0),
    1: (255.0 / 255.0, 165.0 / 255.0, 0.0),
    2: (42.0 / 255.0, 46.0 / 255.0, 139.0 / 255.0),
    3: (0.0 / 255.0, 153.0 / 255.0, 204.0 / 255.0),
    4: (0.0 / 255.0, 221.0 / 255.0, 52.0 / 255.0),
    5: (0.0, 0.75, 0.75),
    6: (0.0, 0.0, 0.0),
}

# ------------------------------------------------------------------------------

color_maps["spring_and_winter"] = {
    0: (93.0 / 255.0, 50.0 / 255.0, 137.0 / 255.0),
    1: (197.0 / 255.0, 43.0 / 255.0, 135.0 / 255.0),
    2: (237.0 / 255.0, 120.0 / 255.0, 159.0 / 255.0),
    3: (241.0 / 255.0, 147.0 / 255.0, 130.0 / 255.0),
    4: (113.0 / 255.0, 187.0 / 255.0, 220.0 / 255.0),
    5: (24.0 / 255.0, 120.0 / 255.0, 187.0 / 255.0),
}

# ------------------------------------------------------------------------------

color_maps["winter_and_spring"] = {
    0: (24.0 / 255.0, 120.0 / 255.0, 187.0 / 255.0),
    1: (113.0 / 255.0, 187.0 / 255.0, 220.0 / 255.0),
    2: (241.0 / 255.0, 147.0 / 255.0, 130.0 / 255.0),
    3: (237.0 / 255.0, 120.0 / 255.0, 159.0 / 255.0),
    4: (197.0 / 255.0, 43.0 / 255.0, 135.0 / 255.0),
    5: (93.0 / 255.0, 50.0 / 255.0, 137.0 / 255.0),
}

# ------------------------------------------------------------------------------

color_maps["summer_sun"] = {
    0: (234.0 / 255.0, 185.0 / 255.0, 185.0 / 255.0),
    1: (234.0 / 255.0, 90.0 / 255.0, 103.0 / 255.0),
    2: (255.0 / 255.0, 231.0 / 255.0, 76.0 / 255.0),
    3: (249.0 / 255.0, 179.0 / 255.0, 52.0 / 255.0),
    4: (55.0 / 255.0, 97.0 / 255.0, 140.0 / 255.0),
    5: (82.0 / 255.0, 158.0 / 255.0, 214.0 / 255.0),
}

# ------------------------------------------------------------------------------

color_maps["summer_sky"] = {
    0: (82.0 / 255.0, 158.0 / 255.0, 214.0 / 255.0),
    1: (55.0 / 255.0, 97.0 / 255.0, 140.0 / 255.0),
    2: (249.0 / 255.0, 179.0 / 255.0, 52.0 / 255.0),
    3: (255.0 / 255.0, 231.0 / 255.0, 76.0 / 255.0),
    4: (234.0 / 255.0, 90.0 / 255.0, 103.0 / 255.0),
    5: (234.0 / 255.0, 185.0 / 255.0, 185.0 / 255.0),
}

# ------------------------------------------------------------------------------

color_maps["autumn_fields"] = {
    0: (50.0 / 255.0, 138.0 / 255.0, 165.0 / 255.0),
    1: (16.0 / 255.0, 135.0 / 255.0, 98.0 / 255.0),
    2: (198.0 / 255.0, 212.0 / 255.0, 60.0 / 255.0),
    3: (255.0 / 255.0, 251.0 / 255.0, 73.0 / 255.0),
    4: (237.0 / 255.0, 118.0 / 255.0, 40.0 / 255.0),
    5: (142.0 / 255.0, 26.0 / 255.0, 26.0 / 255.0),
}

# ------------------------------------------------------------------------------

color_maps["autumn_leaves"] = {
    0: (142.0 / 255.0, 26.0 / 255.0, 26.0 / 255.0),
    1: (237.0 / 255.0, 118.0 / 255.0, 40.0 / 255.0),
    2: (255.0 / 255.0, 251.0 / 255.0, 73.0 / 255.0),
    3: (198.0 / 255.0, 212.0 / 255.0, 60.0 / 255.0),
    4: (16.0 / 255.0, 135.0 / 255.0, 98.0 / 255.0),
    5: (50.0 / 255.0, 138.0 / 255.0, 165.0 / 255.0),
}

# ------------------------------------------------------------------------------

color_maps["shades_of_gray"] = {0: (90.0 / 255.0, 90.0 / 255.0, 90.0 / 255.0)}

# ******************************************************************************
# definition of color interpolation utilities


def color_linear_interpolation(rgb_1, rgb_2, alpha):
    """
    This function performs a linear color interpolation in RGB space.
    alpha has to go from zero to one and is the coordinate.
    """
    _out_color = []
    for _a, _b in zip(rgb_1, rgb_2):
        _out_color.append(_a + (_b - _a) * alpha)
    return tuple(_out_color)


# ******************************************************************************
# definition of the color helper


def nice_colors(
    num,
    colormap="the_gold_standard",
    interpolation_method="linear",
    output_format="RGB",
):
    """
    This function returns a color from a colormap defined above, according to the
    number entered.

    Parameters
    ----------
    num: int or float
        if the number is integer the function returns one of the colors in the
        colormap;
        if the number is a float returns the shade combining the two
        neighbouring colors.

    colormap: str
        name of the colormap

    interpolation_method: str
        method to interpolate between colors;
        options:
            - 'linear': linear interpolation

    output_format: str
        output format of the color
        options:
            - 'HEX'
            - 'RGB'

    Returns
    -------
    _out_color: str or tuple
        string with HEX color or tuple with RGB coordinates
    """
    # get the colormap
    try:
        _cmap = color_maps[colormap]
    except:
        raise ValueError(
            "Requested color map (" + str(colormap) + ") does not exist.",
        )

    # get the indexes of the color map
    _idx_low = int(math.floor(num % len(_cmap)))
    _idx_up = int(math.floor((_idx_low + 1) % len(_cmap)))

    # perform color interpolation
    if interpolation_method == "linear":
        _t = num % len(_cmap) - _idx_low
        _out_color = color_linear_interpolation(_cmap[_idx_low], _cmap[_idx_up], _t)
    else:
        raise ValueError(
            "Requested color interpolation method ("
            + str(interpolation_method)
            + ") does not exist."
        )

    # choose the output format
    if output_format == "HEX":
        _out_color = "#%02x%02x%02x" % tuple([_c * 255.0 for _c in _out_color])
        _out_color = str(_out_color)
    elif output_format == "RGB":
        pass
    else:
        raise ValueError(
            "Requested output format (" + str(output_format) + ") does not exist."
        )

    return _out_color


# vectorize nice_colors over the first argument
nice_colors = np.vectorize(
    nice_colors,
    excluded=["colormap", "interpolation_method", "output_format"],
)

################################################################################
