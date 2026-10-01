"""Notebook plots for the blog's finite Var and Joint distributions."""

import numpy as np
from dist_types import Var, Joint
from intro_answers import expect


def _pyplot():
    """Load Matplotlib only for the legacy static-figure functions."""
    import matplotlib.pyplot as plt
    return plt


def _number(value, signed=False):
    value = round(float(value), 2)
    if value == 0:
        value = 0.0
    return format(value, "+.2f" if signed else ".2f").rstrip("0").rstrip(".")


def _format_ticks(fig):
    from matplotlib.ticker import FuncFormatter
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(FuncFormatter(lambda value, pos: _number(value)))
        if hasattr(ax, "zaxis"):
            ax.zaxis.set_major_formatter(FuncFormatter(lambda value, pos: _number(value)))


def _scalar_mass(rv):
    """Collect scalar probability masses, merging floating-point duplicates."""
    values, mass = rv.values, rv.probs
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Expected a finite scalar distribution")
    if not np.isfinite(mass).all() or np.any(mass < 0) or mass.sum() <= 0:
        raise ValueError("Invalid probability masses")
    support, inverse = np.unique(np.round(values, 12), return_inverse=True)
    mass = np.bincount(inverse, weights=mass)
    keep = mass > 0
    return support[keep], mass[keep]/mass.sum()


def _range_masses(*distributions):
    """Sum masses on a shared, mean-centered grid of width 2% of the range."""
    values = np.concatenate([v for v, p in distributions])
    weights = np.concatenate([p/len(distributions) for v, p in distributions])
    support = np.unique(values)
    if len(support) == 1:
        return [(support.copy(), np.array([mass.sum()])) for values, mass in distributions]
    width = (support[-1]-support[0])*.02
    if np.all(np.diff(support) >= width):
        return [(v.copy(), p.copy()) for v, p in distributions]
    center = float(values @ weights / weights.sum())
    def bins(values):
        return np.floor((values-center)/width + .5 + 1e-12).astype(int)
    result = []
    for values, mass in distributions:
        indices, inverse = np.unique(bins(values), return_inverse=True)
        grouped = np.bincount(inverse, weights=mass)
        keep = grouped > 0
        result.append((center + indices[keep]*width, grouped[keep]))
    return result


def _discrete_dots(ax, support, mass, color="#4c8194", hollow=False):
    ax.vlines(support, 0, mass, color="black", linewidth=2.4, zorder=1)
    ax.scatter(support, mass, s=55,
               facecolors="white" if hollow else color, edgecolors=color,
               alpha=1, linewidths=1, zorder=3)
    ax.set_ylim(0, max(ax.get_ylim()[1], 1.08*float(mass.max())))


def _minimal_histogram_axes(ax, values):
    values = np.unique(values)
    labels = {}
    gap = (values[-1]-values[0])/7
    last = -np.inf
    for value in values:
        if value-last >= gap and _number(value) not in labels:
            labels[_number(value)] = value
            last = value
    ax.set_xticks(list(labels.values()), list(labels))
    ax.set_yticks([])
    ax.spines[:].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=9, colors="#263940", labelsize=14)


def variance_3d(x, **kwargs):
    """Squares with side |x-E[x]|, extruded by probability; volume is variance."""
    if kwargs.get("show", True):
        from IPython.display import display
        from plotly_viz import variance_widget
        widget = variance_widget(x)
        display(widget)
        return widget
    from intro_answers import shared
    return covariance_3d(shared(x), variance=True, **kwargs)


def joint_histogram(j: Joint, **kwargs):
    """Joint probability masses as labeled 3D posts, without covariance boxes."""
    if not isinstance(j, Joint):
        raise TypeError("Expected a Joint")
    return covariance_3d(j, boxes=False, **kwargs)


def covariance_3d(pair, variance=False, show=True, probability_limit=None,
                  max_post_labels=None, title=None, boxes=True):
    """Signed rectangle volumes p*(x-E[x])*(y-E[y]) for a joint scalar pair.

    Raw outcome axes; boxes run from the means to each outcome and up to p.
    Red lines locate the means. No independence is assumed.
    """
    if show:
        from IPython.display import display
        from plotly_viz import covariance_widget
        if isinstance(pair,Var):
            raise TypeError("Covariance needs a Joint")
        widget=covariance_widget(pair,variance=variance,boxes=boxes,max_labels=max_post_labels)
        display(widget)
        return widget
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    if isinstance(pair, Var):
        raise TypeError("Covariance needs a Joint; use shared(x) or indep(x, y)")
    x, y = np.meshgrid(pair._x, pair._y, indexing="ij")
    values = np.column_stack([x.ravel(), y.ravel()])
    mass = pair.probs.ravel()
    if not np.isfinite(values).all() or not np.isfinite(mass).all() or np.any(mass < 0):
        raise ValueError('Expected a finite joint distribution.')
    support, inverse = np.unique(values, axis=0, return_inverse=True)
    mass = np.bincount(inverse, weights=mass)
    keep = mass > 0
    support, mass = support[keep], mass[keep]
    mass = mass/mass.sum()
    center = mass @ support
    deviations = support-center
    covariance = float(mass @ (deviations[:, 0]*deviations[:, 1]))
    fig = _pyplot().figure(figsize=(8, 5.8), layout="constrained")
    fig.set_facecolor("#ffffff")
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    ax.set_facecolor("#ffffff")
    def box_edges(corners, color, width):
        for i, j in ((0, 1), (1, 2), (2, 3), (3, 0),
                     (4, 5), (5, 6), (6, 7), (7, 4),
                     (0, 4), (1, 5), (2, 6), (3, 7)):
            edge = np.array([corners[i], corners[j]])
            ax.plot(edge[:, 0], edge[:, 1], edge[:, 2], color=color, lw=width, alpha=.3)

    def corners(x0, x1, y0, y1, top):
        return [(x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0),
                (x0, y0, top), (x1, y0, top), (x1, y1, top), (x0, y1, top)]

    labeled = set(range(len(support))) if max_post_labels is None else set(
        np.linspace(0, len(support)-1, min(max_post_labels, len(support))).round().astype(int))
    for index, ((x, y), (dx, dy), p) in enumerate(zip(support, deviations, mass)):
        color = "#4c8194" if not boxes or dx*dy >= 0 else "#9a78a5"
        vertices = corners(center[0], x, center[1], y, p)
        faces = [[vertices[i] for i in indices] for indices in
                 ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
                  (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))]
        if boxes:
            ax.add_collection3d(Poly3DCollection(faces, facecolors=color, edgecolors="none", alpha=.1))
            box_edges(vertices, color, .8)
        ax.plot([x, x], [y, y], [0, p], color="black", lw=2.2)
        ax.scatter([x], [y], [p], color=color, s=60, depthshade=False, zorder=10)
        label = f"{_number(x)}" if variance else f"{_number(x)},{_number(y)}"
        if index in labeled:
            ax.text(x, y, 0, f"\n{label}", ha="center", va="top",
                    color="#263940", fontsize=14, clip_on=False, zorder=20)
    low = support.min(axis=0)
    high = support.max(axis=0)
    span = np.maximum(high-low, .5)
    bounds_low = np.where(high > low, low, low-.25)
    bounds_high = np.where(high > low, high, high+.25)
    top = mass.max() if probability_limit is None else probability_limit
    ax.set(xlim=(bounds_low[0], bounds_high[0]),
           ylim=(bounds_low[1], bounds_high[1]), zlim=(0, top),
           xlabel="$x$", ylabel="$x$" if variance else "$y$",
           zlabel="$p$")
    # Grid planes meet outcomes, means, and probability heights exactly.
    for axis, setter in ((0, ax.set_xticks), (1, ax.set_yticks)):
        ticks = np.unique(np.append(support[:, axis], center[axis]))
        if len(ticks) > 9:
            ticks = np.unique(np.append(ticks[np.linspace(0, len(ticks)-1, 8).astype(int)], center[axis]))
        setter(ticks)
    ticks = np.unique(np.append(mass, 0)) if probability_limit is None else np.array([0., top])
    ax.set_zticks(ticks if len(ticks) <= 8 else ticks[np.linspace(0, len(ticks)-1, 8).astype(int)])
    ax.set(xlabel="", ylabel="", zlabel="", xticks=[], yticks=[], zticks=[])
    ax.xaxis.line.set_visible(False)
    ax.yaxis.line.set_visible(False)
    ax.zaxis.line.set_visible(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_edgecolor("none")
        axis.pane.fill = False
    if not boxes:
        # A quiet x-y floor anchors the probability posts without a box frame.
        ax.zaxis.pane.fill = True
        ax.zaxis.set_pane_color((.30, .51, .58, .06))
        ax.zaxis.pane.set_edgecolor("none")
    ax.grid(False)
    red = "#c63737"
    if boxes:
        ax.plot([center[0], center[0]], [center[1], center[1]], [0, top], color=red, lw=2.4)
    ax.set_box_aspect((span[0], span[1], .65*max(span)))
    ax.set_proj_type("ortho")
    # The variance outcomes lie on x=y; look perpendicular to that diagonal
    # so their feet project to one horizontal baseline without perspective.
    ax.view_init(elev=24, azim=-45)
    ax.set_title("" if title is None else title, fontsize=12, pad=12)
    ax.tick_params(labelsize=8)
    _format_ticks(fig)
    if show:
        _pyplot().show()
    return fig


def variance_reduction_3d(j: Joint, show=True, sign=-1, parts_only=False):
    """Signed volumes for Var(A) + Var(B) - 2 Cov(A,B) = Var(A-B).

    Blue squares add volume; red rectangles subtract below the floor.
    Negative covariance contributes upward red volume instead. All panels
    share centered-coordinate scales, so volume comparisons are meaningful.
    """
    if show:
        from IPython.display import display
        from plotly_viz import decomposition_widget
        widget=decomposition_widget(j,sign,parts_only)
        display(widget)
        return widget
    from intro_answers import expect, marginal, transpose
    if not isinstance(j, Joint):
        raise TypeError("Expected Joint with the actual coupling of A and B")
    aa, bb = np.meshgrid(j._x, j._y, indexing="ij")
    keep = j.probs.ravel() > 0
    a, b, p = aa.ravel()[keep], bb.ravel()[keep], j.probs.ravel()[keep]
    da = a - expect(marginal(j))
    db = b - expect(marginal(transpose(j)))
    combined = da + sign*db
    contributions = np.array([p*da**2, p*db**2, 2*sign*p*da*db, p*combined**2])
    totals = contributions.sum(axis=1)
    np.testing.assert_allclose(totals[:3].sum(), totals[3], atol=1e-10)
    fig = _pyplot().figure(figsize=(10, 3.5) if parts_only else (10, 8), layout="constrained")
    fig.set_facecolor("#ffffff")
    terms = [(da, da, p, "A variance", "#4c8194"),
             (db, db, p, "B variance", "#4c8194"),
             (da, db, 2*sign*p*np.sign(da*db), "cross contribution" if sign == 1 else "control contribution", "#c63737"),
             (combined, combined, p, "sum variance" if sign == 1 else "remaining variance", "#4c8194")]
    if parts_only:
        terms = terms[:3]
    extent = max(float(np.max(np.abs(np.r_[da, db, combined]))), .5)
    grouped = []
    for u, v, heights, title, color in terms:
        uv, inverse = np.unique(np.column_stack([u, v]), axis=0, return_inverse=True)
        grouped.append((uv[:, 0], uv[:, 1], np.bincount(inverse, weights=heights), title, color))
    height = max(float(np.abs(term[2]).max()) for term in grouped)
    for index, (u, v, heights, title, color) in enumerate(grouped):
        ax = fig.add_subplot(1, 3, index+1, projection="3d") if parts_only else fig.add_subplot(2, 2, index+1, projection="3d")
        ax.set_facecolor("#ffffff")
        for dx, dy, dz in zip(u, v, heights):
            if dx != 0 and dy != 0:
                ax.bar3d(min(0, dx), min(0, dy), min(0, dz),
                         abs(dx), abs(dy), abs(dz), color=color,
                         alpha=.18, edgecolor=color, linewidth=.45, shade=False)
            ax.plot([dx, dx], [dy, dy], [0, dz], color="black", lw=1.4)
            ax.scatter([dx], [dy], [dz], color=color, s=18)
        # The explicit floor separates added (above) from removed (below) volume.
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        floor = [(-extent, -extent, 0), (extent, -extent, 0),
                 (extent, extent, 0), (-extent, extent, 0)]
        ax.add_collection3d(Poly3DCollection([floor], facecolors="#4c8194",
                                            edgecolors="none", alpha=.04))
        ax.set(xlim=(-extent, extent), ylim=(-extent, extent),
               zlim=(-height, height), xticks=[], yticks=[], zticks=[])
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
            axis.pane.set_edgecolor("none")
            axis.line.set_visible(False)
        ax.grid(False)
        ax.set_proj_type("ortho")
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=24, azim=-45)
        value = f"{_number(totals[index], signed=True)}" if index == 2 else f"{_number(totals[index])}"
        if not parts_only:
            ax.set_title(f"{title} {value}", color=color, fontsize=11)
    _format_ticks(fig)
    if show:
        _pyplot().show()
    return fig


def histogram(rv, title="Distribution", target=None, without=None, comparison_labels=None, show=True):
    """Default discrete-mass view; variance_3d is an explicit alternative."""
    if show:
        from IPython.display import display
        from plotly_viz import histogram_widget
        widget=histogram_widget(rv,without)
        display(widget)
        return widget
    if isinstance(rv, Joint):
        return joint_histogram(rv,show=False)
    if without is not None:
        return control_comparison(rv, without, comparison_labels)
    (support, mass), = _range_masses(_scalar_mass(rv))
    mean = expect(rv)
    fig, ax = _pyplot().subplots(figsize=(7, 2.8), layout="constrained")
    fig.set_facecolor("#ffffff"); ax.set_facecolor("#ffffff")
    _discrete_dots(ax, support, mass)
    ax.axvline(mean, color="#c63737", linewidth=3)
    _minimal_histogram_axes(ax, support)
    _format_ticks(fig)
    _pyplot().show()
    return fig


def histogram_row(*distributions, labels=None):
    """Display two or more separate histograms in horizontal columns."""
    from IPython.display import display
    from plotly_viz import histogram_row_widget
    widget = histogram_row_widget(*distributions, labels=labels)
    display(widget)
    return widget


def density(rv, without=None, comparison_labels=None):
    """Display a smooth density view of exact discrete masses."""
    from IPython.display import display
    from plotly_viz import density_widget
    widget = density_widget(rv, without=without, comparison_labels=comparison_labels)
    display(widget)
    return widget


def show_control(pair: Joint, steps: int = 1, decomposition: bool = False,
                 show=True, density_view=False):
    """Compare raw and adjusted estimates and report joint variance terms."""
    from intro_answers import marginal, transpose, expect, variance, covar, monte_carlo, monte_carlo_with_control
    a, b = marginal(pair), marginal(transpose(pair))
    np.testing.assert_allclose(expect(b), 0, atol=1e-10)
    var_a, var_b = variance(a)/steps, variance(b)/steps
    cross = 2*covar(pair)/steps
    adjusted = monte_carlo_with_control(pair, steps)
    raw = monte_carlo(a, steps)
    print(("Gray" if density_view else "White") + ": raw estimate | Blue: estimate minus control")
    print(f"Var 1 = {var_a:.6g}; Var 2 = {var_b:.6g}; 2 Cov = {cross:.6g}")
    print(f"Var(adjusted) = {var_a:.6g} + {var_b:.6g} - ({cross:.6g}) = {variance(adjusted):.6g}")
    fig = density(adjusted, without=raw) if density_view and show else histogram(adjusted, without=raw, show=show)
    if decomposition:
        print("Single sample, left to right: Var 1 + Var 2 - 2 Cov")
        print(f"{variance(a):.6g} + {variance(b):.6g} - ({2*covar(pair):.6g}) = {variance(pair.sub()):.6g}")
        variance_reduction_3d(pair, parts_only=True)
    return fig


def control_comparison(rv, without, comparison_labels=None):
    """Overlay masses in shared equal-width bins; the mean remains exact."""
    (values, mass), (other, other_mass) = _range_masses(
        _scalar_mass(rv), _scalar_mass(without))
    np.testing.assert_allclose(expect(rv), expect(without), rtol=2e-6, atol=2e-6)
    fig, ax = _pyplot().subplots(figsize=(7, 2.8), layout="constrained")
    fig.set_facecolor("#ffffff"); ax.set_facecolor("#ffffff")
    _discrete_dots(ax, other, other_mass, "#858b90", hollow=True)
    _discrete_dots(ax, values, mass)
    ax.axvline(expect(rv), color="#c63737", linewidth=3)
    _minimal_histogram_axes(ax, np.r_[values, other])
    _format_ticks(fig)
    _pyplot().show()
    return fig
