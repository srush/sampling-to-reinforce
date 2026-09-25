"""Notebook-friendly probability histograms for scalar distribution.RV objects."""

import numpy as np
import matplotlib.pyplot as plt
import distribution as d
from dist_types import Var, Joint


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


def _plot_dist(x):
    return d.pmf(x.values, x.probs) if isinstance(x, Var) else x


def _plot_joint(j):
    if isinstance(j, Joint):
        xx, yy = np.meshgrid(j._x, j._y, indexing="ij")
        return d.pmf(np.column_stack([xx.ravel(), yy.ravel()]), j.probs.ravel())
    return j


def _scalar_mass(rv):
    values, mass = (np.asarray(a) for a in rv.table())
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Expected a finite scalar distribution")
    if not np.isfinite(mass).all() or np.any(mass < 0) or mass.sum() <= 0:
        raise ValueError("Invalid probability masses")
    support, inverse = np.unique(values, return_inverse=True)
    mass = np.bincount(inverse, weights=mass)
    keep = mass > 0
    return support[keep], mass[keep]/mass.sum()


def _count_tokens(mass, limit=80):
    """Small exact count representation, otherwise retain exact weighted atoms."""
    from fractions import Fraction
    from math import lcm
    if len(mass) > limit:
        return np.arange(len(mass)), mass.copy(), 1, False
    denominators = [Fraction(float(p)).limit_denominator(limit).denominator for p in mass]
    n = lcm(*denominators)
    if n <= limit:
        counts = np.rint(mass*n).astype(int)
        if counts.sum() == n and np.all(counts > 0) and np.allclose(counts/n, mass, rtol=0, atol=1e-12):
            indices = np.repeat(np.arange(len(mass)), counts)
            return indices, np.ones(n), n, True
    return np.arange(len(mass)), mass.copy(), 1, False




def _discrete_dots(ax, support, mass, color="#4c8194", hollow=False):
    ax.vlines(support, 0, mass, color="black", linewidth=2.4, zorder=1)
    ax.scatter(support, mass, s=32,
               facecolors="none" if hollow else color, edgecolors=color, alpha=.7, linewidths=1)
    ax.set_ylim(0, max(ax.get_ylim()[1], 1.08*float(mass.max())))


def _geometry_axes():
    fig, ax = plt.subplots(figsize=(7, 3.5), layout="constrained")
    fig.set_facecolor("#faf9f5")
    ax.set_facecolor("#faf9f5")
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def variance_3d(x, **kwargs):
    """Squares with side |x-E[x]|, extruded by probability; volume is variance."""
    if isinstance(x, Var):
        from intro_answers import shared
        return covariance_3d(shared(x), variance=True, **kwargs)
    return covariance_3d(d.joint(x, x), variance=True, **kwargs)


def joint_top_view(j: Joint, show=True):
    """View joint mass from above; faint crosses mark zero-mass grid points."""
    x, y = np.meshgrid(j._x, j._y, indexing="ij")
    keep = j.probs > 0
    fig, ax = plt.subplots(figsize=(6, 6), layout="constrained")
    fig.set_facecolor("#faf9f5")
    ax.set_facecolor("#faf9f5")
    ax.scatter(x[~keep], y[~keep], marker="x", s=20, color="#d3d5d4", linewidths=1)
    ax.scatter(x[keep], y[keep], s=80*j.probs[keep]/j.probs.max(),
               color="#4c8194", edgecolors="black", linewidths=.5)
    ax.set_aspect("equal")
    ax.set_xticks(j._x)
    ax.set_yticks(j._y)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#d3d5d4")
    ax.tick_params(colors="#53616a", labelsize=9)
    _format_ticks(fig)
    if show:
        plt.show()
    return fig


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
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    if isinstance(pair, Var):
        raise TypeError("Covariance needs a Joint; use shared(x) or indep(x, y)")
    pair = _plot_joint(pair)
    if pair.event_shape != (2,):
        raise ValueError('Pass d.joint(x, y) or x @ y.')
    values, mass = (np.asarray(a) for a in pair.table())
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
    fig = plt.figure(figsize=(8, 5.8), layout="constrained")
    fig.set_facecolor("#faf9f5")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#faf9f5")
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
        ax.scatter([x], [y], [p], color=color, s=35)
        label = f"{_number(x)}" if variance else f"{_number(x)}, {_number(y)}"
        if index in labeled:
            ax.text(x, y, 0, f"\n{label}", ha="center", va="top",
                    color="#263940", fontsize=10, clip_on=False, zorder=20)
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
    ax.set(xlabel="", ylabel="", xticks=[], yticks=[])
    ax.xaxis.line.set_visible(False)
    ax.yaxis.line.set_visible(False)
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
    label = "variance " if variance else ""
    default_title = f"{label}{_number(covariance)}" if boxes else ""
    ax.set_title(default_title if title is None else title, fontsize=12, pad=12)
    ax.tick_params(labelsize=8)
    _format_ticks(fig)
    if show:
        plt.show()
    return fig


def variance_sum_3d(j: Joint, show=True):
    """Signed-volume decomposition of the variance of an additive reward."""
    return variance_reduction_3d(j, show=show, sign=1)


def variance_reduction_3d(j: Joint, show=True, sign=-1):
    """Signed volumes for Var(A) + Var(B) - 2 Cov(A,B) = Var(A-B).

    Blue squares add volume; red rectangles subtract below the floor.
    Negative covariance contributes upward red volume instead. All panels
    share centered-coordinate scales, so volume comparisons are meaningful.
    """
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
    fig = plt.figure(figsize=(10, 8), layout="constrained")
    fig.set_facecolor("#faf9f5")
    terms = [(da, da, p, "A variance", "#4c8194"),
             (db, db, p, "B variance", "#4c8194"),
             (da, db, 2*sign*p*np.sign(da*db), "cross contribution" if sign == 1 else "control contribution", "#c63737"),
             (combined, combined, p, "sum variance" if sign == 1 else "remaining variance", "#4c8194")]
    extent = max(float(np.max(np.abs(np.r_[da, db, combined]))), .5)
    grouped = []
    for u, v, heights, title, color in terms:
        uv, inverse = np.unique(np.column_stack([u, v]), axis=0, return_inverse=True)
        grouped.append((uv[:, 0], uv[:, 1], np.bincount(inverse, weights=heights), title, color))
    height = max(float(np.abs(term[2]).max()) for term in grouped)
    for index, (u, v, heights, title, color) in enumerate(grouped):
        ax = fig.add_subplot(2, 2, index+1, projection="3d")
        ax.set_facecolor("#faf9f5")
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
        ax.set_title(f"{title} {value}", color=color, fontsize=11)
    _format_ticks(fig)
    if show:
        plt.show()
    return fig


def coin_average(n):
    """Exact distribution of the mean of n fair -1/+1 flips (binomial counts)."""
    from math import comb
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    return d.pmf((2*np.arange(n+1)-n)/n,
                 [comb(n, k)/2**n for k in range(n+1)], name=f"average.{n}")


def coin_variance_animation(max_flips=20):
    """Notebook GIF using the variance plot with fixed camera and scales."""
    from io import BytesIO
    from PIL import Image
    from IPython.display import Image as NotebookImage
    if not isinstance(max_flips, int) or max_flips < 1:
        raise ValueError("max_flips must be a positive integer")
    frames = []
    for n in range(1, max_flips+1):
        average = coin_average(n)
        np.testing.assert_allclose(average.var(), 1/n, rtol=2e-6, atol=2e-6)
        fig = variance_3d(average, show=False, probability_limit=.5, max_post_labels=5,
                          title=f"{n} {'flip' if n == 1 else 'flips'} · variance {_number(1/n)}")
        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=100)
        buffer.seek(0)
        with Image.open(buffer) as frame:
            frames.append(frame.convert("RGB"))
        plt.close(fig)
    durations = [450]*len(frames)
    durations[0], durations[-1] = 1100, 1800
    output = BytesIO()
    frames[0].save(output, format="GIF", save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, disposal=2)
    return NotebookImage(data=output.getvalue(), format="gif")


def _arrow(ax, v, color="#4c8194", start=(0, 0)):
    ax.annotate("", xy=np.asarray(start)+v, xytext=start,
                arrowprops=dict(arrowstyle="->", color=color, lw=3))


def _two_state_vector(rv):
    values, mass = (np.asarray(a) for a in rv.table())
    keep = mass > 0
    values, mass = values[keep], mass[keep]
    if values.ndim != 1:
        raise ValueError("Use a scalar two-state distribution")
    support, inverse = np.unique(values, return_inverse=True)
    mass = np.bincount(inverse, weights=mass)
    if len(support) != 2:
        raise ValueError("Expected exactly two distinct states")
    return np.sqrt(mass)*(support-mass@support)


def variance_vector(rv):
    v = _two_state_vector(rv)
    fig, ax = _geometry_axes()
    _arrow(ax, v)
    scale = max(np.linalg.norm(v), .1)
    ax.scatter([0], [0], color="#263940", s=18)
    ax.set(xlim=(-1.5*scale, 1.5*scale), ylim=(-.25*scale, 1.3*scale))
    ax.text(0, 1.1*scale, f"variance {_number(float(rv.var()))}", ha="center")
    _format_ticks(fig)
    plt.show()
    return fig


def covariance_vectors(x, y):
    # Align actual joint worlds: separate marginal tables would lose coupling.
    _two_state_vector(x); _two_state_vector(y)
    pair = x.apply(lambda a: np.array([1., 0.])*a) + y.apply(lambda b: np.array([0., 1.])*b)
    values, mass = (np.asarray(a) for a in pair.table())
    keep = mass > 0
    values, mass = values[keep], mass[keep]
    weighted = np.sqrt(mass)[:, None]*(values-mass@values)
    # Isometrically represent their span in 2D, preserving norms and dot product.
    norm = np.linalg.norm(weighted[:, 0])
    covariance = weighted[:, 0] @ weighted[:, 1]
    projection = covariance/norm if norm else 0.
    vx = np.array([norm, 0.])
    vy = np.array([projection, np.sqrt(max(0., (weighted[:, 1]**2).sum()-projection**2))])
    fig, ax = _geometry_axes()
    _arrow(ax, vx); _arrow(ax, vy, "#c63737")
    scale = max(np.linalg.norm(vx), np.linalg.norm(vy), .1)
    foot = np.array([projection, 0.])
    ax.plot([vy[0], foot[0]], [vy[1], foot[1]],
            linestyle="--", color="#858b90", linewidth=1.5)
    _arrow(ax, foot, "#bd822c")
    ax.scatter(*foot, color="#bd822c", s=20, zorder=4)
    if vy[1] > 1e-8*scale and abs(projection) > 1e-8*scale:
        corner = min(.08*scale, vy[1]/3, abs(projection)/3)
        side = -np.sign(projection)*corner
        ax.plot([projection+side, projection+side, projection],
                [0, corner, corner], color="#858b90", linewidth=1)
    ax.text(projection/2, -.13*scale, "projection",
            color="#bd822c", ha="center")
    ax.set(xlim=(-1.5*scale, 1.5*scale), ylim=(-.5*scale, 1.2*scale))
    ax.text(0, scale, f"{_number(covariance)}", ha="center")
    ax.text(vx[0], -.2*scale, "X", color="#4c8194", ha="center")
    ax.text(vy[0], vy[1]+.12*scale, "Y", color="#c63737", ha="center")
    _format_ticks(fig)
    plt.show()
    return fig


def pythagorean_variance(x, y):
    if set(x._sources) & set(y._sources):
        raise ValueError("Use independent variables for the right-angle diagram")
    vx, vy = float(x.var()), float(y.var())
    average = d.mean([x, y])
    average_variance = float(average.var())
    np.testing.assert_allclose(average_variance, (vx+vy)/4, rtol=2e-6, atol=2e-6)
    a, b = np.sqrt(vx)/2, np.sqrt(vy)/2
    fig, ax = _geometry_axes()
    from matplotlib.patches import Polygon
    # Squares on each side of the triangle: areas are variances.
    ax.add_patch(Polygon([(0, 0), (a, 0), (a, -a), (0, -a)], color="#4c8194", alpha=.12))
    ax.add_patch(Polygon([(a, 0), (a, b), (a+b, b), (a+b, 0)], color="#858b90", alpha=.16))
    ax.add_patch(Polygon([(0, 0), (a, b), (a-b, b+a), (-b, a)], color="#c63737", alpha=.1))
    _arrow(ax, np.array([a, 0.])); _arrow(ax, np.array([0., b]), "#858b90", (a, 0))
    _arrow(ax, np.array([a, b]), "#c63737")
    s = max(a, b, .1)
    corner = .12*s
    ax.plot([a-corner, a-corner, a], [0, corner, corner], color="#53616a", lw=1)
    ax.text(a/2, -a/2, f"{_number(vx/4)}", ha="center", va="center", color="#4c8194")
    ax.text(a+b/2, b/2, f"{_number(vy/4)}", ha="center", va="center", color="#73797e")
    ax.text((a-b)/2, (b+a)/2, f"{_number(average_variance)}", ha="center", va="center", color="#c63737")
    ax.set(xlim=(-b-.4*s, a+b+.4*s), ylim=(-a-.6*s, b+a+.6*s))
    _format_ticks(fig)
    plt.show()
    return fig


def histogram(rv, title="Distribution", target=None, without=None, comparison_labels=None):
    """Default discrete-mass view; variance_3d is an explicit alternative."""
    if isinstance(rv, Joint):
        return joint_histogram(rv)
    rv = _plot_dist(rv)
    if without is not None:
        without = _plot_dist(without)
    if rv.event_shape == (2,):
        return gradient_distribution(rv, title, target, without=without)
    if without is not None:
        return control_comparison(rv, without, comparison_labels)
    support, mass = _scalar_mass(rv)
    mean = float(mass @ support)
    variance = float(mass @ (support-mean)**2)
    fig, ax = plt.subplots(figsize=(7, 2.8), layout="constrained")
    fig.set_facecolor("#faf9f5"); ax.set_facecolor("#faf9f5")
    _discrete_dots(ax, support, mass)
    if len(support) <= 12:
        ax.set_xticks(support)
    ax.axvline(mean, color="#c63737", linewidth=3)
    variance_text(ax, variance)
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d3d5d4")
    ax.tick_params(colors="#53616a", labelsize=9)
    _format_ticks(fig)
    plt.show()
    return fig


def gradient_distribution(rv, title, target, without=None):
    """Exact two-dimensional PMF, one draw, and total gradient variance."""
    rv = _plot_joint(rv)
    if without is not None:
        without = _plot_joint(without)
    values, weights = (np.asarray(a) for a in rv.table())
    keep = weights > 0
    values, weights = values[keep], weights[keep]
    weights = weights/weights.sum()
    mean = weights @ values
    variance = weights @ ((values-mean)**2).sum(axis=1)
    sample = values[np.random.default_rng(7).choice(len(values), p=weights)]
    support, inverse = np.unique(np.round(values, 8), axis=0, return_inverse=True)
    mass = np.bincount(inverse, weights=weights)
    fig, ax = plt.subplots(figsize=(7, 4.2), layout="constrained")
    fig.set_facecolor("#faf9f5"); ax.set_facecolor("#faf9f5")
    if without is not None:
        other, other_mass = (np.asarray(a) for a in without.table())
        other_support, other_inverse = np.unique(np.round(other, 8), axis=0, return_inverse=True)
        other_mass = np.bincount(other_inverse, weights=other_mass)
        ax.scatter(other_support[:, 0], other_support[:, 1], s=1500*other_mass,
                   facecolors="none", edgecolors="#858b90", linewidths=2)
    ax.scatter(support[:, 0], support[:, 1], s=1500*mass, color="#4c8194", alpha=.5)
    for v, color, width in ((sample, "#263940", .006),
                             (mean, "#c63737", .012)):
        ax.quiver(0, 0, *v, angles="xy", scale_units="xy", scale=1,
                  color=color, width=width)
    ax.axhline(0, color="#d3d5d4", linewidth=.7)
    ax.axvline(0, color="#d3d5d4", linewidth=.7)
    # Shared limits cover both binary-policy examples.
    ax.set(xlim=(-.85, .1), ylim=(-.1, .85), aspect="equal")
    variance_text(ax, variance, without, total=True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors="#53616a", labelsize=9)
    _format_ticks(fig)
    plt.show()
    return fig


def variance_text(ax, variance, without=None, total=False, comparison_labels=None):
    label = "total variance" if total else "variance"
    if without is None:
        ax.text(1, 1.04, f"{label} {_number(variance)}", transform=ax.transAxes,
                ha="right", fontsize=9, color="#53616a")
    else:
        other_variance = float(np.asarray(without.var()).sum())
        before, after = comparison_labels or ("without", "with")
        ax.text(0, 1.04, f"{label} {before} {_number(other_variance)}",
                transform=ax.transAxes, fontsize=9, color="#73797e")
        ax.text(1, 1.04, f"{after} {_number(variance)}", transform=ax.transAxes,
                ha="right", fontsize=9, color="#4c8194")


def control_comparison(rv, without, comparison_labels=None):
    """Overlay exact discrete masses and their exact variances, without bins."""
    values, mass = _scalar_mass(rv)
    other, other_mass = _scalar_mass(without)
    np.testing.assert_allclose(rv.mean(), without.mean(), rtol=2e-6, atol=2e-6)
    fig, ax = plt.subplots(figsize=(7, 2.8), layout="constrained")
    fig.set_facecolor("#faf9f5"); ax.set_facecolor("#faf9f5")
    _discrete_dots(ax, other, other_mass, "#858b90", hollow=True)
    _discrete_dots(ax, values, mass)
    ax.axvline(float(rv.mean()), color="#c63737", linewidth=3)
    variance_text(ax, float(rv.var()), without, comparison_labels=comparison_labels)
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d3d5d4")
    ax.tick_params(colors="#53616a", labelsize=9)
    _format_ticks(fig)
    plt.show()
    return fig
