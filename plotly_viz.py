"""An isolated Plotly experiment for the linear control-variate puzzle."""

import json
import os
import uuid
from html import escape

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import HTML
from dist_types import Joint

from intro_answers import expect, variance, linear_control, marginal, transpose, covar, monte_carlo
from viz import _range_masses, _scalar_mass

EMBED_PLOTLY_JS = os.environ.get("RL_PUZZLES_EMBED_PLOTLY_JS", "1") != "0"


WHEEL_SCRIPT = """
// Keep native page scrolling; stop Plotly's WebGL wheel handlers.
plot.addEventListener('wheel', e => e.stopImmediatePropagation(), {capture:true, passive:true});
"""


def _label_indices(count, limit=8):
    """Label every small support; otherwise retain eight evenly spaced values."""
    return np.linspace(0, count-1, min(count, limit)).round().astype(int)


def polling_sketch():
    """One dot per resident; colors indicate counties, not observed responses."""
    fig=go.Figure()
    for count,offset,color,label in ((10,0,"#c63737","10 people"),
                                      (20,7,"#4c8194","20 people")):
        i=np.arange(count)
        fig.add_trace(go.Scatter(x=offset+i%5,y=-(i//5),mode="markers",
            marker=dict(size=15,color=color),hoverinfo="skip"))
        fig.add_annotation(x=offset+2,y=1,text=label,showarrow=False,
                           font=dict(size=16,color=color))
    fig.update_layout(height=185,showlegend=False,dragmode=False,
        paper_bgcolor="#ffffff",plot_bgcolor="#ffffff",margin=dict(l=20,r=20,t=5,b=5),
        xaxis=dict(visible=False,range=[-2,13],fixedrange=True),
        yaxis=dict(visible=False,range=[-4,2],fixedrange=True,scaleanchor="x",scaleratio=1))
    return HTML(fig.to_html(full_html=False,include_plotlyjs=EMBED_PLOTLY_JS,
        config=dict(displayModeBar=False,scrollZoom=False,responsive=True),
        post_script="const plot=document.getElementById('{plot_id}');"+WHEEL_SCRIPT))


def _weighted_density(values, masses, grid, bandwidth):
    """Gaussian smoothing of an exact discrete probability distribution."""
    offsets = (grid[:, None] - values[None, :]) / bandwidth
    return np.exp(-0.5 * offsets**2) @ masses / (bandwidth * np.sqrt(2*np.pi))


def _density_grid(distributions):
    values = np.concatenate([values for values, _ in distributions])
    span = float(values.max() - values.min())
    bandwidth = max(span / 60, 0.15)
    padding = 3 * bandwidth
    return np.linspace(values.min() - padding, values.max() + padding, 400), bandwidth


def density_widget(rv, without=None, comparison_labels=None):
    """Smooth exact masses for display, without changing probability calculations."""
    distributions = [_scalar_mass(rv)]
    if without is not None:
        distributions.append(_scalar_mass(without))
    grid, bandwidth = _density_grid(distributions)
    fig = go.Figure()
    before, after = comparison_labels or ("Raw", "Adjusted")
    styles = ([('Distribution', '#4c8194', 'rgba(76,129,148,.25)')]
              if without is None else
              [(before, '#858b90', 'rgba(133,139,144,.20)'),
               (after, '#4c8194', 'rgba(76,129,148,.25)')])
    for (values, masses), (name, color, fill) in zip(distributions[::-1], styles):
        fig.add_trace(go.Scatter(x=grid, y=_weighted_density(values, masses, grid, bandwidth),
            mode="lines", name=name, fill="tozeroy", fillcolor=fill,
            line=dict(color=color, width=3),
            hovertemplate="Value %{x:.2f}<br>Smoothed density %{y:.3f}<extra>%{fullData.name}</extra>"))
    for index, ((values, masses), (name, color, _)) in enumerate(zip(distributions[::-1], styles)):
        mean = float(values @ masses)
        mean_color = color if without is not None else "#c63737"
        fig.add_vline(x=mean, line_color=mean_color, line_width=3,
                      line_dash="dash" if index else "dot")
        fig.add_annotation(x=mean, y=1-index*.13, yref="paper", showarrow=False,
            text=f"{name} mean: {mean:.4f}", font=dict(color=mean_color, size=12),
            bgcolor="#ffffff")
    fig.update_layout(height=320, paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        margin=dict(l=25, r=20, t=15, b=45), dragmode=False,
        showlegend=without is not None,
        xaxis=dict(showgrid=False, zeroline=False, fixedrange=True),
        yaxis=dict(visible=False, fixedrange=True))
    return HTML(fig.to_html(full_html=False, include_plotlyjs=EMBED_PLOTLY_JS,
        config=dict(displayModeBar=False, scrollZoom=False, responsive=True)))


def reinforce_baseline_slider(control, reward, score, steps=5):
    """Vary a constant reward baseline while keeping the sample count fixed."""
    baselines = np.arange(-4., 8.25, .25)
    estimates = [monte_carlo(control(reward, float(b), score), steps) for b in baselines]
    raw = estimates[16]  # b = 0
    masses = [_scalar_mass(rv) for rv in estimates]
    grid, bandwidth = _density_grid(masses)
    curves = [_weighted_density(v, p, grid, bandwidth) for v, p in masses]
    active = int(np.flatnonzero(baselines == 4.0)[0])
    fig = go.Figure()
    for name, curve, color in (("Reinforce", curves[16], "#858b90"),
                                ("Constant baseline", curves[active], "#4c8194")):
        fig.add_trace(go.Scatter(x=grid, y=curve, mode="lines", name=name,
            line=dict(color=color, width=3),
            hovertemplate="Gradient %{x:.3f}<br>Smoothed density %{y:.3f}<extra>%{fullData.name}</extra>"))
    fig.add_vline(x=expect(raw), line_color="#c63737", line_dash="dash")
    def report(i):
        return (f"{steps} samples · Mean = {expect(estimates[i]):.4f} · "
                f"Variance = {variance(estimates[i]):.4f} · Raw variance = {variance(raw):.4f}")
    fig.frames = [go.Frame(name=str(i), data=[go.Scatter(y=curve)], traces=[1],
        layout=dict(annotations=[dict(text=report(i), x=0, y=1.12,
            xref="paper", yref="paper", showarrow=False, xanchor="left")]))
        for i, curve in enumerate(curves)]
    fig.update_layout(height=370, paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=35, r=20, t=45, b=85), dragmode=False,
        annotations=fig.frames[active].layout.annotations,
        xaxis=dict(title="Gradient estimate", range=[float(grid.min()), float(grid.max())], fixedrange=True),
        yaxis=dict(title="Smoothed density", range=[0, max(float(c.max()) for c in curves)*1.08], fixedrange=True),
        sliders=[dict(active=active, x=.08, len=.84, y=-.2,
            currentvalue=dict(prefix="Baseline b = "),
            steps=[dict(label=f"{b:g}", method="animate", args=[[str(i)],
                dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))])
                for i, b in enumerate(baselines)])])
    return HTML(fig.to_html(full_html=False, include_plotlyjs=EMBED_PLOTLY_JS,
        post_script="const plot=document.getElementById('{plot_id}');" + WHEEL_SCRIPT,
        config=dict(displayModeBar=False, scrollZoom=False, responsive=True)))


def monte_carlo_samples_slider(sample, max_samples=5, without=None, labels=None):
    """Show exact Monte Carlo mean distributions as the sample count changes."""
    from intro_answers import monte_carlo
    if max_samples < 1:
        raise ValueError("max_samples must be positive")
    counts = range(1, max_samples + 1)
    estimates = [monte_carlo(sample, count) for count in counts]
    comparisons = ([monte_carlo(without, count) for count in counts]
                   if without is not None else None)
    masses = [_scalar_mass(rv) for rv in estimates]
    other_masses = [_scalar_mass(rv) for rv in comparisons] if comparisons else None
    grid, bandwidth = _density_grid(masses + (other_masses or []))
    densities = [_weighted_density(values, probs, grid, bandwidth)
                 for values, probs in masses]
    other_densities = ([_weighted_density(values, probs, grid, bandwidth)
                        for values, probs in other_masses] if other_masses else None)
    ymax = max(float(y.max()) for y in densities + (other_densities or [])) * 1.08
    fig = go.Figure()
    if other_densities:
        other_label, estimate_label = labels or ("Unadjusted", "Estimate")
        fig.add_trace(go.Scatter(x=grid, y=other_densities[0], mode="lines",
            name=other_label, fill="tozeroy", fillcolor="rgba(133,139,144,.20)",
            line=dict(color="#858b90", width=2),
            hovertemplate="Value %{x:.3f}<br>Smoothed density %{y:.3f}<extra>%{fullData.name}</extra>"))
    else:
        estimate_label = (labels or ("Estimate",))[0]
    fig.add_trace(go.Scatter(x=grid, y=densities[0], mode="lines",
        name=estimate_label, fill="tozeroy", fillcolor="rgba(76,129,148,.25)",
        line=dict(color="#4c8194", width=3),
        hovertemplate="Value %{x:.3f}<br>Smoothed density %{y:.3f}<extra>%{fullData.name}</extra>"))
    mean = expect(sample)
    if comparisons:
        other_mean = expect(without)
        fig.add_trace(go.Scatter(x=[other_mean, other_mean], y=[0, ymax], mode="lines",
            line=dict(color="#858b90", width=3, dash="dot"),
            name=f"{other_label} mean", showlegend=False,
            hovertemplate=f"{other_label} mean: {other_mean:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[mean, mean], y=[0, ymax], mode="lines",
        line=dict(color="#4c8194" if comparisons else "#c63737", width=3, dash="dash"),
        name=f"{estimate_label} mean",
        hovertemplate=f"{estimate_label} mean: {mean:.4f}<extra></extra>",
        showlegend=False))
    fig.frames = [go.Frame(name=str(count),
        data=([go.Scatter(y=other_densities[i])] if other_densities else [])
             + [go.Scatter(y=densities[i])],
        traces=([0, 1] if other_densities else [0]))
        for i, count in enumerate(counts)]
    fig.update_layout(height=350, paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        margin=dict(l=25, r=20, t=15, b=75), dragmode=False,
        showlegend=other_densities is not None,
        xaxis=dict(title="Estimate", range=[float(grid.min()), float(grid.max())], fixedrange=True),
        yaxis=dict(title="Smoothed density", range=[0, ymax], fixedrange=True),
        sliders=[dict(active=0, x=.08, len=.84, y=-.04,
            currentvalue=dict(prefix="Samples = "),
            steps=[dict(label=str(count), method="animate",
                args=[[str(count)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                          transition=dict(duration=0))])
                for count in counts])])
    return HTML(fig.to_html(
        full_html=False, include_plotlyjs=EMBED_PLOTLY_JS,
        config=dict(displayModeBar=False, scrollZoom=False, responsive=True)))


def linear_control_frames(x, f, b=3, steps=5):
    """Compute every slider state with the same exact Var/Joint machinery."""
    mean, var = expect(x), variance(x)
    frames = []
    for strength in np.linspace(0, 2, 21):
        target = lambda a: float(f(a)) + strength*((a-mean)**2-var)
        pair = linear_control(x, target, b)
        a, control = marginal(pair), marginal(transpose(pair))
        adjusted = monte_carlo(pair.sub(), steps)
        raw = monte_carlo(a, steps)
        distributions = (_scalar_mass(adjusted), _scalar_mass(raw))
        va, vb, cross = variance(a), variance(control), 2*covar(pair)
        frames.append(dict(strength=float(strength), distributions=distributions,
                           a=np.array([target(v) for v in x.values]),
                           b=b*(x.values-mean), probs=x.probs,
                           mean=expect(a), terms=(va, vb, cross, variance(pair.sub())),
                           mc_variance=variance(adjusted)))
    return frames


def linear_control_widget(x, f, b=3, steps=5):
    states = linear_control_frames(x, f, b, steps)
    return _control_states_widget(states, steps, "Nonlinearity λ = ")


def ab_control_frames(population, treated, untreated, initial, steps=5):
    """Pair independent people while retaining outcome/control dependence."""
    from dist_types import Var
    u,v=np.meshgrid(population.values,population.values,indexing="ij")
    weights=np.outer(population.probs,population.probs).ravel()
    outcomes=np.array([treated(a)-untreated(b) for a,b in zip(u.ravel(),v.ravel())])
    controls=np.array([initial(a)-initial(b) for a,b in zip(u.ravel(),v.ravel())])
    raw=monte_carlo(Var(outcomes,weights),steps)
    va=variance(Var(outcomes,weights))
    states=[]
    for strength in np.linspace(-4,2,25):
        control=strength*controls
        sample=Var(outcomes-control,weights).op(lambda value: round(float(value),12))
        adjusted=monte_carlo(sample,steps)
        vb=variance(Var(control,weights))
        covariance=float(weights @ ((outcomes-expect(raw))*control))
        states.append(dict(strength=float(strength),
            distributions=(_scalar_mass(adjusted),_scalar_mass(raw)),
            a=outcomes,b=control,probs=weights,mean=expect(raw),
            terms=(va,vb,2*covariance,variance(sample)),mc_variance=variance(adjusted)))
    return states


def ab_control_widget(population, treated, untreated, initial, steps=5):
    states=ab_control_frames(population,treated,untreated,initial,steps)
    return _control_states_widget(states,steps,"b = ",active=20)


def _control_states_widget(states, steps, slider_prefix, active=0):
    distributions = [dist for state in states for dist in state["distributions"]]
    grid, bandwidth = _density_grid(distributions)
    for state in states:
        state["densities"] = [_weighted_density(values, masses, grid, bandwidth)
                              for values, masses in state["distributions"]]
    max_density = max(float(density.max()) for state in states for density in state["densities"])
    fig = make_subplots(rows=2, cols=3, vertical_spacing=.16,
                        row_heights=[.43, .57],
                        specs=[[{"colspan": 3}, None, None],
                               [{"type": "scene"}]*3])
    blue, red = "#4c8194", "#c63737"

    def traces(state):
        result = []
        for density, name, color, fill in zip(state["densities"][::-1],
                ["Raw", "Adjusted"], ["#858b90", blue],
                ["rgba(133,139,144,.20)", "rgba(76,129,148,.25)"]):
            result.append(go.Scatter(x=grid, y=density, mode="lines", name=name,
                fill="tozeroy", fillcolor=fill, line=dict(color=color, width=3),
                hovertemplate="Value %{x:.2f}<br>Smoothed density %{y:.3f}<extra>%{fullData.name}</extra>"))
        result.append(go.Scatter(x=[state["mean"]]*2, y=[0, max_density*1.05], mode="lines",
                                line=dict(color=red, width=4), hoverinfo="skip", showlegend=False))
        da = state["a"] - state["mean"]
        db = state["b"]
        for panel, (u, v, heights, color) in enumerate([
                (da, da, state["probs"], blue),
                (db, db, state["probs"], blue),
                (da, db, state["probs"], blue)]):
            scene = "scene" if panel == 0 else f"scene{panel+1}"
            # Every signed box volume is its contribution to the printed term.
            colors = [blue if dx*dy >= 0 else "#9a78a5" for dx,dy in zip(u,v)] if panel == 2 else [blue]*len(u)
            for dx, dy, dz, box_color in zip(u, v, heights, colors):
                xx = [0, dx, dx, 0, 0, dx, dx, 0]
                yy = [0, 0, dy, dy, 0, 0, dy, dy]
                zz = [0]*4+[dz]*4
                result.append(go.Mesh3d(x=xx, y=yy, z=zz,
                    i=[0,0,4,4,0,0,1,1,2,2,3,3], j=[1,2,5,6,1,5,2,6,3,7,0,4],
                    k=[2,3,6,7,5,4,6,5,7,6,4,7], color=box_color,
                    opacity=.18, flatshading=True, scene=scene, hoverinfo="skip", showlegend=False))
            result.append(go.Scatter3d(x=np.array([[z,z,np.nan] for z in u]).ravel(),
                y=np.array([[z,z,np.nan] for z in v]).ravel(),
                z=np.array([[0,z,np.nan] for z in heights]).ravel(),
                mode="lines", line=dict(color="black", width=4), scene=scene, hoverinfo="skip", showlegend=False))
            result.append(go.Scatter3d(x=u, y=v, z=heights, mode="markers",
                marker=dict(color=colors, size=4), scene=scene,
                hovertemplate="%{x:.2f}, %{y:.2f}<extra></extra>", showlegend=False))
        return result

    extent = max(float(np.abs(s["a"]-s["mean"]).max()) for s in states)
    extent = max(extent, max(float(np.abs(s["b"]).max()) for s in states), .5)
    zmax = max(max(float(s["probs"].max()) for s in states)*2, .1)
    first = traces(states[active])
    for t in first[:3]:
        fig.add_trace(t, row=1, col=1)
    for t in first[3:]:
        fig.add_trace(t)
    fig.frames = [go.Frame(name=str(i), data=traces(s)) for i, s in enumerate(states)]
    fig.update_layout(height=700, paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        margin=dict(l=15, r=15, t=15, b=80), showlegend=True, uirevision="linear-control",
        sliders=[dict(active=active, x=.08, len=.84, y=-.06, currentvalue=dict(prefix=slider_prefix),
            steps=[dict(label=f"{s['strength']:g}", method="animate",
                args=[[str(i)], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))])
                for i,s in enumerate(states)])])
    fig.update_xaxes(range=[float(grid.min()), float(grid.max())],
                     showgrid=False, zeroline=False, showline=False, tickfont=dict(size=16))
    fig.update_yaxes(range=[0,max_density*1.1], visible=False)
    for scene in ("scene", "scene2", "scene3"):
        fig.update_layout(**{scene: dict(
            xaxis=dict(visible=False, range=[-extent,extent]),
            yaxis=dict(visible=False, range=[-extent,extent]),
            zaxis=dict(visible=False, range=[-zmax,zmax]),
            aspectmode="cube", bgcolor="#ffffff", dragmode=False, uirevision="keep-camera",
            camera=dict(eye=dict(x=1.4,y=-1.4,z=.9), projection=dict(type="orthographic")))})
    for x_pos, title in zip((.14,.5,.86),("Var(A)","Var(B)","Cov(A, B)")):
        fig.add_annotation(x=x_pos,y=.57,xref="paper",yref="paper",
                           text=title,showarrow=False)
    reports = []
    for s in states:
        va,vb,cross,total=s["terms"]
        reports.append(f"Gray: raw estimate | Blue: adjusted ({steps} samples)\n"
            f"Single sample: Var(A) = {va:.4g}; Var(B) = {vb:.4g}; Cov(A, B) = {cross/2:.4g}\n"
            f"Covariance: blue positive; purple negative.\n"
            f"{va:.4g} + {vb:.4g} − ({cross:.4g}) = {total:.4g}\n"
            f"Variance of adjusted {steps}-sample mean: {s['mc_variance']:.4g}")
    ident = "linear-"+uuid.uuid4().hex
    script = """const plot=document.getElementById('{plot_id}');
const report=document.getElementById('REPORT_ID');
const reports=REPORTS;
report.textContent=reports[ACTIVE];
plot.on('plotly_sliderchange', e => { report.textContent=reports[Number(e.step.args[0][0])]; });"""
    script = script.replace("REPORT_ID", ident).replace("REPORTS", json.dumps(reports)).replace("ACTIVE", str(active))
    script += WHEEL_SCRIPT
    html = f'<pre id="{ident}" style="white-space:pre-wrap"></pre>'
    html += fig.to_html(full_html=False, include_plotlyjs=EMBED_PLOTLY_JS, post_script=script,
                        config=dict(displayModeBar=False, scrollZoom=False, responsive=True))
    return HTML(html)


def variance_widget(x):
    """Fixed-camera variance boxes; hovering a probability post highlights its box."""
    from intro_answers import shared
    from dist_types import Var
    values, probs = _scalar_mass(x)
    return covariance_widget(shared(Var(values, probs)), variance=True)


def _covariance_figure(pair, variance=False, bounds=None, capacity=None,
                       max_labels=None, boxes=True, height_scale=1, keep_zeros=False):
    """Fixed orthographic projection, with SVG dots/text above all box faces.

    Projecting explicitly avoids WebGL depth-testing hiding hover targets.
    Coordinates still describe the same probability-height 3D boxes.
    """
    if isinstance(pair, Joint):
        xx, yy = np.meshgrid(pair._x, pair._y, indexing="ij")
        values = np.column_stack((xx.ravel(), yy.ravel()))
        probs = pair.probs.ravel()
    else:
        values, probs = pair.table()
    values, probs = np.asarray(values), np.asarray(probs)
    keep = np.ones_like(probs, dtype=bool) if keep_zeros else probs > 0
    values, probs = values[keep], probs[keep]
    mean = probs @ values
    blue = "#4c8194"
    low, high, top = bounds or (values.min(), values.max(), probs.max())
    span = max(float(high-low), .5)
    center = (low+high)/2
    def project(a, b, z):
        a, b, z = np.asarray(a), np.asarray(b), np.asarray(z)
        return ((a+b-2*center)/span/np.sqrt(2),
                (b-a)/span/np.sqrt(2)*np.sin(np.deg2rad(24))
                + height_scale*z/top*.65*np.cos(np.deg2rad(24)))
    count = len(values)
    labeled = set(_label_indices(count,8 if max_labels is None else max_labels))
    capacity = capacity or count
    values = np.pad(values, ((0, capacity-count),(0,0)))
    probs = np.pad(probs, (0, capacity-count))
    active = [index < count and prob > 0 for index, prob in enumerate(probs)]
    fig = go.Figure()
    mapping, box_ids = {}, []
    labels, tips, colors, edges = [], [], [], []
    # Draw every box first. Posts, dots and labels are separate foreground layers.
    for index, ((a,b), prob) in enumerate(zip(values, probs)):
        contribution = height_scale*prob*(a-mean[0])*(b-mean[1])
        color = blue if contribution >= 0 else "#9a78a5"
        fmt = lambda value: f"{value:.2f}".rstrip('0').rstrip('.')
        label = fmt(a) if variance else f"{fmt(a)},{fmt(b)}"
        tooltip = f"{label}<br>{prob:.2f}<br>{contribution:.2f}<extra></extra>"
        labels.append(label); tips.append(tooltip); colors.append(color)
        box_ids.append(len(fig.data))
        cx,cy=project([mean[0],a,a,mean[0]]*2,
                      [mean[1],mean[1],b,b]*2,[0]*4+[prob]*4)
        ex,ey=[],[]
        for start,end in ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),
                          (6,7),(7,4),(0,4),(1,5),(2,6),(3,7)):
            ex.extend([cx[start],cx[end],None])
            ey.extend([cy[start],cy[end],None])
        edges.append((ex,ey))
        # Fill the projected solid's convex silhouette; overlapping face paths
        # otherwise cancel under SVG's fill rule.
        points=sorted(set(zip(cx,cy)))
        def half_hull(points):
            hull=[]
            for p in points:
                while len(hull)>1:
                    a0,b0=hull[-2],hull[-1]
                    if (b0[0]-a0[0])*(p[1]-a0[1])-(b0[1]-a0[1])*(p[0]-a0[0])>0:
                        break
                    hull.pop()
                hull.append(p)
            return hull
        hull=half_hull(points)[:-1]+half_hull(points[::-1])[:-1]
        hull=(hull or points)+[(hull or points)[0]]
        px,py=zip(*hull)
        fig.add_trace(go.Scatter(x=px,y=py,fill="toself",fillcolor=color,
            mode="lines",line=dict(color=color,width=.7),opacity=.12,
            hoverinfo="skip",visible=active[index] and boxes))
    for index, (ex,ey) in enumerate(edges):
        fig.add_trace(go.Scatter(x=ex,y=ey,mode="lines",
            line=dict(color=colors[index],width=.8),opacity=.3,
            hoverinfo="skip",visible=active[index] and boxes))
    for index, ((a,b),prob) in enumerate(zip(values,probs)):
        px,py=project(a,b,np.linspace(0,prob,20))
        mapping[str(len(fig.data))]=index
        fig.add_trace(go.Scatter(x=[float(px)]*20,y=py,mode="lines",
            line=dict(color="black",width=3),hovertemplate=tips[index],visible=active[index]))
    px,py=project(mean[0],mean[1],[0,top/height_scale])
    fig.add_trace(go.Scatter(x=[float(px)]*2,y=py,mode="lines",
        line=dict(color="#c63737",width=4),hoverinfo="skip",visible=boxes))
    for index, ((a,b),prob) in enumerate(zip(values,probs)):
        px,py=project(a,b,prob)
        mapping[str(len(fig.data))]=index
        fig.add_trace(go.Scatter(x=[float(px)],y=[float(py)],mode="markers",
            marker=dict(color=colors[index],size=11),hovertemplate=tips[index],visible=active[index]))
        px,py=project(a,b,0)
        mapping[str(len(fig.data))]=index
        fig.add_trace(go.Scatter(x=[float(px)],y=[float(py)-.06],mode="markers+text",
            marker=dict(size=24,color="rgba(0,0,0,0)"),text=[labels[index] if index in labeled else ""],
            textfont=dict(size=18,color="#263940"),textposition="middle center",
            hovertemplate=tips[index],visible=active[index]))
    fig.update_layout(height=480,showlegend=False,margin=dict(l=10,r=10,t=10,b=20),
        paper_bgcolor="#ffffff",plot_bgcolor="#ffffff",dragmode=False,
        hovermode="closest",hoverdistance=20,
        hoverlabel=dict(font=dict(size=11),align="left"),
        xaxis=dict(visible=False,range=[-.9,.9],fixedrange=True),
        yaxis=dict(visible=False,range=[-.42,1],scaleanchor="x",scaleratio=1,fixedrange=True))
    if not boxes:
        px,py=project([low,high,high,low],[low,low,high,high],0)
        path="M "+" L ".join(f"{a},{b}" for a,b in zip(px,py))+" Z"
        fig.add_shape(type="path",path=path,fillcolor="rgba(76,129,148,.06)",
                      line=dict(width=0),layer="below")
    return fig,box_ids,mapping


def _hover_script(box_ids, mapping):
    script = """const plot=document.getElementById('{plot_id}');
const boxes=BOXES, mapping=MAPPING;
const edges=boxes.map(b => b+boxes.length);
let highlighted=null;
function highlight(index) {
  if(index===highlighted) return;
  highlighted=index;
  Plotly.restyle(plot, {
    opacity: boxes.map(b => b===index ? .65 : .12).concat(boxes.map(b => b===index ? .95 : .3)),
    'line.width': boxes.map(() => .7).concat(boxes.map(b => b===index ? 1.5 : .8))
  }, boxes.concat(edges));
}
plot.on('plotly_hover', e => highlight(mapping[e.points[0].curveNumber] ?? null));
plot.on('plotly_click', e => highlight(mapping[e.points[0].curveNumber] ?? null));
plot.on('plotly_sliderchange', () => { highlighted=null; });
plot.addEventListener('mouseleave', () => highlight(null));"""
    script = script.replace("BOXES",json.dumps(box_ids)).replace("MAPPING",json.dumps(mapping))
    return script + WHEEL_SCRIPT


def covariance_widget(pair, variance=False, boxes=True, max_labels=None):
    fig,box_ids,mapping=_covariance_figure(pair,variance,boxes=boxes,max_labels=max_labels)
    return HTML(fig.to_html(full_html=False,include_plotlyjs=EMBED_PLOTLY_JS,post_script=_hover_script(box_ids,mapping),
        config=dict(displayModeBar=False,scrollZoom=False,responsive=True)))


def covariance_interpolation_slider(x):
    """Move from independent copies of X to two views of the same draw."""
    from intro_answers import covar, indep, shared, variance
    independent, paired = indep(x, x), shared(x)
    strengths = np.linspace(0, 1, 11)
    bounds = (float(x.values.min()), float(x.values.max()),
              float(max(independent.probs.max(), paired.probs.max())))
    joints = [Joint(x.values, x.values,
                    (1-strength)*independent.probs + strength*paired.probs)
              for strength in strengths]
    figures = [_covariance_figure(joint, bounds=bounds, keep_zeros=True)
               for joint in joints]
    fig, box_ids, mapping = figures[0]
    fig.frames = [go.Frame(name=str(i), data=frame.data)
                  for i, (frame, _, _) in enumerate(figures)]
    fig.update_layout(height=550, margin=dict(l=10, r=10, t=10, b=70),
        sliders=[dict(active=0, x=.08, len=.84, y=-.03,
            currentvalue=dict(prefix="Shared fraction = "),
            steps=[dict(label=f"{strength:.1f}", method="animate",
                args=[[str(i)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                      transition=dict(duration=0))])
                for i, strength in enumerate(strengths)])])
    reports = [f"Cov(X,Y) = {covar(joint):.3f};  Var(X) = {variance(x):.3f}"
               for strength, joint in zip(strengths, joints)]
    ident = "covariance-mix-" + uuid.uuid4().hex
    script = _hover_script(box_ids, mapping)
    script += f"""const report=document.getElementById('{ident}');
const reports={json.dumps(reports)};
plot.on('plotly_sliderchange', e => {{
 report.textContent=reports[Number(e.step.args[0][0])];
}});"""
    return HTML(f'<pre id="{ident}">{reports[0]}</pre>' + fig.to_html(
        full_html=False, include_plotlyjs=EMBED_PLOTLY_JS, post_script=script,
        config=dict(displayModeBar=False, scrollZoom=False, responsive=True)))


def sum_variance_slider(x):
    """Show the scalar sum as two copies move from independent to shared."""
    from intro_answers import indep, shared
    independent, paired = indep(x, x), shared(x)
    strengths = np.linspace(0, 1, 11)
    support = np.unique(np.add.outer(x.values, x.values))
    mean = 2 * expect(x)
    states = []
    for strength in strengths:
        joint = Joint(x.values, x.values,
                      (1-strength)*independent.probs + strength*paired.probs)
        values, masses = _scalar_mass(joint.add())
        probabilities = np.zeros(len(support))
        probabilities[np.searchsorted(support, values)] = masses
        covariance = covar(joint)
        if abs(covariance) < 1e-12:
            covariance = 0.0
        states.append((probabilities, covariance, variance(joint.add())))

    ymax = max(float(probs.max()) for probs, _, _ in states) * 1.12
    fig = go.Figure()
    fig.add_trace(go.Bar(x=support, y=states[0][0], width=.75,
        marker_color="#4c8194", name="Sum", hovertemplate="Sum %{x:g}<br>Probability %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[mean, mean], y=[0, ymax], mode="lines",
        line=dict(color="#c63737", width=3), name="Mean", hoverinfo="skip"))
    fig.frames = [go.Frame(name=str(i), data=[go.Bar(y=probs)], traces=[0])
                  for i, (probs, _, _) in enumerate(states)]
    fig.update_layout(height=360, paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        margin=dict(l=30, r=20, t=15, b=75), showlegend=False, dragmode=False,
        xaxis=dict(title="X + Y", tickmode="array", tickvals=support,
                   range=[float(support.min())-.8, float(support.max())+.8], fixedrange=True),
        yaxis=dict(title="Probability", range=[0, ymax], fixedrange=True),
        sliders=[dict(active=0, x=.08, len=.84, y=-.04,
            currentvalue=dict(prefix="Shared fraction = "),
            steps=[dict(label=f"{strength:.1f}", method="animate",
                args=[[str(i)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                      transition=dict(duration=0))])
                for i, strength in enumerate(strengths)])])
    component = variance(x)
    reports = [f"Var(X + Y) = {component:.3f} + {component:.3f} + 2 × {covariance:.3f} = {total:.3f}"
               for _, covariance, total in states]
    ident = "sum-variance-" + uuid.uuid4().hex
    script = f"""const report=document.getElementById('{ident}');
const reports={json.dumps(reports)};
plot.on('plotly_sliderchange', e => {{
 report.textContent=reports[Number(e.step.args[0][0])];
}});"""
    return HTML(f'<pre id="{ident}">{reports[0]}</pre>' + fig.to_html(
        full_html=False, include_plotlyjs=EMBED_PLOTLY_JS, post_script=script,
        config=dict(displayModeBar=False, scrollZoom=False, responsive=True)))


def _histogram_figure(rv, without=None):
    """Shared mass grouping, exact means, and foreground hover targets."""
    distributions=[_scalar_mass(rv)]
    if without is not None:
        distributions.append(_scalar_mass(without))
    grouped=_range_masses(*distributions)
    fig=go.Figure()
    markers=[]
    for index in reversed(range(len(grouped))):
        values,mass=grouped[index]
        xx,yy=[],[]
        for value,p in zip(values,mass):
            xx.extend([value,value,None]); yy.extend([0,p,None])
        fig.add_trace(go.Scatter(x=xx,y=yy,mode="lines",line=dict(color="black",width=3),hoverinfo="skip"))
        markers.append(go.Scatter(x=values,y=mass,mode="markers",
            marker=dict(size=11,color="white" if index else "#4c8194",line=dict(color="#4c8194",width=1.5)),
            hovertemplate="%{x:.2f}<br>%{y:.2f}<extra></extra>"))
    mean=float(distributions[0][0] @ distributions[0][1])
    ymax=max(float(m.max()) for v,m in grouped)*1.12
    fig.add_trace(go.Scatter(x=[mean,mean],y=[0,ymax],mode="lines",
        line=dict(color="#c63737",width=4),hoverinfo="skip"))
    for marker in markers:
        fig.add_trace(marker)
    support=np.unique(np.concatenate([v for v,m in grouped]))
    ticks=support[_label_indices(len(support))]
    from viz import _number
    fig.update_layout(height=290,showlegend=False,dragmode=False,
        paper_bgcolor="#ffffff",plot_bgcolor="#ffffff",margin=dict(l=20,r=20,t=15,b=40),
        hoverlabel=dict(font=dict(size=11)),
        xaxis=dict(showgrid=False,zeroline=False,showline=False,fixedrange=True,
                   tickmode="array",tickvals=ticks,ticktext=[_number(v) for v in ticks],tickfont=dict(size=18)),
        yaxis=dict(visible=False,range=[0,ymax],fixedrange=True))
    return fig


def temperature_policy_slider(policy, temperature=3.0):
    """Exact class probabilities as softmax temperature changes."""
    temperatures = np.unique(np.r_[0.25, np.arange(0.5, 20.5, 0.5), temperature])
    active = int(np.flatnonzero(temperatures == temperature)[0])
    figures = [_histogram_figure(policy(float(t))) for t in temperatures]
    # Keep the probability scale fixed, including the red expectation marker.
    for frame in figures:
        frame.data[1].y = [0, 1.05]
        frame.data[2].hovertemplate = "Class %{x:.0f}<br>Probability %{y:.4f}<extra></extra>"
    fig = figures[active]
    fig.frames = [go.Frame(name=f"{t:g}", data=frame.data)
                  for t, frame in zip(temperatures, figures)]
    fig.update_layout(height=350, margin=dict(l=55, r=20, t=15, b=95),
        xaxis=dict(title="Output class", range=[-0.5, 7.5]),
        yaxis=dict(visible=True, title="Probability", range=[0, 1.05],
                   showgrid=False, zeroline=False),
        sliders=[dict(active=active, x=.08, len=.84, y=-.2,
            currentvalue=dict(prefix="Temperature T = "),
            steps=[dict(label=f"{t:g}", method="animate",
                args=[[f"{t:g}"], dict(mode="immediate",
                    frame=dict(duration=0, redraw=True), transition=dict(duration=0))])
                for t in temperatures])])
    return HTML(fig.to_html(full_html=False, include_plotlyjs=EMBED_PLOTLY_JS,
        post_script="const plot=document.getElementById('{plot_id}');" + WHEEL_SCRIPT,
        config=dict(displayModeBar=False, scrollZoom=False, responsive=True)))


def histogram_widget(rv, without=None):
    if isinstance(rv, Joint) or getattr(rv, "event_shape", None) == (2,):
        widgets=[covariance_widget(rv,boxes=False).data]
        if without is not None:
            widgets.insert(0,covariance_widget(without,boxes=False).data)
        return HTML(''.join(widgets))
    fig=_histogram_figure(rv,without)
    return HTML(fig.to_html(full_html=False,include_plotlyjs=EMBED_PLOTLY_JS,
        post_script="const plot=document.getElementById('{plot_id}');"+WHEEL_SCRIPT,
        config=dict(displayModeBar=False,scrollZoom=False,responsive=True)))


def histogram_row_widget(*distributions, labels=None):
    """Render independent histograms in equal-width columns."""
    if len(distributions) < 2:
        raise ValueError("Pass at least two distributions")
    if labels is not None and len(labels) != len(distributions):
        raise ValueError("Provide one label per distribution")
    panels = []
    for index, rv in enumerate(distributions):
        label = '' if labels is None else f'<div style="font-weight:600;margin:0 0 8px">{escape(str(labels[index]))}</div>'
        panels.append(f'<div style="min-width:0">{label}{histogram_widget(rv).data}</div>')
    return HTML('<div style="display:grid;grid-template-columns:repeat('
                + str(len(distributions)) + ',minmax(0,1fr));gap:20px">'
                + ''.join(panels) + '</div>')


def decomposition_widget(pair, sign=-1, parts_only=False):
    from intro_answers import shared
    a,b=marginal(pair),marginal(transpose(pair))
    result=pair.add() if sign==1 else pair.sub()
    print(f"Var 1 = {variance(a):.2f}; Var 2 = {variance(b):.2f}; Cov = {covar(pair):.2f}")
    print(f"Var(result) = {variance(a):.2f} + {variance(b):.2f} {'+' if sign==1 else '−'} 2 × ({covar(pair):.2f}) = {variance(result):.2f}")
    specs=[(shared(a),True),(shared(b),True),(pair,False)]
    if not parts_only:
        specs.append((shared(result),True))
    values=np.concatenate([np.r_[j._x,j._y] for j,_ in specs])
    bounds=(float(values.min()),float(values.max()),
            float(max(j.probs.max() for j,_ in specs)))
    titles=["A", "B", "Cov(A, B)", "A + B" if sign==1 else "A − B"]
    html=[]
    for index,(j,is_var) in enumerate(specs):
        fig,boxes,mapping=_covariance_figure(j,is_var,bounds=bounds)
        fig.update_layout(height=310, title=dict(text=titles[index], x=.5),
                          margin=dict(t=40))
        html.append('<div style="min-width:0">'+fig.to_html(full_html=False,include_plotlyjs=EMBED_PLOTLY_JS,
            post_script=_hover_script(boxes,mapping),
            config=dict(displayModeBar=False,scrollZoom=False,responsive=True))+'</div>')
    return HTML('<div style="display:grid;grid-template-columns:repeat(2,minmax(0,1fr))">'+''.join(html)+'</div>')


def square_slider(x):
    from intro_answers import square
    strengths=np.linspace(0,2,21)
    figs=[_histogram_figure(square(x,float(b))) for b in strengths]
    fig=figs[10]
    fig.frames=[go.Frame(name=str(i),data=f.data,layout=dict(xaxis=dict(
        tickmode="array",tickvals=f.layout.xaxis.tickvals,ticktext=f.layout.xaxis.ticktext))) for i,f in enumerate(figs)]
    fig.update_xaxes(range=[-1,max(1,max(x.values**2))+1])
    fig.update_yaxes(range=[0,1.1])
    fig.update_layout(height=350,margin=dict(l=20,r=20,t=15,b=80),
        sliders=[dict(active=10,x=.08,len=.84,y=-.15,currentvalue=dict(prefix="b = "),
            steps=[dict(label=f"{b:.1f}",method="animate",args=[[str(i)],dict(mode="immediate",
                frame=dict(duration=0,redraw=True),transition=dict(duration=0))]) for i,b in enumerate(strengths)])])
    return HTML(fig.to_html(full_html=False,include_plotlyjs=EMBED_PLOTLY_JS,
        post_script="const plot=document.getElementById('{plot_id}');"+WHEEL_SCRIPT,
        config=dict(displayModeBar=False,scrollZoom=False,responsive=True)))


def scaled_variance_slider(x):
    """Scale the outcome values while keeping their probabilities fixed."""
    from intro_answers import shared
    from dist_types import Var
    strengths=range(1,5)
    scaled_values=np.concatenate([b*x.values for b in strengths])
    bounds=(float(scaled_values.min()),float(scaled_values.max()),float(x.probs.max()))
    figures=[_covariance_figure(shared(Var(b*x.values,x.probs)),True,bounds=bounds)
             for b in strengths]
    fig,boxes,mapping=figures[0]
    fig.frames=[go.Frame(name=str(i),data=f.data) for i,(f,_,_) in enumerate(figures)]
    fig.update_layout(height=540,margin=dict(l=10,r=10,t=10,b=70),
        sliders=[dict(active=0,x=.08,len=.84,y=-.03,currentvalue=dict(prefix="b = "),
            steps=[dict(label=str(b),method="animate",args=[[str(i)],dict(mode="immediate",
                frame=dict(duration=0,redraw=True),transition=dict(duration=0))])
                for i,b in enumerate(strengths)])])
    reports=[f"Var(bX) = {b*b*variance(x):.2f}" for b in strengths]
    ident="scaled-variance-"+uuid.uuid4().hex
    script=_hover_script(boxes,mapping)
    script+=f"""const report=document.getElementById('{ident}');
const reports={json.dumps(reports)};
plot.on('plotly_sliderchange', e => {{
 report.textContent=reports[Number(e.step.args[0][0])];
}});"""
    return HTML(f'<pre id="{ident}">{reports[0]}</pre>'+fig.to_html(
        full_html=False,include_plotlyjs=EMBED_PLOTLY_JS,post_script=script,
        config=dict(displayModeBar=False,scrollZoom=False,responsive=True)))


def coin_variance_slider(max_flips=20):
    """Exact coin-average variances, with shared camera/scales and hover targets."""
    from math import comb
    from intro_answers import shared
    from dist_types import Var
    figures=[]
    for n in range(1,max_flips+1):
        values=(2*np.arange(n+1)-n)/n
        probs=np.array([comb(n,k)/2**n for k in range(n+1)])
        figures.append(_covariance_figure(shared(Var(values,probs)),True,
                       bounds=(-1,1,.5),capacity=max_flips+1,max_labels=5))
    fig,boxes,mapping=figures[0]
    fig.frames=[go.Frame(name=str(n),data=f.data) for n,(f,_,_) in enumerate(figures,1)]
    fig.update_layout(height=540,margin=dict(l=10,r=10,t=10,b=70),
        sliders=[dict(active=0,x=.08,len=.84,y=-.03,currentvalue=dict(prefix="Coin flips: "),
            steps=[dict(label=str(n),method="animate",args=[[str(n)],dict(mode="immediate",
                frame=dict(duration=0,redraw=True),transition=dict(duration=0))])
                for n in range(1,max_flips+1)])])
    ident="coin-"+uuid.uuid4().hex
    script=_hover_script(boxes,mapping)
    script+=f"""const report=document.getElementById('{ident}');
plot.on('plotly_sliderchange', e => {{
 const n=Number(e.step.label); report.textContent='Variance = 1/'+n+' = '+(1/n).toFixed(2);
}});"""
    return HTML(f'<pre id="{ident}">Variance = 1/1 = 1.00</pre>'+fig.to_html(
        full_html=False,include_plotlyjs=EMBED_PLOTLY_JS,post_script=script,
        config=dict(displayModeBar=False,scrollZoom=False,responsive=True)))
