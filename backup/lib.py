"""Finite two-parameter models and variance-reduction reference previews."""

from dataclasses import dataclass
from itertools import product

import numpy as np
import matplotlib.pyplot as plt


def normal_pdf(x, mean, sigma):
    return np.exp(-0.5 * ((x - mean) / sigma)**2) / (sigma * np.sqrt(2*np.pi))


def softmax(logits):
    weights = np.exp(logits - np.max(logits))
    return weights / weights.sum()


@dataclass
class Model:
    name: str
    theta: np.ndarray
    probabilities: np.ndarray
    rewards: np.ndarray
    scores: np.ndarray
    groups: np.ndarray
    labels: list

    @property
    def gradient(self):
        return self.probabilities @ (-self.rewards[:, None] * self.scores)

    @property
    def loss(self):
        return -self.probabilities @ self.rewards


def make_model(kind, theta):
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (2,) or not np.isfinite(theta).all():
        raise ValueError("theta must contain exactly two finite parameters")
    # Stable sigmoid, including large negative logits.
    p = np.exp(-np.logaddexp(0, -theta))
    if kind == "binary":
        outcomes = np.array(list(product(range(2), repeat=2)))
        context, z = outcomes.T
        probs = 0.5 * np.where(z, p[context], 1-p[context])
        rewards = normal_pdf(np.array([0.2, 0.8])[context], z, 0.5)
        scores = (z-p[context])[:, None] * np.eye(2)[context]
        groups = context
        labels = [f"context={c}, coin={a}" for c, a in outcomes]
        name = "Binary: two context-specific coins"
    elif kind == "ordered":
        actions = np.arange(5)
        u = (actions-2)/2
        features = np.column_stack([u, -u**2])
        probs = softmax(features @ theta)
        scores = features - probs @ features
        rewards = normal_pdf(3.2, actions, 0.65)
        groups = actions // 2
        labels = [f"class={a}" for a in actions]
        name = "Five ordered classes: linear and quadratic preferences"
    elif kind == "walk":
        paths = np.array(list(product((0, 1), repeat=4)))
        parameters = np.arange(4) % 2
        step_p = p[parameters]
        probs = np.prod(np.where(paths, step_p, 1-step_p), axis=1)
        scores = np.column_stack([
            (paths[:, parameters == j]-p[j]).sum(axis=1) for j in range(2)
        ])
        rewards = normal_pdf(2.5, (2*paths-1).sum(axis=1), 1.0)
        groups = paths[:, :3] @ np.array([4, 2, 1])
        labels = ["".join("R" if a else "L" for a in path) for path in paths]
        name = "Random walk: alternating step biases"
    else:
        raise ValueError(f"Unknown model: {kind}")
    return Model(name, theta.copy(), probs, rewards, scores, groups, labels)


def _stratify(n, rng):
    return (np.arange(n) + rng.random(n)) / n


def _control(rewards, scores, baseline):
    return -((rewards-baseline)[:, None]*scores).mean(axis=0)


def _conditional(probabilities, rewards, scores):
    return probabilities @ (-rewards[:, None]*scores)


def _moments(probabilities, vectors):
    mean = probabilities @ vectors
    centered = vectors-mean
    covariance = centered.T @ (probabilities[:, None]*centered)
    return mean, covariance


def estimator_reference(model, n, baseline):
    """Exact mean/covariance of all four estimators, including stratification."""
    q, r, s = model.probabilities, model.rewards, model.scores
    vectors = -r[:, None]*s
    mean, covariance = _moments(q, vectors)
    result = {"Naive": {"mean": mean, "covariance": covariance/n, "cost": float(n)}}
    cv_mean, cv_cov = _moments(q, -(r-baseline)[:, None]*s)
    result["Control variate"] = {"mean": cv_mean, "covariance": cv_cov/n, "cost": float(n)}
    cdf = np.r_[0, np.cumsum(q)]
    cdf[-1] = 1.0
    strat_mean, strat_cov = np.zeros(2), np.zeros((2, 2))
    for i in range(n):
        overlap = np.maximum(0, np.minimum(cdf[1:], (i+1)/n)-np.maximum(cdf[:-1], i/n))
        conditional = n*overlap
        m, c = _moments(conditional, vectors)
        strat_mean += m/n
        strat_cov += c/n**2
    result["Stratification"] = {"mean": strat_mean, "covariance": strat_cov, "cost": float(n)}
    group_indices = [np.flatnonzero(model.groups == g) for g in np.unique(model.groups)]
    group_probs = np.array([q[idx].sum() for idx in group_indices])
    # Zero-mass groups can arise from floating-point underflow at extreme logits.
    active = group_probs > 0
    group_indices = [idx for idx, keep in zip(group_indices, active) if keep]
    group_probs = group_probs[active]
    group_vectors = np.array([
        _conditional(q[idx]/mass, r[idx], s[idx])
        for idx, mass in zip(group_indices, group_probs)
    ])
    m, c = _moments(group_probs, group_vectors)
    result["Derandomization"] = {
        "mean": m, "covariance": c/n,
        "cost": float(n*(group_probs @ np.array([len(idx) for idx in group_indices]))),
    }
    return result, group_indices, group_probs


def compare_estimators(model, n=16, baseline=0.15, repeats=600, seed=7,
                       stratify=None, control=None, conditional=None):
    """Arrows and repeated-estimate errors; exact metrics refer to reference methods."""
    if not isinstance(n, int) or n < 1 or repeats < 2:
        raise ValueError("Use integer n >= 1 and repeats >= 2")
    stratify = _stratify if stratify is None else stratify
    control = _control if control is None else control
    conditional = _conditional if conditional is None else conditional
    stats, groups, masses = estimator_reference(model, n, baseline)
    rng = np.random.default_rng(seed)
    q, r, s = model.probabilities, model.rewards, model.scores
    cdf = np.cumsum(q)
    cdf[-1] = 1.0
    vectors = -r[:, None]*s
    names = ["Naive", "Stratification", "Control variate", "Derandomization"]
    samples = {name: [] for name in names}
    for _ in range(repeats):
        indices = rng.choice(len(q), n, p=q)
        samples["Naive"].append(vectors[indices].mean(axis=0))
        samples["Control variate"].append(control(r[indices], s[indices], baseline))
        uniforms = np.asarray(stratify(n, rng))
        if uniforms.shape != (n,) or not np.all((uniforms >= 0) & (uniforms < 1)):
            raise ValueError("Stratification must return n uniforms in [0,1)")
        strat_indices = np.searchsorted(cdf, uniforms, side="right")
        samples["Stratification"].append(vectors[strat_indices].mean(axis=0))
        sampled_groups = rng.choice(len(groups), n, p=masses)
        contributions = []
        for g in sampled_groups:
            idx = groups[g]
            contributions.append(conditional(q[idx]/masses[g], r[idx], s[idx]))
        samples["Derandomization"].append(np.mean(contributions, axis=0))
    for name in names:
        samples[name] = np.asarray(samples[name])
        if samples[name].shape != (repeats, 2) or not np.isfinite(samples[name]).all():
            raise ValueError(f"{name} must return finite gradient vectors of shape (2,)")
    truth = model.gradient
    arrow_limit = 1.25*max(np.max(np.abs(truth)),
                          max(np.max(np.abs(samples[name][0])) for name in names), 1e-5)
    error_limit = 1.12*max(max(np.max(np.abs(samples[name]-truth)) for name in names), 1e-5)
    fig, axes = plt.subplots(4, 2, figsize=(10, 13), layout="constrained")
    fig.suptitle(f"{model.name}\nTwo parameters; {n} draws per estimate; fixed baseline {baseline:g}", fontsize=14)
    for row, name in enumerate(names):
        estimated = samples[name][0]
        ax, spread = axes[row]
        for vector, color, label in ((truth, "#182d3b", "Exact gradient"),
                                     (estimated, "#d7733b", "One estimate")):
            ax.quiver(0, 0, *vector, angles="xy", scale_units="xy", scale=1,
                      color=color, width=.012, label=label)
        ax.set(xlim=(-arrow_limit, arrow_limit), ylim=(-arrow_limit, arrow_limit),
               title=f"{name} · {stats[name]['cost']:g} expected reward evaluations",
               xlabel="Gradient coordinate 0", ylabel="Gradient coordinate 1")
        ax.legend(fontsize=8, loc="upper left")
        errors = samples[name]-truth
        spread.scatter(errors[:, 0], errors[:, 1], s=9, alpha=.16, color="#467b9c")
        spread.scatter([0], [0], marker="+", s=120, linewidth=2, color="#182d3b", label="Exact")
        spread.scatter(*errors[0], s=40, color="#d7733b", label="Same estimate")
        variance = float(np.trace(stats[name]["covariance"]))
        spread.set(title=f"Exact reference variance: {variance:.5g}",
                   xlabel="Estimated g₀ − exact g₀", ylabel="Estimated g₁ − exact g₁",
                   xlim=(-error_limit, error_limit), ylim=(-error_limit, error_limit))
        spread.legend(fontsize=8, loc="upper left")
        for panel in (ax, spread):
            panel.axhline(0, color="0.85", lw=.6, zorder=0)
            panel.axvline(0, color="0.85", lw=.6, zorder=0)
            panel.set_aspect("equal", adjustable="box")
        stats[name]["estimates"] = samples[name]
        stats[name]["variance"] = variance
        stats[name]["bias"] = stats[name]["mean"]-truth
        stats[name]["empirical_bias"] = samples[name].mean(axis=0)-truth
        stats[name]["empirical_variance"] = float(samples[name].var(axis=0).sum())
        print(f"{name}: exact variance={variance:.6g}; "
              f"empirical variance={stats[name]['empirical_variance']:.6g}; "
              f"empirical bias norm={np.linalg.norm(stats[name]['empirical_bias']):.3g}")
    plt.show()
    return fig, stats


def check_strategy(name, fn):
    """Exercise contracts: deterministic numeric checks, not noisy pass/fail tests."""
    rng = np.random.default_rng(23)
    if name == "stratification":
        for n in (1, 5, 16):
            seed = 19
            expected = _stratify(n, np.random.default_rng(seed))
            actual = np.asarray(fn(n, np.random.default_rng(seed)))
            assert actual.shape == (n,)
            # This exercise specifies one independent Generator.random draw per stratum.
            np.testing.assert_allclose(actual, expected, atol=1e-12)
    elif name == "control_variate":
        for n in (1, 5, 16):
            rewards, scores = rng.normal(size=n), rng.normal(size=(n, 2))
            for b in (0.0, .15, -.4):
                actual = fn(rewards, scores, b)
                assert np.shape(actual) == (2,)
                np.testing.assert_allclose(actual, _control(rewards, scores, b), atol=1e-12)
    elif name == "derandomization":
        for n in (1, 2, 5):
            probabilities = rng.dirichlet(np.ones(n))
            rewards, scores = rng.normal(size=n), rng.normal(size=(n, 2))
            actual = fn(probabilities, rewards, scores)
            assert np.shape(actual) == (2,)
            np.testing.assert_allclose(actual, _conditional(probabilities, rewards, scores), atol=1e-12)
    else:
        raise ValueError(name)
    print(f"{name}: passed")
