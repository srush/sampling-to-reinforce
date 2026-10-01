"""Mystery functions and independent full-distribution checks for the puzzles."""

from collections import Counter
from itertools import combinations_with_replacement
from math import factorial, prod

import numpy as np

from viz import density, histogram, variance_3d
from dist_types import Var, Joint


def _table(rv):
    if isinstance(rv, Var):
        return rv.values, rv.probs
    if isinstance(rv, Joint):
        x, y = np.meshgrid(rv._x, rv._y, indexing="ij")
        return np.column_stack([x.ravel(), y.ravel()]), rv.probs.ravel()
    raise TypeError("Expected Var or Joint")


def f(x):
    return np.array([9., 10., 13., 15., 20., 26.])[np.asarray(x, dtype=int)-1]


def polling_response(person: float) -> float:
    """Hidden toy responses: ten high-scoring and twenty low-scoring residents."""
    responses = (4, 5, 5, 5, 6, 6, 6, 6, 6, 6,
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 4)
    return float(responses[int(person)])


def linear_f(x):
    return 10 + 3*x + np.array([1., -1., 0., 1., -1., 0.])[np.asarray(x, dtype=int)-1]


def quadratic_f(x):
    return x**2 + 2*x + np.array([0., 1., -1., 1., 0., -1.])[np.asarray(x, dtype=int)-1]


def _mean_distribution(values, n, offset=0.0, probabilities=None):
    """Independent reference: unordered index sequences and their multiplicities."""
    values = np.asarray(values, dtype=float)
    probabilities = np.ones(len(values))/len(values) if probabilities is None else probabilities
    outcomes, masses = [], []
    for indices in combinations_with_replacement(range(len(values)), n):
        multiplicities = Counter(indices)
        outcomes.append(values[list(indices)].mean(axis=0) + offset)
        masses.append(factorial(n) / prod(factorial(v) for v in multiplicities.values())
                      * prod(probabilities[i]**c for i, c in multiplicities.items()))
    return np.array(outcomes), np.array(masses)


def _reference(name):
    die = np.arange(1, 7)
    fx = np.array([float(f(int(x))) for x in die])
    if name == "single":
        return fx, np.ones(6)/6, fx.mean()
    if name == "ten":
        v, p = _mean_distribution(fx, 10)
        return v, p, fx.mean()
    if name == "linear":
        values = np.array([float(linear_f(int(x))) for x in die])
        v, p = _mean_distribution(values-3*die, 5, offset=3*3.5)
        return v, p, values.mean()
    if name == "quadratic":
        values = np.array([float(quadratic_f(int(x))) for x in die])
        v, p = _mean_distribution(values-die**2-2*die, 5, offset=91/6+7)
        return v, p, values.mean()
    if name in {"kl", "k3", "topk"}:
        p, q = die/21, np.ones(6)/6
        values = np.log(p/q)
        target = p @ values
        if name == "k3":
            values = values + q/p - 1
        if name == "topk":
            exact = p[3:] @ values[3:]
            tail_mass = p[:3].sum()
            values = exact + tail_mass * values[:3]
            p = p[:3] / tail_mass
        v, mass = _mean_distribution(values, 5, probabilities=p)
        return v, mass, target
    raise KeyError(name)


def assert_distribution(rv, expected_values, expected_mass):
    """Check every support point and its total probability, allowing roundoff."""
    actual, mass = _table(rv)
    actual, mass = np.asarray(actual), np.asarray(mass, dtype=float)
    if actual.ndim not in (1, 2) or not np.isfinite(actual).all():
        raise AssertionError("Expected finite scalar or vector outcomes")
    if not np.isfinite(mass).all() or (mass < 0).any():
        raise AssertionError("Invalid probability masses")
    np.testing.assert_allclose(mass.sum(), 1, atol=2e-6)
    support, inverse = np.unique(np.round(expected_values, 7), axis=0, return_inverse=True)
    target = np.bincount(inverse, weights=expected_mass)
    keep = mass > 0
    actual, mass = actual[keep], mass[keep]
    if actual.ndim == 1:
        upper = np.clip(np.searchsorted(support, actual), 0, len(support)-1)
        lower = np.maximum(upper-1, 0)
        nearest = np.where(abs(actual-support[lower]) < abs(actual-support[upper]), lower, upper)
    else:
        nearest = ((actual[:, None, :]-support[None, :, :])**2).sum(axis=2).argmin(axis=1)
    np.testing.assert_allclose(actual, support[nearest], rtol=2e-6, atol=2e-6,
                               err_msg="Unexpected estimator outcome")
    observed = np.bincount(nearest, weights=mass, minlength=len(support))
    np.testing.assert_allclose(observed, target, rtol=2e-5, atol=2e-7,
                               err_msg="Incorrect probability distribution")


TITLES = {
    "single": "4 · One sample",
    "ten": "5 · Ten samples",
    "linear": "6 · Linear control variate",
    "quadratic": "7 · Quadratic control variate",
    "kl": "11 · Monte Carlo KL",
    "k3": "12 · k3 KL control variate",
    "topk": "15 · Unbiased top-k KL",
}


def check(name, rv, plot=True, without=None, verbose=False,
          variance_diagram=False, density_view=False):
    values, mass, target = _reference(name)
    # Check the target expectation independently of the estimator's PMF.
    # For dice-construction exercises, use the reference distribution's mean.
    expected_mean = np.asarray(mass) @ np.asarray(values) if target is None else np.asarray(target)
    actual_values, actual_mass = _table(rv)
    actual_mean = np.asarray(actual_mass) @ np.asarray(actual_values)
    bias = actual_mean - expected_mean
    np.testing.assert_allclose(
        actual_mean, expected_mean, rtol=2e-6, atol=2e-6,
        err_msg=f"Biased estimator: E[estimate]={actual_mean}, target={expected_mean}, bias={bias}",
    )
    assert_distribution(rv, values, mass)
    if verbose:
        print(f"✓ {TITLES[name]} — full distribution matches")
        label = "mean matches" if target is None else "unbiased (within numerical tolerance)"
        fmt = lambda a: np.array2string(np.asarray(a), precision=6, suppress_small=True)
        print(f"  {label}: E[estimate]={fmt(actual_mean)}, "
              f"target={fmt(expected_mean)}, bias={fmt(bias)}")
    if plot:
        if variance_diagram:
            if without is not None:
                raise ValueError("Use the comparison plot for control variates")
            variance_3d(rv)
        elif density_view:
            density(rv, without=without)
        else:
            histogram(rv, TITLES[name], target=target, without=without)
