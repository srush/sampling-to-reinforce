"""Filled-in single-function answers for the Var/Joint exercises."""
from typing import Callable
from operator import index
import numpy as np
from dist_types import Fn, Var, Joint, _joint


def uniform(a: int, b: int) -> Var:
    """Uniform distribution on the integers a, a+1, ..., b-1."""
    if isinstance(a, bool) or isinstance(b, bool):
        raise TypeError("Uniform bounds must be integers")
    a, b = index(a), index(b)
    if b <= a:
        raise ValueError("The interval [a, b) must contain at least one integer")
    return Var(np.arange(a, b), np.full(b-a, 1/(b-a)))


def expect(x: Var) -> float:
    return float(x.values @ x.probs)


def variance(x: Var) -> float:
    mean = expect(x)
    return expect(x.op(lambda a: (a - mean)**2))


def shared(x: Var) -> Joint:
    return _joint(x.values, x.values, np.diag(x.probs))


def indep(x: Var, y: Var) -> Joint:
    return _joint(x.values, y.values, np.outer(x.probs, y.probs))


def op(f: Fn, x: Var) -> Var:
    values = np.array([f(a) for a in x.values])
    support, inverse = np.unique(values, return_inverse=True)
    return Var(support, np.bincount(inverse, weights=x.probs))


Var.op = lambda self, f: op(f, self)


def condition(predicate: Callable[[float], bool], x: Var) -> Var:
    keep = np.array([predicate(a) for a in x.values])
    probs = x.probs[keep]
    return Var(x.values[keep], probs / probs.sum())


Var.cond = lambda self, predicate: condition(predicate, self)


def binop(f: Callable[[float, float], float], j: Joint) -> Var:
    values = np.array([f(x, y) for x in j._x for y in j._y])
    support, inverse = np.unique(values, return_inverse=True)
    prob = np.bincount(inverse, weights=j.probs.ravel())
    return Var(support, prob)


Joint._binop = lambda self, f: binop(f, self)


def op_joint(f: Fn, g: Fn,
                j: Joint) -> Joint:
    xs = np.array([f(x) for x in j._x])
    ys = np.array([g(y) for y in j._y])
    x, rows = np.unique(xs, return_inverse=True)
    y, cols = np.unique(ys, return_inverse=True)
    prob = np.zeros((len(x), len(y)))
    np.add.at(prob, (rows[:, None], cols[None, :]), j.probs)
    return _joint(x, y, prob)


Joint.op = lambda self, f, g: op_joint(f, g, self)


def add(j: Joint) -> Var:
    return j._binop(lambda x, y: x+y)


def sub(j: Joint) -> Var:
    return j._binop(lambda x, y: x-y)


Joint.add = lambda self: add(self)
Joint.sub = lambda self: sub(self)
Joint.mul = lambda self: self._binop(lambda x, y: x*y)


def transpose(j: Joint) -> Joint:
    return _joint(j._y, j._x, j.probs.T)


def marginal(j: Joint) -> Var:
    return Var(j._x, j.probs.sum(axis=1))


def covar(j: Joint) -> float:
    mx, my = expect(marginal(j)), expect(marginal(transpose(j)))
    return expect(j._binop(lambda x, y: (x-mx)*(y-my)))


def monte_carlo(f_x: Var, steps: int) -> Var:
    total = f_x.op(lambda value: 0)
    for _ in range(steps):
        total = add(indep(total, f_x))
    return total.op(lambda value: value / steps)


def control_variate(x: Var, f: Fn,
                    h: Fn, known_mean: float,
                    b: float = 1) -> Joint:
    """Pair f(X) with b*(h(X)-known_mean), using the same draw of X."""
    return shared(x).op(f, lambda a: b*(h(a)-known_mean))


def monte_carlo_with_control(pair: Joint, steps: int) -> Var:
    return monte_carlo(pair.sub(), steps)


def stratify(red_responses: Var, blue_responses: Var,
             red_share: float, red_steps: int, blue_steps: int) -> Var:
    """Weighted average of independent draws from each group's response law."""
    red_poll = monte_carlo(red_responses, red_steps)
    blue_poll = monte_carlo(blue_responses, blue_steps)
    return indep(red_poll, blue_poll)._binop(lambda r, b: red_share*r + (1-red_share)*b)


def linear_control(x: Var, f: Fn, b: float) -> Joint:
    return control_variate(x, f, lambda a: a, expect(x), b)


def quadratic_control(x: Var, f: Fn, a: float, b: float) -> Joint:
    h = lambda z: a*z*z + b*z
    center = expect(x.op(h))
    return control_variate(x, f, h, center)


def weighted_die(weights: list[int]) -> Var:
    if not weights or any(int(w) != w or w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("Use nonnegative integer weights with positive total")
    total = int(sum(weights))
    coin = Var([-1, 1], [.5, .5])
    candidate = Var([0], [1.0])
    for _ in range((total - 1).bit_length()):
        candidate = indep(candidate, coin)._binop(lambda a, b: 2*a + (b + 1)/2)
    accepted = candidate.cond(lambda a: a < total)
    edges = np.cumsum(weights)
    return accepted.op(lambda a: np.searchsorted(edges, a, side="right") + 1)


def ab_sampling(population: Var, treated: Fn,
                untreated: Fn, steps: int) -> Var:
    treated_estimate = monte_carlo(population.op(treated), steps)
    untreated_estimate = monte_carlo(population.op(untreated), steps)
    return indep(treated_estimate, untreated_estimate).sub()


def ab_test(population: Var, treated: Fn,
            untreated: Fn, initial: Fn,
            b: float, steps: int) -> Var:
    center = expect(population.op(initial))
    treated_pair = control_variate(population, treated, initial, center, b)
    untreated_pair = control_variate(population, untreated, initial, center, b)
    treated_estimate = monte_carlo_with_control(treated_pair, steps)
    untreated_estimate = monte_carlo_with_control(untreated_pair, steps)
    return indep(treated_estimate, untreated_estimate).sub()


def kl(p: Var, q: Var) -> float:
    """Compute D_KL(p || q) by summing over the finite support of p."""
    total = 0.0
    for value in np.unique(p.values):
        probability = p.prob(value)
        if probability == 0:
            continue
        reference = q.prob(value)
        if reference == 0:
            return float("inf")
        total += probability * np.log(probability / reference)
    return float(total)


def topk(x: Var, f: Fn, k: int) -> Var:
    """Sum the top-k contribution exactly and sample the conditional remainder."""
    x = x.op(lambda a: a)
    if not 0 <= k <= len(x.values):
        raise ValueError("k must be between zero and the support size")
    indices = np.argsort(-x.probs, kind="stable")[:k]
    top = x.values[indices]
    exact = sum(x.probs[i]*f(x.values[i]) for i in indices)
    remaining = ~np.isin(x.values, top)
    mass = float(x.probs[remaining].sum())
    if mass == 0:
        return Var([exact], [1.0])
    tail = Var(x.values[remaining], x.probs[remaining] / mass)
    return tail.op(lambda a: exact + mass*f(a))


def temperature_policy(temperature: float) -> Var:
    """Eight classes with fixed logits 0,...,7 and positive temperature."""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    logits = np.arange(8, dtype=float)
    weights = np.exp((logits - logits.max()) / temperature)
    return Var(logits, weights / weights.sum())


def leave_one_out(temperature: float, n: int = 3, seed: int | None = 11) -> float:
    """Mean reward of n-1 independent other draws; rewards equal class indices."""
    if n < 2:
        raise ValueError("Leave-one-out needs at least two samples")
    action = temperature_policy(temperature)
    others = np.random.default_rng(seed).choice(
        action.values, size=n-1, p=action.probs)
    return float(others.mean())
