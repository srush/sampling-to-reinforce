"""Filled-in single-function answers for the Var/Joint exercises."""
from typing import Callable
import numpy as np
from dist_types import Var, Joint, _joint


def expect(x: Var) -> float:
    return float(x.values @ x.probs)


def shift(x: Var, amount: float) -> Var:
    return x.op(lambda a: a + amount)


def six_from_eight(x: Var) -> Var:
    return x.cond(lambda a: a <= 6)


def variance(x: Var) -> float:
    mean = expect(x)
    return expect(x.op(lambda a: (a - mean)**2))


def shared(x: Var) -> Joint:
    return _joint(x.values, x.values, np.diag(x.probs))


def indep(x: Var, y: Var) -> Joint:
    return _joint(x.values, y.values, np.outer(x.probs, y.probs))


def op(f: Callable[[float], float], x: Var) -> Var:
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


def op_joint(f: Callable[[float], float], g: Callable[[float], float],
                j: Joint) -> Joint:
    xs = np.array([f(x) for x in j._x])
    ys = np.array([g(y) for y in j._y])
    x, rows = np.unique(xs, return_inverse=True)
    y, cols = np.unique(ys, return_inverse=True)
    prob = np.zeros((len(x), len(y)))
    np.add.at(prob, (rows[:, None], cols[None, :]), j.probs)
    return _joint(x, y, prob)


Joint.op = lambda self, f, g: op_joint(f, g, self)


def condition_joint(predicate: Callable[[float, float], bool], j: Joint) -> Joint:
    keep = np.array([[predicate(a, b) for b in j._y] for a in j._x])
    probs = j.probs * keep
    return _joint(j._x, j._y, probs / probs.sum())


Joint.cond = lambda self, predicate: condition_joint(predicate, self)


def add(j: Joint) -> Var:
    return j._binop(lambda x, y: x+y)


def sub(j: Joint) -> Var:
    return j._binop(lambda x, y: x-y)


def mul(j: Joint) -> Var:
    return j._binop(lambda x, y: x*y)


def div(j: Joint) -> Var:
    return j._binop(lambda x, y: x/y)


Joint.add = lambda self: add(self)
Joint.sub = lambda self: sub(self)
Joint.mul = lambda self: mul(self)
Joint.div = lambda self: div(self)


def transpose(j: Joint) -> Joint:
    return _joint(j._y, j._x, j.probs.T)


def marginal(j: Joint) -> Var:
    return Var(j._x, j.probs.sum(axis=1))


def covar(j: Joint) -> float:
    mx, my = expect(marginal(j)), expect(marginal(transpose(j)))
    return expect(j._binop(lambda x, y: (x-mx)*(y-my)))


def four_sides(coin: Var) -> Var:
    return indep(coin.op(lambda a: 2*a), coin).add().op(lambda a: a + 1)


def six_sides(coin: Var) -> Var:
    eight = indep(four_sides(coin), coin.op(lambda a: 4*a)).add()
    return eight.cond(lambda a: a <= 6)


def single_sample(x: Var, f: Callable[[float], float]) -> Var:
    return x.op(f)


def monte_carlo(x: Var, steps: int) -> Var:
    total = Var([0], [1.0])
    for _ in range(steps):
        total = add(indep(total, x))
    return div(indep(total, Var([steps], [1.0])))


def ten_samples(x: Var, f: Callable[[float], float]) -> Var:
    return monte_carlo(x.op(f), 10)


def linear_control(x: Var, f: Callable[[float], float], b: float) -> Joint:
    pair = shared(x)
    return pair.op(f, lambda a: b*a - b*expect(x))


def quadratic_control(x: Var, f: Callable[[float], float], a: float, b: float) -> Joint:
    h = lambda z: a*z*z + b*z
    center = expect(x.op(h))
    return shared(x).op(f, lambda z: h(z) - center)


def marginal_control(j: Joint) -> Joint:
    y = marginal(transpose(j))
    center = expect(y)
    return j.op(lambda x: x, lambda y: y-center)


def two_variables(x: Var, f: Callable[[float, float], float]) -> Var:
    averaged = x.op(lambda a: expect(x.op(lambda b: f(a, b))))
    return monte_carlo(averaged, 5)


def independent_two_variables(x: Var, y: Var,
                              f: Callable[[float, float], float]) -> Var:
    sample = indep(x, y)._binop(f)
    return monte_carlo(sample, 5)


def additive(x: Var, f: Callable[[float], float],
             g: Callable[[float, float], float]) -> Var:
    offset = expect(x.op(f))
    return add(indep(two_variables(x, g), Var([offset], [1.0])))


def two_dice(x: Var) -> Var:
    return indep(x, x).add()


def circle(x: Var, y: Var, radius: float, center: tuple[float, float] = (5, 5)) -> Joint:
    cx, cy = center
    return indep(x, y).cond(lambda a, b: (a-cx)**2 + (b-cy)**2 <= radius**2)


def triangular(x: Var, u: Var) -> Var:
    accepted = indep(x, u).cond(lambda a, b: b <= (6 - abs(a - 5))/6)
    return marginal(accepted)


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


def ab_sampling(population: Var, treatment: Callable[[float], float],
                control: Callable[[float], float], steps: int) -> Var:
    treated = monte_carlo(population.op(treatment), steps)
    untreated = monte_carlo(population.op(control), steps)
    return indep(treated, untreated).sub()


def ab_test(population: Var, treatment: Callable[[float], float],
            control: Callable[[float], float], initial: Callable[[float], float],
            b: float, steps: int) -> Var:
    center = expect(population.op(initial))
    def adjusted(outcome: Callable[[float], float]) -> Var:
        pair = shared(population).op(outcome, lambda person: b*(initial(person)-center))
        return pair.sub()
    treated = monte_carlo(adjusted(treatment), steps)
    untreated = monte_carlo(adjusted(control), steps)
    return indep(treated, untreated).sub()


def fit_cuped(initial: np.ndarray, outcome: np.ndarray) -> float:
    z = initial - initial.mean()
    y = outcome - outcome.mean()
    return float(np.linalg.lstsq(z[:, None], y, rcond=None)[0][0])


def fit_baseline(rewards: np.ndarray) -> float:
    return float(rewards.mean())


def estimated_baseline(x: Var, baseline: float) -> Joint:
    p = expect(x)
    return shared(x).op(lambda r: r*score(r, p), lambda r: baseline*score(r, p))


def kl_estimate(x: Var, p: Callable[[float], float], q: Callable[[float], float]) -> Var:
    return x.op(lambda a: k1(q(a) / p(a)))


def k1(r: float) -> float:
    return -np.log(r)


def k3(r: float) -> float:
    return r - 1 - np.log(r)


def kl_k3(x: Var, p: Callable[[float], float], q: Callable[[float], float]) -> Joint:
    ratios = x.op(lambda a: q(a) / p(a))
    return shared(ratios).op(lambda r: -np.log(r), lambda r: 1-r)


def topk(x: Var, f: Callable[[float], float], k: int) -> Var:
    x = x.op(lambda a: a)
    if not 0 <= k <= len(x.values):
        raise ValueError("k must be between zero and the support size")
    indices = np.argsort(-x.probs, kind="stable")[:k]
    top = x.values[indices]
    exact = sum(x.probs[i]*f(x.values[i]) for i in indices)
    return x.op(lambda a: exact + (0 if a in top else f(a)))


def kl_topk(x: Var, p: Callable[[float], float], q: Callable[[float], float], k: int) -> Var:
    return topk(x, lambda a: np.log(p(a)/q(a)), k)


def group_rewards(model: Var, reward_a: Callable[[float], float], reward_b: Callable[[float], float]) -> Joint:
    return shared(model).op(reward_a, reward_b)


def group_variance(rewards: Joint) -> float:
    a = marginal(rewards)
    b = marginal(transpose(rewards))
    return variance(a) + variance(b) + 2*covar(rewards)


def propagate(state: Var, transition: Joint) -> Var:
    """State values must match the transition rows; every row has positive mass."""
    conditional = transition.probs / transition.probs.sum(axis=1, keepdims=True)
    return Var(transition._y, state.probs @ conditional)


def markov_chain(state: Var, T: Joint, steps: int = 3) -> Var:
    for _ in range(steps):
        state = propagate(state, T)
    return state


def markov_unigram(state: Var, T: Joint, steps: int = 3) -> Joint:
    unigram = propagate(state, T)
    final = markov_chain(state, T, steps)
    final_cdf = np.cumsum(final.probs)
    unigram_cdf = np.cumsum(unigram.probs)
    final_cdf[-1] = unigram_cdf[-1] = 1.
    edges = np.unique(np.r_[0., final_cdf, unigram_cdf])
    uniform = Var((edges[:-1]+edges[1:])/2, np.diff(edges))
    return shared(uniform).op(
        lambda u: final.values[np.searchsorted(final_cdf, u)],
        lambda u: unigram.values[np.searchsorted(unigram_cdf, u)]-expect(unigram))


def binary_policy(theta: np.ndarray) -> Var:
    theta = np.asarray(theta)
    probs = np.exp(theta - np.max(theta))
    return Var([0, 1], probs / probs.sum())


def score(reward: float, p: float) -> float:
    """Second logit component of the binary log-probability gradient."""
    return reward - p


def gradient_pair(component: Var) -> Joint:
    """The two binary-logit gradient components are negatives of each other."""
    return shared(component).op(lambda g: -g, lambda g: g)


def loo_control(mean_reward: float, p: float, n: int) -> float:
    """Mean of leave-one-out baseline times score, for n >= 2 binary rewards."""
    mean_pair_product = (n*mean_reward**2 - mean_reward)/(n-1)
    return mean_pair_product - p*mean_reward


def reinforce(theta: np.ndarray, n: int = 1) -> Joint:
    action = binary_policy(theta)
    p = expect(action)
    component = monte_carlo(action.op(lambda r: r*score(r, p)), n)
    return gradient_pair(component)


def reinforce_loo(theta: np.ndarray, n: int = 3) -> Joint:
    action = binary_policy(theta)
    p = expect(action)
    fraction = monte_carlo(action, n)
    return shared(fraction).op(lambda m: m*score(1, p),
                               lambda m: loo_control(m, p, n))
