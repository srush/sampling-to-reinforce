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


def shift(x: Var, amount: float) -> Var:
    return x.op(lambda a: a + amount)


def square(x: Var, b: float) -> Var:
    return x.op(lambda a: a ** b)


def six_from_eight(x: Var) -> Var:
    return x.cond(lambda a: a <= 6)


def variance(x: Var) -> float:
    mean = expect(x)
    return expect(x.op(lambda a: (a - mean)**2))


def shared(x: Var) -> Joint:
    return _joint(x.values, x.values, np.diag(x.probs))


def indep(x: Var, y: Var) -> Joint:
    return _joint(x.values, y.values, np.outer(x.probs, y.probs))


def _sample_indices(code: float, base: int, steps: int) -> tuple[int, ...]:
    """Decode a scalar batch ID into support indices, including leading zeros."""
    indices = []
    code = int(code)
    for _ in range(steps):
        code, digit = divmod(code, base)
        indices.append(digit)
    return tuple(reversed(indices))


def _sorted_sample_code(code: float, base: int, steps: int) -> int:
    """Canonical ID for an unordered batch; binop merges its probability mass."""
    result = 0
    for digit in sorted(_sample_indices(code, base, steps)):
        result = result*base + digit
    return result


def iid_statistic(draw: Var, steps: int,
                  statistic: Callable[[tuple[float, ...]], float],
                  *, symmetric: bool = False) -> Var:
    """Compose scalar batch IDs; symmetric=True requires an order-invariant statistic."""
    base = len(draw.values)
    if base**steps > 2**53:
        raise OverflowError("Batch IDs must fit exactly in the scalar float representation")
    indices = Var(np.flatnonzero(draw.probs), draw.probs[draw.probs > 0])
    batches = Var([0], [1])
    for size in range(1, steps + 1):
        def append(code, digit):
            result = int(code)*base + int(digit)
            return _sorted_sample_code(result, base, size) if symmetric else result
        batches = indep(batches, indices)._binop(append)
    return batches.op(lambda code: round(float(statistic(tuple(
        draw.values[i] for i in _sample_indices(code, base, steps)
    ))), 12))


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


def condition_joint(predicate: Callable[[float, float], bool], j: Joint) -> Joint:
    keep = np.array([[predicate(a, b) for b in j._y] for a in j._x])
    probs = j.probs * keep
    return _joint(j._x, j._y, probs / probs.sum())


Joint.cond = lambda self, predicate: condition_joint(predicate, self)


def add(j: Joint) -> Var:
    return j._binop(lambda x, y: x+y)


def sub(j: Joint) -> Var:
    return j._binop(lambda x, y: x-y)


def div(j: Joint) -> Var:
    return j._binop(lambda x, y: x/y)


Joint.add = lambda self: add(self)
Joint.sub = lambda self: sub(self)
Joint.mul = lambda self: self._binop(lambda x, y: x*y)
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


def additive(x: Var, f: Fn,
             g: Callable[[float, float], float]) -> Var:
    offset = expect(x.op(f))
    return add(indep(two_variables(x, g), Var([offset], [1.0])))


def two_dice(x: Var) -> Var:
    return indep(x, x).add()


def two_triangles(x: Var) -> Var:
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


def fit_cuped(initial: np.ndarray, outcome: np.ndarray) -> float:
    z = initial - initial.mean()
    y = outcome - outcome.mean()
    return float(np.linalg.lstsq(z[:, None], y, rcond=None)[0][0])


def cuped(population: Var, outcome: Fn,
          initial: Fn, known_mean: float, steps: int) -> Var:
    """Full sampling distribution with leave-one-out fits; steps >= 2."""
    def estimate(samples):
        z = np.array([initial(person) for person in samples])
        y = np.array([outcome(person) for person in samples])
        adjusted = []
        for i in range(len(samples)):
            others = np.arange(len(samples)) != i
            b = fit_cuped(z[others], y[others])
            adjusted.append(y[i] - b*(z[i] - known_mean))
        return sum(adjusted)/len(adjusted)
    return iid_statistic(population, steps, estimate, symmetric=True)


def fit_baseline(rewards: np.ndarray) -> float:
    return float(rewards.mean())


def estimated_baseline(x: Var, baseline: float) -> Joint:
    p = expect(x)
    return shared(x).op(lambda r: r*score(r, p), lambda r: baseline*score(r, p))


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


def k1(p: Var, q: Var) -> Var:
    """One-sample KL estimate under p."""
    return p.op(lambda a: p.log_prob(a) - q.log_prob(a))


def k2(p: Var, q: Var) -> Var:
    """Squared-log approximation under p."""
    return p.op(lambda a: 0.5 * (p.log_prob(a) - q.log_prob(a))**2)


def kl_k3(p: Var, q: Var) -> Joint:
    """Pair the KL target and zero-mean control; p and q share positive support."""
    return control_variate(
        p, lambda a: p.log_prob(a) - q.log_prob(a),
        lambda a: q.prob(a) / p.prob(a), known_mean=1, b=-1)


def k3(p: Var, q: Var) -> Var:
    """KL estimate with a zero-mean control; p and q share positive support."""
    return kl_k3(p, q).sub()


def topk(x: Var, f: Fn, k: int) -> Var:
    x = x.op(lambda a: a)
    if not 0 <= k <= len(x.values):
        raise ValueError("k must be between zero and the support size")
    indices = np.argsort(-x.probs, kind="stable")[:k]
    top = x.values[indices]
    exact = sum(x.probs[i]*f(x.values[i]) for i in indices)
    return x.op(lambda a: exact + (0 if a in top else f(a)))


def kl_topk(p: Var, q: Var, k: int) -> Var:
    """On-policy value estimator from Algorithm 1 of arXiv:2602.04417."""
    if not np.isfinite(kl(p, q)):
        raise ValueError("q must assign positive probability wherever p does")
    p = p.op(lambda a: a)  # Combine repeated outcomes before ranking.
    if not 0 <= k <= len(p.values):
        raise ValueError("k must be between zero and the support size")
    indices = np.argsort(-p.probs, kind="stable")[:k]
    top = p.values[indices]  # Select by p(a), before computing any KL terms.

    def log_ratio(a):
        return p.log_prob(a) - q.log_prob(a) if p.prob(a) else 0.0

    exact = sum(p.prob(a)*log_ratio(a) for a in top)
    # X ~ p over the full support; mask the sampled term when X is in top.
    sampled = p.op(lambda a: 0.0 if a in top else log_ratio(a))
    return sampled.op(lambda tail: exact + tail)


def group_rewards(model: Var, reward_a: Fn, reward_b: Fn) -> Joint:
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


def temperature_policy(temperature: float) -> Var:
    """Eight classes with fixed logits 0,...,7 and positive temperature."""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    logits = np.arange(8, dtype=float)
    weights = np.exp((logits - logits.max()) / temperature)
    return Var(logits, weights / weights.sum())


def temperature_score(action: float, mean_action: float, temperature: float) -> float:
    """Derivative of log p_T(action) with respect to T."""
    return (mean_action - action) / temperature**2


def temperature_reinforce(temperature: float, n: int = 1) -> Var:
    action = temperature_policy(temperature)
    mean_action = expect(action)
    return monte_carlo(action.op(
        lambda a: a * temperature_score(a, mean_action, temperature)), n)


def reinforce_control(r: Fn, b: float, score: Joint) -> Var:
    """Subtract a constant baseline from the reward, then multiply by the score."""
    return score.op(lambda a: r(a) - b, lambda s: s).mul()


def leave_one_out(temperature: float, n: int = 3, seed: int | None = 11) -> float:
    """Mean reward of n-1 independent other draws; rewards equal class indices."""
    if n < 2:
        raise ValueError("Leave-one-out needs at least two samples")
    action = temperature_policy(temperature)
    others = np.random.default_rng(seed).choice(
        action.values, size=n-1, p=action.probs)
    return float(others.mean())


def _leave_one_out_gradient(r: Fn, score: Joint, n: int = 3) -> Var:
    """Enumerate independent batches and apply a constant baseline to each draw."""
    from itertools import product
    if n < 2:
        raise ValueError("Leave-one-out needs at least two samples")
    rows, cols = np.nonzero(score.probs)
    actions, scores = score._x[rows], score._y[cols]
    mass = score.probs[rows, cols]
    estimates, probabilities = [], []
    for indices in product(range(len(mass)), repeat=n):
        rewards = np.array([r(actions[i]) for i in indices])
        adjusted = []
        for position, i in enumerate(indices):
            baseline = float(np.mean(np.delete(rewards, position)))
            draw = Joint([actions[i]], [scores[i]], [[1.0]])
            adjusted.append(expect(reinforce_control(r, baseline, draw)))
        estimates.append(float(np.mean(adjusted)))
        probabilities.append(float(np.prod(mass[list(indices)])))
    return Var(estimates, probabilities)


def temperature_loo(temperature: float, n: int = 3) -> Var:
    """Compatibility wrapper for the temperature-policy example."""
    action = temperature_policy(temperature)
    score = shared(action).op(
        lambda a: a,
        lambda a: temperature_score(a, expect(action), temperature))
    return _leave_one_out_gradient(lambda a: a, score, n)


def temperature_baseline(temperature: float, baseline: float) -> Joint:
    action = temperature_policy(temperature)
    mean_action = expect(action)
    return shared(action).op(
        lambda a: a * temperature_score(a, mean_action, temperature),
        lambda a: baseline * temperature_score(a, mean_action, temperature))
