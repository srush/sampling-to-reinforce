"""Minimal answers. Open puzzle.py as a Jupytext notebook for the exercises."""

import distribution as d
import jax.numpy as jnp
import jax


def four_sides(name="four"):
    a = (d.flip(f"{name}.a")+1)//2
    b = (d.flip(f"{name}.b")+1)//2
    return 1 + 2*a + b


def six_sides(name="six"):
    four = four_sides(name)
    candidate = four + 4*((d.flip(f"{name}.c")+1)//2)
    return candidate.given(candidate <= 6)


def two_dice(x):
    return x + six_sides("second")


def single_sample(x, f):
    return x.apply(f)


def ten_samples(x, f):
    return d.mean([six_sides(f"x.{i}").apply(f) for i in range(10)])


def linear_control(x, f, b):
    draws = [six_sides(f"x.{i}") for i in range(5)]
    return d.mean([a.apply(f)-b*a for a in draws]) + b*x.mean()


def quadratic_control(x, f, a, b):
    draws = [six_sides(f"x.{i}") for i in range(5)]
    return d.mean([z.apply(f)-a*z**2-b*z for z in draws]) + a*(x**2).mean() + b*x.mean()


def two_variables(x, f):
    y = six_sides("y")
    draws = [six_sides(f"x.{i}") for i in range(5)]
    return d.mean([z.apply(lambda a: y.apply(lambda b: f(a, b)).mean()) for z in draws])


def additive(x, f, g):
    return x.apply(f).mean() + two_variables(x, g)


def weighted_die(weights):
    weights = tuple(weights)
    if not weights or any(int(w) != w or w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("Use nonnegative integer weights with positive total")
    total = int(sum(weights))
    candidate = d.constant(0)
    for i in range((total-1).bit_length()):
        candidate = 2*candidate + (d.flip(f"weighted.{i}")+1)//2
    accepted = candidate.given(candidate < total)
    edges = jnp.cumsum(jnp.array(weights))
    return accepted.apply(lambda u: jnp.minimum(jnp.searchsorted(edges, u, side="right"), len(weights)-1)+1)


def kl_estimate(p, q, n=5):
    p, q = jnp.asarray(p), jnp.asarray(q)
    draws = [d.pmf(jnp.arange(len(p)), p, name=f"x.{i}") for i in range(n)]
    return d.mean([x.apply(lambda a: jnp.log(p[a]/q[a])) for x in draws])


def k3(r):
    return (r-1) - jnp.log(r)


def kl_k3(p, q, n=5):
    p, q = jnp.asarray(p), jnp.asarray(q)
    draws = [d.pmf(jnp.arange(len(p)), p, name=f"x.{i}") for i in range(n)]
    return d.mean([x.apply(lambda a: k3(q[a]/p[a])) for x in draws])


def markov_chain(steps=3):
    state = d.constant(0)
    for i in range(steps):
        u = four_sides(f"step.{i}")-1
        state = (u < 1+2*state)*1
    return state


def markov_unigram(steps=3):
    state = d.constant(0)
    for i in range(steps):
        u = four_sides(f"step.{i}")-1
        state = (u < 1+2*state)*1
    unigram = (u < 1)*1
    return state - unigram + 0.25


def log_prob(theta, action):
    return jax.nn.log_softmax(theta)[action]


def policy_terms(theta):
    a = d.from_logits([0, 1], theta)
    reward = a
    score = a.apply(lambda action: jax.grad(log_prob)(theta, action))
    return reward, score


def reinforce(theta):
    a = d.from_logits([0, 1], theta)
    reward = a
    score = a.apply(lambda action: jax.grad(log_prob)(theta, action))
    return reward*score


def reinforce_loo(theta, n=3):
    if n < 2:
        raise ValueError("Leave-one-out needs at least two samples")
    # Each pair shares its action; different pairs use independent actions.
    pairs = [policy_terms(theta) for _ in range(n)]
    total_reward = sum(r for r, s in pairs)
    return sum((r-(total_reward-r)/(n-1))*s for r, s in pairs)/n
