"""Lea-style finite random variables, evaluated with JAX arrays.

Install: pip install 'jax[cpu]'
Run examples: python distribution.py
Run checks: python distribution.py --test

    import distribution as jl
    x = jl.flip("x")
    y = x.apply(lambda v: v ** 2)
    average = jl.mean([jl.flip(f"x.{i}") for i in range(3)])
    average.mean(), average.var()  # 0, 1/3
    (x + x).var()                 # shared randomness: 4
    (x + jl.flip("y")).var()      # independent flips: 2
    y.given(x > 0).mean()          # 1
    average.names                 # ('x.0', 'x.1', 'x.2')

Design: eager JAX arrays plus named source axes. No expression graph, parent
pointers, or deferred evaluation. Arithmetic aligns named axes and immediately
computes outcome arrays; conditioning aligns and combines Boolean masks. Source
probability vectors remain factored until queries. Duplicate outcomes are kept
to preserve derivatives and provenance. No random sampling.

Outcomes may be scalars, vectors, or tensors. Leading axes identify random
sources; trailing axes are the shape of one outcome (event_shape). Arithmetic
broadcasts outcome shapes separately from random-source axes. apply(f) uses vmap
to apply a JAX-compatible function to each individual outcome. mean() and var()
return the outcome shape; covariance is an expectation of centered products.
given() requires a scalar Boolean per world; e.g. (v > 0).apply(jnp.all).

Limits: finite support, exponential enumeration for arbitrary arithmetic.
mean(draws) uses convolution for large independent tables with concrete outcomes.
Joint tables are bounded by MAX_WORLDS. Compression retains named provenance.
Build expressions inside a function passed to jit/grad; return JAX arrays, not
RV objects. Structure/support sizes must be static under jit.
Masses are normalized at construction. Invalid masses and zero-probability
evidence produce NaN results (also under jit), not Python exceptions.
All outcome functions must be defined on every enumerated world: conditioning
does not make division by zero or another invalid branch gradient-safe.
Support comparisons are discrete; gradients are meaningful through masses and
smooth value transformations, not across changes to hard evidence boundaries.
Uses ordinary JAX precision; exact enumeration does not mean exact arithmetic.
This is a research prototype, not API-compatible with the complete Lea library.
"""

from dataclasses import dataclass
import math
import operator
from itertools import count

import jax
import jax.numpy as jnp

MAX_WORLDS = 1_000_000
_draw_ids = count()


@dataclass(frozen=True)
class Draw:
    name: str
    kind: str
    id: int
    parent_id: int | None = None


def _draw(name, kind, parent_id=None):
    return Draw(name, kind, next(_draw_ids), parent_id)


@dataclass(eq=False, frozen=True)
class _Source:
    values: object
    mass: object
    draws: tuple = ()


def _merge(*groups):
    # Source equality is object identity, never overloaded RV equality.
    sources = tuple(dict.fromkeys(s for group in groups for s in group))
    owners = {}
    for source in sources:
        for draw in source.draws:
            if draw.id in owners and owners[draw.id] is not source:
                raise ValueError('Cannot recombine a compressed average with its original draws.')
            owners[draw.id] = source
    return sources


def _weights(raw):
    raw = jnp.asarray(raw)
    if jnp.iscomplexobj(raw):
        raise TypeError('Masses must be real.')
    raw = raw.astype(jnp.result_type(raw, jnp.float32))
    total = raw.sum()
    valid = jnp.all(jnp.isfinite(raw) & (raw >= 0)) & (total > 0)
    return jnp.where(valid, raw / jnp.where(valid, total, 1), jnp.nan)


def _shape(sources):
    shape = tuple(s.values.shape[0] for s in sources)
    if math.prod(shape) > MAX_WORLDS:
        raise ValueError(f'Joint table exceeds {MAX_WORLDS:,} worlds.')
    return shape


def _align(array, old_sources, new_sources):
    """Align source axes, leaving trailing outcome axes unchanged."""
    old_positions = {s: i for i, s in enumerate(old_sources)}
    order = tuple(old_positions[s] for s in new_sources if s in old_positions)
    event_shape = array.shape[len(old_sources):]
    array = jnp.transpose(array, order + tuple(range(len(old_sources), array.ndim)))
    shape = tuple(s.values.shape[0] if s in old_positions else 1
                  for s in new_sources)
    return array.reshape(shape + event_shape)


def _pad_events(array, n_sources, event_ndim):
    """Insert leading singleton event axes for ordinary NumPy broadcasting."""
    event_shape = array.shape[n_sources:]
    return array.reshape(array.shape[:n_sources] +
                         (1,) * (event_ndim - len(event_shape)) + event_shape)


class RV:
    """Eager outcome tensor, source axes, and an evidence mask; no graph."""

    __hash__ = None
    __array_priority__ = 1000

    def __init__(self, sources, values, mask=True, source=None):
        self._sources = sources
        shape = _shape(sources)
        values = jnp.asarray(values)
        self.event_shape = values.shape[len(sources):]
        self.values = jnp.broadcast_to(values, shape + self.event_shape)
        self.mask = jnp.broadcast_to(jnp.asarray(mask, dtype=bool), shape)
        self._source = source

    def __bool__(self):
        raise TypeError('Use &, |, ~ and .prob(); an RV is not a Python bool.')

    @property
    def draws(self):
        return tuple(dict.fromkeys(draw for s in self._sources for draw in s.draws))

    @property
    def names(self):
        return tuple(draw.name for draw in self.draws)

    @property
    def flips(self):
        return tuple(draw for draw in self.draws if draw.kind == 'flip')

    def apply(self, f):
        """Apply f to each single outcome using vmap; supports vector outputs."""
        count = math.prod(_shape(self._sources))
        result = jax.vmap(f)(self.values.reshape((count,) + self.event_shape))
        result = jnp.asarray(result)
        return RV(self._sources,
                  result.reshape(_shape(self._sources) + result.shape[1:]),
                  self.mask)

    def __getitem__(self, key):
        """Index outcome components, not random-source axes: g[0], g[:2], etc."""
        return self.apply(lambda value: value[key])

    def __matmul__(self, other):
        """Independent product: x @ y, even when x and y share sources."""
        return independent(self, other)

    def _binary(self, other, f):
        other = other if isinstance(other, RV) else constant(other)
        sources = _merge(self._sources, other._sources)
        _shape(sources)  # Check the world budget BEFORE allocating the product.
        left = _align(self.values, self._sources, sources)
        right = _align(other.values, other._sources, sources)
        ndim = max(len(self.event_shape), len(other.event_shape))
        left = _pad_events(left, len(sources), ndim)
        right = _pad_events(right, len(sources), ndim)
        mask = (_align(self.mask, self._sources, sources) &
                _align(other.mask, other._sources, sources))
        return RV(sources, f(left, right), mask)

    def given(self, event):
        """Align and attach Boolean evidence; repeated evidence is conjoined.

        Combining conditioned RVs conjoins their masks. No marginalization is
        performed, so subsequent operations retain all shared source axes.
        """
        event = event if isinstance(event, RV) else constant(event)
        if event.values.dtype != jnp.bool_ or event.event_shape:
            raise TypeError('given() requires a scalar Boolean per outcome; '
                            'reduce vector events with .apply(jnp.all/any).')
        sources = _merge(self._sources, event._sources)
        _shape(sources)
        mask = (_align(self.mask, self._sources, sources) &
                _align(event.mask & event.values, event._sources, sources))
        return RV(sources, _align(self.values, self._sources, sources),
                  mask, self._source)

    def _repeat(self, n):
        """Sum n independent copies, collapsing permutations into count vectors.

        The result is a fresh source, independent of the original RV. Fixed
        support (including zero-mass worlds) preserves JIT and autodiff.
        Equal numerical outcomes are not merged: their derivatives can differ.
        """
        if not isinstance(n, int) or n < 0:
            raise ValueError('n must be a static nonnegative Python integer.')
        if n == 0:
            return constant(jnp.zeros(self.event_shape, dtype=self.values.dtype))
        values, weights = self.table()
        k = values.shape[0]
        size = math.comb(n+k-1, k-1)
        if size * max(k, math.prod(self.event_shape)) > MAX_WORLDS:
            raise ValueError(f'Count table exceeds {MAX_WORLDS:,} entries.')
        from itertools import combinations
        import numpy as np
        counts = np.array([np.diff((-1, *bars, n+k-1))-1
                           for bars in combinations(range(n+k-1), k-1)])
        coefficients = np.array([math.factorial(n) / math.prod(math.factorial(int(c))
                                for c in row) for row in counts])
        counts = jnp.asarray(counts)
        # Polynomial multiplication keeps higher derivatives finite at zero
        # masses; array-valued powers can produce 0 * infinity there.
        factors = jnp.ones(counts.shape, dtype=weights.dtype)
        for draw in range(1, n+1):
            factors = factors * jnp.where(counts >= draw, weights, 1)
        mass = jnp.asarray(coefficients) * jnp.prod(factors, axis=1)
        totals = (counts @ values.reshape(k, -1)).reshape((size,) + self.event_shape)
        draws = tuple(_draw(f'{draw.name}[{i}]', draw.kind, draw.id)
                      for i in range(n) for draw in self.draws)
        return _pmf(totals, mass, draws)

    def _raw_table(self, overrides=None):
        shape = _shape(self._sources)
        mass = jnp.ones(shape)
        for axis, source in enumerate(self._sources):
            view_shape = [1] * len(shape)
            view_shape[axis] = shape[axis]
            weights = (overrides[source] if overrides is not None
                       and source in overrides else source.mass)
            mass = mass * weights.reshape(view_shape)
        return self.values, jnp.where(self.mask, mass, 0)

    def table(self):
        """Arrays shaped (worlds, *event_shape) and (worlds,); no merging."""
        values, mass = self._raw_table()
        total = mass.sum()
        valid = jnp.isfinite(total) & (total > 0)
        mass = jnp.where(valid, mass / jnp.where(valid, total, 1), jnp.nan)
        return values.reshape((mass.size,) + self.event_shape), mass.ravel()

    def mean(self):
        values, mass = self.table()
        weights = mass.reshape((-1,) + (1,) * len(self.event_shape))
        return jnp.sum(weights * values, axis=0)

    def var(self):
        """Componentwise variance, with the same shape as one outcome."""
        values, mass = self.table()
        weights = mass.reshape((-1,) + (1,) * len(self.event_shape))
        mean = jnp.sum(weights * values, axis=0)
        return jnp.sum(weights * (values - mean) ** 2, axis=0)

    def prob(self):
        """Probability of a Boolean expression, subject to its evidence."""
        values, mass = self.table()
        if values.dtype != jnp.bool_ or self.event_shape:
            raise TypeError('prob() requires a scalar Boolean per outcome.')
        return jnp.sum(mass * values)

    def posterior_via_grad(self, event):
        """For a primitive RV, return (source values, posterior masses).

        Differentiates the unnormalized evidence mass Z with respect to this
        source's independent mass coordinates, then returns p * dZ/dp / Z.
        Other sources remain fixed even if their probabilities share parameters.
        This returns a marginal posterior, not the complete conditional joint.
        """
        if self._source is None:
            raise TypeError('Call posterior_via_grad on a primitive RV.')
        conditioned = self.given(event)

        def z(weights):
            _, mass = conditioned._raw_table({self._source: weights})
            return mass.sum()

        mass = self._source.mass
        total, gradient = jax.value_and_grad(z)(mass)
        valid = jnp.isfinite(total) & (total > 0)
        posterior = jnp.where(valid, mass * gradient /
                              jnp.where(valid, total, 1), jnp.nan)
        return self._source.values, posterior


def constant(value):
    value = jnp.asarray(value)
    return RV((), value)


def joint(x, y):
    """Pair two scalar RVs, preserving their actual shared randomness/evidence."""
    if x.event_shape or y.event_shape:
        raise ValueError('joint() currently pairs scalar RVs.')
    return x._binary(y, lambda a, b: jnp.stack(jnp.broadcast_arrays(a, b), axis=-1))


def independent(x, y):
    """Independent product of the two marginals; equivalent to x @ y.

    Both source sets are freshly named and retain parent draw IDs. In contrast,
    joint(x, y) retains the existing coupling between its arguments.
    """
    def copy(rv, side):
        replacements = {
            source: _Source(source.values, source.mass,
                            tuple(_draw(f'{side}.{draw.name}', draw.kind, draw.id)
                                  for draw in source.draws))
            for source in rv._sources
        }
        return RV(tuple(replacements.values()), rv.values, rv.mask,
                  replacements.get(rv._source))
    return joint(copy(x, 'left'), copy(y, 'right'))


def mean(draws):
    """Average explicitly constructed RVs, preserving sharing in small tables.

    Large independent tables use convolution with equal outcomes coalesced.
    Such a compressed average cannot be recombined with its original inputs.
    Traced outcomes cannot be coalesced; their tables obey MAX_WORLDS.
    """
    draws = list(draws)
    if not draws:
        raise ValueError('mean() needs at least one draw.')
    sources = _merge(*(rv._sources for rv in draws))
    if math.prod(s.values.shape[0] for s in sources) <= MAX_WORLDS:
        return sum(draws)/len(draws)
    seen = set()
    for rv in draws:
        ids = {draw.id for draw in rv.draws}
        if seen & ids:
            raise ValueError('Large averages require independent draws.')
        seen.update(ids)
    import numpy as np
    total = constant(jnp.zeros(draws[0].event_shape))
    for rv in draws:
        combined = total + rv
        values, weights = combined.table()
        if isinstance(values, jax.core.Tracer):
            total = combined
            continue
        _, indices, inverse = np.unique(np.asarray(values), axis=0,
                                        return_index=True, return_inverse=True)
        mass = jnp.zeros(len(indices), dtype=weights.dtype).at[inverse].add(weights)
        total = _pmf(values[indices], mass, combined.draws)
    return total/len(draws)


def pmf(values, weights, name="pmf"):
    return _pmf(values, weights, (_draw(name, 'categorical'),))


def _pmf(values, weights, draws):
    """Create a fresh source from (K, *event_shape) values and (K,) weights.
    Duplicated values are allowed and remain separate latent outcomes.
    """
    values, weights = jnp.asarray(values), jnp.asarray(weights)
    if values.ndim < 1 or weights.ndim != 1 or values.shape[0] != weights.shape[0]:
        raise ValueError('Expected values shaped (K, *event_shape) and K weights.')
    if values.size == 0:
        raise ValueError('Empty support is not supported.')
    source = _Source(values, _weights(weights), draws)
    return RV((source,), values, source=source)


def _uniform(*values):
    """Uniform distribution over the supplied outcomes."""
    return pmf(values, jnp.ones(len(values)))


def flip(name):
    """A fresh fair -1/+1 flip (mean 0, variance 1). Reuse the RV to share it."""
    if not isinstance(name, str) or not name:
        raise ValueError('A flip needs a nonempty string name.')
    return _pmf([-1, 1], [1, 1], (_draw(name, 'flip'),))


def from_logits(values, logits, name="action"):
    return pmf(values, jax.nn.softmax(jnp.asarray(logits)), name=name)


# Operators immediately compute aligned JAX outcome tensors.
def _install_operators():
    for name, fn in {
        'add': operator.add, 'sub': operator.sub, 'mul': operator.mul,
        'truediv': operator.truediv, 'floordiv': operator.floordiv,
        'pow': operator.pow, 'mod': operator.mod,
        'eq': operator.eq, 'ne': operator.ne, 'lt': operator.lt,
        'le': operator.le, 'gt': operator.gt, 'ge': operator.ge,
        'and': operator.and_, 'or': operator.or_, 'xor': operator.xor,
    }.items():
        setattr(RV, '__' + name + '__',
                lambda self, other, f=fn: self._binary(other, f))
        if name not in {'eq', 'ne', 'lt', 'le', 'gt', 'ge'}:
            setattr(RV, '__r' + name + '__',
                    lambda self, other, f=fn:
                    self._binary(other, lambda a, b: f(b, a)))
    RV.__neg__ = lambda self: self.apply(operator.neg)
    RV.__abs__ = lambda self: self.apply(operator.abs)
    RV.__invert__ = lambda self: self.apply(operator.invert)


_install_operators()


def demo():
    x = _uniform(-2, -1, 0, 1, 2)
    average = (x ** 2)._repeat(3) / 3
    print('Average: mean, variance =', average.mean(), average.var())
    print('Conditional square mean =', (x ** 2).given(x > 0).mean())
    a, b = _uniform(0, 1), pmf([0, 1], [0.2, 0.8])
    print('P(X | X+Y=1) =', a.posterior_via_grad(a + b == 1)[1])

    def objective(logits):
        z = from_logits([-2., -1., 0., 1., 2.], logits)
        return (z ** 2)._repeat(3).var() / 9

    print('Gradient of average variance =',
          jax.jit(jax.grad(objective))(jnp.zeros(5)))

    theta = jnp.array([jnp.log(3.), 0.])
    p = jax.nn.softmax(theta)
    action = pmf([0, 1], p)
    reward = action  # reward 0 for action 0, reward 1 for action 1
    def log_prob(theta, a):
        return jax.nn.log_softmax(theta)[a]

    score = action.apply(lambda a: jax.grad(log_prob)(theta, a))
    g = reward * score
    print('REINFORCE vector outcomes and masses =', g.table())
    print('Expected gradient =', g.mean())
    centered = g-g.mean()
    print('Gradient covariance =', centered.apply(lambda v: jnp.outer(v, v)).mean())


def self_test():
    import unittest
    import numpy as np

    def covariance(rv):
        return (rv-rv.mean()).apply(lambda v: jnp.outer(v, v)).mean()

    class Checks(unittest.TestCase):
        def test_vector_reinforce(self):
            def gradient_rv(theta):
                p = jax.nn.softmax(theta)
                a = pmf([0, 1], p)
                score = a.apply(lambda i: jax.nn.one_hot(i, 2) - p)
                return a * score

            theta = jnp.array([jnp.log(3.), 0.])
            g = gradient_rv(theta)
            v, p = g.table()
            np.testing.assert_allclose(v, [[0., 0.], [-.75, .75]], atol=1e-6)
            np.testing.assert_allclose(p, [.75, .25], atol=1e-6)
            expected = jax.grad(lambda t: jax.nn.softmax(t)[1])(theta)
            np.testing.assert_allclose(g.mean(), expected, atol=1e-6)
            np.testing.assert_allclose(g.var(), [.10546875]*2, atol=1e-6)
            cov = .10546875 * np.array([[1., -1.], [-1., 1.]])
            np.testing.assert_allclose(covariance(g), cov, atol=1e-6)
            np.testing.assert_allclose(covariance(g._repeat(10)/10), cov/10, atol=1e-6)
            compiled = jax.jit(lambda t: gradient_rv(t).mean())
            np.testing.assert_allclose(compiled(theta), expected, atol=1e-6)
            # Differentiating the expectation again recovers the Hessian.
            np.testing.assert_allclose(jax.jacrev(compiled)(theta),
                jax.hessian(lambda t: jax.nn.softmax(t)[1])(theta), atol=1e-6)

        def test_vector_axes_and_conditioning(self):
            x, y = _uniform(0., 1.), _uniform(1., 2., 3.)
            u = x.apply(lambda a: jnp.array([a, 2*a]))
            v = y.apply(lambda b: jnp.array([b, -b]))
            np.testing.assert_allclose(((u+v)-(v+u)).var(), [0., 0.])
            np.testing.assert_allclose((u*y).mean(), [1., 2.])
            np.testing.assert_allclose(u.given(x+y == 2).mean(), [.5, 1.])
            np.testing.assert_allclose(u.given(x == 1).mean(), [1., 2.])
            np.testing.assert_allclose(u[1].mean(), 1.)
            np.testing.assert_allclose(covariance(u-u), np.zeros((2, 2)))
            np.testing.assert_allclose(covariance(u-u._repeat(1)), 2*covariance(u))
            np.testing.assert_allclose(u._repeat(0).mean(), [0., 0.])
            np.testing.assert_allclose((u > 0).apply(jnp.all).prob(), .5)
            with self.assertRaises(TypeError):
                u.given(u > 0)

        def test_vector_sources_and_tensor_broadcasting(self):
            v = pmf([[1., 2.], [3., 4.]], [.25, .75])
            np.testing.assert_allclose(v.mean(), [2.5, 3.5])
            np.testing.assert_allclose(covariance(v), np.full((2, 2), .75))
            np.testing.assert_allclose(v.posterior_via_grad(v[0] == 3)[1], [0, 1])
            # Outcome broadcasting is independent of the support's size.
            matrix = constant(jnp.array([[1.], [2.], [3.]])) * v
            self.assertEqual(matrix.event_shape, (3, 2))
            np.testing.assert_allclose(matrix.mean(),
                np.array([[1.], [2.], [3.]]) * np.array([2.5, 3.5]))
            np.testing.assert_allclose(v.apply(lambda z: 7.).mean(), 7.)

        def test_independence_and_reuse(self):
            x = _uniform(0, 1)
            self.assertAlmostEqual(float((x + x).var()), 1.)
            self.assertAlmostEqual(float((x + x._repeat(1)).var()), .5)
            self.assertEqual(float((x - x).var()), 0.)
            # Operands can carry the same axes in opposite orders and sizes.
            y = _uniform(1, 2, 3)
            self.assertEqual(float(((x+y)-(y+x)).var()), 0.)
            self.assertAlmostEqual(float(y.given(x+y == 2).mean()), 1.5)

        def test_nonlinear_and_iid_average(self):
            x = _uniform(-2, -1, 0, 1, 2)
            m = (x ** 2)._repeat(3) / 3
            self.assertAlmostEqual(float(m.mean()), 2., places=5)
            self.assertAlmostEqual(float(m.var()), 2.8 / 3, places=5)
            self.assertAlmostEqual(float((1 / _uniform(1, 2)).mean()), .75)

        def test_conditioning_and_clone(self):
            x, y = _uniform(0, 1), pmf([0, 1], [.2, .8])
            event = x + y == 1
            self.assertAlmostEqual(float(event.prob()), .5)
            self.assertAlmostEqual(float(x.given(event).mean()), .2)
            self.assertAlmostEqual(float((x+y).given(event).var()), 0.)
            c = x.given(event)
            self.assertAlmostEqual(float((c-c._repeat(1)).var()), .32, places=6)
            self.assertTrue(bool(jnp.isnan(x.given(x > 2).mean())))
            with self.assertRaises(TypeError):
                bool(x)

        def test_reverse_inference_and_jit(self):
            def posterior(p, q):
                x, y = pmf([0, 1], p), pmf([0, 1], q)
                return x.posterior_via_grad(x + y == 1)[1]
            p, q = jnp.array([.5, .5]), jnp.array([.2, .8])
            np.testing.assert_allclose(jax.jit(posterior)(p, q), [.8, .2])
            x = _uniform(0, 1)
            np.testing.assert_allclose(x.posterior_via_grad(x+x == 2)[1], [0, 1])

        def test_gradients(self):
            def objective(logits):
                x = from_logits([-2., -1., 0., 1., 2.], logits)
                return (x**2)._repeat(3).var() / 9
            # d Var(Y)/d logit_i = p_i ((y_i-mu)^2 - Var(Y)).
            expected = np.array([1.2, -1.8, 1.2, -1.8, 1.2]) / 15
            np.testing.assert_allclose(jax.jit(jax.grad(objective))(jnp.zeros(5)),
                                       expected, atol=1e-6)
            # Moving support values also retain their pathwise derivatives.
            def moving(t):
                return (_uniform(-1., 1.).apply(lambda v: t*v)).var()
            self.assertAlmostEqual(float(jax.grad(moving)(2.)), 4.)

        def test_conditional_gradient(self):
            def objective(t):
                x = from_logits([0., 1., 2.], jnp.array([0., t, 2*t]))
                return x.given(x > 0).mean()
            # Conditional probability of 2 is sigmoid(t).
            t = jnp.array(.3)
            expected = jax.nn.sigmoid(t) * (1-jax.nn.sigmoid(t))
            np.testing.assert_allclose(jax.grad(objective)(t), expected, atol=1e-6)

        def test_validation_and_limits(self):
            with self.assertRaises(ValueError):
                _uniform()
            with self.assertRaises(TypeError):
                _uniform(0, 1).given(1).mean()
            with self.assertRaises(ValueError):
                _uniform(0, 1)._repeat(MAX_WORLDS).mean()
            self.assertTrue(bool(jnp.isnan(pmf([0, 1], [-1., 2.]).mean())))

        def test_compact_times(self):
            die = _uniform(1., 2., 3., 4., 5., 6.)
            average = die._repeat(10) / 10
            self.assertEqual(average.table()[0].shape, (3003,))
            np.testing.assert_allclose(average.mean(), die.mean(), rtol=1e-5)
            np.testing.assert_allclose(average.var(), die.var()/10, rtol=1e-5)
            # The new sum is independent of its original sources.
            copy = die._repeat(1)
            np.testing.assert_allclose((copy-die).var(), 2*die.var(), rtol=1e-5)
            tensor = die.apply(lambda a: jnp.array([[a, 2*a], [-a, a*a]]))
            np.testing.assert_allclose(tensor._repeat(2).mean(), 2*tensor.mean(), rtol=1e-5)
            np.testing.assert_allclose(tensor._repeat(2).var(), 2*tensor.var(), rtol=1e-5)

        def test_compact_times_conditioned_autodiff(self):
            def compact(t):
                x = from_logits([0., 1., 2.], jnp.array([0., t, -t]))
                return (x.given(x > 0)._repeat(3)/3).var()
            def reference(t):
                x = from_logits([0., 1., 2.], jnp.array([0., t, -t]))
                return x.given(x > 0).var()/3
            for order in range(3):
                left, right = compact, reference
                for _ in range(order):
                    left, right = jax.grad(left), jax.grad(right)
                np.testing.assert_allclose(jax.jit(left)(.3), jax.jit(right)(.3), atol=1e-6)

    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    import sys
    self_test() if '--test' in sys.argv else demo()
