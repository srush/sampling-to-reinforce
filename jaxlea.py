"""Lea-style finite random variables, evaluated with JAX arrays.

Install: pip install 'jax[cpu]'
Run examples: python jaxlea.py
Run checks: python jaxlea.py --test

    import jaxlea as jl
    x = jl.vals(-2, -1, 0, 1, 2)
    y = x.map(lambda v: v ** 2)
    average = y.times(3) / 3
    average.mean(), average.var()  # 2, 2.8/3
    (x + x).var()                 # shared randomness: 8
    (x + x.new()).var()           # independent copies: 4
    y.given(x > 0).mean()         # 2.5

Design: eager JAX arrays plus named source axes. No expression graph, parent
pointers, or deferred evaluation. Arithmetic aligns named axes and immediately
computes outcome arrays; conditioning aligns and combines Boolean masks. Source
probability vectors remain factored until queries. Duplicate outcomes are kept
to preserve derivatives and provenance. No random sampling.

Outcomes may be scalars, vectors, or tensors. Leading axes identify random
sources; trailing axes are the shape of one outcome (event_shape). Arithmetic
broadcasts outcome shapes separately from random-source axes. map(f) uses vmap
to apply a JAX-compatible function to each individual outcome. mean() and var()
return the outcome shape; cov() returns a covariance matrix for vector outcomes.
given() requires a scalar Boolean per world; e.g. (v > 0).map(jnp.all).

Limits: finite support, exponential enumeration (at most MAX_WORLDS).
Build expressions inside a function passed to jit/grad; return JAX arrays, not
RV objects. Structure/support sizes and times(n) must be static under jit.
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

import jax
import jax.numpy as jnp

MAX_WORLDS = 1_000_000


@dataclass(eq=False, frozen=True)
class _Source:
    values: object
    mass: object


def _merge(*groups):
    # Source equality is object identity, never overloaded RV equality.
    return tuple(dict.fromkeys(s for group in groups for s in group))


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

    def map(self, f):
        """Apply f to each single outcome using vmap; supports vector outputs."""
        count = math.prod(_shape(self._sources))
        result = jax.vmap(f)(self.values.reshape((count,) + self.event_shape))
        result = jnp.asarray(result)
        return RV(self._sources,
                  result.reshape(_shape(self._sources) + result.shape[1:]),
                  self.mask)

    def __getitem__(self, key):
        """Index outcome components, not random-source axes: g[0], g[:2], etc."""
        return self.map(lambda value: value[key])

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
                            'reduce vector events with .map(jnp.all/any).')
        sources = _merge(self._sources, event._sources)
        _shape(sources)
        mask = (_align(self.mask, self._sources, sources) &
                _align(event.mask & event.values, event._sources, sources))
        return RV(sources, _align(self.values, self._sources, sources),
                  mask, self._source)

    def new(self):
        """Relabel all source axes to make an independent copy.

        Outcome and mask arrays are reused, not recomputed. Parameter values
        remain shared for autodiff; the random sources become independent.
        """
        replacements = {s: _Source(s.values, s.mass) for s in self._sources}
        return RV(tuple(replacements.values()), self.values, self.mask,
                  replacements.get(self._source))

    def times(self, n):
        """Sum n independent copies; exponential enumeration, no convolution.
        The result has fresh source axes, independent of this RV's axes.
        """
        if not isinstance(n, int) or n < 0:
            raise ValueError('n must be a static nonnegative Python integer.')
        if math.prod(s.values.shape[0] for s in self._sources) ** n > MAX_WORLDS:
            raise ValueError(f'Joint table exceeds {MAX_WORLDS:,} worlds.')
        result = constant(jnp.zeros(self.event_shape, dtype=self.values.dtype))
        for _ in range(n):
            result = result + self.new()
        return result

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

    def cov(self):
        """Covariance matrix for 1D vector outcomes; scalar variance for scalars."""
        if not self.event_shape:
            return self.var()
        if len(self.event_shape) != 1:
            raise ValueError('cov() requires vector outcomes; use .map(jnp.ravel).')
        values, mass = self.table()
        centered = values - jnp.sum(mass[:, None] * values, axis=0)
        return jnp.einsum('n,ni,nj->ij', mass, centered, centered)

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


def pmf(values, weights):
    """Create a fresh source from (K, *event_shape) values and (K,) weights.
    Duplicated values are allowed and remain separate latent outcomes.
    """
    values, weights = jnp.asarray(values), jnp.asarray(weights)
    if values.ndim < 1 or weights.ndim != 1 or values.shape[0] != weights.shape[0]:
        raise ValueError('Expected values shaped (K, *event_shape) and K weights.')
    if values.size == 0:
        raise ValueError('Empty support is not supported.')
    source = _Source(values, _weights(weights))
    return RV((source,), values, source=source)


def vals(*values):
    """Uniform distribution over the supplied outcomes."""
    return pmf(values, jnp.ones(len(values)))


def from_logits(values, logits):
    return pmf(values, jax.nn.softmax(jnp.asarray(logits)))


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
    RV.__neg__ = lambda self: self.map(operator.neg)
    RV.__abs__ = lambda self: self.map(operator.abs)
    RV.__invert__ = lambda self: self.map(operator.invert)


_install_operators()


def demo():
    x = vals(-2, -1, 0, 1, 2)
    average = (x ** 2).times(3) / 3
    print('Average: mean, variance =', average.mean(), average.var())
    print('Conditional square mean =', (x ** 2).given(x > 0).mean())
    a, b = vals(0, 1), pmf([0, 1], [0.2, 0.8])
    print('P(X | X+Y=1) =', a.posterior_via_grad(a + b == 1)[1])

    def objective(logits):
        z = from_logits([-2., -1., 0., 1., 2.], logits)
        return (z ** 2).times(3).var() / 9

    print('Gradient of average variance =',
          jax.jit(jax.grad(objective))(jnp.zeros(5)))

    theta = jnp.array([jnp.log(3.), 0.])
    p = jax.nn.softmax(theta)
    action = pmf([0, 1], p)
    reward = action  # reward 0 for action 0, reward 1 for action 1
    def log_prob(theta, a):
        return jax.nn.log_softmax(theta)[a]

    score = action.map(lambda a: jax.grad(log_prob)(theta, a))
    g = reward * score
    print('REINFORCE vector outcomes and masses =', g.table())
    print('Expected gradient =', g.mean())
    print('Gradient covariance =', g.cov())
    print('10-sample gradient covariance =', (g.times(10)/10).cov())


def self_test():
    import unittest
    import numpy as np

    class Checks(unittest.TestCase):
        def test_vector_reinforce(self):
            def gradient_rv(theta):
                p = jax.nn.softmax(theta)
                a = pmf([0, 1], p)
                score = a.map(lambda i: jax.nn.one_hot(i, 2) - p)
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
            np.testing.assert_allclose(g.cov(), cov, atol=1e-6)
            np.testing.assert_allclose((g.times(10)/10).cov(), cov/10, atol=1e-6)
            compiled = jax.jit(lambda t: gradient_rv(t).mean())
            np.testing.assert_allclose(compiled(theta), expected, atol=1e-6)
            # Differentiating the expectation again recovers the Hessian.
            np.testing.assert_allclose(jax.jacrev(compiled)(theta),
                jax.hessian(lambda t: jax.nn.softmax(t)[1])(theta), atol=1e-6)

        def test_vector_axes_and_conditioning(self):
            x, y = vals(0., 1.), vals(1., 2., 3.)
            u = x.map(lambda a: jnp.array([a, 2*a]))
            v = y.map(lambda b: jnp.array([b, -b]))
            np.testing.assert_allclose(((u+v)-(v+u)).var(), [0., 0.])
            np.testing.assert_allclose((u*y).mean(), [1., 2.])
            np.testing.assert_allclose(u.given(x+y == 2).mean(), [.5, 1.])
            np.testing.assert_allclose(u.given(x == 1).mean(), [1., 2.])
            np.testing.assert_allclose(u[1].mean(), 1.)
            np.testing.assert_allclose((u-u).cov(), np.zeros((2, 2)))
            np.testing.assert_allclose((u-u.new()).cov(), 2*u.cov())
            np.testing.assert_allclose(u.times(0).mean(), [0., 0.])
            np.testing.assert_allclose((u > 0).map(jnp.all).prob(), .5)
            with self.assertRaises(TypeError):
                u.given(u > 0)

        def test_vector_sources_and_tensor_broadcasting(self):
            v = pmf([[1., 2.], [3., 4.]], [.25, .75])
            np.testing.assert_allclose(v.mean(), [2.5, 3.5])
            np.testing.assert_allclose(v.cov(), np.full((2, 2), .75))
            np.testing.assert_allclose(v.posterior_via_grad(v[0] == 3)[1], [0, 1])
            # Outcome broadcasting is independent of the support's size.
            matrix = constant(jnp.array([[1.], [2.], [3.]])) * v
            self.assertEqual(matrix.event_shape, (3, 2))
            np.testing.assert_allclose(matrix.mean(),
                np.array([[1.], [2.], [3.]]) * np.array([2.5, 3.5]))
            np.testing.assert_allclose(v.map(lambda z: 7.).mean(), 7.)

        def test_independence_and_reuse(self):
            x = vals(0, 1)
            self.assertAlmostEqual(float((x + x).var()), 1.)
            self.assertAlmostEqual(float((x + x.new()).var()), .5)
            self.assertEqual(float((x - x).var()), 0.)
            # Operands can carry the same axes in opposite orders and sizes.
            y = vals(1, 2, 3)
            self.assertEqual(float(((x+y)-(y+x)).var()), 0.)
            self.assertAlmostEqual(float(y.given(x+y == 2).mean()), 1.5)

        def test_nonlinear_and_iid_average(self):
            x = vals(-2, -1, 0, 1, 2)
            m = (x ** 2).times(3) / 3
            self.assertAlmostEqual(float(m.mean()), 2., places=5)
            self.assertAlmostEqual(float(m.var()), 2.8 / 3, places=5)
            self.assertAlmostEqual(float((1 / vals(1, 2)).mean()), .75)

        def test_conditioning_and_clone(self):
            x, y = vals(0, 1), pmf([0, 1], [.2, .8])
            event = x + y == 1
            self.assertAlmostEqual(float(event.prob()), .5)
            self.assertAlmostEqual(float(x.given(event).mean()), .2)
            self.assertAlmostEqual(float((x+y).given(event).var()), 0.)
            c = x.given(event)
            self.assertAlmostEqual(float((c-c.new()).var()), .32, places=6)
            self.assertTrue(bool(jnp.isnan(x.given(x > 2).mean())))
            with self.assertRaises(TypeError):
                bool(x)

        def test_reverse_inference_and_jit(self):
            def posterior(p, q):
                x, y = pmf([0, 1], p), pmf([0, 1], q)
                return x.posterior_via_grad(x + y == 1)[1]
            p, q = jnp.array([.5, .5]), jnp.array([.2, .8])
            np.testing.assert_allclose(jax.jit(posterior)(p, q), [.8, .2])
            x = vals(0, 1)
            np.testing.assert_allclose(x.posterior_via_grad(x+x == 2)[1], [0, 1])

        def test_gradients(self):
            def objective(logits):
                x = from_logits([-2., -1., 0., 1., 2.], logits)
                return (x**2).times(3).var() / 9
            # d Var(Y)/d logit_i = p_i ((y_i-mu)^2 - Var(Y)).
            expected = np.array([1.2, -1.8, 1.2, -1.8, 1.2]) / 15
            np.testing.assert_allclose(jax.jit(jax.grad(objective))(jnp.zeros(5)),
                                       expected, atol=1e-6)
            # Moving support values also retain their pathwise derivatives.
            def moving(t):
                return (vals(-1., 1.).map(lambda v: t*v)).var()
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
                vals()
            with self.assertRaises(TypeError):
                vals(0, 1).given(1).mean()
            with self.assertRaises(ValueError):
                vals(0, 1).times(21).mean()
            self.assertTrue(bool(jnp.isnan(pmf([0, 1], [-1., 2.]).mean())))

    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    import sys
    self_test() if '--test' in sys.argv else demo()
