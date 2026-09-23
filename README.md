# RL puzzles

`jaxlea` is a small Lea-style library for exact enumeration of finite random
variables, built from eager JAX arrays. It supports nonlinear transformations,
shared randomness, independent copies, conditioning, vector-valued outcomes, and
autodiff. It does not draw random samples or maintain a separate expression graph.

## Install and run

```bash
python -m pip install -e .
python jaxlea.py
python jaxlea.py --test
```

The last command runs the ten embedded tests. The prototype has been checked with
Python 3.12 and JAX 0.11.2 on CPU. JAX's default precision is used; for higher
precision, enable `jax_enable_x64` before constructing arrays.

## Random variables and independent copies

```python
import jaxlea as jl

x = jl.vals(0., 1.)

(x + x).var()        # 1.0: reuse the same coin
(x + x.new()).var()  # 0.5: two independent coins

mc = x.times(10) / 10
mc.mean()            # approximately 0.5
mc.var()             # approximately 0.025 = Var(X) / 10
```

`times(n)` constructs the distribution of the sum of `n` independent copies. It
does not execute a Monte Carlo simulation. Copies share parameter values for
autodiff while having independent random-source identities.

`pmf(values, weights)` accepts nonnegative, unnormalized weights;
`from_logits(values, logits)` uses softmax probabilities. `map(f)` applies a
JAX-compatible function to each outcome using `jax.vmap`.

## Conditioning and stratification

```python
x = jl.pmf([0., 1.], [0.8, 0.2])
y = jl.vals(0., 1.)
f = 10 + 0.9*x + 0.1*y

# Ten copies contain twenty source bits, including the conditioned bits.
jl.MAX_WORLDS = 2**20

e0, e1 = x == 0, x == 1
mean0 = f.given(e0).times(5) / 5
mean1 = f.given(e1).times(5) / 5
stratified = e0.prob()*mean0 + e1.prob()*mean1

stratified.mean()       # approximately 10.23
stratified.var()        # approximately 0.00034
(f.times(10)/10).var()  # approximately 0.01321
```

`given(event)` masks worlds and normalizes within the stratum. The original
stratum weight must be supplied separately, as above. Multiple conditions and
conditions on operands of an expression are conjoined. This represents the
accepted distribution of rejection sampling, not the number of retries.

## A vector-valued REINFORCE estimator

```python
import jax
import jax.numpy as jnp

def log_prob(theta, action):
    return jax.nn.log_softmax(theta)[action]

def gradient_distribution(theta):
    action = jl.from_logits([0, 1], theta)
    reward = action  # rewards 0 and 1
    score = action.map(lambda a: jax.grad(log_prob)(theta, a))
    return reward * score

theta = jnp.array([jnp.log(3.), 0.])
g = gradient_distribution(theta)

g.mean()  # approximately [-0.1875, 0.1875]
g.var()   # componentwise variance: [0.10546875, 0.10546875]
g.cov()   # [[0.10546875, -0.10546875], [-0.10546875, 0.10546875]]

(g.times(10)/10).cov()  # g.cov() / 10

# Build RVs inside transformed functions and return ordinary JAX arrays.
expected_gradient = jax.jit(lambda t: gradient_distribution(t).mean())
expected_gradient(theta)
```

For rewards independent of the parameters, the mean of this vector equals the
gradient of expected reward. `g[0]` selects an outcome component. `map` can return
scalars, vectors, or tensors; `mean` and `var` preserve the outcome shape.
`cov` takes vector outcomes (flatten tensors explicitly with `map(jnp.ravel)`).

## Posterior marginals through autodiff

```python
x = jl.vals(0, 1)
y = jl.pmf([0, 1], [0.2, 0.8])
support, posterior = x.posterior_via_grad(x + y == 1)
# support: [0, 1], posterior: [0.8, 0.2]
```

For a primitive source with mass vector `p`, this differentiates the
unnormalized evidence probability `Z` with respect to independent mass
coordinates, returning `p * grad(Z) / Z`. It is not differentiation with respect
to softmax logits. Other source weights are held fixed for this partial
derivative, even when their values share model parameters.

## Representation and limits

- Each independent source has a named leading tensor axis. Arithmetic aligns
  source axes and computes outcome arrays immediately. Trailing axes describe
  the scalar/vector/tensor outcome. Probabilities remain factored by source until
  a query forms the joint masses.
- `table()` returns arrays shaped `(worlds, *event_shape)` and `(worlds,)`.
  Duplicate outcomes remain separate so that dependencies and derivatives are
  preserved. There is no automatic coalescing or convolution optimization.
- Enumeration grows exponentially: ten independent coins have 1,024 worlds;
  ten pairs of coins have 1,048,576. The default cap is 1,000,000 worlds, and is
  checked before joint-array allocation. Vector outcomes consume additional
  memory per world. Conditioning masks worlds but does not remove their axes.
- `jit` requires static structure, support sizes, and repetition counts. RV
  objects themselves are not registered as JAX pytrees; return numerical queries
  from transformed functions.
- Evidence must be a scalar Boolean per world. Reduce vector predicates with
  `.map(jnp.all)` or `.map(jnp.any)` before conditioning.
- Invalid weights or zero-probability evidence produce NaNs. Functions must be
  defined on every represented outcome: conditioning does not make division by
  zero in a discarded world safe for differentiation.
- Hard comparisons do not provide meaningful gradients across moving evidence
  boundaries. Smooth outcome functions and probability weights are
  differentiable. Exact enumeration still has floating-point rounding error.

This is an experimental teaching/research prototype, not the complete Lea API.
