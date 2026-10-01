# RL Puzzles

All notebook exercises use two simple NumPy containers: `Var` holds
equal-length values/probability arrays; `Joint` holds two 1D supports and a 2D
probability table. They have no overloaded operators or random-source names.
The opening sections are Random Variables, Joint variables, and Elementary
Sampling, followed by Monte Carlo, Control Variates, and KL Approximations.
Operation implementations are hidden; exercise answers remain visible.
Sampling exercises take a fair 0/1 coin. The triangular rejection example uses
an independent six-point uniform threshold, exactly matching continuous-uniform
acceptance for the specified heights.
`Var` exposes `values` and `probs`; `Joint` exposes its probability table as `probs`.
`intro_answers.py` contains the typed,
filled-in function exercises. `shared(x)` makes a diagonal joint and `indep(x, y)`
makes an outer-product joint. `x.op(f)` transforms an `Var` into an `Var`;
`x.cond(predicate)` filters and renormalizes it.
`binop(f, joint)` collapses a `Joint` to a `Var`; `add`, `sub`, and `div` specialize it.
Use `joint.mul()` to multiply the paired values.
`div` requires nonzero denominator values.
`j.op(f, g)` transforms each coordinate and retains the joint,
merging duplicate rows and columns. Use `shared(x).op(F, G)` for
`[F(X), G(X)]` from the same draw. `transpose(j)` swaps coordinates and transposes
the probability table. `marginal(j)` keeps the first coordinate; use
`marginal(transpose(j))` for the second. `covar(shared(x))` is variance.
Plotting accepts these containers directly.
Control-variate functions return `Joint` of `(estimate, zero-mean control)`.
Use `sub(pair)` to correct the estimate, then `monte_carlo` when averaging is needed.
The final REINFORCE section uses eight classes with fixed logits `0,...,7`,
rewards equal to class indices, and one positive temperature parameter `T`.
The policy is `softmax(logits / T)` and the scalar score is `(E[A] - a) / T**2`.
The `temperature_*` helpers implement the policy, gradient, leave-one-out control,
and fitted baseline. Leave-one-out directly subtracts the mean of the other
rewards from each reward, multiplies by its score, and averages the results.
`monte_carlo(x, steps)` averages independent copies. `condition` filters and
renormalizes an `Var`. Earlier binary-policy helpers remain available for regression tests.
The older JAX implementation remains in `puzzles.py` and `distribution.py` for
reference and regression tests. Run `python -m unittest test_intro.py test_puzzles.py`.

Start with `puzzle.py`, a percent-format Jupytext notebook with twenty short
distribution and variance-reduction exercises. `intro_answers.py` contains the answers,
`checks.py` compares full probability distributions, and `viz.py` renders shared
notebook histograms and two-dimensional gradient plots. Every estimator is also
checked for unbiasedness. The final six exercises cover weighted dice from fair
coins, Monte Carlo KL, Markov chains and a coupled unigram control variate,
and temperature-parameter REINFORCE with leave-one-out and fitted baselines.
Previous experiments are preserved in `backup/`.

```bash
python -m pip install -e .
jupytext --to notebook puzzle.py
python puzzle.py
```

The examples use exact distributions of random estimators, not realized samples.
`d.mean([make_draw(i) for i in range(n)])` builds an average from explicitly
constructed draws. Reusing an RV shares its randomness. Small tables preserve
the full joint distribution; large independent averages use convolution with
equal outcomes coalesced. Compressed results keep their names and flip records,
but cannot be recombined with their original inputs. Traced outcomes are not
coalesced, so JIT/autodiff workloads must fit the joint-table budget.

To preview the illustrated HTML while editing the Python notebook:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python preview.py
```

Open `http://127.0.0.1:8765`. Edits to `puzzle.py` or its Python helpers
rebuild the page and refresh the browser automatically. Keep this command
running while we rearrange the notebook together. If you open `build/index.html`
directly, run `.venv/bin/python preview.py --watch-only` to rebuild on each save;
reload that file tab to see changes. For a one-time export, run
`.venv/bin/python build_html.py`; it writes `build/index.html`. Add `--profile`
to see the slowest notebook cells.
If a build fails, the watcher reports the error, keeps the last successful HTML,
and retries on the next save. It also keeps running if the initial build fails.
The HTML build uses Plotly. Matplotlib is only needed for the older static plot
functions and their tests; install it with `.venv/bin/python -m pip install -e '.[static-plots]'` if needed.

`distribution` is a small Lea-style library for exact enumeration of finite random
variables, built from eager JAX arrays. It supports nonlinear transformations,
shared randomness, independent copies, conditioning, vector-valued outcomes, and
autodiff. It does not draw random samples or maintain a separate expression graph.

## Install and run

```bash
python -m pip install -e .
python distribution.py
python distribution.py --test
python -m unittest test_puzzles.py
```

The last command runs the twelve embedded tests. The prototype has been checked with
Python 3.12 and JAX 0.11.2 on CPU. JAX's default precision is used; for higher
precision, enable `jax_enable_x64` before constructing arrays.

## Random variables and independent copies

`flip(name)` returns −1 or +1 with equal probability: mean 0, variance 1.
Use `(flip(name)+1)//2` when a 0/1 bit is needed. Names are labels, not a global cache.
Reuse the returned object to share randomness. Every RV exposes `.names`,
`.draws`, and `.flips`. Draw records contain a name, kind, unique ID, and optional
parent ID. Arithmetic, transformations, and conditioning preserve these records.
List comprehensions construct independent draws with explicit names. Their
records survive averaging, including compressed averages.
Named categorical sources remain available through `pmf(..., name=...)` and
`from_logits(..., name=...)`; these are not counted as fair flips.

```python
import distribution as jl

x = jl.flip("x")

(x + x).var()        # 4.0: reuse the same coin
(x + jl.flip("y")).var()  # 2.0: two independent coins

mc = jl.mean([jl.flip(f"x.{i}") for i in range(10)])
mc.mean()            # 0
mc.var()             # 0.1 = Var(X) / 10
```

`jl.mean(draws)` averages RVs; `rv.mean()` takes an exact expectation.

`pmf(values, weights)` accepts nonnegative, unnormalized weights;
`from_logits(values, logits)` uses softmax probabilities. `apply(f)` applies a
JAX-compatible function to each outcome using `jax.vmap`.

## Conditioning and stratification

```python
x = jl.pmf([0., 1.], [0.8, 0.2])
y = (jl.flip("y")+1)//2
f = 10 + 0.9*x + 0.1*y

e0, e1 = x == 0, x == 1
def conditional_draw(state, i):
    a = jl.pmf([0., 1.], [0.8, 0.2], name=f"state.{state}.{i}")
    b = (jl.flip(f"noise.{state}.{i}")+1)//2
    return (10 + 0.9*a + 0.1*b).given(a == state)

mean0 = jl.mean([conditional_draw(0, i) for i in range(5)])
mean1 = jl.mean([conditional_draw(1, i) for i in range(5)])
stratified = e0.prob()*mean0 + e1.prob()*mean1

stratified.mean()       # approximately 10.23
stratified.var()        # approximately 0.00034
```

`given(event)` masks worlds and normalizes within the stratum. The original
stratum weight must be supplied separately, as above. Multiple conditions and
conditions on operands of an expression are conjoined. This represents the
accepted distribution of rejection sampling, not the number of retries.

## Joint distributions and covariance

```python
x = jl.flip("x")
y = 2*((x+jl.flip("z")) > 0)-1
d_joint = jl.joint(x, y)  # preserve coupling
product = x @ y           # independent product of the marginals
(d_joint-d_joint.mean()).apply(lambda v: v[0]*v[1]).mean()  # 0.5
(product-product.mean()).apply(lambda v: v[0]*v[1]).mean()  # 0.0
```

`@` is this library's Python product notation, not Haskell syntax.
It is equivalent to `jl.independent(x, y)` and makes fresh copies of both
marginals, including when the inputs already share flips. `joint` currently
pairs scalar RVs; the result has two components. Compute covariance by centering
the joint and taking the expectation of the product of its components.

`viz.variance_3d(x)` draws squared deviations. `viz.covariance_3d(joint)` draws
rectangles running from the means to each raw outcome, with probability height.
Their signed volumes sum to covariance (purple negative, blue positive); red
lines mark the means. Diagrams use the
actual joint table, not a product of marginals unless explicitly requested.

## A vector-valued REINFORCE estimator

```python
import jax
import jax.numpy as jnp

def log_prob(theta, action):
    return jax.nn.log_softmax(theta)[action]

def gradient_distribution(theta):
    action = jl.from_logits([0, 1], theta)
    reward = action  # rewards 0 and 1
    score = action.apply(lambda a: jax.grad(log_prob)(theta, a))
    return reward * score

theta = jnp.array([jnp.log(3.), 0.])
g = gradient_distribution(theta)

g.mean()  # approximately [-0.1875, 0.1875]
g.var()   # componentwise variance: [0.10546875, 0.10546875]
(g-g.mean()).apply(lambda v: jnp.outer(v, v)).mean()  # covariance matrix

# Build RVs inside transformed functions and return ordinary JAX arrays.
expected_gradient = jax.jit(lambda t: gradient_distribution(t).mean())
expected_gradient(theta)
```

For rewards independent of the parameters, the mean of this vector equals the
gradient of expected reward. `g[0]` selects an outcome component. `apply` can return
scalars, vectors, or tensors; `mean` and `var` preserve the outcome shape.
An outer product gives the full covariance matrix without a special method.

## Posterior marginals through autodiff

```python
x = jl.flip("x")
y = jl.pmf([-1, 1], [0.2, 0.8])
support, posterior = x.posterior_via_grad(x + y == 0)
# support: [-1, 1], posterior: [0.8, 0.2]
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
  preserved outside explicit average compression.
- Arbitrary arithmetic still grows exponentially. `mean(draws)` compresses
  large independent averages when values are concrete. Traced values are not
  merged because their derivatives can differ. Tables have a 1,000,000-entry
  budget. Conditioning masks worlds but does not remove their axes.
- `jit` requires static structure, support sizes, and repetition counts. RV
  objects themselves are not registered as JAX pytrees; return numerical queries
  from transformed functions.
- Evidence must be a scalar Boolean per world. Reduce vector predicates with
  `.apply(jnp.all)` or `.apply(jnp.any)` before conditioning.
- Invalid weights or zero-probability evidence produce NaNs. Functions must be
  defined on every represented outcome: conditioning does not make division by
  zero in a discarded world safe for differentiation.
- Hard comparisons do not provide meaningful gradients across moving evidence
  boundaries. Smooth outcome functions and probability weights are
  differentiable. Exact enumeration still has floating-point rounding error.

This is an experimental teaching/research prototype, not the complete Lea API.
