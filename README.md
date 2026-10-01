# From Sampling to Reinforce

[Read the blog](https://srush.github.io/sampling-to-reinforce/).

An interactive introduction to finite random variables, Monte Carlo sampling,
and variance reduction, ending with KL estimators and Reinforce.

The article lives in `puzzle.py`, a percent-format Jupytext notebook.
Its examples compute exact distributions of estimators, so the plots show all
possible outcomes and their probabilities.

- `dist_types.py`: NumPy containers for a random variable (`Var`) and a joint
  distribution (`Joint`).
- `intro_answers.py`: distribution operations and helpers used by the article.
- `viz.py` and `plotly_viz.py`: static plots and interactive diagrams.
- `checks.py`: independent probability-distribution checks for the examples.
- `build_html.py` and `preview.py`: HTML export and live preview.

`shared(x)` pairs a draw with itself; `indep(x, y)` pairs independent draws.
`x.op(f)` transforms a variable, and `x.cond(predicate)` filters and renormalizes
it. `joint.op(f, g)` transforms both coordinates while preserving their coupling.
`joint.add()`, `joint.sub()`, and `joint.mul()` combine the paired values.
`monte_carlo(x, n)` computes the distribution of the average of independent draws.

The final example uses eight classes, a softmax temperature parameter, and a
constant reward baseline. `leave_one_out(temperature, n)` returns the mean reward
of `n-1` other sampled actions; the article computes the gradient itself.

## Run and preview

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[static-plots]'
.venv/bin/python preview.py
```

Open `http://127.0.0.1:8765`. Edits to the notebook or its Python helpers rebuild
the page and refresh the browser. If a build fails, the preview keeps the last
successful page and retries on the next save.

For a one-time export:

```bash
.venv/bin/python build_html.py
```

This writes `build/index.html`. Add `--profile` to list the slowest cells.
Use `preview.py --watch-only` to rebuild without starting a server.

To open the article as a notebook:

```bash
.venv/bin/jupytext --to notebook puzzle.py
```

## Checks and publishing

```bash
MPLBACKEND=Agg .venv/bin/python -m unittest test_intro.py test_preview.py
```

The tests cover the distribution operations, article estimators, plots, and
preview recovery. GitHub Actions runs the tests, builds the article, and
publishes `build/` to GitHub Pages on pushes to `main`.
