# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Variance Reduction Puzzles
#
# Three tiny models, exactly two learnable parameters each:
# binary -> five ordered classes -> random walk.
#
# The goal is to improve a gradient ESTIMATOR, not implement an optimizer.
# Every model has fixed rewards r(z), probabilities q_theta(z), and a score
# s(z) = grad_theta log q_theta(z), a vector with shape (2,).
#
# We minimize J(theta) = -E[r(z)]. The exact gradient is
# g = -sum_z q_theta(z) r(z) s(z).
# Ordinary NumPy operations are allowed. The model and plotting code is in lib.py.
#
# Each preview shows the exact gradient and ONE estimated gradient as arrows.
# A second panel shows repeated estimates, centered on the exact gradient.
# Arrows show gradients; a gradient-descent update would point the opposite way.
# Variance means trace(Cov[g_hat]) = E[||g_hat - E[g_hat]||²].
# We compute bias and variance exactly, not just from the displayed scatter.

# %%
import numpy as np
from lib import make_model, compare_estimators, check_strategy

theta = np.array([0.4, -0.3])
samples_per_estimate = 16

# %% [markdown]
# ## Model 1 - binary, with two contexts
#
# First observe c ~ Uniform({0, 1}), whose distribution is fixed.
# Then z | c ~ Bernoulli(sigmoid(theta[c])).
# Each parameter controls a different coin. The observation model is
# x_c | z ~ Normal(z, sigma²), with fixed targets x_0=0.2 and x_1=0.8.
# Reward is the Gaussian density at x_c. Neither the targets nor sigma is learned.
#
# There are four joint outcomes (c,z). With only one context and fixed rewards,
# a single Bernoulli could not provide two independent policy parameters.
# Derandomization will enumerate z while still sampling the context.

# %%
binary = make_model("binary", theta)

# %% [markdown]
# ## Puzzle 1 - stratification
#
# Inverse-CDF sampling maps U ~ Uniform(0,1) to a discrete model outcome.
# Rather than draw n independent uniforms, split [0,1) into n equal intervals
# and draw ONE independent uniform inside each interval:
# U_i ~ Uniform(i/n, (i+1)/n).
#
# Return shape (n,). The harness maps uniforms to outcomes and averages their
# gradient contributions. Do not replace the random draws with midpoints:
# that would generally introduce bias. Equal averaging is correct here because
# each interval has probability mass 1/n.
# Outcome order affects variance; the models use their natural enumeration order.

# %%
def stratified_uniforms(n, rng):
    raise NotImplementedError


# check_strategy("stratification", stratified_uniforms)

# %% [markdown]
# ## Puzzle 2 - a simple control variate
#
# The score has expectation zero. Subtract a fixed reward baseline b:
# g_hat = -mean_i[(r_i - b) s_i].
#
# Inputs: rewards (n,), scores (n,2), and a scalar baseline.
# Return shape (2,). The previews use b=0.15, held fixed independently of the
# sampled outcomes. Try changing it: a poor baseline can INCREASE variance.
# This changes the gradient estimator, not the underlying reward or objective.
# No gradient is taken through b. A pilot batch or previous batches could
# estimate b independently, but fitting it on these same samples can cause bias.

# %%
def control_variate_gradient(rewards, scores, baseline):
    raise NotImplementedError


# check_strategy("control_variate", control_variate_gradient)

# %% [markdown]
# ## Puzzle 3 - derandomize one latent choice
#
# Sample a group of outcomes, then enumerate all outcomes INSIDE that group.
# The harness provides conditional probabilities within the chosen group,
# rewards (m,), and scores (m,2). Return their exact weighted loss gradient.
#
# This is conditional averaging (Rao-Blackwellization). Use the ORIGINAL
# joint-distribution scores supplied by the harness, not a new conditional score.
# It remains unbiased and cannot increase variance for the same number of
# independent group draws. It can require more reward evaluations per draw.
#
# Binary: sample the context, enumerate the coin.
# Ordered classes: sample one of {0,1}, {2,3}, {4}, enumerate within it.
# Walk: sample a prefix of T-1 steps, enumerate the final left/right step.

# %%
def conditional_gradient(probabilities, rewards, scores):
    raise NotImplementedError


# check_strategy("derandomization", conditional_gradient)

# %% [markdown]
# ## Compare the strategies
#
# The reference preview runs before the exercises are solved. To test your
# solutions visually, pass stratify=stratified_uniforms,
# control=control_variate_gradient, conditional=conditional_gradient.
# Checks above compare your outputs with the estimator contract. The exact
# metrics shown in plots always describe the reference mathematical estimators;
# empirical metrics also measure the supplied implementations.
#
# Each row uses n draws. Derandomization's label reports expected reward
# evaluations per estimate, so its extra work is visible.
# All plots within one comparison share scales; zoom by changing theta/model.

# %%
binary_figure, binary_report = compare_estimators(
    binary, n=samples_per_estimate, baseline=0.15
)

# %% [markdown]
# ## Model 2 - five ordered classes
#
# a in {0,1,2,3,4}, u_a = (a-2)/2.
# q_theta(a) = softmax(theta[0]*u_a - theta[1]*u_a²).
# The two parameters control a linear and a quadratic preference over position.
# They are not five independent class logits.
#
# Class a selects Normal(a, 0.65²); reward is its density at fixed target x=3.2.
# Reuse the SAME three estimator functions.

# %%
ordered = make_model("ordered", theta)
ordered_figure, ordered_report = compare_estimators(
    ordered, n=samples_per_estimate, baseline=0.15
)

# %% [markdown]
# ## Model 3 - random binary walk
#
# Start at position zero and take T=4 binary steps.
# On steps 1 and 3, move right with sigmoid(theta[0]);
# on steps 2 and 4, move right with sigmoid(theta[1]).
# Left moves by -1; right by +1. Exactly two parameters control the whole walk.
#
# The endpoint selects Normal(s_T, 1²), with terminal reward its density at x=2.5.
# Score coordinate j sums z_t-p_t over steps controlled by theta[j].
# We enumerate all 16 paths for the exact reference gradient. Paths with the
# same endpoint can have different scores, so endpoint alone is not sufficient.
#
# Derandomization averages the two possible final steps for each sampled prefix.
# It uses terminal rewards, not a learned value function or a baseline.

# %%
walk = make_model("walk", theta)
walk_figure, walk_report = compare_estimators(
    walk, n=samples_per_estimate, baseline=0.15
)
