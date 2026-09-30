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
# # Reinforce without Randomness
#
# Much has been written about how remarkable it is that 
# language model are pretrained to "predict the next word". 
# It is underappreciated, though, that RL post-training 
# is even simpler at heart. The model effectively guesses a 
# solution and is updated based on how well it does.
# Mathematically, the REINFORCE objective really is just, 
#

# update = sum ( guess reward * guess direction )
 
#
# Even though this is objective is intuitive, 
# the RL training process is awash in complexity.
# Read any tutorial and you will get an alphabet soup of 
# different explanations and tweaks to make it work in practice. Unfortunately like with 
# many complex systems, 5% of these methods are useful, but 
# smart (and loud) people disagree on which ones. 
# 
# While smartness eludes me, I am quite loud. My hunch is that 
# reinforce is hard because, unlike pretraining, it involves randomness,
# and randomness is confusing. There is a programmer mentality to just 
# code your way to understanding, but that becomes hard when things stop 
# being deterministic.  
# 
# So in this blog, we will do just that. We are going to code our way to 
# a baby version of reinforce without using any randomness. This is a 
# bait and switch as we will not be building cool things 
# like coding agents. Instead the focus is entirely on really boring things like
# elementary variance reduction techniques. The blog assumes no knowledge of RL or 
# math. The text and comments are written by a person, the code is written by AI. 

# %% tags=["hide"]
import jax
jax.config.update("jax_enable_x64", True)

from checks import check, f, linear_f, quadratic_f, fxy, gxy
import numpy as np
from viz import variance_3d, covariance_3d, coin_variance_animation
from IPython.display import display

# %% tags=["hide"]
from dist_types import Var, Joint
from intro_answers import (
    square, expect, six_from_eight, variance, indep, shared, marginal, covar,
    transpose, add, sub, mul, div, monte_carlo, four_sides, six_sides,
    two_triangles, triangular, circle, single_sample, ten_samples, linear_control,
    independent_two_variables,
    quadratic_control, marginal_control, two_variables, additive, weighted_die, ab_sampling, ab_test, kl_estimate,
    kl_k3, kl_topk, markov_chain, markov_unigram, group_rewards, group_variance, reinforce, reinforce_loo,
    fit_cuped, fit_baseline, estimated_baseline, topk, propagate,
    binary_policy, score, gradient_pair, loo_control,
)
from viz import histogram, variance_reduction_3d, variance_sum_3d, joint_top_view, show_control

# %% [markdown]
# ## Section 1 · Discrete Random Variables
#


# We are going to start from scratch by implementing a mini-language for 
# working with random variables. Our language will allow use to apply 
# some basic operations on these variables and propagate key properties. 
# There is, obviously, some confusion in the terminology, as these are random 
# in the statistical sense, but will not require drawing pseudorandom values in 
# the CS sense. 

#
# The main object in the language is a finite-sized, discrete random variable. 
# It is represented by aligned arrays of values and probabilities. 
# We visualize it as a histogram.

# $$ $$ 
# %%
eight = Var(np.arange(1, 9), 
            np.array([1, 3, 1, 3, 1, 3, 1, 3])/16)

histogram(eight)

# %% [markdown]

# We can transform random variables to create new ones. 
# Mathematically, this is written, 

# $$ $$ 

# However, to make it more clear what is going on we write it 
# as an explicit map over values. 

# We also define a filter over values that renomalizes the distribution. 

# %%

histogram(eight.op(lambda a: a * 2))
histogram(eight.cond(lambda a: a < 10))


# %% [markdown]
# 
# A key operation will be taking an expectation. 
# This is represented as the red line in the histogram
# and computed as 
#
# $$
# \mathbb E[X]=\sum_i p_i x_i
# $$

# %%
assert expect(eight) == 4.75

# %% [markdown]
# ## Exercise · square
#
# Raise each value to the power b, from 0 through 2.
#
# $$
# Y=X^b,\qquad 0\le b\le2
# $$

# %%
from plotly_viz import square_slider
display(square_slider(eight))


# %% [markdown]
# 
# The main reason for these definitions is to study variance 
# which will be the quantity we eventually aim to minimize. 
#
# $$
# \operatorname{Var}(X)=\mathbb E[(X-\mathbb E[X])^2]
# $$
#

# The visualization for variance will be a weigted sum of squares. 
# The base of each box with side length distance from the mean. The height 
# is the same height as the histogram. If it is helpful, you can think of a random 
# variable as expectation + noise. The variance is a way of quantifying the noise. 

# %%
assert np.isclose(variance(eight), 5.1875)
from plotly_viz import scaled_variance_slider
display(scaled_variance_slider(eight))

# %% [markdown]
# ## Section 2 · Joint variables
#
# 

# %%
coins = Joint([0, 1], [0, 1], [[1/8, 3/8], [1/8, 3/8]])
histogram(coins)
transformed = coins.op(lambda a: 2*a, lambda b: b + 1)
histogram(transformed)
histogram(transformed.add())

# %% [markdown]
# ## Exercise · indep
#
# Construct the joint distribution of two independent draws.
#
# $$
# P_{ij}=p_iq_j
# $$

# %%
coin = Var([0, 1], [.5, .5])
biased = Var([0, 1], [.25, .75])
np.testing.assert_allclose(indep(coin, biased).probs, coins.probs)

# %% [markdown]
# ## Exercise · shared
#
# Use the same draw in both coordinates.
#
# $$
# P_{ij}=p_i\,\mathbf 1\{i=j\}
# $$

# %%
histogram(shared(coin))

# %% [markdown]
# ## Exercise · marginal
#
# Recover the distribution of the first variable by summing over the second.
# `transpose(j)` swaps the coordinates, so `marginal(transpose(j))`
# recovers the second variable.
#
# $$
# p_i=\sum_jP_{ij}
# $$

# %%
np.testing.assert_allclose(marginal(coins).probs, coin.probs)
np.testing.assert_allclose(marginal(transpose(coins)).probs, biased.probs)

# %% [markdown]
# ## Exercise · covar
#
# Compute covariance from the joint distribution.
#
# $$
# \operatorname{Cov}(X,Y)=\sum_{i,j}P_{ij}
# (x_i-\mathbb E[X])(y_j-\mathbb E[Y])
# $$

# %%
assert np.isclose(covar(coins), 0)
assert covar(shared(coin)) == .25
covariance_3d(shared(coin))

# %% [markdown]
# ## Section 3 · Elementary Sampling
#
# The coin argument in these exercises takes values 0 and 1 with equal probability.

# %% [markdown]
# ## Four sides
#
# Given the fair coin, construct a fair four-sided die.
#
# $$
# B_1,B_2\overset{\mathrm{iid}}{\sim}\operatorname{Bernoulli}(1/2),
# \qquad X=1+2B_1+B_2
# $$

# %%
die4 = four_sides(coin)
check("four", die4)
variance_3d(die4)

# %% [markdown]
# ## Six sides
#
# Use one more independent flip and rejection to construct a fair six-sided die.
#
# $$
# U\sim\operatorname{Uniform}\{1,\ldots,8\},\qquad X\sim(U\mid U\le6)
# $$

# %%
die = six_sides(coin)
check("six", die)

# %% [markdown]
# ## Triangular distribution
#
# Draw X uniformly from 0 through 10 and accept it using an independent U.
# `Joint.cond(predicate)` conditions on a predicate of both coordinates.
# A continuous uniform U would work; the six-point grid below gives exactly
# the same acceptance probabilities for these thresholds.
#
# $$
# \begin{aligned}
# U&\sim\operatorname{Uniform}\{1/6,2/6,\ldots,1\},\\
# \text{accept}&\iff U\le\frac{6-|X-5|}{6},\\
# \Pr(X=k\mid\text{accept})&=\frac{6-|k-5|}{36},\qquad k=0,\ldots,10.
# \end{aligned}
# $$

# %%
candidate = Var(np.arange(11), np.ones(11)/11)
uniform = Var(np.arange(1, 7)/6, np.ones(6)/6)
triangle = triangular(candidate, uniform)
check("triangular", triangle)
variance_3d(triangle)

# %% [markdown]
# ## Exercise · circle
#
# Draw two independent coordinates uniformly from 0 through 10. Reject points
# farther than radius 5 from (5,5). This gives a uniform disk on our discrete
# grid, including its boundary.
#
# $$
# X,Y\overset{\mathrm{iid}}{\sim}\operatorname{Uniform}\{0,\ldots,10\},\qquad
# (X,Y)\mid (X-5)^2+(Y-5)^2\le r^2,\quad r=5.
# $$

# %%
coordinate = Var(np.arange(11), np.ones(11)/11)
disk = circle(coordinate, coordinate, radius=5)
histogram(disk)

# %% tags=["hide"]
accepted = (np.arange(11)[:, None]-5)**2 + (np.arange(11)[None, :]-5)**2 <= 25
np.testing.assert_allclose(disk.probs, accepted/accepted.sum())

# %% [markdown]
# ## Weighted dice from fair coins
#
# $$
# w=(1,2,3,4,5,6),\quad
# \Pr(X=k)=\frac{w_k}{\sum_j w_j}=\frac{k}{21}
# $$

# %%
check("weighted", weighted_die([1, 2, 3, 4, 5, 6]), variance_diagram=True)

# %% [markdown]
# ## Section 4 · Monte Carlo
#
# $$
# \mu=\mathbb E[f(X)],\qquad X_1,\ldots,X_n\overset{\mathrm{iid}}{\sim}X
# $$

# %% [markdown]
# ## Two triangles
#
# Sample the triangular distribution twice independently and return the sum.
#
# $$
# X,Y\overset{\mathrm{iid}}{\sim}\operatorname{Triangular}\{0,\ldots,10\},
# \qquad S=X+Y
# $$

# %%
from plotly_viz import covariance_widget
display(covariance_widget(indep(triangle, triangle)))
triangle_sum = two_triangles(triangle)
np.testing.assert_allclose(triangle_sum.probs, np.convolve(triangle.probs, triangle.probs))
assert np.isclose(expect(triangle_sum), 2 * expect(triangle))
variance_3d(triangle_sum)

# %% [markdown]
# ## One sample
#
# $$
# \begin{aligned}
# \hat\mu_1&=f(X_1), & \mathbb E[\hat\mu_1]&=\mu,\\
# \operatorname{Var}(\hat\mu_1)&=\operatorname{Var}(f(X)).
# \end{aligned}
# $$

# %%
check("single", single_sample(die, f), variance_diagram=True)

# %% [markdown]
# ## Monte Carlo · ten samples
#
# $$
# \begin{aligned}
# \hat\mu_{10}&=\frac1{10}\sum_{i=1}^{10}f(X_i), & \mathbb E[\hat\mu_{10}]&=\mu,\\
# \operatorname{Var}(\hat\mu_{10})&=\frac{\operatorname{Var}(f(X))}{10}.
# \end{aligned}
# $$

# %%
check("ten", ten_samples(die, f))

# %% [markdown]
# ## Independent two-variable Monte Carlo
#
# Draw a fresh independent pair for each sample.
#
# $$
# \hat\mu=\frac15\sum_{i=1}^5 f(X_i,Y_i),\qquad
# (X_i,Y_i)\overset{\mathrm{iid}}{\sim}P_X\otimes P_Y
# $$

# %%
check("xy_mc", independent_two_variables(die, die, fxy))

# %% [markdown]
# ## Section 5 · Control Variates
#
# Subtract a zero-mean variable using the same draw.
# The expectation stays fixed; positive covariance can reduce the variance.
#
# $$
# \begin{aligned}
# \mathbb E[B]&=0,\qquad \mathbb E[A-B]=\mathbb E[A],\\
# \operatorname{Var}(A-B)&=\operatorname{Var}(A)+\operatorname{Var}(B)-2\operatorname{Cov}(A,B),\\
# X&\sim\operatorname{Uniform}\{-1,1\},\quad A=2X,\quad B=X,\quad 4+1-4=1.
# \end{aligned}
# $$

# %%
centered_coin = Var([-1, 1], [.5, .5])
decomposition = shared(centered_coin).op(lambda a: 2*a, lambda b: b)
variance_reduction_3d(decomposition)

# %% [markdown]
# ## A roughly linear control variate
#
# $$
# \text{Target: }\mathbb E_{X\sim\mathrm{Uniform}\{1,\ldots,6\}}[f(X)]
# $$
# $$
# \text{Control: }B=3(X-\mathbb E[X]).
# $$
#
# Drag the slider to add nonlinearity while keeping the mean and control fixed.
# The 3D panels use a fixed view.
#
# $$
# f_\lambda(x)=f(x)+\lambda\bigl[(x-\mathbb E[X])^2-\operatorname{Var}(X)\bigr].
# $$

# %%
pair: Joint = linear_control(die, linear_f, b=3)
from plotly_viz import linear_control_widget
display(linear_control_widget(die, linear_f, b=3, steps=5))

# %% tags=["hide"]
check("linear", monte_carlo(sub(pair), 5), plot=False)

# %% [markdown]
# ## Five samples, a roughly parabolic function
#
# $$
# \text{Target: }\mathbb E_{X\sim\mathrm{Uniform}\{1,\ldots,6\}}[f(X)]
# $$
# $$
# \text{Control: }B=h(X)-\mathbb E[h(X)],\qquad h(x)=x^2+2x.
# $$

# %%
pair: Joint = quadratic_control(die, quadratic_f, a=1, b=2)
show_control(pair, steps=5)
check("quadratic", monte_carlo(sub(pair), 5), plot=False)

# %% [markdown]
# ## Exercise · marginal_control
#
# Marginalize out X to get the mean of Y.
# $$
# \text{Target: }\mathbb E_{(X,Y)\sim\mathrm{measurements}}[X]
# $$
# $$
# \text{Control: }B=Y-\mathbb E_{Y\sim\mathrm{marginal}_Y}[Y].
# $$

# %%
measurements = Joint([0, 1, 3, 4], [0, 3],
                     [[.25, 0], [.25, 0], [0, .25], [0, .25]])
pair = marginal_control(measurements)
show_control(pair, steps=5)
adjusted = monte_carlo(pair.sub(), 5)
unadjusted = monte_carlo(marginal(measurements), 5)
assert np.isclose(expect(marginal(transpose(pair))), 0)
assert np.isclose(expect(adjusted), 2)
assert np.isclose(variance(unadjusted), .5)
assert np.isclose(variance(adjusted), .05)

# %% [markdown]
# ## Five samples, two variables
#
# Average out Y exactly; sample only X. Both dice are independent.
# $$
# \text{Target: }\mathbb E_{X,Y\sim\mathrm{Uniform}\{1,\ldots,6\}}[f(X,Y)]
# $$
# $$
# \text{Control: }B=f(X,Y)-\mathbb E_Y[f(X,Y)].
# $$

# %%
check("xy", two_variables(die, fxy))

# %% [markdown]
# ## Five samples, an additive function
#
# Compute the known f contribution and the average over Y exactly.
# $$
# \text{Target: }\mathbb E_{X,Y\sim\mathrm{Uniform}\{1,\ldots,6\}}[f(X)+g(X,Y)]
# $$
# $$
# \text{Control: }B=f(X)-\mathbb E_X[f(X)]+g(X,Y)-\mathbb E_Y[g(X,Y)].
# $$

# %%
check("additive", additive(die, f, gxy))

# %% [markdown]
# ## Section 6 · Example: A/B Tests
#
# Each arm pairs an outcome with a pre-treatment measurement. Estimate the
# difference in mean outcomes using independent samples in the two arms.
# The initial population supplies the pre-treatment measurement Z.

# %% [markdown]
# ## Exercise · ab_sampling
#
# Draw independent people for the treatment and control arms.
# $$
# \text{Target: }\mathbb E_{U,V\overset{\mathrm{iid}}{\sim}\mathrm{population}}
# [\mathrm{treatment}(U)-\mathrm{control}(V)].
# $$

# %%
population = Var([0, 1, 2, 3], [.25]*4)

def initial(person: float) -> float:
    return person // 2

def control(person: float) -> float:
    return 2*initial(person) + person % 2

def treatment(person: float) -> float:
    return control(person) + 1

raw_difference = ab_sampling(population, treatment, control, steps=5)
histogram(raw_difference)
assert np.isclose(expect(raw_difference), 1)
assert np.isclose(variance(raw_difference), .5)

# %% [markdown]
# ## Exercise · ab_test
#
# Use each person's initial measurement, with b=2.
# $$
# \text{Target: }\mathbb E_{U,V\overset{\mathrm{iid}}{\sim}\mathrm{population}}
# [\mathrm{treatment}(U)-\mathrm{control}(V)]
# $$
# $$
# \text{Control: }B=b\bigl(\mathrm{initial}(U)-\mathrm{initial}(V)\bigr).
# $$

# %%
adjusted_difference = ab_test(population, treatment, control, initial, b=2, steps=5)
histogram(adjusted_difference, without=raw_difference)
assert np.isclose(expect(adjusted_difference), 1)
assert np.isclose(variance(raw_difference), .5)
assert np.isclose(variance(adjusted_difference), .1)

# %% [markdown]
# ## Exercise · fit_cuped
#
# Fit b by regressing centered historical outcomes on centered initial measurements.
# Freeze it before drawing evaluation samples.
# $$
# \text{Target: }\mathbb E_{U,V\overset{\mathrm{iid}}{\sim}\mathrm{population}}
# [\mathrm{treatment}(U)-\mathrm{control}(V)]
# $$
# $$
# \text{Control: }B=\hat b\bigl(\mathrm{initial}(U)-\mathrm{initial}(V)\bigr).
# $$

# %%
rng = np.random.default_rng(7)
history = rng.choice(population.values, size=128, p=population.probs)
initial_history = np.array([initial(person) for person in history])
outcome_history = np.array([control(person) for person in history])
learned_b = fit_cuped(initial_history, outcome_history)
cuped_difference = ab_test(population, treatment, control, initial,
                          b=learned_b, steps=5)
histogram(cuped_difference, without=raw_difference,
          comparison_labels=("plain MC", "learned CUPED"))
assert np.isclose(expect(cuped_difference), 1)
assert variance(cuped_difference) < variance(raw_difference)

# %% [markdown]
# ## Section 7 · KL Approximations

# %% [markdown]
# ## k1 KL estimate
#
# $$
# X_i\overset{\mathrm{iid}}{\sim}p,\quad
# k_1(r)=-\log r,\quad r(x)=q(x)/p(x),\quad
# \hat D=\frac15\sum_{i=1}^5\log\frac{p(X_i)}{q(X_i)},\quad
# \mathbb E[\hat D]=D_{\mathrm{KL}}(p\Vert q)
# $$

# %% tags=["hide"]
p_distribution = weighted_die([1, 2, 3, 4, 5, 6])
q_distribution = six_sides(Var([0, 1], [.5, .5]))

def p(a: float) -> float:
    return float(p_distribution.probs[p_distribution.values == a].sum())

def q(a: float) -> float:
    return float(q_distribution.probs[q_distribution.values == a].sum())

x = p_distribution

# %%
k1_samples = monte_carlo(kl_estimate(x, p, q), 5)
check("kl", k1_samples)

# %% [markdown]
# ## k3 KL control variate
#
# $$
# \text{Target: }\mathbb E_{X\sim p}\!\left[\log\frac{p(X)}{q(X)}\right]
# $$
# $$
# \text{Control: }B=1-\frac{q(X)}{p(X)}.
# $$

# %%
pair: Joint = kl_k3(x, p, q)
show_control(pair, steps=5)
k3_samples = monte_carlo(sub(pair), 5)
check("k3", k3_samples, plot=False)

# %% [markdown]
# ## Exercise · topk
#
# Sum the most probable k outcomes exactly; sample the remaining contribution.
#
# $$
# \text{Target: }\mathbb E_{X\sim p}[f(X)],\qquad
# \hat\mu=\sum_{a\in T_k}p(a)f(a)+\mathbf1\{X\notin T_k\}f(X).
# $$

# %% [markdown]
# ## Unbiased top-k KL
#
# $$
# \begin{aligned}
# T_k&=\operatorname{TopK}_a p(a),\qquad X\sim p,\\
# \hat D_k&=\sum_{a\in T_k}p(a)\log\frac{p(a)}{q(a)}
# +\mathbf 1\{X\notin T_k\}\log\frac{p(X)}{q(X)},\\
# \mathbb E_p[\hat D_k]&=D_{\mathrm{KL}}(p\Vert q).
# \end{aligned}
# $$

# %%
topk_samples = monte_carlo(kl_topk(x, p, q, 3), 5)
check("topk", topk_samples, plot=False)
histogram(topk_samples, without=k1_samples, comparison_labels=("k1", "top-k"))

# %% [markdown]
# ## Section 8 · Markov Chains

# %% [markdown]
# ## Helper · propagate
#
# Take one step using the row-conditional probabilities of the transition joint.
# Return the full next-state distribution, preserving its variance.
# State values must match the transition rows in order; each row needs positive mass.
#
# $$
# p_{t+1}(y)=\sum_x p_t(x)\,T(y\mid x).
# $$

# %% [markdown]
# ## A two-state Markov chain
#
# Starting from the supplied state distribution, repeatedly call propagate with T.
#
# $$
# S_0=0,\quad P(S_{t+1}\mid S_t)=\begin{pmatrix}3/4&1/4\\1/4&3/4\end{pmatrix},\quad
# \hat\mu=S_3,\quad \mathbb E[\hat\mu]=\frac7{16}
# $$

# %%
T = Joint([0, 1], [0, 1], [[3/8, 1/8], [1/8, 3/8]])
initial_state = Var([0, 1], [1., 0.])
check("markov", markov_chain(initial_state, T), variance_diagram=True)

# %% [markdown]
# ## A unigram control variate
#
# Use the one-step distribution as a unigram approximation. Draw the final
# state and the unigram with the same uniform quantile; center the unigram.
# $$
# \text{Target: }\mathbb E_{\mathrm{chain}}[S_3]
# $$
# $$
# \text{Control: }B=U-\mathbb E[U],\qquad U\sim\mathrm{propagate}(\mathrm{initial\_state},T).
# $$

# %%
pair: Joint = markov_unigram(initial_state, T)
show_control(pair)
check("unigram", sub(pair), plot=False)

# %% [markdown]
# ## Section 9 · Group Variance
#
# Draw one output from a model and score it with two sub-rewards.
# Their sum depends on both marginal variances and their covariance.

# %% [markdown]
# ## Additive rewards from one model draw
#
# $$
# X\sim p,\quad (A,B)=(r_a(X),r_b(X)),\quad R=A+B,\qquad
# \mathbb E[R]=\mathbb E[A]+\mathbb E[B].
# $$

# %%
model = four_sides(coin)
reward_a = lambda x: float(x >= 3)
reward_b = lambda x: float(x >= 2)
rewards = group_rewards(model, reward_a, reward_b)
histogram(rewards)
covariance_3d(rewards)

# %% [markdown]
# ## Exercise · group_variance
#
# Compute the variance of the total reward from the joint sub-rewards.
#
# $$
# \operatorname{Var}(R)=\operatorname{Var}(A)+\operatorname{Var}(B)
# +2\operatorname{Cov}(A,B).
# $$

# %%
assert np.isclose(group_variance(rewards), variance(rewards.add()))
variance_sum_3d(rewards)

# %% [markdown]
# ## Positive and negative reward dependence
#
# Both examples have the same marginal rewards and mean total reward.
# Compare each against rewards from independent model draws.
#
# $$
# \operatorname{Var}(A)=\tfrac14,\quad \operatorname{Var}(B)=\tfrac3{16},\quad
# \operatorname{Cov}(A,B)=\pm\tfrac18,\quad
# \operatorname{Var}(A+B)=\tfrac7{16}\pm\tfrac14.
# $$

# %%
independent_rewards = indep(marginal(rewards), marginal(transpose(rewards)))
histogram(rewards.add(), without=independent_rewards.add(),
          comparison_labels=("independent", "positive"))
opposing_rewards = group_rewards(model, reward_a, lambda x: float(x <= 3))
covariance_3d(opposing_rewards)
variance_sum_3d(opposing_rewards)
histogram(opposing_rewards.add(), without=independent_rewards.add(),
          comparison_labels=("independent", "negative"))
assert np.isclose(group_variance(rewards), 11/16)
assert np.isclose(group_variance(independent_rewards), 7/16)
assert np.isclose(group_variance(opposing_rewards), 3/16)

# %% [markdown]
# ## Section 10 · REINFORCE

# %% [markdown]
# ## Helper · binary_policy
#
# Convert two logits to a binary reward distribution.

# %% [markdown]
# ## Helper · score
#
# For this binary example, the second logit score is R-p. The first is its negative.

# %% [markdown]
# ## Helper · gradient_pair
#
# Turn the scalar estimate into the two gradient components.

# %% [markdown]
# ## REINFORCE with a two-dimensional gradient
#
# $$
# \begin{aligned}
# a&\sim p_\theta=\operatorname{softmax}(\theta),\quad r(a)=a,\quad p=\Pr(a=1),\\
# \hat g&=a\nabla_\theta\log p_\theta(a)=a(a-p)(-1,1),\\
# \mathbb E[\hat g]&=\nabla_\theta\mathbb E_{p_\theta}[a].
# \end{aligned}
# $$

# %%
theta = np.array([np.log(3.), 0.])
check("reinforce", reinforce(theta))

# %% [markdown]
# ## Helper · loo_control
#
# For binary rewards, compute the leave-one-out control from their sample mean.
# This is the average of each reward's score times the mean of the other rewards (n >= 2).

# %% [markdown]
# ## REINFORCE with leave-one-out
#
# Each baseline is the mean of the other two rewards.
# $$
# \text{Target: }\mathbb E_{R\sim\mathrm{Bernoulli}(p)}[R(R-p)]
# $$
# $$
# \text{Control: }B=\frac13\sum_{i=1}^3\bar R_{-i}(R_i-p),\qquad
# \bar R_{-i}=\frac12\sum_{j\ne i}R_j.
# $$

# %%
pair: Joint = reinforce_loo(theta)
show_control(pair)
gradient = gradient_pair(sub(pair))
check("loo", gradient, plot=False)

# %% [markdown]
# ## Exercise · fit_baseline
#
# Fit a constant predictor of reward on a separate training batch.
#
# $$
# \hat v=\arg\min_v\sum_i(R_i-v)^2=\frac1m\sum_i R_i.
# $$

# %%
reward = Var([0, 1], [.75, .25])
training_rewards = np.random.default_rng(11).choice(
    reward.values, size=128, p=reward.probs)
baseline = fit_baseline(training_rewards)

# %% [markdown]
# ## Exercise · estimated_baseline
#
# Freeze the fitted reward mean before evaluating on fresh draws.
# $$
# \text{Target: }\mathbb E_{R\sim\mathrm{Bernoulli}(p)}[R(R-p)],\qquad p=\tfrac14
# $$
# $$
# \text{Control: }B=\hat v(R-p).
# $$

# %%
baseline_pair = estimated_baseline(reward, baseline)
show_control(baseline_pair, steps=5)
corrected = monte_carlo(baseline_pair.sub(), 5)
uncorrected = monte_carlo(marginal(baseline_pair), 5)
assert np.isclose(expect(marginal(transpose(baseline_pair))), 0)
assert np.isclose(expect(corrected), .1875)
assert variance(corrected) < variance(uncorrected)

# %% [markdown]
# ## Geometry

# %% [markdown]
# ## Geometry · Off-diagonal covariance
#
# $$
# X\sim\operatorname{Uniform}\{-1,1\},\qquad
# (A,B)=(X,-X),\qquad \operatorname{Cov}(A,B)=-1
# $$

# %%
coin = Var([-1, 1], [0.5, 0.5])
opposites: Joint = shared(coin).op(lambda a: a, lambda a: -a)
assert covar(opposites) == -1
histogram(opposites)
covariance_3d(opposites)

# %% [markdown]
# ## Slider · 1–20 coin flips
#
# $$
# X_i\overset{\mathrm{iid}}{\sim}\operatorname{Uniform}\{-1,1\},\quad
# \bar X_n=\frac1n\sum_{i=1}^n X_i,\quad
# \operatorname{Var}(\bar X_n)=\frac1n,\quad n=1,\ldots,20
# $$

# %%
from plotly_viz import coin_variance_slider
display(coin_variance_slider(20))
