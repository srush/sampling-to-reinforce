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
# # From Sampling to Reinforce
# 
# With the growing importance of LLM post-training, 
# mastery of [Reinforce](https://en.wikipedia.org/wiki/Policy_gradient_method) has become arguably as useful 
# as attention. The algorithm is central to producing models 
# that reason, act, and discover. 
# 
# 
# Despite being mathematically concise, 
# reinforce is tricky to master. It is not uncommon 
# to read papers in the area that are confused or buggy. 
# My personal theory for this is that 
# random sampling can actually hide a bunch of issues.
# It make it hard to just "code your way" through problems.
#  
# The goal of this blog is to build up to reinforce 
# from scratch. It takes the non-standard approach
# of doing it without any random numbers.  
# Obviously, this cannot scale, but it will let use build 
# intuition for boring things like
# elementary variance reduction techniques. The blog assumes no knowledge of RL or 
# math. The text and comments are written by a person, the code is written by AI. 

# %% tags=["hide"]
import jax
jax.config.update("jax_enable_x64", True)

from checks import check, f, linear_f, quadratic_f, fxy
import numpy as np
from viz import variance_3d, covariance_3d, coin_variance_animation
from IPython.display import display

# %% tags=["hide"]
from dist_types import Var, Joint
from intro_answers import (
    control_variate, monte_carlo_with_control, iid_statistic, post_stratify, stratify,
    expect, six_from_eight, uniform, variance, indep, shared, marginal, covar,
    transpose, add, sub, mul, div, four_sides, six_sides,
    two_triangles, triangular, circle, linear_control,
    independent_two_variables,
    quadratic_control, two_variables, weighted_die, ab_sampling, ab_test,
    kl_topk, markov_chain, markov_unigram, group_rewards, group_variance, reinforce, reinforce_loo,
    fit_cuped, cuped, fit_baseline, estimated_baseline, kl, topk, propagate,
    binary_policy, score, gradient_pair, loo_control,
)
from viz import density, histogram, histogram_row, variance_reduction_3d, variance_sum_3d, joint_top_view, show_control
from plotly_viz import (
    scaled_variance_slider, covariance_interpolation_slider,
    coin_variance_slider, linear_control_widget, sum_variance_slider,
    monte_carlo_samples_slider,
)

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

# %%
eight = Var(np.arange(1, 9), 
            np.array([1, 3, 1, 3, 1, 3, 1, 3])/16)

histogram(eight)

# %% [markdown]

# We can transform random variables to create new ones. 
# Mathematically, this is written, 

# $$ 
# Y=2X.
# $$


# However, to make it more clear what is going on we write it 
# as an explicit map over values. 

# We also define a filter over values that renomalizes the distribution. 

# %%

histogram_row(eight.op(lambda a: a * 2), eight.cond(lambda a: a < 10),
              labels=("Double the value", "Keep values below 10"))


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
def expect(x: Var) -> float:
    return float(x.values @ x.probs)

print(f"E[X] = {expect(eight):.2f}")

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
# is the same height as the histogram.
#
# $$
# \operatorname{Var}(bX)=b^2\operatorname{Var}(X),\qquad
# \operatorname{Var}(2X)=4\operatorname{Var}(X).
# $$
#
# If it is helpful, you can think of a random
# variable as expectation + noise. The variance is a way of quantifying the noise. 

# %%
def variance(x: Var) -> float:
    mean = expect(x)
    return expect(x.op(lambda value: (value - mean)**2))

print(f"Var(X) = {variance(eight):.4f}")
display(scaled_variance_slider(eight))

# %% [markdown]
# ## Section 2 · Joint random variables
#
# Next we will consider the relationship between random variables. 
# We'll be explicit about this, because it will get
# a bit tricky. Our object here will keep track of the probability of 
# every shared state in a table. 



# %%
joint_example = Joint([0, 1, 2], [0, 1],
                      [[1/12, 1/12], [1/6, 1/6], [1/4, 1/4]])
histogram(joint_example)

# %% [markdown]
# Just as with random variables we can map over the values. 

# %%
transformed = joint_example.op(lambda a: 2*a, lambda b: b + 1)
histogram(transformed)

# %% [markdown]
# More interestingly, we can create a random variable by an operation on the joint. 

# %%
histogram(transformed.add())


# %% [markdown]
# 
# Instead of specifying a joint directly we can construct them through 
# random variables. Here is a joint of independent dice. 

# %% 
die = uniform(1, 7)
two_dice = indep(die, die) 
histogram(two_dice)

# %% [markdown]
# 
# And here is one of the same dice with itself. 

# %% 
two_dice_shared = shared(die)
histogram(two_dice_shared)

# %% [markdown]
# 
# Note that they yield different sums. 

# %%
histogram_row(two_dice.add(), two_dice_shared.add(),
              labels=("Independent", "Shared"))


# %% [markdown]
# 
# One reason to define joint distributions explicitly is to compute covariance:
#
# $$
# \operatorname{Cov}(X,Y)=\sum_{i,j}P_{ij}
# (x_i-\mathbb E[X])(y_j-\mathbb E[Y])
# $$
# 
# Covariance measures how each paired outcome in our joint distribution
# relates to the two means. Unlike variance, contributions can have a
# positive or negative contribution to the covariance, depending on which side 
# of the mean they are. 
# 
# In the sense that variance represents the norm of noise, covariance is the dot product. 
# For the dice above, the two ways of drawing give these rules:
#
# $$
# \begin{aligned}
# X,Y\text{ independent}: && \operatorname{Cov}(X,Y)&=0,\\
# Y=X\text{ (shared draw)}: && \operatorname{Cov}(X,Y)&=\operatorname{Var}(X).
# \end{aligned}
# $$
# Move the slider to see how the joint distribution changes between them.


# %%
assert np.isclose(covar(joint_example), 0)
assert np.isclose(covar(two_dice), 0)
assert np.isclose(covar(two_dice_shared), variance(die))
display(covariance_interpolation_slider(die))

candidate = uniform(0, 11)
threshold = Var(np.arange(1, 7)/6, np.ones(6)/6)
triangle = triangular(candidate, threshold)
check("triangular", triangle)

# %% [markdown]
# The variance of the resulting, one-dimensional variable depends on covariance:
#
# $$
# \begin{aligned}
# \operatorname{Var}(X+Y)&=\operatorname{Var}(X)+\operatorname{Var}(Y)+2\operatorname{Cov}(X,Y),\\
# \operatorname{Var}(X-Y)&=\operatorname{Var}(X)+\operatorname{Var}(Y)-2\operatorname{Cov}(X,Y).
# \end{aligned}
# $$
# For the two dice, move between independent and shared draws to see the
# distribution and variance of their sum.

# %%
display(sum_variance_slider(die))


# %% [markdown]
# ### Helper · iid_statistic
#
# To compute with several independent draws, give `iid_statistic` a `Var`,
# a number of draws, and a function of the resulting `samples`.
# The function sees only the sampled values; the helper returns a `Var` of its result.
# Each value is an independent draw from the input variable.
#
# $$
# P(Z_1=z_1,\ldots,Z_n=z_n)=\prod_{i=1}^n P(Z=z_i).
# $$
# Like `monte_carlo`, build the result with `indep` and `_binop`. Instead of a
# running sum, the scalar state is an integer batch ID: appending support index
# i changes code to k*code+i. For a coin, code 5 with three draws decodes to 1,0,1.
# `_sample_indices` decodes the ID before calling the statistic; no Var stores tuples.
# Rounding the final result to 12 decimal places merges floating-point duplicates.
# With k possible values this takes k**n batches, so use small examples.
# For an order-invariant statistic, `symmetric=True` combines permutations of
# each batch by sorting its indices in `_sorted_sample_code`. `_binop` adds their
# probabilities as usual. This does not group the population.

# %%
def sample_mean(samples):
    return sum(samples) / len(samples)

histogram(iid_statistic(die, 2, sample_mean))

# %% tags=["hide"]
batch_mean = iid_statistic(die, 2, sample_mean)
assert np.isclose(expect(batch_mean), 3.5)
assert np.isclose(variance(batch_mean), variance(die) / 2)


# %% [markdown]
# ## Section 3 · Monte Carlo
# 
# Monte Carlo refers to the approach of using random sampling to
# estimate the mean of an unknown random variable. 
#
# $$
# \mu=\mathbb E[f(X)],\qquad X_1,\ldots,X_n\overset{\mathrm{iid}}{\sim}X
# $$
#
# The standard code itself is almost too simple to be that informative. 

# %% 
def monte_carlo_with_randomness(f, T):
    import random
    return sum(f(random.randint(1, 6)) for _ in range(T)) / T

# %%
def monte_carlo(x: Var, steps: int) -> Var:
    total = uniform(0, 1)
    for _ in range(steps):
        total = add(indep(total, x))
    return div(indep(total, uniform(steps, steps + 1)))

# %% [markdown]
#
# The core laws of probability will tell us that with enough samples
# this will produce the expectation of the distribution. 
# 
# 
# The part that is a bit 
# more intuitively challenging to understand is the distribution of its noise. 
# We can use our framework to calculate this explicitly. 


# %% [markdown]
# 
# $$
# \begin{aligned}
# \hat\mu_1&=f(X_1), & \mathbb E[\hat\mu_1]&=\mu,\\
# \operatorname{Var}(\hat\mu_1)&=\operatorname{Var}(f(X)).
# \end{aligned}
# $$

# %%
check("single", monte_carlo(die.op(f), 1), variance_diagram=True)

# %% [markdown]
# We can decrease the variance by averaging independent samples. 
# 
# As we saw, independent variables have covariance zero, while dividing by $n$ shrinks variance by $n^2$.
#
# $$
# \begin{aligned}
# \hat\mu_{10}&=\frac1{10}\sum_{i=1}^{10}f(X_i), & \mathbb E[\hat\mu_{10}]&=\mu,\\
# \operatorname{Var}(\hat\mu_{10})&=\frac{\operatorname{Var}(f(X))}{10}.
# \end{aligned}
# $$

# %%
check("ten", monte_carlo(die.op(f), 10))


# %% [markdown]
# We can visualize
#
# $$
# X_i\overset{\mathrm{iid}}{\sim}\operatorname{Uniform}\{-1,1\},\quad
# \bar X_n=\frac1n\sum_{i=1}^n X_i,\quad
# \operatorname{Var}(\bar X_n)=\frac1n,\quad n=1,\ldots,20
# $$

# %%
display(coin_variance_slider(20))


# %% [markdown]
# The same approach can be applied to functions of multiple variables. 
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
# ## Section 4 · Control Variates
#
# Monte Carlo provides a scalable method for reducing variance: just
# take more independent samples and it goes down. However, independent 
# samples are expensive, and the variance decreases slowly with more samples. 
# It would be really nice if we could do better. 

# Control variates are a class of techniques for cheaper variance reduction through 
# arithmetic with dependent random variables that preserve the mean. 
#
# Consider subtracting a zero-mean variable using the same draw.
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
centered_coin = uniform(0, 2).op(lambda a: 2*a - 1)
decomposition = shared(centered_coin).op(lambda a: 2*a, lambda b: b)
variance_reduction_3d(decomposition)

# %% [markdown]
# ## Helper · control_variate
#
# Pair the response with a centered control from the same person or draw.
# `known_mean` must be the population mean of `h`, so the control has mean zero.

# %% [markdown]
# ## Helper · monte_carlo_with_control
#
# Subtract the control, then average independent draws of the pair.

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
# The top curves smooth the exact distributions for display; calculations use their exact masses.
# The 3D panels use a fixed view.
#
# $$
# f_\lambda(x)=f(x)+\lambda\bigl[(x-\mathbb E[X])^2-\operatorname{Var}(X)\bigr].
# $$

# %%
pair: Joint = linear_control(die, linear_f, b=3)
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
show_control(pair, steps=5, density_view=True)
check("quadratic", monte_carlo(sub(pair), 5), plot=False)


# %% [markdown]
# ## Example - Polling

# %% tags=["hide"]
from checks import polling_response as response
from plotly_viz import polling_sketch
display(polling_sketch())

# %% [markdown]
# Each dot is a person. The red county tends to answer higher than the blue
# county. We want the average response across all 30 people.
# `population` represents **one random person**, not a fixed set of observations.
# We take six independent draws (with replacement). The response function is
# hidden; the fitting statistic can use only the six responses in its sample.

# %%
population = uniform(0, 30)
group = lambda person: person < 10
ordinary_poll = monte_carlo(population.op(response), 6)

# %% [markdown]
# Use `iid_statistic` from the Joint section to compute with all six draws
# together, retaining their shared role in fitting the control.

# %% [markdown]
# ### Exercise · post_stratify
#
# Leave each draw out when fitting the two group means. Subtract its group's
# fitted mean, then add back the fitted means weighted by 1/3 red and 2/3 blue.
# A group absent from the other five draws gets the fixed fallback 0.
# Return the distribution of the average of all six corrected responses.

# %%
adjusted_poll = post_stratify(population, steps=6, red_share=1/3,
                            group=group, response=response)
print(f"Expectation: {expect(ordinary_poll):.3f} → {expect(adjusted_poll):.3f}")

# %% [markdown]
# These are sampling distributions of the complete six-draw estimators.
# Every possible sample has its own fitted means. The corrected terms share
# fitting data, so their dependence is retained when computing the variance.

# %%
print(f"Sampling variance: {variance(ordinary_poll):.3f} → {variance(adjusted_poll):.3f}")
density(adjusted_poll, without=ordinary_poll,
        comparison_labels=("ordinary poll", "leave-one-out control"))

# %% tags=["hide"]
assert np.isclose(expect(adjusted_poll), expect(ordinary_poll))
assert variance(adjusted_poll) < variance(ordinary_poll)

# %% [markdown]
# ## Exercise · stratify
#
# Instead, fix the allocation beforehand: **2 red and 4 blue draws**.
# Each Var still represents one random response; both groups' means are
# computed from their draws, not by taking population expectations.

# %%
red = population.cond(group).op(response)
blue = population.cond(lambda person: not group(person)).op(response)
stratified_poll = stratify(red, blue, red_share=1/3, red_steps=2, blue_steps=4)
print(f"Sampling variance: {variance(ordinary_poll):.3f} → {variance(stratified_poll):.3f}")
density(stratified_poll, without=ordinary_poll,
        comparison_labels=("ordinary poll", "stratified poll"))

# %% tags=["hide"]
assert np.isclose(expect(stratified_poll), expect(ordinary_poll))
assert variance(stratified_poll) < variance(ordinary_poll)

# %% [markdown]
# ## Section 6 · Example: A/B Tests
#
# Draw independent people for the treatment and control arms.
# $$
# \text{Target: }\mathbb E_{U,V\overset{\mathrm{iid}}{\sim}\mathrm{population}}
# [\mathrm{treatment}(U)-\mathrm{control}(V)].
# $$

# %%
population = uniform(0, 4)

def initial(person: float) -> float:
    return person // 2

def control(person: float) -> float:
    return 2*initial(person) + person % 2

def treatment(person: float) -> float:
    return control(person) + 1

raw_difference = ab_sampling(population, treatment, control, steps=5)
density(raw_difference)
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
density(adjusted_difference, without=raw_difference)
assert np.isclose(expect(adjusted_difference), 1)
assert np.isclose(variance(raw_difference), .5)
assert np.isclose(variance(adjusted_difference), .1)

# %% [markdown]
# ## Exercise · fit_cuped
#
# Fit b from the sampled initial measurements and outcomes.

# %% [markdown]
# ### Helper · cuped
#
# For each person, fit b on the other four sampled people, then correct that
# person's outcome. The pre-treatment population mean is known to be 0.5.
# The graph includes the randomness of every fit, not one frozen coefficient.
# $$
# \text{Target: }\mathbb E_{U,V\overset{\mathrm{iid}}{\sim}\mathrm{population}}
# [\mathrm{treatment}(U)-\mathrm{control}(V)]
# $$
# $$
# \text{Control in each arm: }B_i=\hat b_{-i}(\mathrm{initial}(U_i)-0.5).
# $$

# %%
treated = cuped(population, treatment, initial, known_mean=0.5, steps=5)
untreated = cuped(population, control, initial, known_mean=0.5, steps=5)
cuped_difference = indep(treated, untreated).sub()
print(f"Expectation: {expect(cuped_difference):.3f}")
print(f"Variance: {variance(raw_difference):.3f} → {variance(cuped_difference):.3f}")
density(cuped_difference, without=raw_difference,
        comparison_labels=("plain MC", "leave-one-out CUPED"))
assert np.isclose(expect(cuped_difference), 1)
assert variance(cuped_difference) < variance(raw_difference)


# %% [markdown]
# ## Example: KL Divergence
#
# We now turn to some applications of control variates in 
# machine learning. One case they often come up in is estimating
# entropy and divergences. This is particularly important in 
# applications like language modeling where the size of the token 
# set makes exact computation expensive. 

# $$
# \mathrm{KL}(p\Vert q)=E_{x} \log\frac{p(x)}{q(x)}.
# $$
#
# Here the random variable that we are estimating is the log ratio itself. 
# We can treat p and q as determistic mappings over a shared x.  
# Otherwise we handle it like we have in the past. 

# %%
p = weighted_die([1, 2, 3, 4, 5, 6])
q = uniform(1, 7)
exact_kl = kl(p, q)
print(f"Exact KL(p || q) = {exact_kl:.6f}")

# %% [markdown]
# 
# The terminology for KL estimators draws from the 
# John Schulman's [blog](http://joschu.net/blog/kl-approx.html). 
# The first known as k1 is just what you would do with monte carlo. 


# $$
# X_i\overset{\mathrm{iid}}{\sim}p,\quad
# k_1(r)=-\log r,\quad r(x)=q(x)/p(x),\quad
# $$

# %%
def k1(p: Var, q: Var) -> Var:
    return p.op(lambda a: p.log_prob(a) - q.log_prob(a))

k1_draw = k1(p, q)
k1_samples = monte_carlo(k1_draw, 5)
check("kl", k1_samples, plot=False)
display(monte_carlo_samples_slider(k1_draw, labels=("k1",)))
assert np.isclose(expect(k1_samples), exact_kl)

# %% [markdown]
# ## k2 KL estimate
#
# Squaring the log ratio gives a nonnegative approximation. Its expectation
# generally differs from exact KL; taking more samples reduces variance but
# does not remove this bias.
#
# $$
# k_2(r)=\frac12(\log r)^2.
# $$

# %%
def k2(p: Var, q: Var) -> Var:
    return p.op(lambda a: 0.5 * (p.log_prob(a) - q.log_prob(a))**2)

k2_draw = k2(p, q)
print(f"k2 expectation = {expect(k2_draw):.6f}; exact KL = {exact_kl:.6f}")
display(monte_carlo_samples_slider(k2_draw, without=k1_draw,
                                   labels=("k1", "k2")))

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
def k3(p: Var, q: Var) -> Var:
    pair = control_variate(
        p, lambda a: p.log_prob(a) - q.log_prob(a),
        lambda a: 1 - q.prob(a) / p.prob(a), known_mean=0)
    return pair.sub()

k3_draw = k3(p, q)
k3_samples = monte_carlo(k3_draw, 5)
check("k3", k3_samples, plot=False)
display(monte_carlo_samples_slider(k3_draw, without=k1_draw,
                                   labels=("k1", "k3")))
assert np.isclose(expect(k3_samples), exact_kl)

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
topk_draw = kl_topk(p, q, 3)
topk_samples = monte_carlo(topk_draw, 5)
check("topk", topk_samples, plot=False)
assert np.isclose(expect(topk_samples), exact_kl)
display(monte_carlo_samples_slider(topk_draw, without=k1_draw,
                                   labels=("k1", "top-k")))

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
show_control(pair, density_view=True)
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
coin = uniform(0, 2)
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
density(rewards.add(), without=independent_rewards.add(),
        comparison_labels=("independent", "positive"))
opposing_rewards = group_rewards(model, reward_a, lambda x: float(x <= 3))
covariance_3d(opposing_rewards)
variance_sum_3d(opposing_rewards)
density(opposing_rewards.add(), without=independent_rewards.add(),
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
show_control(pair, density_view=True)
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
show_control(baseline_pair, steps=5, density_view=True)
corrected = monte_carlo(baseline_pair.sub(), 5)
uncorrected = monte_carlo(marginal(baseline_pair), 5)
assert np.isclose(expect(marginal(transpose(baseline_pair))), 0)
assert np.isclose(expect(corrected), .1875)
assert variance(corrected) < variance(uncorrected)
