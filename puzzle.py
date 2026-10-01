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
# mastery of [Reinforce](https://en.wikipedia.org/wiki/Policy_gradient_method)
# has become arguably as central as techniques like attention. 
# The algorithm is critical to producing models 
# that reason, act, and discover. 
# 
# 
# Despite being mathematically concise, 
# Reinforce is tricky to master. It is not uncommon 
# to read papers in the area that are confused or buggy. 
# My personal theory for this is that 
# random sampling can actually hide a bunch of issues.
# It makes it hard to just "code your way" through problems.
#  
# The goal of this blog is to build up to Reinforce 
# from scratch. It takes the non-standard approach
# of doing it without any random numbers.  Obviously, this cannot scale, but it will let us build 
# intuition for boring things like
# elementary variance reduction techniques. # Code is available at [srush/sampling-to-reinforce](https://github.com/srush/sampling-to-reinforce). The blog assumes no knowledge of RL or 
# math. The text and comments are written by a person; the code is written by AI. 
# 
# This project is inspired by [Haskell distributions](https://hackage.haskell.org/package/distribution-1.1.1.0/docs/Data-Distribution.html), 
# [lea](https://archive.fosdem.org/2015/schedule/event/lea,_a_probability_engine_in_python/), 
# and [Monte Carlo theory, methods and examples](https://artowen.su.domains/mc/).

# - srush
# %% tags=["hide"]
import jax
jax.config.update("jax_enable_x64", True)

from checks import check, f, linear_f, quadratic_f, fxy
import numpy as np
from viz import variance_3d, covariance_3d, coin_variance_animation
from IPython.display import display

# %% tags=["hide"]
from dist_types import Fn, Var, Joint
from intro_answers import (
    stratify, expect, six_from_eight, uniform, variance, indep, shared, marginal,
    covar, transpose, add, sub, div, four_sides, six_sides, two_triangles,
    triangular, circle, linear_control, independent_two_variables, quadratic_control,
    two_variables, weighted_die, ab_sampling, ab_test, markov_chain,
    markov_unigram, group_rewards, group_variance,
    fit_baseline, kl, topk, propagate, leave_one_out,
)
from viz import density, histogram, histogram_row, variance_reduction_3d, variance_sum_3d, joint_top_view, show_control
from plotly_viz import (
    scaled_variance_slider, covariance_interpolation_slider,
    coin_variance_slider, linear_control_widget, sum_variance_slider,
    monte_carlo_samples_slider, ab_control_widget, temperature_policy_slider,
    reinforce_baseline_slider,
)

# %% [markdown]
# ## Discrete Random Variables
#


# We are going to start from scratch by implementing a mini-language for 
# working with random variables. Our language will allow us to apply 
# some basic operations on these variables and propagate key properties. 
# These are random in the statistical sense, but will not require drawing pseudorandom values in
# the CS sense. Instead, we keep all possible values and their probabilities.

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
# Mathematically, this is written as 
# $$ 
# Y=2X.
# $$
# However, to make it clearer what is going on, we write it
# as an explicit map over values. Similarly, we define a filter
# over values that renormalizes the distribution. 

# %%

histogram_row(eight.op(lambda a: a * 2), eight.cond(lambda a: a < 6),
              labels=("Double the value", "Keep values below 6"))


# %% [markdown]
# 
# We will be interested in estimating expectations of random variables.
# The expectation is represented by the red line in the histogram
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
# As we build our way to Reinforce, our primary goal will be to 
# build low-variance, unbiased estimators. Unbiased means that 
# they have the same expectation as a target quantity. 
# Variance measures the spread of a random variable around its mean.
# It's the quantity we aim to minimize. 
#
# $$
# \operatorname{Var}(X)=\mathbb E[(X-\mathbb E[X])^2]
# $$
#

# The visualization for variance will be a weighted sum of squares.
# The base of each box is a square whose side length is the distance from the mean.
# Its height is the same as in the histogram.
#
# $$
# \operatorname{Var}(bX)=b^2\operatorname{Var}(X)
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
# ## Joint Random Variables
#
# Next, we will consider the relationship between random variables. 
# We'll be overly pedantic here to show how it works. 
# Our object here will keep track of the probability of 
# every shared state in a table. 



# %%
joint_example = Joint([0, 2, 5], [0, 1],
                      [[1/12, 1/12], [1/6, 1/4], [1/4, 1/6]])
histogram(joint_example)

# %% [markdown]
# Just as with random variables, we can map over the values. 

# %%
transformed = joint_example.op(lambda a: 2 * a, lambda b: -1 * b + 3)
histogram(transformed)

# %% [markdown]
# We can also combine them. 

# %%
histogram(transformed.add())


# %% [markdown]
# 
# Or construct joints from independent random variables. 

# %% 
die = uniform(1, 7)
two_dice = indep(die, die) 
histogram(two_dice)

# %% [markdown]
# 
# Or from the same shared random variable.

# %% 
two_dice_shared = shared(die)
histogram(two_dice_shared)

# %% [markdown]
# 
# Notably, these joints have different sums.

# %%
histogram_row(two_dice.add(), two_dice_shared.add(),
              labels=("Independent", "Shared"))


# %% [markdown]
# 
# The key property of joints will be their covariance. 
#
# $$
# \operatorname{Cov}(X,Y)=\sum_{i,j}P_{ij}
# (x_i-\mathbb E[X])(y_j-\mathbb E[Y])
# $$
# 
# Covariance measures how each paired outcome in our joint distribution
# relates to the two means. Unlike variance contributions, covariance contributions
# can be positive or negative, depending on which side of their respective
# means the paired values lie. 
# 
# In the sense that variance represents the squared norm of noise, covariance is the dot product.
# For the dice above, the two ways of drawing give these rules:
#
# $$
# \begin{aligned}
# \operatorname{Cov}(X,Y)&=0 &&\text{if }X,Y\text{ are independent},\\
# \operatorname{Cov}(X,X)&=\operatorname{Var}(X) &&\text{for the same shared draw}.
# \end{aligned}
# $$
# In this example we interpolate between shared and independent dice and show the covariance. 
# Magenta indicates a negative contribution. 

# %%
assert np.isclose(covar(two_dice), 0)
assert np.isclose(covar(two_dice_shared), variance(die))
display(covariance_interpolation_slider(die))


# %% [markdown]
# The variance of the resulting, one-dimensional variable depends on covariance:
#
# $$
# \begin{aligned}
# \operatorname{Var}(X+Y)&=\operatorname{Var}(X)+\operatorname{Var}(Y)+2\operatorname{Cov}(X,Y),\\
# \operatorname{Var}(X-Y)&=\operatorname{Var}(X)+\operatorname{Var}(Y)-2\operatorname{Cov}(X,Y).
# \end{aligned}
# $$
# As the slider moves from independent to shared, each die has the same variance
# but their covariance increases. You can see this intuitively in the histogram:
# the independent sum spreads out less than the shared one. 

# %%
display(sum_variance_slider(die))


# %% [markdown]
# ## Monte Carlo Sampling
# 
# Monte Carlo sampling is used to
# estimate the expectation of an unknown random variable. 
#
# $$
# \mu=\mathbb E[f(X)],\qquad X_1,\ldots,X_n\overset{\mathrm{iid}}{\sim}X
# $$
#
# The standard code itself is trivial. 

# %% 
def monte_carlo_with_randomness(f, T):
    import random
    return sum(f(random.randint(1, 6)) for _ in range(T)) / T

# %% [markdown]
# Our version will use joint distribution primitives 
# to explicitly track independent draws. The result is a random variable itself, called the estimator.

# %%
def monte_carlo(f_x: Var, steps: int) -> Var:
    total = f_x.op(lambda value: 0)
    for _ in range(steps):
        total = add(indep(total, f_x))
    return total.op(lambda value: value / steps)

# %% [markdown]
# 
# The Monte Carlo estimator will have the same expectation as the true random variable. 
# The interesting question, though, is how it can deviate from this value. This is captured 
# by the variance of the estimator.


# %%
histogram(die.op(f))
check("single", monte_carlo(die.op(f), 1), variance_diagram=True)

# %% [markdown]
# We decrease the variance by averaging independent samples. 
# This works because of the three properties above:
# * independent variables have covariance zero,
# * adding independent variables therefore adds their variances, and 
# * dividing by n shrinks variance by the square of n.
#

# %%
check("ten", monte_carlo(die.op(f), 10))


# %% [markdown]
# Here is a full example of a set of coin flips showing the variance of the estimator.

# %%
display(coin_variance_slider(20))




# %% [markdown]
# ## Example: Polling

# %% tags=["hide"]
from checks import polling_response as response
from plotly_viz import polling_sketch
display(polling_sketch())

# %% [markdown]
# Sampling makes more sense if the function `f` is completely unknown. 
# We can consider the example of estimating a population-level value by polling 
# a subset of people. Here, imagine two counties, one that leans red (6) and one that leans blue (1).
# Each individual's value is unknown.

# %%
population = uniform(0, 30)
group = lambda person: person < 10
ordinary_poll = monte_carlo(population.op(response), 6)
histogram(ordinary_poll)


# %% [markdown]
# One intuitive way to think about variance here 
# is to consider what is happening in the worst-case scenario where 
# the estimator yields 1. That can only really occur if all the samples
# landed on the blue side of the population. 
#
# In this case we can actually reduce variance without requiring more samples through 
# stratified sampling. 
# We do this by exploiting the known structure of the population, i.e. the individual groups 
# may have lower internal variance. Note that this does not use the direction of each group's lean.
#
# We see that this yields an unbiased estimator that doesn't sample the outliers at all.

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
# ## Control Variates
#
# Monte Carlo provides a scalable method for reducing variance: just
# take more independent samples and it goes down. However, independent 
# samples are expensive, and the variance decreases slowly with more samples. 
# We saw that stratified sampling helped, but required specific knowledge of the function.

# Control variates are flexible techniques for variance reduction. They work by 
# adding in a term to the random variable that preserves expectations while reducing variance. 
#

# Consider subtracting a zero-mean variable using the same draw.
# The expectation stays fixed; positive covariance can reduce the variance.

# $$
# \mathbb E[X-B]=\mathbb E[X]-\mathbb E[B]=\mathbb E[X]
# $$


# %%
def control_variate(x: Var, f: Fn, h: Fn,
                    known_mean: float, b: float = 1) -> Joint:
    return shared(x).op(f, lambda a: b*(h(a)-known_mean))

def monte_carlo_with_control(pair: Joint, steps: int) -> Var:
    return monte_carlo(pair.sub(), steps)

# %% [markdown]
# 
# Recall our variance decomposition. For a control variate to reduce variance,
# twice its covariance with the target must exceed its own variance.

#
# $$
# \begin{aligned}
# \operatorname{Var}(A-B)&=\operatorname{Var}(A)+\operatorname{Var}(B)-2\operatorname{Cov}(A,B).
# \end{aligned}
# $$

# %%
centered_coin = uniform(0, 2).op(lambda a: 2*a - 1)
decomposition = shared(centered_coin).op(lambda a: 2*a, lambda b: b)
variance_reduction_3d(decomposition)




# %% [markdown]
# Generally, we would like to pick a control variate that is 
# close to the target random variable. There is some art to 
# this, and it can also be learned from data. 
#
# Let's consider a simple example where we have a function that we know 
# is roughly linear with slope 3. We use a linear function with shared randomness 
# as a control variate. 
# 
# In the best case, we get it exactly right and the estimator variance goes to zero.
# Worst case, we get it wrong and add variance. In any case, the expectation is preserved.



# %%
pair: Joint = linear_control(die, linear_f, b=3)
display(linear_control_widget(die, linear_f, b=3, steps=5))

# %% tags=["hide"]
check("linear", monte_carlo(sub(pair), 5), plot=False)

# %% [markdown]
#  
#

# %%
pair: Joint = quadratic_control(die, quadratic_f, a=1, b=2)
show_control(pair, steps=5, density_view=True)
check("quadratic", monte_carlo(sub(pair), 5), plot=False)



# %% [markdown]
# ## Example: A/B Tests
#
# Now let's consider an example of an A/B test. 
# We independently sample `U` and `V` from the population and compare 
# their responses to a treatment. Notably, we do this when we are 
# unable to get the value for the same person both treated and untreated.

# $$
# \mathbb E_{U,V}
# [\mathrm{treated}(U)-\mathrm{untreated}(V)].
# $$


# %%
population = uniform(0, 4)

def initial(person: float) -> float:
    "Initial preference. Think of this as user features."
    return -0.25 * (person % 2)

def untreated(person: float) -> float:
    "Change from baseline"
    return 2*initial(person) + person % 2

def treated(person: float) -> float:
    "Change from treatment"
    return untreated(person) + 1

raw_difference = ab_sampling(population, treated, untreated, steps=5)
density(raw_difference)

# %% [markdown]
#
# To try to reduce the variance of this estimator, we can use information 
# that we know about the population before the test. Here we will use the 
# initial preference as a control variate, assuming we know the mean. 
# 
# $$
# \text{Control: }B=\bigl(\mathrm{initial}(U)-\mathrm{initial}(V)\bigr).
# $$

# %%
adjusted_difference = ab_test(population, treated, untreated, initial, b=1, steps=5)
density(adjusted_difference, without=raw_difference)


# %% [markdown]
# Hmm, this felt like a really good idea, but it actually increased the variance!
# The problem is the correlation had the wrong sign. 

# We're on the right track, though. Instead of just using the initial preference 
# we can instead scale it without changing the expectation. 
#
# $$
# \text{Control: }B=b\bigl(\mathrm{initial}(U)-\mathrm{initial}(V)\bigr).
# $$
#
# In practice, this scaling factor could be learned or estimated from the sampled group (if 
# you are careful). 

# %%
display(ab_control_widget(population, treated, untreated, initial, steps=5))


# %% [markdown]
# ## Example: KL Divergence
#
# We now turn to some applications of control variates in 
# machine learning. They often come up when estimating
# entropy and divergences. This is particularly important in 
# applications like language modeling where the size of the token 
# set makes exact computation expensive. 

# $$
# \mathrm{KL}(p\Vert q)=\mathbb E_{X\sim p}\left[\log\frac{p(X)}{q(X)}\right].
# $$
#
# Here the random variable that we are estimating is the log ratio itself. 
# We can treat `p` and `q` as deterministic mappings over a shared `x`.  
# Otherwise, we handle it like we have in the past. 

# %%
p = weighted_die([1, 2, 3, 4, 5, 6])
q = uniform(1, 7)
exact_kl = kl(p, q)
print(f"Exact KL(p || q) = {exact_kl:.6f}")

# %% [markdown]
# 
# This section is based loosely on John Schulman's 
# [Approximating KL Divergence](http://joschu.net/blog/kl-approx.html). 
# We won't go into the details of the derivations, but just connect his pieces to our 
# framework. 
# 
# The most obvious estimator is just to run Monte Carlo.

# $$
# X_i\overset{\mathrm{iid}}{\sim}p,\quad
# k_1(r)=-\log r,\quad r(x)=q(x)/p(x).
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
# Interestingly, this estimator has the property that it can be negative. 
# While this is fine in expectation, it is not ideal in practice as it is acting as a divergence. 
#
#
# Squaring the log ratio gives a nonnegative approximation. 
# 
# $$
# k_2(r)=\frac12(\log r)^2.
# $$

# %%
def k2(p: Var, q: Var) -> Var:
    return p.op(lambda a: 0.5 * (p.log_prob(a) - q.log_prob(a))**2)

k2_draw = k2(p, q)
print(f"k2 expectation = {expect(k2_draw):.6f}; exact KL = {exact_kl:.6f}")
histogram(k2_draw)


# %% 
display(monte_carlo_samples_slider(k2_draw, without=k1_draw,
                                   labels=("k1", "k2")))

# %% [markdown]
# Unfortunately, the k2 estimator changes the expectation, yielding
# a biased estimator. We really want to maintain the expectation. 
#
# The approach in k3 is to use the ratio of the two distributions as a control variate.
# The benefit of this is that we can easily show that its expectation is 1. 
# 
# $$
# \text{Control: }B=\frac{q(X)}{p(X)}.
# $$
# 
# 


# %%
def kl_k3(p: Var, q: Var, b: float = -1) -> Joint:
    return control_variate(
        p, lambda a: p.log_prob(a) - q.log_prob(a),
        lambda a: q.prob(a) / p.prob(a), known_mean=1, b=b)

pair: Joint = kl_k3(p, q, b=1)
covariance_3d(pair, max_post_labels=0)

# %% [markdown]
# Again, we see that the term has negative covariance. This seems bad at first, but 
# we can fix it by setting the coefficient to a negative value.  Specifically, setting 
# it to -1 gives
# 
# $$
# k_3(r)=-\log r+r-1.
# $$
# 
# This yields a better unbiased estimator for our example. 

# %%
pair = kl_k3(p, q, b=-1)
show_control(pair, steps=5, density_view=True, show=False)
k3_draw = sub(pair)
k3_samples = monte_carlo(k3_draw, 5)
check("k3", k3_samples, plot=False)
display(monte_carlo_samples_slider(k3_draw, without=k1_draw,
                                   labels=("k1", "k3")))
assert np.isclose(expect(k3_samples), exact_kl)



# %% [markdown]
# There are other estimators of KL that extend these ideas further.
# Inspired by the [Top-K KL Estimator](https://arxiv.org/abs/2602.04417), we sum
# the top-k highest-probability outcomes exactly. Here we sample only from the
# remaining outcomes and multiply the log ratio by their total probability mass.
#
# $$
# \hat D_k=\sum_{a\in T_k}p(a)\log\frac{p(a)}{q(a)}
# +m\log\frac{p(X)}{q(X)},\qquad
# X\sim p(\cdot\mid X\notin T_k),\quad m=\sum_{a\notin T_k}p(a).
# $$
# If the remaining mass is zero, the exact sum is the whole answer.
# 
# We can see this approach as a variant of stratified sampling (as in polling) where we combine
# random sampling with deterministic enumeration. 

# %%
def kl_topk(p: Var, q: Var, k: int) -> Var:
    return topk(p, lambda a: np.log(p.prob(a)/q.prob(a)) if p.prob(a) else 0.0, k)


topk_draw = kl_topk(p, q, 3)
topk_samples = monte_carlo(topk_draw, 5)
check("topk", topk_samples, plot=False)
assert np.isclose(expect(topk_samples), exact_kl)
display(monte_carlo_samples_slider(topk_draw, without=k1_draw,
                                   labels=("k1", "top-k")))


# %% [markdown]
# ## Reinforce
#
# The conclusion of this exercise is to apply what we have learned to 
# Reinforce. Reinforce gives us a general form for computing the derivative of 
# the expected reward. The reward is fixed with respect to the parameters.
# The second step uses the log-derivative identity.

# $$
# \begin{aligned}
# \nabla_\theta\mathbb E_{A\sim p_\theta}[r(A)]
# &=\sum_a r(a)\nabla_\theta p_\theta(a)\\
# &=\sum_a p_\theta(a)r(a)\nabla_\theta\log p_\theta(a)\\
# &=\mathbb E_{A\sim p_\theta}
#   [r(A)\nabla_\theta\log p_\theta(A)].
# \end{aligned}
# $$

# In this final term, the second part, known as the score, has an important property. 
# Here we use the same trick in reverse. 
# 
# $$
# \begin{aligned}
# \mathbb E_{A\sim p_\theta}[\nabla_\theta\log p_\theta(A)]
# &=\sum_a p_\theta(a)\nabla_\theta\log p_\theta(a)\\
# &=\sum_a\nabla_\theta p_\theta(a) =\nabla_\theta 1=0.
# \end{aligned}
# $$
#
# The implication of this property is that any constant offset to the 
# first term does not change the expectation. 
#
# $$
# \begin{aligned}
# \mathbb E_{A\sim p_\theta}[(r(A)-b)\nabla_\theta\log p_\theta(A)]
# &=\mathbb E_{A\sim p_\theta}[r(A)\nabla_\theta\log p_\theta(A)]\\
#   &\qquad -b\underbrace{\mathbb E_{A\sim p_\theta}[\nabla_\theta\log p_\theta(A)]}_{0}\\
# &=\nabla_\theta\mathbb E_{A\sim p_\theta}[r(A)],\qquad b\text{ constant}.
# \end{aligned}
# $$

# This allows us to use baselines whose means need not be known,
# as long as the baselines are independent of the sampled action A. 


# %% [markdown]
# ## Example: Model Training
#  
# Now let's consider training the simplest model possible.
# The model has one parameter, temperature, and uses that parameter 
# to control a discrete policy with eight possible outcomes. 
#
# $$
# p_T(a)=\frac{\exp(a/T)}{\sum_{j}\exp(j/T)}\quad T>0.
# $$

# %%
def policy(temperature: float) -> Var:
    logits = np.arange(8, dtype=float)
    weights = np.exp((logits - logits.max()) / temperature)
    return Var(logits, weights / weights.sum())

temperature = 3.0
action = policy(temperature)
display(temperature_policy_slider(policy, temperature))

# %% [markdown]
# The score is the derivative of the sampled class's log probability with
# respect to temperature. It has expectation zero.
#
# $$
# s_T(a)=\frac{\partial}{\partial T}\log p_T(a).
# $$

# %%
def temperature_score(action: float, mean_action: float, temperature: float) -> float:
    """Derivative of log p_T(action) with respect to T."""
    return (mean_action - action) / temperature**2


score: Var = action.op(lambda a: temperature_score(a, expect(action), temperature))
histogram(score)
assert np.isclose(expect(score), 0)

# %% [markdown]
#
# We now put this together to form
#
# $$
# \mathbb E_{A\sim p_\theta}[r(A)\nabla_\theta\log p_\theta(A)].
# $$

# %%
def r(a): 
    "Set reward to the value."
    return a

score_joint: Joint = shared(action).op(
    lambda a: a,
    lambda a: temperature_score(a, expect(action), temperature))
reinforce: Joint = score_joint.op(r, lambda s: s)
gradient_draw = reinforce.mul()
gradient_samples = monte_carlo(gradient_draw, 5)

exact_gradient = -variance(action) / temperature**2
histogram(gradient_samples)
display(monte_carlo_samples_slider(gradient_draw, labels=("Reinforce",)))

# %% [markdown]
#
# Now that we have the full formula, we can consider methods for variance reduction. 
# We start by playing with some possible constants to modify our formula with. 
#
# $$
#  (r(A)-b) s_T(A).
# $$

# %%
def reinforce_control(r: Fn, b: float, score: Joint) -> Var:
    """Subtract a constant baseline from the reward, then multiply by the score."""
    return score.op(lambda a: r(a) - b, lambda s: s).mul()


controlled_draw = reinforce_control(r, 4.0, score_joint)
controlled_samples = monte_carlo(controlled_draw, 5)
display(reinforce_baseline_slider(reinforce_control, r, score_joint, steps=2))

# %% [markdown]

# Playing around with the above example, you get a sense that putting the constant value 
# around the mean of the original distribution reduces the variance of our estimator. 
# The math here is beyond the blog, but in general, approximating the expected reward is a useful way 
# to set this value. 
#
# Of course, if we are in the process of learning our model, it is hard to know what the 
# expected reward is, since we cannot enumerate our policy. Luckily, this too can be done 
# with Monte Carlo sampling. We simply take independent samples and use their mean
# to estimate the expected reward. 

# In fact, we can reuse the samples that we are already using for Reinforce to compute this
# value. The trick is to exclude the current sample: the baseline b must be independent
# of that sample for the identity to hold. 

# $$
# (r(A_i)-\bar r_{-i})s_T(A_i).
# $$


# %%
baseline = leave_one_out(temperature, n=3)
gradient = reinforce_control(r, baseline, score_joint)

display(monte_carlo_samples_slider(gradient, without=gradient_draw,
                                   labels=("Reinforce", "Leave-one-out")))
assert np.isclose(expect(gradient), exact_gradient)

# %% [markdown]
# There are, of course, several more things you need to extend this approach to full-scale LLMs. 
# We need to extend it to multiple parameters as well as sequences of random variables. 
# There are also many interesting details in the choice and interaction of rewards.
# That being said, this is roughly the main math underlying much of what we now call post-training. 
