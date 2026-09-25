import unittest
from unittest.mock import patch
import numpy as np
from dist_types import Var, Joint
from intro_answers import expect, shared, indep, op, binop, op_joint, add, mul, marginal, transpose, covar


class IntroTests(unittest.TestCase):
    def setUp(self):
        self.x = Var([-1, 1], [.5, .5])
        self.y = Var([0, 2, 5], [.2, .3, .5])

    def assertDist(self, actual, values, prob):
        np.testing.assert_allclose(actual.values, values)
        np.testing.assert_allclose(actual.probs, prob)

    def test_expect_and_constant(self):
        self.assertEqual(expect(self.x), 0)
        self.assertEqual(expect(self.y), 3.1)
        self.assertDist(Var([7], [1.0]), [7], [1])

    def test_four_sides(self):
        from intro_answers import four_sides
        from checks import check
        die = four_sides(Var([0, 1], [.5, .5]))
        self.assertIsInstance(die, Var)
        self.assertDist(die, [1, 2, 3, 4], [.25]*4)
        self.assertEqual(expect(die), 2.5)
        self.assertEqual(covar(shared(die)), 1.25)
        check("four", die, plot=False)
        with self.assertRaises(AssertionError):
            check("four", Var([1, 4], [.5, .5]), plot=False)

    def test_six_sides(self):
        from intro_answers import six_sides
        from checks import check
        die = six_sides(Var([0, 1], [.5, .5]))
        self.assertIsInstance(die, Var)
        self.assertDist(die, [1, 2, 3, 4, 5, 6], [1/6]*6)
        self.assertAlmostEqual(expect(die), 3.5)
        self.assertAlmostEqual(covar(shared(die)), 35/12)
        check("six", die, plot=False)

    def test_monte_carlo(self):
        from intro_answers import six_sides, single_sample, ten_samples, linear_control
        from checks import check, f, linear_f
        die = six_sides(Var([0, 1], [.5, .5]))
        one = single_sample(die, f)
        ten = ten_samples(die, f)
        pair = linear_control(die, linear_f, 3)
        self.assertIsInstance(pair, Joint)
        from intro_answers import monte_carlo, sub
        control = monte_carlo(sub(pair), 5)
        self.assertAlmostEqual(covar(pair), 25.25)
        self.assertAlmostEqual(expect(marginal(transpose(pair))), 0)
        for name, dist in (("single", one), ("ten", ten), ("linear", control)):
            self.assertIsInstance(dist, Var)
            check(name, dist, plot=False)
        self.assertAlmostEqual(covar(shared(ten)), covar(shared(one))/10)
        plain = op(linear_f, die)
        self.assertLess(covar(shared(control)), covar(shared(plain))/5)
        self.assertAlmostEqual(expect(control), expect(plain))
        self.assertDist(ten_samples(Var([2], [1.0]), lambda a: 3*a), [6], [1])

    def test_monte_carlo_helper(self):
        from intro_answers import monte_carlo
        self.assertDist(monte_carlo(self.x, 1), [-1, 1], [.5, .5])
        self.assertDist(monte_carlo(self.x, 2), [-1, 0, 1], [.25, .5, .25])
        self.assertDist(monte_carlo(Var([7], [1.0]), 10), [7], [1])
        for n in (3, 5, 10):
            average = monte_carlo(self.y, n)
            self.assertAlmostEqual(expect(average), expect(self.y))
            self.assertAlmostEqual(covar(shared(average)), covar(shared(self.y))/n)

    def test_problems_seven_to_nine(self):
        from intro_answers import six_sides, quadratic_control, two_variables, additive, monte_carlo
        from checks import check, f, quadratic_f, fxy, gxy
        die = six_sides(Var([0, 1], [.5, .5]))
        from intro_answers import sub
        quadratic = monte_carlo(sub(quadratic_control(die, quadratic_f, 1, 2)), 5)
        for name, result in (("quadratic", quadratic), ("xy", two_variables(die, fxy)),
                             ("additive", additive(die, f, gxy))):
            self.assertIsInstance(result, Var)
            check(name, result, plot=False)
        self.assertLess(covar(shared(quadratic)), covar(shared(monte_carlo(op(quadratic_f, die), 5))))
        # Nonuniform inputs: integrate one independent coordinate exactly.
        result = two_variables(self.y, lambda a, b: a + b)
        self.assertAlmostEqual(expect(result), 2*expect(self.y))
        self.assertAlmostEqual(covar(shared(result)), covar(shared(self.y))/5)

    def test_remaining_puzzles(self):
        import intro_answers as a
        from checks import check
        die = a.six_sides(Var([0, 1], [.5, .5]))
        p, q = np.arange(1., 7.)/21, np.ones(6)/6
        theta = np.array([np.log(3.), 0.])
        loo = shared(a.sub(a.reinforce_loo(theta))).op(lambda g: -g, lambda g: g)
        examples = {"sum": a.two_dice(die), "weighted": a.weighted_die([1, 2, 3, 4, 5, 6]),
                    "kl": a.monte_carlo(a.kl_estimate(Var(np.arange(6), p), lambda z: p[int(z)], lambda z: q[int(z)]), 5),
                    "k3": a.monte_carlo(a.sub(a.kl_k3(Var(np.arange(6), p), lambda z: p[int(z)], lambda z: q[int(z)])), 5),
                    "markov": a.markov_chain(Var([0, 1], [1., 0.]), Joint([0, 1], [0, 1], [[3/8, 1/8], [1/8, 3/8]])), "unigram": a.sub(a.markov_unigram(Var([0, 1], [1., 0.]), Joint([0, 1], [0, 1], [[3/8, 1/8], [1/8, 3/8]]))),
                    "reinforce": a.reinforce(theta), "loo": loo}
        for name, result in examples.items():
            self.assertIsInstance(result, Joint if name in {"reinforce", "loo"} else Var)
            check(name, result, plot=False)
        self.assertDist(a.weighted_die([0, 1, 3]), [2, 3], [.25, .75])
        self.assertLess(covar(shared(examples["unigram"])), covar(shared(examples["markov"])))

    def test_binary_gradients(self):
        from itertools import product
        import intro_answers as a
        from checks import assert_distribution
        for theta in (np.array([0., 0.]), np.array([-.4, .8])):
            probs = a.binary_policy(theta).probs
            for n in (2, 3, 4):
                values, masses = [], []
                for outcomes in product((0, 1), repeat=n):
                    rewards = np.array(outcomes, dtype=float)
                    scores = np.eye(2)[list(outcomes)] - probs
                    baseline = (rewards.sum()-rewards)/(n-1)
                    values.append(((rewards-baseline)[:, None]*scores).mean(axis=0))
                    masses.append(np.prod(probs[list(outcomes)]))
                pair = a.reinforce_loo(theta, n)
                self.assertAlmostEqual(expect(marginal(transpose(pair))), 0)
                corrected = shared(a.sub(pair)).op(lambda g: -g, lambda g: g)
                assert_distribution(corrected, np.array(values), np.array(masses))
                result = a.reinforce(theta, n)
                self.assertAlmostEqual(expect(a.marginal(result)), -probs.prod())
                self.assertAlmostEqual(expect(a.marginal(a.transpose(result))), probs.prod())

    def test_control_pairs(self):
        import intro_answers as a
        from checks import linear_f, quadratic_f
        x = a.six_sides(Var([0, 1], [.5, .5]))
        p, q = np.arange(1., 7.)/21, np.ones(6)/6
        for pair in (a.linear_control(x, linear_f, 3), a.quadratic_control(x, quadratic_f, 1, 2),
                     a.kl_k3(Var(np.arange(6), p), lambda z: p[int(z)], lambda z: q[int(z)]),
                     a.markov_unigram(Var([0, 1], [1., 0.]), Joint([0, 1], [0, 1], [[3/8, 1/8], [1/8, 3/8]])), a.reinforce_loo(np.array([np.log(3.), 0.]))):
            self.assertIsInstance(pair, Joint)
            estimate, control = marginal(pair), marginal(transpose(pair))
            self.assertAlmostEqual(expect(control), 0)
            self.assertAlmostEqual(expect(a.sub(pair)), expect(estimate))
            self.assertAlmostEqual(covar(shared(a.sub(pair))),
                                   covar(shared(estimate)) + covar(shared(control)) - 2*covar(pair))

    def test_independent_two_variable_monte_carlo(self):
        from intro_answers import independent_two_variables, six_sides, variance
        from checks import check, fxy
        die = six_sides(Var([0, 1], [.5, .5]))
        result = independent_two_variables(die, die, fxy)
        check("xy_mc", result, plot=False)
        one = indep(die, die)._binop(fxy)
        self.assertAlmostEqual(expect(result), expect(one))
        self.assertAlmostEqual(variance(result), variance(one)/5)
        result = independent_two_variables(self.x, self.y, lambda a, b: a*b)
        one = indep(self.x, self.y)._binop(lambda a, b: a*b)
        self.assertAlmostEqual(expect(result), expect(one))
        self.assertAlmostEqual(variance(result), variance(one)/5)

    def test_kl_callable_interface(self):
        import intro_answers as a
        x = Var([2, 7], [.3, .7])
        p = lambda z: .3 if z == 2 else .7
        q = lambda z: .6 if z == 2 else .4
        estimate = a.kl_estimate(x, p, q)
        target = .3*np.log(.3/.6) + .7*np.log(.7/.4)
        self.assertAlmostEqual(expect(estimate), target)
        pair = a.kl_k3(x, p, q)
        self.assertAlmostEqual(expect(marginal(transpose(pair))), 0)
        self.assertAlmostEqual(expect(a.sub(pair)), target)
        self.assertAlmostEqual(a.variance(a.monte_carlo(estimate, 5)), a.variance(estimate)/5)

    def test_markov_joint_transition(self):
        import intro_answers as a
        transition = np.array([[.75, .25], [.25, .75]])
        T = Joint([2, 7], [2, 7], np.array([.2, .8])[:, None]*transition)
        state = Var([2, 7], [.3, .7])
        self.assertDist(a.propagate(state, T), [2, 7], state.probs @ transition)
        for weights in ([.5, .5], [.2, .8]):
            T = Joint([2, 7], [2, 7], np.array(weights)[:, None]*transition)
            for steps in (0, 1, 3, 5):
                for probs in ([1., 0.], [.3, .7]):
                    result = a.markov_chain(Var([2, 7], probs), T, steps)
                    self.assertDist(result, [2, 7], np.array(probs) @ np.linalg.matrix_power(transition, steps))

    def test_unigram_with_supplied_state_and_transition(self):
        import intro_answers as a
        T = Joint([2, 7], [2, 7], [[.3, .1], [.12, .48]])
        state = Var([2, 7], [.3, .7])
        for steps in (0, 1, 3, 6):
            pair = a.markov_unigram(state, T, steps)
            expected = a.markov_chain(state, T, steps)
            self.assertDist(marginal(pair), expected.values, expected.probs)
            self.assertAlmostEqual(expect(marginal(transpose(pair))), 0)
            self.assertAlmostEqual(expect(pair.sub()), expect(expected))
        self.assertAlmostEqual(a.variance(a.markov_unigram(state, T, 1).sub()), 0)

    def test_generic_topk(self):
        import intro_answers as a
        x = Var([2, 7, 9], [.2, .5, .3])
        f = lambda z: z*z-3
        target = expect(x.op(f))
        for k in range(4):
            self.assertAlmostEqual(expect(a.topk(x, f, k)), target)
        self.assertAlmostEqual(a.variance(a.topk(x, f, 3)), 0)
        self.assertDist(a.topk(x, f, 0), x.op(f).values, x.probs)

    def test_topk_kl(self):
        import intro_answers as a
        from checks import check
        x = a.weighted_die([1, 2, 3, 4, 5, 6])
        p, q = lambda z: z/21, lambda z: 1/6
        baseline = a.kl_estimate(x, p, q)
        target = expect(baseline)
        for k in range(7):
            estimate = a.kl_topk(x, p, q, k)
            self.assertAlmostEqual(expect(estimate), target)
        zero = a.kl_topk(x, p, q, 0)
        self.assertDist(zero, baseline.values, baseline.probs)
        self.assertAlmostEqual(a.variance(a.kl_topk(x, p, q, 6)), 0)
        check("topk", a.monte_carlo(a.kl_topk(x, p, q, 3), 5), plot=False)
        self.assertLess(a.variance(a.sub(a.kl_k3(x, p, q))), a.variance(baseline))
        for k in (-1, 7):
            with self.assertRaises(ValueError):
                a.kl_topk(x, p, q, k)
        x = Var([2, 7], [.3, .7])
        p, q = lambda z: .3 if z == 2 else .7, lambda z: .6 if z == 2 else .4
        result = a.kl_topk(x, p, q, 1)
        exact = .7*np.log(.7/.4)
        self.assertDist(result, [exact+np.log(.3/.6), exact], [.3, .7])

    def test_circle_rejection(self):
        from intro_answers import circle
        coordinate = Var(np.arange(11), np.ones(11)/11)
        for radius in (0, 5, 10):
            disk = circle(coordinate, coordinate, radius)
            accepted = np.array([[(x-5)**2+(y-5)**2 <= radius*radius for y in range(11)]
                                 for x in range(11)])
            np.testing.assert_allclose(disk.probs, accepted/accepted.sum())
            self.assertAlmostEqual(disk.probs.sum(), 1)
        disk = circle(coordinate, coordinate, 5)
        self.assertGreater(disk.probs[0, 5], 0)
        self.assertEqual(disk.probs[10, 10], 0)
        self.assertEqual(np.count_nonzero(disk.probs), 81)
        from viz import joint_top_view
        import matplotlib.pyplot as plt
        fig = joint_top_view(disk, show=False)
        self.assertEqual(len(fig.axes[0].collections[1].get_offsets()), 81)
        plt.close(fig)

    def test_explicit_marginal_control(self):
        import intro_answers as a
        j = Joint([0, 1, 3, 4], [0, 3],
                  [[.25, 0], [.25, 0], [0, .25], [0, .25]])
        pair = a.marginal_control(j)
        self.assertDist(marginal(transpose(pair)), [-1.5, 1.5], [.5, .5])
        self.assertAlmostEqual(expect(marginal(transpose(pair))), 0)
        self.assertAlmostEqual(covar(pair), 2.25)
        result = a.monte_carlo(pair.sub(), 5)
        self.assertAlmostEqual(expect(result), expect(marginal(j)))
        self.assertAlmostEqual(a.variance(result), .05)
        keep = result.probs > 0
        np.testing.assert_allclose(result.values[keep], np.arange(6)/5+1.5)
        np.testing.assert_allclose(result.probs[keep], np.array([1, 5, 10, 10, 5, 1])/32)

    def test_ab_sampling_and_initial_control(self):
        import intro_answers as a
        population = Var([0, 1, 2, 3], [.25]*4)
        initial = lambda person: person//2
        control = lambda person: 2*initial(person) + person%2
        treatment = lambda person: control(person)+1
        for n in (1, 5):
            raw = a.ab_sampling(population, treatment, control, n)
            adjusted = a.ab_test(population, treatment, control, initial, 2, n)
            self.assertAlmostEqual(expect(raw), 1)
            self.assertAlmostEqual(expect(adjusted), 1)
            self.assertAlmostEqual(a.variance(raw), 2.5/n)
            self.assertAlmostEqual(a.variance(adjusted), .5/n)
        result = a.ab_test(population, treatment, control, initial, 2, 1)
        keep = result.probs > 0
        np.testing.assert_allclose(result.values[keep], [0, 1, 2])
        np.testing.assert_allclose(result.probs[keep], [.25, .5, .25])

    def test_learned_cuped_and_baseline(self):
        import intro_answers as a
        z = np.array([0., 0., 1., 1.])
        y = np.array([0., 1., 2., 3.])
        self.assertAlmostEqual(a.fit_cuped(z, y), 2)
        self.assertAlmostEqual(a.fit_cuped(np.ones(4), y), 0)
        population = Var([0, 1, 2, 3], [.25]*4)
        b = a.fit_cuped(z, y)
        estimate = a.ab_test(population, lambda s: s+1, lambda s: s,
                             lambda s: s//2, b, 5)
        self.assertAlmostEqual(expect(estimate), 1)
        self.assertAlmostEqual(a.variance(estimate), .1)
        baseline = a.fit_baseline(np.array([0., 0., 0., 1.]))
        self.assertAlmostEqual(baseline, .25)
        reward = Var([0, 1], [.75, .25])
        for fitted in (baseline, 0., 1., -2.):
            pair = a.estimated_baseline(reward, fitted)
            self.assertAlmostEqual(expect(marginal(transpose(pair))), 0)
            self.assertAlmostEqual(expect(pair.sub()), .1875)
        pair = a.estimated_baseline(reward, baseline)
        self.assertAlmostEqual(a.variance(pair.sub()), .046875)
        self.assertLess(a.variance(pair.sub()), a.variance(marginal(pair)))

    def test_group_rewards(self):
        import intro_answers as a
        from viz import variance_sum_3d
        import matplotlib.pyplot as plt
        model = a.four_sides(Var([0, 1], [.5, .5]))
        positive = a.group_rewards(model, lambda x: float(x >= 3), lambda x: float(x >= 2))
        negative = a.group_rewards(model, lambda x: float(x >= 3), lambda x: float(x <= 3))
        independent = indep(marginal(positive), marginal(transpose(positive)))
        for pair, cross, var in ((positive, 1/8, 11/16), (negative, -1/8, 3/16),
                                 (independent, 0, 7/16)):
            self.assertAlmostEqual(expect(pair.add()), 1.25)
            self.assertAlmostEqual(covar(pair), cross)
            self.assertAlmostEqual(a.group_variance(pair), var)
            self.assertAlmostEqual(a.variance(pair.add()), var)
            fig = variance_sum_3d(pair, show=False)
            self.assertTrue(fig.axes[-1].get_title().startswith("sum variance"))
            plt.close(fig)
        self.assertDist(positive.add(), [0, 1, 2], [.25, .25, .5])
        self.assertDist(negative.add(), [0, 1, 2], [0, .75, .25])

    def test_plot_rounding(self):
        from viz import _number, covariance_3d
        import matplotlib.pyplot as plt
        self.assertEqual(_number(1.23456), "1.23")
        self.assertEqual(_number(1.2), "1.2")
        self.assertEqual(_number(-.001), "0")
        self.assertEqual(_number(2), "2")
        fig = covariance_3d(shared(self.x), show=False)
        for axis in (fig.axes[0].xaxis, fig.axes[0].yaxis, fig.axes[0].zaxis):
            self.assertEqual(axis.get_major_formatter()(1.23456, 0), "1.23")
        plt.close(fig)

    def test_div(self):
        from intro_answers import div
        self.assertDist(div(indep(self.x, Var([2], [1.0]))), [-.5, .5], [.5, .5])
        self.assertDist(div(shared(self.x)), [-1, 1], [0, 1])
        self.assertDist(div(indep(self.x, self.x)), [-1, 1], [.5, .5])

    def test_attached_operations(self):
        import intro_answers as a
        self.assertDist(self.x.op(lambda z: z+2), [1, 3], [.5, .5])
        self.assertDist(self.x.cond(lambda z: z > 0), [1], [1])
        pair = shared(self.x).op(lambda z: z+2, lambda z: -z)
        self.assertDist(pair._binop(lambda x, y: x+y), [0, 2, 4], [0, 1, 0])
        with patch.object(a, "op", return_value=Var([9], [1.0])) as replacement:
            self.assertDist(self.x.op(lambda z: z), [9], [1])
            replacement.assert_called_once()

    def test_reorganized_exercises(self):
        import intro_answers as a
        from checks import check
        eight = Var(np.arange(1, 9), np.ones(8)/8)
        self.assertDist(a.shift(eight, 2), np.arange(3, 11), np.ones(8)/8)
        self.assertEqual(a.variance(eight), 5.25)
        check("six", a.six_from_eight(eight), plot=False)
        coins = Joint([0, 1], [0, 1], [[1/8, 3/8], [1/8, 3/8]])
        self.assertEqual(expect(marginal(coins)), .5)
        self.assertEqual(expect(marginal(transpose(coins))), .75)
        self.assertDist(coins.add(), [0, 1, 2], [1/8, 1/2, 3/8])
        candidate = Var(np.arange(11), np.ones(11)/11)
        uniform = Var(np.arange(1, 7)/6, np.ones(6)/6)
        triangle = a.triangular(candidate, uniform)
        check("triangular", triangle, plot=False)
        self.assertAlmostEqual(expect(triangle), 5)
        self.assertAlmostEqual(a.variance(triangle), 35/6)
        # Conditioning an independent pair creates dependence.
        accepted = indep(candidate, uniform).cond(lambda x, u: u <= (6-abs(x-5))/6)
        self.assertAlmostEqual(accepted.probs.sum(), 1)

    def test_condition(self):
        from intro_answers import condition
        self.assertDist(condition(lambda a: a > 0, self.x), [1], [1])
        self.assertDist(condition(lambda a: a < 5, self.y), [0, 2], [.4, .6])
        self.assertDist(condition(lambda a: True, self.x), [-1, 1], [.5, .5])
        with self.assertRaises(ValueError):
            condition(lambda a: False, self.x)

    def test_transpose_and_sub(self):
        from intro_answers import sub
        from dist_types import _joint
        pair = _joint([-1, 1], [0, 2, 5], [[.1, .2, 0], [0, .3, .4]])
        swapped = transpose(pair)
        np.testing.assert_allclose(swapped.probs, pair.probs.T)
        np.testing.assert_allclose(transpose(swapped).probs, pair.probs)
        self.assertDist(marginal(swapped), [0, 2, 5], [.1, .5, .4])
        self.assertAlmostEqual(covar(swapped), covar(pair))
        self.assertDist(sub(indep(Var([5], [1.0]), Var([2], [1.0]))), [3], [1])
        self.assertDist(sub(transpose(indep(Var([5], [1.0]), Var([2], [1.0])))), [-3], [1])
        self.assertDist(sub(shared(self.x)), [-2, 0, 2], [0, 1, 0])

    def test_joint_and_marginals(self):
        pair = indep(self.x, self.y)
        self.assertEqual(pair.probs.shape, (2, 3))
        np.testing.assert_allclose(pair.probs, [[.1, .15, .25], [.1, .15, .25]])
        self.assertDist(marginal(pair), [-1, 1], [.5, .5])
        self.assertDist(marginal(transpose(pair)), [0, 2, 5], [.2, .3, .5])
        np.testing.assert_allclose(shared(self.y).probs, np.diag([.2, .3, .5]))

    def test_apply_add_mul(self):
        self.assertDist(op(lambda a: a + 2, self.x), [1, 3], [.5, .5])
        self.assertDist(op(lambda a: a*a, self.x), [1], [1])
        self.assertDist(op(lambda a: 4., self.y), [4], [1])
        self.assertDist(add(shared(self.x)), [-2, 0, 2], [.5, 0, .5])
        self.assertDist(add(indep(self.x, self.x)), [-2, 0, 2], [.25, .5, .25])
        self.assertDist(mul(shared(self.x)), [-1, 1], [0, 1])
        self.assertDist(mul(indep(self.x, self.x)), [-1, 1], [.5, .5])
        combined = binop(lambda a, b: a*a+2*b, shared(self.x))
        self.assertDist(combined, [-1, 3], [.5, .5])
        collapsed = binop(lambda a, b: 4., indep(self.x, self.y))
        self.assertDist(collapsed, [4], [1])

    def test_apply_joint(self):
        f, g = lambda a: a + 2, lambda a: 2*a
        joint = op_joint(f, g, shared(self.x))
        np.testing.assert_allclose(joint.probs, [[.5, 0], [0, .5]])
        self.assertEqual(covar(joint), 2)
        self.assertDist(marginal(joint), [1, 3], [.5, .5])
        self.assertDist(marginal(transpose(joint)), [-2, 2], [.5, .5])
        self.assertEqual(covar(indep(op(f, self.x), op(g, self.x))), 0)
        merged = op_joint(lambda a: a*a, lambda a: 7., shared(self.x))
        np.testing.assert_allclose(merged.probs, [[1]])
        self.assertDist(marginal(merged), [1], [1])
        # Many-to-one transforms on both axes of a rectangular joint.
        merged = op_joint(lambda a: a*a, lambda a: a % 2, indep(self.x, self.y))
        np.testing.assert_allclose(merged.probs, [[.5, .5]])

    def test_covariance(self):
        self.assertEqual(covar(shared(self.x)), 1)
        self.assertEqual(covar(indep(self.x, self.x)), 0)
        self.assertAlmostEqual(covar(shared(self.y)), np.dot(self.y.probs, (self.y.values-3.1)**2))
        self.assertAlmostEqual(covar(indep(self.x, self.y)), 0)
        self.assertEqual(covar(shared(Var([4], [1.0]))), 0)

    def test_validation_and_no_operators(self):
        for values, prob in (([1, 2], [1]), ([1], [-1]), ([1], [np.nan]), ([1], [.5]), ([], [])):
            with self.assertRaises(ValueError):
                Var(values, prob)
        with self.assertRaises(TypeError):
            self.x + self.x
        with self.assertRaises(TypeError):
            self.x @ self.x
        with self.assertRaises(TypeError):
            Joint()
        self.assertFalse(hasattr(self.x, "given"))
        self.assertFalse(hasattr(self.x, "names"))

    def test_visualizations(self):
        import matplotlib.pyplot as plt
        from viz import histogram, variance_3d, covariance_3d
        with patch.object(plt, "show"):
            figs = [histogram(self.x), variance_3d(self.x), covariance_3d(indep(self.x, self.y))]
            with self.assertRaises(TypeError):
                covariance_3d(self.x)
            figs.append(histogram(shared(self.x)))
        self.assertEqual(figs[1].axes[0].get_title(), "variance 1")
        for fig in figs:
            plt.close(fig)

    def test_variance_reduction_geometry(self):
        import matplotlib.pyplot as plt
        from viz import variance_reduction_3d
        for slope in (1, -1):
            pair = op_joint(lambda a: 2*a, lambda b: slope*b, shared(self.x))
            fig = variance_reduction_3d(pair, show=False)
            self.assertEqual(len(fig.axes), 4)
            self.assertEqual(fig.axes[2].get_title(), f"control contribution {-4*slope:+g}")
            self.assertEqual(fig.axes[3].get_title(), f"remaining variance {(2-slope)**2}")
            self.assertEqual(fig.axes[0].get_zlim(), fig.axes[2].get_zlim())
            plt.close(fig)
        for pair in (indep(self.x, self.y), shared(Var([3], [1.0]))):
            plt.close(variance_reduction_3d(pair, show=False))

    def test_joint_histogram_posts(self):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from viz import histogram
        with patch.object(plt, "show"):
            fig = histogram(indep(self.x, self.y))
        ax = fig.axes[0]
        self.assertFalse(any(isinstance(c, Poly3DCollection) for c in ax.collections))
        self.assertEqual(len(ax.lines), 6)
        self.assertTrue(ax.zaxis.pane.fill)
        self.assertAlmostEqual(ax.zaxis.pane.get_facecolor()[3], .06)
        self.assertFalse(ax.xaxis.pane.fill)
        self.assertFalse(ax.yaxis.pane.fill)
        self.assertEqual(ax.get_zlim()[0], 0)
        for line, (x, y, p) in zip(ax.lines, [(x, y, px*py) for x, px in zip(self.x.values, self.x.probs)
                                             for y, py in zip(self.y.values, self.y.probs)]):
            np.testing.assert_allclose(line.get_data_3d(), [[x, x], [y, y], [0, p]])
            self.assertEqual(line.get_color(), "black")
        plt.close(fig)

    def test_histogram_posts_and_zero_baseline(self):
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from viz import histogram
        with patch.object(plt, "show"):
            for dist in (self.x, self.y, Var([0, 1], [.123456789, .876543211])):
                fig = histogram(dist)
                ax = fig.axes[0]
                self.assertEqual(ax.get_ylim()[0], 0)
                posts = [c for c in ax.collections if isinstance(c, LineCollection)]
                self.assertEqual(len(posts), 1)
                np.testing.assert_allclose(posts[0].get_colors()[0], [0, 0, 0, 1])
                for segment, value, prob in zip(posts[0].get_segments(), dist.values, dist.probs):
                    np.testing.assert_allclose(segment, [[value, 0], [value, prob]])
                plt.close(fig)
            fig = histogram(add(indep(self.x, self.x)), without=add(shared(self.x)))
            self.assertEqual(fig.axes[0].get_ylim()[0], 0)
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
