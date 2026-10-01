import unittest
from unittest.mock import patch
import numpy as np
from dist_types import Var, Joint
from intro_answers import expect, uniform, shared, indep, op, binop, op_joint, add, marginal, transpose, covar

import ast
from pathlib import Path
from types import SimpleNamespace
import intro_answers

def blog_functions():
    """Load the article's actual function definitions without running its demos."""
    source = Path(__file__).with_name("puzzle.py").read_text()
    tree = ast.parse(source)
    definitions = ast.Module(
        body=[node for node in tree.body if isinstance(node, ast.FunctionDef)],
        type_ignores=[])
    namespace = dict(vars(intro_answers))
    exec(compile(definitions, "puzzle.py", "exec"), namespace)
    return SimpleNamespace(**namespace)

blog = blog_functions()

class IntroTests(unittest.TestCase):
    def test_temperature_gradients(self):
        for temperature in (0.7, 3.0, 8.0):
            action = blog.policy(temperature)
            np.testing.assert_allclose(
                action.probs, intro_answers.temperature_policy(temperature).probs)
            self.assertEqual(len(action.values), 8)
            delta = 1e-5
            derivative = (expect(blog.policy(temperature + delta))
                          - expect(blog.policy(temperature - delta))) / (2 * delta)
            scores = np.array([blog.temperature_score(x, expect(action), temperature)
                               for x in action.values])
            numerical_scores = (np.log(blog.policy(temperature + delta).probs)
                                - np.log(blog.policy(temperature - delta).probs)) / (2 * delta)
            np.testing.assert_allclose(scores, numerical_scores, atol=1e-8)
            self.assertAlmostEqual(float(action.probs @ scores), 0)
            score = shared(action).op(
                lambda a: a, lambda a: blog.temperature_score(a, expect(action), temperature))
            for baseline in (0., expect(action), -2.):
                gradient = blog.reinforce_control(lambda a: a, baseline, score)
                self.assertAlmostEqual(expect(gradient), derivative)
        for temperature in (0., -1., np.inf, np.nan):
            with self.assertRaises(ValueError):
                intro_answers.temperature_policy(temperature)

    def test_leave_one_out_baseline(self):
        from intro_answers import leave_one_out, temperature_policy
        for temperature in (0.7, 3.0, 8.0):
            action = temperature_policy(temperature)
            others = np.random.default_rng(11).choice(
                action.values, size=4, p=action.probs)
            self.assertEqual(leave_one_out(temperature, n=5), float(others.mean()))
        with self.assertRaises(ValueError):
            leave_one_out(3., n=1)

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
        repeated = Var([2, 2, 7], [.2, .3, .5])
        self.assertAlmostEqual(repeated.prob(2), .5)
        self.assertEqual(repeated.prob(3), 0)
        self.assertAlmostEqual(repeated.log_prob(2), np.log(.5))

    def test_uniform_half_open_interval(self):
        self.assertDist(uniform(1, 4), [1, 2, 3], [1/3]*3)
        self.assertDist(uniform(-2, -1), [-2], [1])
        with self.assertRaises(ValueError):
            uniform(3, 3)
        with self.assertRaises(TypeError):
            uniform(0.5, 3)

    def test_monte_carlo(self):
        from intro_answers import monte_carlo, linear_control
        from checks import check, f, linear_f
        die = uniform(1, 7)
        one = monte_carlo(die.op(f), 1)
        ten = monte_carlo(die.op(f), 10)
        pair = linear_control(die, linear_f, 3)
        self.assertIsInstance(pair, Joint)
        from intro_answers import sub
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
        self.assertDist(monte_carlo(Var([2], [1.0]).op(lambda a: 3*a), 10), [6], [1])

    def test_monte_carlo_helper(self):
        from intro_answers import monte_carlo
        self.assertDist(monte_carlo(self.x, 1), [-1, 1], [.5, .5])
        self.assertDist(monte_carlo(self.x, 2), [-1, 0, 1], [.25, .5, .25])
        self.assertDist(monte_carlo(Var([7], [1.0]), 10), [7], [1])
        for n in (3, 5, 10):
            average = monte_carlo(self.y, n)
            self.assertAlmostEqual(expect(average), expect(self.y))
            self.assertAlmostEqual(covar(shared(average)), covar(shared(self.y))/n)

    def test_control_pairs(self):
        import intro_answers as a
        from checks import linear_f, quadratic_f
        x = uniform(1, 7)
        p, q = np.arange(1., 7.)/21, np.ones(6)/6
        p_dist, q_dist = Var(np.arange(6), p), Var(np.arange(6), q)
        for pair in (a.linear_control(x, linear_f, 3), a.quadratic_control(x, quadratic_f, 1, 2),
                     a.control_variate(p_dist,
                         lambda z: p_dist.log_prob(z)-q_dist.log_prob(z),
                         lambda z: 1-q_dist.prob(z)/p_dist.prob(z), 0)):
            self.assertIsInstance(pair, Joint)
            estimate, control = marginal(pair), marginal(transpose(pair))
            self.assertAlmostEqual(expect(control), 0)
            self.assertAlmostEqual(expect(a.sub(pair)), expect(estimate))
            self.assertAlmostEqual(covar(shared(a.sub(pair))),
                                   covar(shared(estimate)) + covar(shared(control)) - 2*covar(pair))

    def test_kl_var_interface(self):
        import intro_answers as a
        p = Var([2, 7], [.3, .7])
        q = Var([7, 2], [.4, .6])
        estimate = blog.k1(p, q)
        target = .3*np.log(.3/.6) + .7*np.log(.7/.4)
        self.assertAlmostEqual(a.kl(p, q), target)
        self.assertEqual(a.kl(p, p), 0)
        self.assertEqual(a.kl(p, Var([2, 9], [.6, .4])), float("inf"))
        self.assertEqual(a.kl(Var([2, 7], [1, 0]), Var([2], [1])), 0)
        self.assertEqual(a.kl(Var([2, 2], [.3, .7]), Var([2], [1])), 0)
        self.assertAlmostEqual(expect(estimate), target)
        corrected = blog.kl_k3(p, q).sub()
        pair = blog.kl_k3(p, q)
        self.assertIsInstance(pair, Joint)
        self.assertAlmostEqual(expect(marginal(pair)), target)
        self.assertAlmostEqual(expect(marginal(transpose(pair))), 0)
        self.assertDist(pair.sub(), corrected.values, corrected.probs)
        self.assertAlmostEqual(expect(corrected), target)
        self.assertTrue(np.all(corrected.values[corrected.probs > 0] >= -1e-12))
        squared = blog.k2(p, q)
        self.assertAlmostEqual(expect(squared),
            .5*(.3*np.log(.3/.6)**2 + .7*np.log(.7/.4)**2))
        self.assertFalse(np.isclose(expect(squared), target))
        self.assertAlmostEqual(a.variance(a.monte_carlo(estimate, 5)), a.variance(estimate)/5)

    def test_generic_topk(self):
        import intro_answers as a
        x = Var([2, 7, 9], [.2, .5, .3])
        f = lambda z: z*z-3
        target = expect(x.op(f))
        for k in range(4):
            self.assertAlmostEqual(expect(a.topk(x, f, k)), target)
        self.assertAlmostEqual(a.variance(a.topk(x, f, 3)), 0)
        self.assertDist(a.topk(x, f, 0), x.op(f).values, x.probs)
        self.assertDist(a.topk(x, f, 1), [23.5, 62.], [.4, .6])
        # A zero-mass remainder is deterministic even if its support is nonempty.
        self.assertDist(a.topk(Var([2, 7], [1., 0.]), f, 1), [1.], [1.])

    def test_topk_kl(self):
        import intro_answers as a
        from checks import check
        p = a.weighted_die([1, 2, 3, 4, 5, 6])
        q = uniform(1, 7)
        baseline = blog.k1(p, q)
        target = expect(baseline)
        for k in range(7):
            estimate = blog.kl_topk(p, q, k)
            self.assertAlmostEqual(expect(estimate), target)
        zero = blog.kl_topk(p, q, 0)
        self.assertDist(zero, baseline.values, baseline.probs)
        self.assertAlmostEqual(a.variance(blog.kl_topk(p, q, 6)), 0)
        check("topk", a.monte_carlo(blog.kl_topk(p, q, 3), 5), plot=False)
        self.assertLess(a.variance(blog.kl_k3(p, q).sub()), a.variance(baseline))
        for k in (-1, 7):
            with self.assertRaises(ValueError):
                blog.kl_topk(p, q, k)
        p = Var([2, 7], [.3, .7])
        q = Var([2, 7], [.6, .4])
        result = blog.kl_topk(p, q, 1)
        exact = .7*np.log(.7/.4)
        self.assertDist(result, [exact+.3*np.log(.3/.6)], [1.0])

    def test_topk_kl_selects_by_p_probability(self):
        import intro_answers as a
        p = Var([0, 1, 2], [.6, .3, .1])
        q = Var([0, 1, 2], [.89, .1, .01])
        # Outcome 0 has the highest p, but outcome 1 contributes the most KL.
        terms = p.probs * np.log(p.probs / q.probs)
        self.assertEqual(int(np.argmax(terms)), 1)
        exact = terms[0]
        expected = Var([exact + .4*np.log(.3/.1), exact + .4*np.log(.1/.01)],
                       [.75, .25])
        actual = blog.kl_topk(p, q, 1)
        self.assertDist(actual, expected.values, expected.probs)
        self.assertAlmostEqual(expect(actual), a.kl(p, q))

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

    def test_histogram_merges_float_duplicates(self):
        from viz import _scalar_mass
        x = Var([.7999999999999998, .8, .8000000000000003, .804, .82],
                [.1, .2, .3, .1, .3])
        values, mass = _scalar_mass(x)
        np.testing.assert_allclose(values, [.8, .804, .82])
        np.testing.assert_allclose(mass, [.6, .1, .3])
        self.assertEqual(len(x.values), 5)
        import intro_answers as a
        population = Var([0, 1, 2, 3], [.25]*4)
        result = a.ab_sampling(population, lambda s: s+1, lambda s: s, 5)
        values, mass = _scalar_mass(result)
        self.assertEqual(len(values), 31)
        self.assertAlmostEqual(mass.sum(), 1)

    def test_equal_width_plot_bins(self):
        from viz import _range_masses
        values = np.arange(1000.)**2
        mass = np.ones(1000)/1000
        (centers, grouped), = _range_masses((values, mass))
        width = (values.max()-values.min())*.02
        self.assertLessEqual(len(centers), 51)
        edges = np.r_[centers-width/2, centers[-1]+width/2]
        expected, _ = np.histogram(values, bins=edges, weights=mass)
        np.testing.assert_allclose(grouped, expected)
        self.assertAlmostEqual(grouped.sum(), 1)
        np.testing.assert_allclose(np.diff(centers), width)
        a, b = _range_masses((values, mass), (values, mass))
        np.testing.assert_allclose(a, b)
        (centers, grouped), = _range_masses((np.array([1., 2.]), np.array([.3, .7])))
        np.testing.assert_allclose(centers, [1, 2])
        np.testing.assert_allclose(grouped, [.3, .7])
        (centers, grouped), = _range_masses((np.array([2.]), np.array([1.])))
        np.testing.assert_allclose(centers, [2])
        np.testing.assert_allclose(grouped, [1])
        blue = (np.array([.95, .99, 1., 1.01, 1.05]), np.ones(5)/5)
        gray = (np.array([-2., 4.]), np.array([.5, .5]))
        (centers, grouped), _ = _range_masses(blue, gray)
        np.testing.assert_allclose(centers, [1.])
        np.testing.assert_allclose(grouped, [1.])

    def test_shared_control_display(self):
        from contextlib import redirect_stdout
        from io import StringIO
        import matplotlib.pyplot as plt
        from viz import show_control
        pair = shared(self.x).op(lambda x: 2*x, lambda x: x)
        output = StringIO()
        with redirect_stdout(output):
            fig = show_control(pair, steps=2, show=False)
        self.assertIn("Var 1 = 2; Var 2 = 0.5; 2 Cov = 2", output.getvalue())
        self.assertIn("= 0.5", output.getvalue())
        self.assertEqual(len(fig.axes[0].texts), 0)
        np.testing.assert_allclose(fig.axes[0].collections[1].get_facecolors()[0], [1, 1, 1, 1])
        plt.close(fig)

    def test_linear_control_slider(self):
        from plotly_viz import linear_control_frames
        from checks import linear_f
        frames = linear_control_frames(Var(np.arange(1, 7), np.ones(6)/6), linear_f)
        self.assertEqual(len(frames), 21)
        for frame in frames:
            va, vb, cross, total = frame["terms"]
            self.assertAlmostEqual(va + vb - cross, total)
            self.assertAlmostEqual(frame["mc_variance"], total / 5)
            self.assertAlmostEqual(frame["mean"], frames[0]["mean"])
            np.testing.assert_allclose(frame["b"], frames[0]["b"])
            for values, masses in frame["distributions"]:
                self.assertAlmostEqual(masses.sum(), 1)
        self.assertGreater(frames[-1]["terms"][-1], frames[0]["terms"][-1])

    def test_weighted_density(self):
        from plotly_viz import _weighted_density
        grid = np.linspace(-5, 6, 1000)
        density = _weighted_density(np.array([0., 1.]), np.array([.25, .75]), grid, .4)
        self.assertAlmostEqual(np.trapezoid(density, grid), 1, places=5)
        self.assertGreater(density[np.argmin(abs(grid-1))], density[np.argmin(abs(grid))])

    def test_interactive_variance(self):
        from plotly_viz import variance_widget
        html = variance_widget(self.x).data
        self.assertIn('"dragmode":false', html)
        self.assertIn("plotly_hover", html)
        self.assertNotIn("Variance contribution", html)
        self.assertIn("Plotly.restyle", html)

    def test_covariance_interpolation_slider(self):
        from plotly_viz import covariance_interpolation_slider, _covariance_figure
        from intro_answers import variance
        independent, paired = indep(self.x, self.x), shared(self.x)
        for strength in (0, .25, .5, 1):
            mixed = Joint(self.x.values, self.x.values,
                (1-strength)*independent.probs + strength*paired.probs)
            self.assertAlmostEqual(covar(mixed), strength*variance(self.x))
        first, _, _ = _covariance_figure(independent, keep_zeros=True)
        last, _, _ = _covariance_figure(paired, keep_zeros=True)
        self.assertEqual(len(first.data), len(last.data))
        html = covariance_interpolation_slider(self.x).data
        self.assertIn("Cov(X,Y) = 0.000", html)
        self.assertIn("Cov(X,Y) = 1.000", html)

    def test_sum_variance_slider_uses_scalar_dice_sums(self):
        from intro_answers import variance
        from plotly_viz import sum_variance_slider
        die = uniform(1, 7)
        self.assertAlmostEqual(variance(indep(die, die).add()), 35/6)
        self.assertAlmostEqual(variance(shared(die).add()), 35/3)
        html = sum_variance_slider(die).data
        self.assertIn("Var(X + Y) = 2.917 + 2.917 + 2 × 0.000 = 5.833", html)
        self.assertIn("11.667", html)
        self.assertIn('"title":{"text":"X + Y"}', html)

    def test_scaled_variance_slider(self):
        from plotly_viz import scaled_variance_slider, _covariance_figure
        from intro_answers import variance
        x=Var(np.arange(1,9),np.array([1,3,1,3,1,3,1,3])/16)
        for b in range(1,5):
            self.assertAlmostEqual(variance(x.op(lambda a: b*a)),b*b*variance(x))
        html=scaled_variance_slider(x).data
        self.assertIn('"active":0',html)
        self.assertIn('"label":"4"',html)
        self.assertNotIn('"label":"0"',html)
        self.assertIn('Var(bX) = 83.00',html)
        self.assertIn('Var(bX) = 20.75',html)
        bounds=(1,32,float(x.probs.max()))
        f1,_,_=_covariance_figure(shared(x),True,bounds=bounds)
        f4,_,_=_covariance_figure(shared(x.op(lambda a: 4*a)),True,bounds=bounds)
        dots1=[t for t in f1.data if t.mode=="markers"]
        dots4=[t for t in f4.data if t.mode=="markers"]
        self.assertFalse(np.allclose(dots1[0].x,dots4[0].x))
        self.assertIn('32',html)

    def test_plotly_histogram(self):
        from plotly_viz import _histogram_figure, _covariance_figure
        fig=_histogram_figure(self.x)
        self.assertEqual(fig.layout.yaxis.range[0],0)
        self.assertEqual(fig.data[-1].mode,"markers")
        np.testing.assert_allclose(fig.data[-1].y,self.x.probs)
        np.testing.assert_allclose(fig.layout.xaxis.tickvals,self.x.values)
        dense=Var(np.arange(12),np.ones(12)/12)
        ticks=_histogram_figure(dense).layout.xaxis.tickvals
        self.assertEqual(len(ticks),8)
        np.testing.assert_allclose([ticks[0],ticks[-1]],[0,11])
        joint,boxes,mapping=_covariance_figure(indep(self.x,self.y),boxes=False)
        self.assertTrue(all(joint.data[i].visible is False for i in boxes))
        self.assertEqual(len(joint.layout.shapes),1)

    def test_foreground_variance_and_coin_slider(self):
        from plotly_viz import _covariance_figure, coin_variance_slider
        fig, boxes, mapping = _covariance_figure(indep(self.x,self.x))
        targets = [int(index) for index in mapping]
        self.assertGreater(min(targets), max(boxes))
        labels = [trace for trace in fig.data if trace.mode == "markers+text"]
        self.assertEqual(len(labels),4)
        self.assertTrue(all(", " not in trace.text[0] for trace in labels))
        self.assertTrue(all(str(i) in mapping for i,t in enumerate(fig.data) if t.mode == "markers+text"))
        for trace in labels:
            self.assertEqual(trace.hovertemplate.count("<br>"), 2)
            self.assertNotIn("Mass", trace.hovertemplate)
        self.assertEqual(fig.layout.hoverlabel.font.size,11)
        self.assertTrue(all(fig.data[len(boxes)+b].opacity == .3 for b in boxes))
        html=coin_variance_slider(20).data
        self.assertIn('"label":"20"',html)
        self.assertIn('Coin flips:',html)
        self.assertIn('"fixedrange":true',html)

    def test_ab_control_slider(self):
        from plotly_viz import ab_control_frames
        from intro_answers import ab_test, variance
        population=uniform(0,4)
        initial=lambda person: -.25*(person % 2)
        untreated=lambda person: 2*initial(person)+person % 2
        treated=lambda person: untreated(person)+1
        states=ab_control_frames(population,treated,untreated,initial)
        for state in states:
            expected=ab_test(population,treated,untreated,initial,state['strength'],5)
            values,probs=state['distributions'][0]
            self.assertAlmostEqual(float(values @ probs),1)
            self.assertAlmostEqual(state['mc_variance'],variance(expected))
            va,vb,cross,total=state['terms']
            self.assertAlmostEqual(va+vb-cross,total)
            self.assertAlmostEqual(total/5,state['mc_variance'])
        by_b={s['strength']:s for s in states}
        self.assertAlmostEqual(by_b[1]['mc_variance'],.05625)
        self.assertAlmostEqual(by_b[0]['mc_variance'],.025)
        self.assertAlmostEqual(by_b[-2]['mc_variance'],0)

    def test_decomposition_common_scale(self):
        import plotly_viz as v
        pair=shared(self.x).op(lambda x: 2*x, lambda x: x)
        figures=[]
        original=v._covariance_figure
        def capture(*args, **kwargs):
            result=original(*args, **kwargs)
            figures.append(result[0])
            return result
        with patch.object(v, '_covariance_figure', side_effect=capture):
            v.decomposition_widget(pair)
        # Var(2X-X)=1: the result's square has half the side of Var(2X)=4.
        first_width=np.ptp(figures[0].data[0].x)
        result_width=np.ptp(figures[3].data[0].x)
        self.assertAlmostEqual(result_width/first_width, .5)

    def test_polling(self):
        from checks import polling_response
        from intro_answers import monte_carlo, variance, control_variate, monte_carlo_with_control
        population=Var(np.arange(30),np.ones(30)/30)
        responses=population.op(polling_response)
        np.testing.assert_allclose(responses.probs,np.array([12,5,2,2,3,6])/30)
        self.assertAlmostEqual(expect(responses),87/30)
        estimates=monte_carlo(responses,50)
        self.assertAlmostEqual(expect(estimates),expect(responses))
        self.assertAlmostEqual(variance(estimates),variance(responses)/50)
        guess=lambda person: 6 if person<10 else 2
        known_mean=expect(population.op(guess))
        self.assertAlmostEqual(known_mean,10/3)
        pair=control_variate(population,polling_response,guess,known_mean)
        control=marginal(transpose(pair))
        corrected=monte_carlo_with_control(pair,50)
        self.assertAlmostEqual(expect(control),0)
        self.assertAlmostEqual(expect(corrected),expect(responses))
        self.assertLess(variance(corrected),variance(estimates))

    def test_control_variate_helpers(self):
        from intro_answers import control_variate, monte_carlo_with_control, variance
        x=Var([0,1],[.25,.75])
        pair=control_variate(x,lambda a: 2*a+1,lambda a: a,.75,b=2)
        self.assertIsInstance(pair,Joint)
        self.assertAlmostEqual(expect(marginal(transpose(pair))),0)
        self.assertAlmostEqual(covar(pair),.75)
        result=monte_carlo_with_control(pair,5)
        self.assertAlmostEqual(expect(result),2.5)
        self.assertAlmostEqual(variance(result),0)
        unadjusted=monte_carlo_with_control(control_variate(x,lambda a: 2*a+1,lambda a: a,.75,b=0),5)
        self.assertAlmostEqual(variance(unadjusted),.75/5)

    def test_stratified_polling(self):
        from intro_answers import stratify, variance
        red=Var([5,6],[.5,.5])
        blue=Var([1,2,3],[.5,.25,.25])
        with patch('checks.polling_response',side_effect=AssertionError('No population access')):
            result=stratify(red,blue,1/3,2,4)
        self.assertIsInstance(result,Var)
        self.assertAlmostEqual(expect(result),3)
        self.assertAlmostEqual(variance(result),variance(red)/18+variance(blue)/9)

    def test_dense_plot_labels(self):
        from plotly_viz import _covariance_figure, _label_indices
        np.testing.assert_array_equal(_label_indices(8),np.arange(8))
        self.assertEqual(len(_label_indices(9)),8)
        fig,_,_=_covariance_figure(shared(Var(np.arange(12),np.ones(12)/12)),True)
        labels=[t.text[0] for t in fig.data if t.mode=="markers+text"]
        self.assertEqual(sum(bool(t) for t in labels),8)
        self.assertEqual(len(labels),12)

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


    def test_attached_operations(self):
        import intro_answers as a
        self.assertDist(self.x.op(lambda z: z+2), [1, 3], [.5, .5])
        self.assertDist(self.x.cond(lambda z: z > 0), [1], [1])
        pair = shared(self.x).op(lambda z: z+2, lambda z: -z)
        self.assertDist(pair._binop(lambda x, y: x+y), [0, 2, 4], [0, 1, 0])
        with patch.object(a, "op", return_value=Var([9], [1.0])) as replacement:
            self.assertDist(self.x.op(lambda z: z), [9], [1])
            replacement.assert_called_once()

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
        self.assertDist(shared(self.x).mul(), [-1, 1], [0, 1])
        self.assertDist(indep(self.x, self.x).mul(), [-1, 1], [.5, .5])
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
            figs = [histogram(self.x, show=False), variance_3d(self.x, show=False), covariance_3d(indep(self.x, self.y), show=False)]
            with self.assertRaises(TypeError):
                covariance_3d(self.x)
            figs.append(histogram(shared(self.x), show=False))
        self.assertEqual(figs[1].axes[0].get_title(), "")
        self.assertEqual(figs[1].axes[0].get_zlabel(), "")
        self.assertEqual(len(figs[1].axes[0].get_zticks()), 0)
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
            fig = histogram(indep(self.x, self.y), show=False)
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
                fig = histogram(dist, show=False)
                ax = fig.axes[0]
                self.assertEqual(ax.get_ylim()[0], 0)
                posts = [c for c in ax.collections if isinstance(c, LineCollection)]
                self.assertEqual(len(posts), 1)
                np.testing.assert_allclose(posts[0].get_colors()[0], [0, 0, 0, 1])
                for segment, value, prob in zip(posts[0].get_segments(), dist.values, dist.probs):
                    np.testing.assert_allclose(segment, [[value, 0], [value, prob]])
                plt.close(fig)
            fig = histogram(add(indep(self.x, self.x)), without=add(shared(self.x)), show=False)
            self.assertEqual(fig.axes[0].get_ylim()[0], 0)
            plt.close(fig)

if __name__ == "__main__":
    unittest.main()
