"""Exact checks for the new exercises and their variance-reduction claims."""
import unittest
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import distribution as d
import puzzles as p
from checks import check, assert_distribution


class NewPuzzles(unittest.TestCase):
    def test_coin_average_animation_distributions(self):
        from viz import coin_average
        for n in range(1, 21):
            rv = coin_average(n)
            np.testing.assert_allclose(rv.mean(), 0, atol=1e-12)
            np.testing.assert_allclose(rv.var(), 1/n, atol=1e-12)
            if n <= 6:
                explicit = d.mean([d.flip(f"x.{i}") for i in range(n)])
                assert_distribution(explicit, *rv.table())

    def test_covariance_post_labels(self):
        import matplotlib.pyplot as plt
        from unittest.mock import patch
        from viz import covariance_3d
        x = d.flip("x")
        with patch.object(plt, "show"):
            fig = covariance_3d(x @ x, show=False)
        ax = fig.axes[0]
        self.assertEqual([t.get_text().strip() for t in ax.texts],
                         ["-1,-1", "-1,1", "1,-1", "1,1"])
        self.assertEqual(len(ax.get_xticks()), 0)
        self.assertEqual(len(ax.get_yticks()), 0)
        red = [line for line in ax.lines if line.get_color() == "#c63737"]
        self.assertEqual(len(red), 1)
        np.testing.assert_allclose(red[0].get_data_3d()[2], [0, .25])
        plt.close(fig)

    def test_centered_flip(self):
        x = d.flip("centered")
        values, mass = x.table()
        np.testing.assert_array_equal(values, [-1, 1])
        np.testing.assert_allclose(mass, [.5, .5])
        np.testing.assert_allclose([x.mean(), x.var()], [0, 1])
        np.testing.assert_array_equal(((x+1)//2).table()[0], [0, 1])
        average = d.mean([d.flip(f"x.{i}") for i in range(3)])
        np.testing.assert_allclose([average.mean(), average.var()], [0, 1/3], atol=1e-12)

    def test_raw_3d_axes_and_boxes(self):
        import matplotlib.pyplot as plt
        from unittest.mock import patch
        from viz import variance_3d
        with patch.object(plt, "show"):
            fig = variance_3d(d.flip("x"), show=False)
        ax = fig.axes[0]
        self.assertEqual(len(ax.get_xticks()), 0)
        self.assertEqual(len(ax.get_yticks()), 0)
        self.assertFalse(ax.xaxis.line.get_visible())
        self.assertFalse(ax.yaxis.line.get_visible())
        self.assertEqual([t.get_text().strip() for t in ax.texts], ["-1", "1"])
        outlines = [line for line in ax.lines if line.get_color() == "#4c8194"]
        self.assertTrue(all(line.get_alpha() == .3 for line in outlines))
        self.assertEqual(len(ax.get_zticks()), 0)
        self.assertEqual(len(ax.lines), 2*12+2+1)
        red = [line for line in ax.lines if line.get_color() == "#c63737"]
        self.assertEqual(len(red), 1)
        for line in red:
            xs, ys, zs = line.get_data_3d()
            np.testing.assert_allclose(xs, [0, 0])
            np.testing.assert_allclose(ys, [0, 0])
            np.testing.assert_allclose(zs, [0, .5])
        black = [line for line in ax.lines if line.get_color() == "black"]
        self.assertEqual(len(black), 2)
        for value, line in zip([-1, 1], black):
            xs, ys, zs = line.get_data_3d()
            np.testing.assert_allclose(xs, [value, value])
            np.testing.assert_allclose(ys, [value, value])
            np.testing.assert_allclose(zs, [0, .5])
        for line in ax.lines:
            xs, ys, zs = line.get_data_3d()
            self.assertTrue(np.all(np.asarray(xs) >= -1))
            self.assertTrue(np.all(np.asarray(ys) >= -1))
        plt.close(fig)

    def test_joint_and_covariance(self):
        def covariance(rv):
            return (rv-rv.mean()).apply(lambda v: jnp.outer(v, v)).mean()
        x = d.flip("x")
        self.assertFalse(hasattr(x, "cov"))
        y = 2*((x+d.flip("z")) > 0)-1
        coupled, independent = d.joint(x, y), x @ y
        np.testing.assert_allclose(covariance(coupled), [[1, .5], [.5, .75]])
        np.testing.assert_allclose(covariance(independent), [[1, 0], [0, .75]])
        np.testing.assert_allclose(covariance(x @ x), [[1, 0], [0, 1]])
        np.testing.assert_allclose(covariance(d.joint(x, x)), [[1, 1], [1, 1]])
        self.assertEqual(len(independent.flips), 3)
        self.assertEqual(independent.names, ("left.x", "right.x", "right.z"))
        assert_distribution(independent, np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]]),
                            np.array([.375, .125, .375, .125]))
        a = (x+y).given(y == -1)
        np.testing.assert_allclose(covariance(a @ a), np.eye(2)*float(a.var()), atol=1e-12)
        np.testing.assert_allclose(covariance(d.joint(x, y.given(x == 1)))[0, 1], 0)
        def scaled_covariance(scale):
            a = scale*d.flip("a")
            return covariance(d.joint(a, a))[0, 1]
        np.testing.assert_allclose(jax.jit(jax.grad(scaled_covariance))(2.), 4.)

    def test_four_sided_3d_variance(self):
        import matplotlib.pyplot as plt
        from unittest.mock import patch
        from viz import variance_3d
        die = p.four_sides()
        check("four", die, plot=False)
        with patch.object(plt, "show"):
            fig = variance_3d(die, show=False)
        ax = fig.axes[0]
        self.assertEqual(len(ax.get_xticks()), 0)
        self.assertEqual(len(ax.get_yticks()), 0)
        self.assertEqual([t.get_text().strip() for t in ax.texts], ["1", "2", "3", "4"])
        black = [line for line in ax.lines if line.get_color() == "black"]
        self.assertEqual(len(black), 4)
        from mpl_toolkits.mplot3d import proj3d
        _, screen_y, _ = proj3d.proj_transform(np.arange(1, 5), np.arange(1, 5),
                                              np.zeros(4), ax.get_proj())
        np.testing.assert_allclose(screen_y, screen_y[0], atol=1e-12)
        volume = 0.
        for value, line in zip([1, 2, 3, 4], black):
            xs, ys, zs = line.get_data_3d()
            np.testing.assert_allclose(xs, [value, value])
            np.testing.assert_allclose(ys, [value, value])
            np.testing.assert_allclose(zs, [0, .25])
            volume += (xs[-1]-2.5)*(ys[-1]-2.5)*zs[-1]
        np.testing.assert_allclose(volume, die.var())
        np.testing.assert_allclose(volume, 1.25)
        self.assertEqual(ax.get_title(), "")
        plt.close(fig)

    def test_named_flip_provenance(self):
        self.assertFalse(hasattr(d, "vals"))
        x, y = d.flip("x"), d.flip("y")
        self.assertFalse(hasattr(x, "new"))
        self.assertFalse(hasattr(x, "times"))
        self.assertEqual((x+x).names, ("x",))
        z = (2*x+y).apply(lambda v: v*v).given(y == 1)
        self.assertEqual(z.names, ("x", "y"))
        self.assertEqual(z.flips, x.flips+y.flips)
        draws = [2*d.flip(f"x.{i}")+d.flip(f"y.{i}") for i in range(3)]
        average = d.mean(draws)
        self.assertEqual(len(average.flips), 6)
        self.assertEqual(average.names, ("x.0", "y.0", "x.1", "y.1", "x.2", "y.2"))
        self.assertEqual(average.flips, tuple(f for draw in draws for f in draw.flips))
        # Names label fresh draws; reusing the object shares randomness.
        self.assertAlmostEqual(float((x+d.flip("x")).var()), 2.)
        self.assertEqual(len(p.six_sides().flips), 3)
        self.assertEqual(len(p.markov_chain().flips), 6)

    def test_variance_geometry(self):
        from viz import _two_state_vector
        x = d.flip("geometry")
        v = _two_state_vector(x)
        np.testing.assert_allclose(v @ v, x.var())
        with self.assertRaises(ValueError):
            d.mean([])
        self.assertEqual(x.apply(lambda v: 0).flips, x.flips)
        conditioned = d.constant(1).given(x == 1)
        self.assertEqual(conditioned.names, ("geometry",))
        categorical = d.from_logits([0, 1], jnp.zeros(2), name="policy")
        self.assertEqual(categorical.names, ("policy",))
        self.assertEqual(categorical.flips, ())
        z = d.flip("extra")
        y = 2*((x+z) > 0)-1
        np.testing.assert_allclose([x.var(), y.var(), ((x-x.mean())*(y-y.mean())).mean()],
                                   [1, .75, .5])
        np.testing.assert_allclose(d.mean([x, z]).var(), .5)

    def test_functional_mean(self):
        x = d.flip("shared")
        np.testing.assert_allclose(d.mean([x, x]).var(), x.var())
        draws = [p.six_sides(f"die.{i}") for i in range(10)]
        average = d.mean(draws)
        np.testing.assert_allclose(average.mean(), 3.5)
        np.testing.assert_allclose(average.var(), (35/12)/10)
        self.assertEqual(len(average.flips), 30)
        self.assertEqual(average.flips, tuple(f for draw in draws for f in draw.flips))
        with self.assertRaises(ValueError):
            average + draws[0]
        def variance(scale):
            return d.mean([scale*d.flip(f"grad.{i}") for i in range(3)]).var()
        np.testing.assert_allclose(jax.jit(jax.grad(variance))(2.), 4/3)

    def test_new_distributions(self):
        theta = jnp.array([jnp.log(3.), 0.])
        examples = {
            "weighted": p.weighted_die([1, 2, 3, 4, 5, 6]),
            "kl": p.kl_estimate(jnp.arange(1., 7.)/21, jnp.ones(6)/6),
            "markov": p.markov_chain(), "unigram": p.markov_unigram(),
            "reinforce": p.reinforce(theta), "loo": p.reinforce_loo(theta),
        }
        for name, rv in examples.items():
            with self.subTest(name=name):
                check(name, rv, plot=False)

    def test_arbitrary_integer_weights(self):
        for weights in ([0, 3, 0, 1, 2, 0], [0, 0, 1, 0, 0, 0], [1]*6, [3, 7, 2, 1, 0, 4]):
            assert_distribution(p.weighted_die(weights), np.arange(1, 7), np.array(weights)/sum(weights))
        for weights in ([], [0]*6, [-1, 2], [.5, 1.5]):
            with self.assertRaises(ValueError):
                p.weighted_die(weights)

    def test_unigram_variance(self):
        np.testing.assert_allclose(p.markov_chain().var(), 63/256)
        np.testing.assert_allclose(p.markov_unigram().var(), 39/256)

    def test_k3(self):
        prob, ref = jnp.arange(1., 7.)/21, jnp.ones(6)/6
        estimate = p.kl_k3(prob, ref)
        check("k3", estimate, plot=False)
        self.assertGreaterEqual(float(estimate.table()[0].min()), -1e-12)
        self.assertLess(float(estimate.var()), float(p.kl_estimate(prob, ref).var()))
        np.testing.assert_allclose(p.kl_k3(prob, prob).var(), 0, atol=1e-12)
        for a, b in ((prob, ref), (ref, prob)):
            np.testing.assert_allclose(p.kl_k3(a, b).mean(), jnp.sum(a*jnp.log(a/b)), atol=1e-12)

    def test_gradients_against_autodiff(self):
        def objective(theta):
            return jax.nn.softmax(theta)[1]
        for theta in (jnp.array([.4, -.3]), jnp.zeros(2), jnp.array([-1., .7])):
            target = jax.grad(objective)(theta)
            for n in (2, 3):
                np.testing.assert_allclose(d.mean([p.reinforce(theta) for i in range(n)]).mean(), target, atol=1e-10)
                np.testing.assert_allclose(p.reinforce_loo(theta, n).mean(), target, atol=1e-10)
        theta = jnp.array([jnp.log(3.), 0.])
        np.testing.assert_allclose(p.reinforce(theta).mean(), [-.1875, .1875])
        naive = d.mean([p.reinforce(theta) for i in range(3)])
        self.assertLess(float(p.reinforce_loo(theta).var().sum()), float(naive.var().sum()))

    def test_biased_and_wrong_distribution_rejected(self):
        with self.assertRaises(AssertionError):
            check("markov", p.markov_chain()+.1, plot=False)
        # Correct mean alone is insufficient.
        with self.assertRaises(AssertionError):
            check("markov", d.constant(7/16), plot=False)
        with self.assertRaises(AssertionError):
            check("loo", p.reinforce_loo(jnp.array([.4, -.3]))+jnp.array([0., .1]), plot=False)


if __name__ == "__main__":
    unittest.main()
