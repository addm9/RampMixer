import os
os.environ.setdefault('NUMBA_NUM_THREADS', '4')
import unittest
import numpy as np
import torch
from rampmixer.cam import ANOVAResponse, lag_states, causal_trend
from rampmixer.crd import decompose
from rampmixer.ttm import RampMixer
from rampmixer.data import sample_indices
from rampmixer.pipeline import metrics


class ImplementationTests(unittest.TestCase):
    def test_causal_sg(self):
        rng = np.random.default_rng(11)
        p = rng.normal(size=80)
        baseline = causal_trend(p)
        p[40:] += 10000
        np.testing.assert_array_equal(baseline[:40], causal_trend(p)[:40])

    def test_window_lags_do_not_reach_outside(self):
        x = np.arange(24.).reshape(12, 2)
        states = lag_states(x)
        np.testing.assert_array_equal(states[0], np.repeat(x[0, :, None], 4, axis=1))
        np.testing.assert_array_equal(states[3, :, 3], x[0])

    def test_anova_effect_allocation(self):
        rng = np.random.default_rng(4)
        env = rng.normal(size=(300, 3))
        env[:, 1] = env[:, 0]*.8+env[:, 1]*.2
        s = lag_states(env)
        response = ANOVAResponse().fit(s, env[:, 0]+env[:, 0]*env[:, 1])
        main, pair = response.effects(s)
        strength = np.abs(main).copy()
        for k, (i, j) in enumerate(response.pairs):
            w = (abs(main[:, i])+1e-6)/(abs(main[:, i])+abs(main[:, j])+2e-6)
            strength[:, i] += w*abs(pair[:, k])
            strength[:, j] += (1-w)*abs(pair[:, k])
        expected = strength/(strength.sum(1, keepdims=True)+1e-6)
        np.testing.assert_allclose(expected, response.attribute(s))
        np.testing.assert_allclose(strength.sum(1), abs(main).sum(1)+abs(pair).sum(1))
        x, _ = response.design(response.bases(s))
        np.testing.assert_allclose(main.sum(1)+pair.sum(1)+response.model.intercept_, response.model.predict(x), atol=1e-10)

    def test_crd_reconstruction_and_descent(self):
        t = np.arange(24.)
        power = np.stack([2+np.sin(t*.2), 1+np.cos(t*.5)])
        weights = np.empty((2, 24, 2))
        weights[:, :, 0] = .5+.2*np.sin(t*.3)
        weights[:, :, 1] = 1-weights[:, :, 0]
        modes, centers, stats = decompose(power, weights, [2000., 2200.], max_iter=300)
        np.testing.assert_allclose(modes.sum(-1), power, atol=1e-10)
        self.assertTrue(np.isfinite(modes).all())
        self.assertTrue(np.all(stats[:, 5] <= stats[:, 4]+1e-8))
        self.assertTrue(np.all((centers >= 0)&(centers <= np.pi)))

    def test_mode_encoder_isolation(self):
        torch.manual_seed(5)
        model = RampMixer(4, length=12)
        modes = torch.randn(2, 12, 4)
        desc = torch.randn(2, 4, 2)
        before = model.encode(modes, desc)
        modes[:, :, 0] += 50
        after = model.encode(modes, desc)
        torch.testing.assert_close(before[:, 1:], after[:, 1:])
        prediction = model(modes, desc, torch.randn(2, 12, 3))
        self.assertEqual(tuple(prediction.shape), (2, 12))
        prediction.square().mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_window_split_and_future_isolation(self):
        times = np.datetime64('2020-01-01T08:00')+np.arange(50)*np.timedelta64(15, 'm')
        split = np.array(['train']*25+['validation']*10+['test']*15)
        origins = sample_indices(times, split, length=6, horizon=3, minutes=15)
        for i in origins:
            self.assertEqual(len(set(split[i:i+3])), 1)
        # Independent frontend calls on the SAME prefix cannot see appended future data.
        rng = np.random.default_rng(6)
        s = lag_states(rng.normal(size=(100, 2)))
        response = ANOVAResponse().fit(s, rng.normal(size=100))
        env = rng.normal(size=(50, 2))
        first = response.attribute(lag_states(env[5:17]))
        env[17:] += 1e6
        np.testing.assert_array_equal(first, response.attribute(lag_states(env[5:17])))

    def test_metrics_identity(self):
        x = np.arange(1., 13.)
        result = metrics(x, x)
        self.assertEqual(result['RMSE'], 0.)
        self.assertEqual(result['R2'], 1.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
