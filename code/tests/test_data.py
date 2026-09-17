import os
os.environ.setdefault("NUMBA_NUM_THREADS", "4")
import json
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from rampmixer.data import (validate_frame, chronological_split, sample_indices,
                           select_windows, check_target_disjoint)
from rampmixer.pipeline import fit_frontend, frontend
from rampmixer.crd import decompose

ROOT = Path(__file__).resolve().parents[1]


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = json.loads((ROOT/"configs/synthetic_demo.json").read_text())
        cls.frame = validate_frame(pd.read_csv(ROOT.parent/cls.cfg["data"]), cls.cfg["drivers"], 15)

    def test_validation_rejects_missing_values_duplicates_and_off_grid_times(self):
        d = self.frame.copy()
        d.loc[0, "power"] = np.nan
        with self.assertRaises(ValueError):
            validate_frame(d, self.cfg["drivers"], 15)
        d = pd.concat([self.frame, self.frame.iloc[:1]])
        with self.assertRaises(ValueError):
            validate_frame(d, self.cfg["drivers"], 15)
        d = self.frame.copy()
        d.loc[0, "timestamp"] += pd.Timedelta(minutes=1)
        with self.assertRaises(ValueError):
            validate_frame(d, self.cfg["drivers"], 15)

    def test_target_sets_are_disjoint(self):
        times = self.frame.timestamp.to_numpy()
        split = chronological_split(times)
        origins = sample_indices(times, split, 96, 12, 15)
        selected = select_windows(origins, split, [16, 8, 8])
        counts = check_target_disjoint(times, selected, split, 12)
        self.assertTrue(all(v > 0 for v in counts.values()))
        for i in selected:
            self.assertLess(times[i-1], times[i])
            self.assertEqual(times[i].astype("datetime64[D]"), times[i+11].astype("datetime64[D]"))

    def test_missing_intraday_point_and_missing_day_break_windows(self):
        for gap in ([100], list(range(108, 144))):
            d = self.frame.drop(gap).reset_index(drop=True)
            times = d.timestamp.to_numpy()
            split = chronological_split(times)
            origins = sample_indices(times, split, 96, 12, 15)
            for i in origins:
                segment = times[i-96:i+12]
                elapsed = np.diff(segment).astype("timedelta64[m]").astype(int)
                self.assertTrue(np.all(np.isin(elapsed, [15, 15*60+15])))

    def test_test_data_cannot_change_fitted_frontend(self):
        times = self.frame.timestamp.to_numpy()
        split = chronological_split(times)
        a, alpha_a, scale_a, _ = fit_frontend(self.frame, split, self.cfg)
        changed = self.frame.copy()
        changed.loc[split != "train", ["power"]+self.cfg["drivers"]] += 1e6
        b, alpha_b, scale_b, _ = fit_frontend(changed, split, self.cfg)
        np.testing.assert_array_equal(a.model.coef_, b.model.coef_)
        np.testing.assert_array_equal(alpha_a, alpha_b)
        self.assertEqual(scale_a, scale_b)

    def test_independent_window_decomposition_and_input_validation(self):
        rng = np.random.default_rng(9)
        power = rng.random((2, 12))
        weights = np.full((2, 12, 2), .5)
        together, _, _ = decompose(power, weights, [2000., 2100.], max_iter=50)
        single, _, _ = decompose(power[:1], weights[:1], [2000., 2100.], max_iter=50)
        np.testing.assert_allclose(together[:1], single, atol=1e-10)
        with self.assertRaises(ValueError):
            decompose(power, weights, [2000., 2100.], max_iter=0)
        with self.assertRaises(ValueError):
            decompose(power, -weights, [2000., 2100.])


if __name__ == "__main__":
    unittest.main(verbosity=2)
