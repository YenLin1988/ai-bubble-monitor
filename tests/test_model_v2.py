import unittest

import numpy as np
import pandas as pd

from bubble_monitor import model_v2 as m


def _monthly(values):
    return pd.Series(values, index=pd.date_range("2000-01-31", periods=len(values), freq="ME"), dtype=float)


class ModelV2Tests(unittest.TestCase):
    def test_curve_uninversion_phases(self):
        curve = _monthly([1.0, 0.5, -0.2, -0.4, 0.1, 0.6])
        self.assertEqual(m.curve_uninversion(curve).tolist(), [0, 0, 50, 50, 100, 100])

    def test_curve_uninversion_memory_expires(self):
        curve = _monthly([-0.1] + [0.5] * 4)
        self.assertEqual(m.curve_uninversion(curve, memory_months=2).tolist(), [50, 100, 0, 0, 0])

    def test_trailing_percentile_only_uses_past(self):
        s = _monthly([1, 2, 3, 4, 100])
        pct = m.trailing_percentile(s, window=3, min_periods=3)
        self.assertTrue(np.isnan(pct.iloc[1]))
        self.assertEqual(pct.iloc[3], 1.0)
        # Changing a later value must not change earlier percentiles.
        s2 = s.copy(); s2.iloc[-1] = -100
        self.assertEqual(m.trailing_percentile(s2, window=3, min_periods=3).iloc[3], 1.0)

    def test_extreme_score_ignores_below_median(self):
        self.assertEqual(m.extreme_score(pd.Series([0.3, 0.5, 0.95, 1.0])).tolist(), [0, 0, 100, 100])

    def test_sahm_gap(self):
        unrate = _monthly([4.0] * 15 + [4.6, 4.6, 4.6])
        self.assertAlmostEqual(m.sahm_gap(unrate).iloc[-1], 0.6)

    def test_weights_sum_to_one_and_lagging_factors_halved(self):
        self.assertAlmostEqual(sum(m.V2_WEIGHTS.values()), 1.0)
        self.assertAlmostEqual(m.V2_WEIGHTS["Sahm Rule"] * 2, m.V2_WEIGHTS["Breadth"])


if __name__ == "__main__":
    unittest.main()
