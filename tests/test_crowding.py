import unittest

import numpy as np
import pandas as pd

from bubble_monitor import crowding as cr


class CrowdingTests(unittest.TestCase):
    def test_z1_value_only_visible_after_publication(self):
        index = pd.date_range("2020-01-31", "2020-07-31", freq="ME")
        q = pd.Series([40.0, 45.0], index=pd.to_datetime(["2019-10-01", "2020-01-01"]))
        raw = {
            "household_alloc": q, "corp_equity": q * 1000, "total_debt": q * 2000, "mmf": q * 100,
        }
        m = cr.build_measures(raw, index)
        # Q1 2020 (dated 2020-01-01) publishes ~2020-06-10, so May still shows Q4 2019.
        self.assertEqual(m.loc["2020-05-31", "household_alloc"], 40.0)
        self.assertEqual(m.loc["2020-06-30", "household_alloc"], 45.0)
        self.assertTrue(m["margin_yoy"].isna().all())

    def test_expanding_percentile_has_no_look_ahead(self):
        s = pd.Series([1.0, 3.0, 2.0, 5.0, 4.0])
        pct = cr.expanding_percentile(s, min_periods=1)
        self.assertEqual(pct.tolist(), [1.0, 1.0, 2 / 3, 1.0, 0.8])

    def test_rank_corr_uses_non_overlapping_samples(self):
        x = pd.Series(np.arange(48, dtype=float))
        c, n = cr.rank_corr(x, -x, step=12)
        self.assertEqual(n, 4)
        self.assertTrue(np.isnan(c))  # fewer than 8 samples
        c, n = cr.rank_corr(x, -x, step=1)
        self.assertAlmostEqual(c, -1.0)

    def test_forward_monthly_drawdown(self):
        spx = pd.Series([100, 120, 90, 130], index=pd.date_range("2020-01-31", periods=4, freq="ME"), dtype=float)
        fwd = cr.forward_monthly(spx, horizons=(3,))
        self.assertAlmostEqual(fwd["ret_3"].iloc[0], 0.30)
        self.assertAlmostEqual(fwd["maxdd_3"].iloc[0], 90 / 120 - 1)


if __name__ == "__main__":
    unittest.main()
