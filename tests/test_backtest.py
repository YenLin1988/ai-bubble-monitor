import unittest

import numpy as np
import pandas as pd

from bubble_monitor import backtest as bt


def _daily(values, start="2020-01-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="D"), dtype=float)


class DrawdownTests(unittest.TestCase):
    def test_finds_only_drawdowns_beyond_threshold(self):
        prices = _daily([100, 110, 90, 80, 115, 120, 110, 125])
        events = bt.find_drawdown_events(prices, 0.20)
        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertEqual(ev.peak_date, prices.index[1])
        self.assertEqual(ev.trough_date, prices.index[3])
        self.assertEqual(ev.recovery_date, prices.index[4])
        self.assertAlmostEqual(ev.depth, 80 / 110 - 1)

    def test_unrecovered_drawdown_has_no_recovery_date(self):
        events = bt.find_drawdown_events(_daily([100, 70, 75]), 0.20)
        self.assertEqual(len(events), 1)
        self.assertIsNone(events[0].recovery_date)

    def test_forward_max_drawdown_measures_from_running_peak(self):
        prices = pd.Series([100, 120, 90, 130], index=pd.to_datetime(["2020-01-31", "2020-03-31", "2020-06-30", "2021-02-28"]))
        out = bt.forward_max_drawdown(prices, pd.DatetimeIndex(["2020-01-31", "2020-06-30"]), 12)
        self.assertAlmostEqual(out.iloc[0], 90 / 120 - 1)
        self.assertTrue(np.isnan(out.iloc[1]))  # window runs past the last price


class CompositeTests(unittest.TestCase):
    def test_weights_renormalise_over_available_factors(self):
        scores = pd.DataFrame({"A": [100.0, 100.0], "B": [0.0, np.nan]})
        stress, coverage = bt.composite_stress(scores, {"A": 0.25, "B": 0.75})
        self.assertAlmostEqual(stress.iloc[0], 25.0)
        self.assertAlmostEqual(stress.iloc[1], 100.0)
        self.assertAlmostEqual(coverage.iloc[1], 0.25)

    def test_publication_lag_hides_unreleased_unemployment(self):
        index = pd.DatetimeIndex(["2020-01-31", "2020-02-29"])
        unrate = pd.Series([3.5, 9.9], index=pd.to_datetime(["2019-12-01", "2020-01-01"]))
        raw = {k: pd.Series(dtype=float, index=pd.DatetimeIndex([])) for k in [
            "gs10", "fedfunds", "walcl", "wtregen", "rrp", "m2", "usdjpy", "t10y2y", "hy", "baa10y",
            "nfci", "corp_equity", "gdp", "vxn", "vix", "umcsent", "dxy", "oil", "gold", "spx"]}
        raw["unrate"] = unrate
        values = bt.build_factor_values(raw, index)
        # January's rate is released in early February, so January month-end still sees December.
        self.assertEqual(values["unrate"].tolist(), [3.5, 9.9])


class ThresholdTests(unittest.TestCase):
    def test_precision_and_event_recall(self):
        index = pd.date_range("2000-01-31", periods=4, freq="ME")
        stress = pd.Series([70, 70, 20, 20], index=index)
        fwd_dd = pd.Series([-0.30, -0.05, -0.30, -0.05], index=index)
        event = bt.DrawdownEvent(pd.Timestamp("2000-02-15"), pd.Timestamp("2000-05-01"), None, -0.3)
        tbl = bt.threshold_table(stress, fwd_dd, [event], thresholds=[50])
        row = tbl.iloc[0]
        self.assertAlmostEqual(row["precision"], 0.5)
        self.assertAlmostEqual(row["lift"], 1.0)
        self.assertEqual(row["events_caught"], "1/1")


if __name__ == "__main__":
    unittest.main()
