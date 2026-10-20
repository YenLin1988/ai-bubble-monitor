import unittest

import numpy as np
import pandas as pd

from bubble_monitor import overlay as ov


def _sig(levels, near=True, entry=None):
    idx = pd.bdate_range("2020-01-01", periods=len(levels))
    return pd.DataFrame({
        "warning_level": levels, "near_high": near,
        "e_capitulation": entry if entry is not None else [False] * len(levels),
    }, index=idx)


class OverlayTests(unittest.TestCase):
    def test_enter_defensive_and_return_after_calm_days(self):
        exp = ov.warning_overlay(_sig([0, 4, 3, 1, 1, 0, 1, 1, 0]), enter_level=4, hold_days=5)
        self.assertEqual(exp.tolist(), [1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1, 1])

    def test_entry_signal_restores_full_exposure(self):
        exp = ov.warning_overlay(_sig([4, 4, 4, 4], entry=[False, False, True, False]), enter_level=4)
        self.assertEqual(exp.tolist(), [0.5, 0.5, 1, 0.5])

    def test_no_entry_when_not_near_high(self):
        exp = ov.warning_overlay(_sig([5, 5], near=False), enter_level=4)
        self.assertEqual(exp.tolist(), [1, 1])

    def test_combined_exposure_with_entry_floor(self):
        idx = pd.bdate_range("2020-01-01", periods=6)
        trend = pd.Series([1, 1, 0, 0, 0, 1], index=idx, dtype=float)
        warn = pd.Series([1, 0.5, 0.5, 1, 1, 1], index=idx, dtype=float)
        entry = pd.Series([False, False, False, True, False, False], index=idx)
        self.assertEqual(ov.combined_exposure(trend, warn).tolist(), [1, 0.5, 0, 0, 0, 1])
        # entry while the trend is out lifts exposure to 50% until the trend turns back in
        self.assertEqual(ov.combined_exposure(trend, warn, entry).tolist(), [1, 0.5, 0, 0.5, 0.5, 1])

    def test_simulate_trades_one_day_after_signal(self):
        idx = pd.bdate_range("2020-01-01", periods=5)
        price = pd.Series([100, 100, 100, 110, 121], index=idx, dtype=float)
        exposure = pd.Series([1, 0, 0, 0, 0], index=idx, dtype=float)  # decided at close of day 1
        sim = ov.simulate(price, exposure, cost_per_trade=0.0)
        # Exit traded at close of day 2, so day-3 and day-4 gains are missed; nothing before that.
        self.assertEqual(sim["exposure"].tolist(), [1, 1, 1, 0, 0])
        self.assertAlmostEqual(sim["equity"].iloc[-1], 1.0)

    def test_price_only_entry_after_vol_spike_fades(self):
        rng = np.random.default_rng(0)
        calm = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.005, 300)))
        crash = calm[-1] * np.exp(np.cumsum(rng.normal(-0.01, 0.04, 30)))
        quiet = crash[-1] * np.exp(np.cumsum(rng.normal(0, 0.005, 30)))
        px = pd.Series(np.r_[calm, crash, quiet], index=pd.bdate_range("2019-01-01", periods=360))
        sig = ov.price_only_signals(px)
        self.assertTrue(sig["e_entry"].iloc[330:].any())
        self.assertFalse(sig["e_entry"].iloc[:300].any())


if __name__ == "__main__":
    unittest.main()
