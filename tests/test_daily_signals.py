import unittest

import numpy as np
import pandas as pd

from bubble_monitor import daily_signals as ds


def _frame(spx, spread=None, rut_rel=0.4, vix=15.0, vix3m=18.0, curve=1.0):
    idx = pd.bdate_range("2020-01-01", periods=len(spx))
    spx = pd.Series(spx, index=idx, dtype=float)
    return pd.DataFrame({
        "spx": spx,
        "spread": pd.Series(spread if spread is not None else 3.0, index=idx, dtype=float),
        "rut": spx * rut_rel, "vix": vix, "vix3m": vix3m, "curve": curve,
    }, index=idx)


class WarningTests(unittest.TestCase):
    def test_calm_uptrend_has_no_warning(self):
        sig = ds.compute_signals(_frame(np.linspace(100, 130, 300)))
        self.assertEqual(int(sig["warning_level"].iloc[-1]), 0)

    def test_credit_widening_and_trend_break(self):
        n = 300
        spx = np.r_[np.linspace(100, 130, n - 10), np.linspace(129, 125, 10)]
        spread = np.r_[np.full(n - 20, 3.0), np.linspace(3.0, 3.6, 20)]
        sig = ds.compute_signals(_frame(spx, spread=spread))
        last = sig.iloc[-1]
        self.assertTrue(last["w_credit"])
        self.assertTrue(last["w_trend"])
        self.assertFalse(last["w_vol"])
        self.assertEqual(int(last["warning_level"]), 2)

    def test_target_index_drives_drawdown_and_trend_only(self):
        n = 300
        spx = np.linspace(100, 130, n)
        spread = np.r_[np.full(n - 20, 3.0), np.linspace(3.0, 3.6, 20)]
        d = _frame(spx, spread=spread)
        tech = pd.Series(np.r_[np.linspace(100, 160, n - 20), np.linspace(160, 130, 20)], index=d.index)
        base, on_tech = ds.compute_signals(d), ds.compute_signals(d, tech)
        self.assertAlmostEqual(on_tech["dd_1y"].iloc[-1], 130 / 160 - 1)
        self.assertTrue(on_tech["w_trend"].iloc[-1])
        self.assertFalse(base["w_trend"].iloc[-1])
        self.assertEqual(base["w_credit"].iloc[-1], on_tech["w_credit"].iloc[-1])  # market-wide input

    def test_curve_condition_needs_prior_inversion(self):
        curve = np.r_[np.full(100, -0.3), np.full(200, 0.4)]
        sig = ds.compute_signals(_frame(np.linspace(100, 130, 300), curve=curve))
        self.assertTrue(sig["w_curve"].iloc[-1])
        self.assertFalse(ds.compute_signals(_frame(np.linspace(100, 130, 300)))["w_curve"].iloc[-1])


class EntryTests(unittest.TestCase):
    def test_capitulation_after_vix_spike_recedes(self):
        n = 300
        spx = np.r_[np.full(250, 100.0), np.linspace(100, 80, 40), np.full(10, 81.0)]
        vix = np.r_[np.full(285, 15.0), np.full(5, 45.0), np.full(10, 30.0)]
        sig = ds.compute_signals(_frame(spx, vix=vix))
        self.assertTrue(sig["e_capitulation"].iloc[-1])
        self.assertFalse(sig["e_capitulation"].iloc[287])  # VIX still at the spike

    def test_combo_needs_two_signals_within_20_days(self):
        n = 300
        spx = np.r_[np.full(250, 100.0), np.linspace(100, 80, 40), np.full(10, 81.0)]
        vix = np.r_[np.full(285, 15.0), np.full(5, 45.0), np.full(10, 30.0)]
        # VIX term structure inverted during the spike, back to contango with the capitulation.
        vix3m = np.r_[np.full(285, 18.0), np.full(5, 40.0), np.full(10, 33.0)]
        sig = ds.compute_signals(_frame(spx, vix=vix, vix3m=vix3m))
        self.assertTrue(sig["e_vix_ts_flip"].iloc[290])
        self.assertTrue(sig["e_combo"].any())
        alone = ds.compute_signals(_frame(spx, vix=vix))  # no term-structure flip
        self.assertFalse(alone["e_combo"].any())


class PositionTests(unittest.TestCase):
    def test_trend_in_current_month_keeps_last_month_end_decision(self):
        # Rising for a year, then a crash in the final (incomplete) month.
        idx = pd.bdate_range("2019-01-01", "2020-03-20")
        spx = pd.Series(np.linspace(100, 150, len(idx)), index=idx)
        spx.loc["2020-03-01":] = 60.0
        sig = ds.compute_signals(_frame(spx.values))
        sig.index = idx
        pos = ds.position_frame(spx, sig)
        self.assertEqual(pos["trend"].iloc[-1], 1.0)  # March not finished yet
        self.assertLess(pos.attrs["price"] / pos.attrs["ma10_provisional"], 1)  # but the preview shows it below


class ExitSignalTests(unittest.TestCase):
    def _crash_into(self, end):
        idx = pd.bdate_range("2019-01-01", end)
        spx = pd.Series(np.linspace(100, 150, len(idx)), index=idx)
        spx.loc[pd.Timestamp(end).replace(day=1):] = 60.0
        sig = ds.compute_signals(_frame(spx.values))
        sig.index = idx
        return spx, sig

    def test_last_trading_day_detection(self):
        self.assertTrue(ds.is_last_trading_day_of_month(pd.Timestamp("2026-10-30")))   # Friday
        self.assertFalse(ds.is_last_trading_day_of_month(pd.Timestamp("2026-10-29")))
        self.assertTrue(ds.is_last_trading_day_of_month(pd.Timestamp("2026-02-27")))   # Fri, 28th is Sat

    def test_exit_fires_on_month_end_close_only(self):
        spx, sig = self._crash_into("2020-03-31")
        pos = ds.position_frame(spx, sig)
        self.assertEqual(pos["trend"].iloc[-1], 0.0)
        kinds = [e.kind for e in ds.position_events(pos, "2020-03-31")]
        self.assertIn("exit", kinds)
        mid = ds.position_frame(spx.loc[:"2020-03-20"], sig.loc[:"2020-03-20"])
        self.assertEqual(mid["trend"].iloc[-1], 1.0)
        self.assertNotIn("exit", [e.kind for e in ds.position_events(mid, "2020-03-20")])

    def test_exit_watch_when_price_dips_below_average_mid_month(self):
        spx, sig = self._crash_into("2020-03-31")
        first_dip = ds.position_frame(spx.loc[:"2020-03-02"], sig.loc[:"2020-03-02"])
        self.assertLess(first_dip["ma10_gap"].iloc[-1], 0)
        self.assertEqual([e.kind for e in ds.position_events(first_dip, "2020-03-02")], ["exit_watch"])

    def test_exit_stats_counts_saved_and_whipsaws(self):
        idx = pd.bdate_range("2015-01-01", "2020-12-31")
        p = pd.Series(100.0, index=idx)
        p.loc["2015-01-01":"2016-12-31"] = np.linspace(100, 200, len(p.loc["2015-01-01":"2016-12-31"]))
        p.loc["2017-01-01":"2018-06-30"] = np.linspace(200, 80, len(p.loc["2017-01-01":"2018-06-30"]))
        p.loc["2018-07-01":] = np.linspace(80, 220, len(p.loc["2018-07-01":]))
        st = ds.exit_stats(p)
        self.assertGreaterEqual(st["exits"], 1)
        self.assertEqual(st["saved"], st["closed"])  # bought back lower after the crash
        self.assertGreaterEqual(st["big_drops_avoided"], 1)


class EventTests(unittest.TestCase):
    def _sig(self, levels):
        idx = pd.bdate_range("2020-01-01", periods=len(levels))
        sig = pd.DataFrame({"warning_level": levels, "near_high": True, "dd_1y": -0.02, "vix": 20.0}, index=idx)
        for c in ds.WARNING_CONDITIONS:
            sig[f"w_{c}"] = False
        for k in ds.ENTRY_SIGNALS:
            sig[f"e_{k}"] = False
        fwd = pd.DataFrame({"maxdd_126": -0.05, "ret_126": 0.05, "ret_252": 0.1, "lowest_126": -0.02}, index=idx)
        return sig, fwd

    def test_raise_once_then_hold(self):
        sig, fwd = self._sig([0] * 10 + [3, 1, 3])
        self.assertEqual([e.kind for e in ds.todays_events(sig.iloc[:11], fwd.iloc[:11])], ["warning_up"])
        self.assertEqual(ds.todays_events(sig.iloc[:12], fwd.iloc[:12]), [])
        self.assertEqual(ds.todays_events(sig, fwd), [])  # back to 3 within the hold window

    def test_clear_after_hold_days_of_normal(self):
        sig, fwd = self._sig([0] * 10 + [3] + [1] * 5)
        self.assertEqual([e.kind for e in ds.todays_events(sig, fwd)], ["warning_clear"])
        self.assertEqual(ds.todays_events(sig.iloc[:15], fwd.iloc[:15]), [])

    def test_level_below_minimum_is_silent(self):
        sig, fwd = self._sig([0] * 10 + [2])
        self.assertEqual(ds.todays_events(sig, fwd), [])
        self.assertEqual(len(ds.todays_events(sig, fwd, min_warning_level=2)), 1)

    def test_episode_starts_merges_close_runs(self):
        mask = pd.Series([True, False, True] + [False] * 70 + [True], index=pd.bdate_range("2020-01-01", periods=74))
        self.assertEqual(len(ds.episode_starts(mask, gap_days=60)), 2)


if __name__ == "__main__":
    unittest.main()
