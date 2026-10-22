import datetime as dt
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from bubble_monitor import daily_signals as ds
from bubble_monitor import data_sources, health, signal_log


def _series(end: str, n: int = 30) -> pd.Series:
    return pd.Series(np.arange(n, dtype=float), index=pd.bdate_range(end=end, periods=n))


class HealthTests(unittest.TestCase):
    def test_reports_failures_and_stale_series_without_raising(self):
        def fake_yahoo(ticker):
            if ticker == "^BAD":
                raise ConnectionError("timeout")
            return _series("2026-10-19")

        with mock.patch.object(health, "load_yahoo", fake_yahoo), \
                mock.patch.object(health, "load_fred", lambda sid: _series("2026-09-01")):
            raw, problems = health.load_with_health({"hy": "HY"}, {"spx": "^GSPC", "bad": "^BAD"}, None, 0,
                                                    today=dt.date(2026, 10, 20))
        self.assertIn("spx", raw)
        self.assertNotIn("bad", raw)
        self.assertIn("hy", raw)  # stale data is still returned, only flagged
        self.assertEqual(len(problems), 2)
        self.assertTrue(any("^BAD" in p and "下載失敗" in p for p in problems))
        self.assertTrue(any("HY" in p and "沒更新" in p for p in problems))


    def test_falls_back_to_old_cache_when_download_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            _series("2026-10-16").to_frame("baa10y").to_csv(cache / "baa10y.csv")
            old = dt.datetime(2026, 10, 18).timestamp()
            import os
            os.utime(cache / "baa10y.csv", (old, old))

            def down(sid):
                raise data_sources.FredUnavailable("FRED 回傳網頁而不是資料（Site Maintenance）")

            with mock.patch.object(health, "load_fred", down):
                raw, problems = health.load_with_health({"baa10y": "BAA10Y"}, {}, cache, 3600,
                                                        today=dt.date(2026, 10, 22))
        self.assertIn("baa10y", raw)
        self.assertEqual(raw["baa10y"].index[-1], pd.Timestamp("2026-10-16"))
        self.assertTrue(any("改用 2026-10-16 的舊資料" in p and "Maintenance" in p for p in problems))

    def test_fred_maintenance_page_is_reported_clearly(self):
        page = b"<!doctype html>\n<html><head><title>Site Maintenance | St. Louis Fed</title></head></html>"

        class Resp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return page

        with mock.patch("urllib.request.urlopen", return_value=Resp()):
            with self.assertRaises(data_sources.FredUnavailable) as ctx:
                data_sources.load_fred("BAA10Y")
        self.assertIn("Site Maintenance", str(ctx.exception))

    def test_signals_compute_without_credit_data(self):
        idx = pd.bdate_range("2025-01-01", periods=300)
        raw = {"spx": pd.Series(np.linspace(100, 130, 300), index=idx),
               "vix_daily": pd.Series(15.0, index=idx), "vix3m_daily": pd.Series(18.0, index=idx)}
        sig = ds.compute_signals(ds.build_daily_frame(raw))
        self.assertFalse(sig["w_credit"].any())
        self.assertEqual(int(sig["warning_inputs"].iloc[-1]), 2)  # only vol and trend are known


class RetryMarkerTests(unittest.TestCase):
    def test_marker_keeps_first_failure_time_and_clears(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import daily_alert as da

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(da, "PENDING_PATH", Path(tmp) / "p.json"):
            self.assertIsNone(da.read_pending())
            da.write_pending(dt.datetime(2026, 10, 22, 1, 0, tzinfo=dt.timezone.utc), ["FRED down"])
            da.write_pending(dt.datetime(2026, 10, 22, 3, 30, tzinfo=dt.timezone.utc), ["FRED still down"])
            marker = da.read_pending()
            self.assertEqual(marker["since"], "2026-10-22 01:00 UTC")  # first failure, not the latest retry
            self.assertEqual(marker["problems"], ["FRED still down"])
            da.clear_pending()
            self.assertIsNone(da.read_pending())
            da.clear_pending()  # clearing twice is fine


class SignalLogTests(unittest.TestCase):
    def _sig(self, levels, dd=-0.02, entry=None):
        idx = pd.bdate_range("2026-01-01", periods=len(levels))
        sig = pd.DataFrame({"warning_level": levels, "dd_1y": dd, "near_high": True, "vix": 20.0}, index=idx)
        for c in ds.WARNING_CONDITIONS:
            sig[f"w_{c}"] = False
        for k in ds.ENTRY_SIGNALS:
            sig[f"e_{k}"] = False
        if entry is not None:
            sig["e_capitulation"] = entry
        pos = pd.DataFrame({"trend": 1.0, "target": 1.0}, index=idx)
        return sig, pos

    def test_append_replaces_same_day_and_keeps_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.csv"
            sig, pos = self._sig([0, 3])
            signal_log.append_row(path, signal_log.make_row("ndx", sig.iloc[:1], pos.iloc[:1], 100, 0))
            signal_log.append_row(path, signal_log.make_row("ndx", sig, pos, 101, 0))
            log = signal_log.append_row(path, signal_log.make_row("ndx", sig, pos, 102, 1))  # rerun same day
            self.assertEqual(len(log), 2)
            self.assertEqual(log["price"].tolist(), [100, 102])
            self.assertEqual(log["data_problems"].tolist(), [0, 1])

    def test_scorecard_grades_warning_against_later_prices(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.csv"
            sig, pos = self._sig([0, 0, 4])
            for i in range(1, 4):
                log = signal_log.append_row(path, signal_log.make_row("ndx", sig.iloc[:i], pos.iloc[:i], 100, 0))
        idx = pd.bdate_range("2026-01-01", periods=200)
        prices = pd.Series(100.0, index=idx)
        prices.iloc[10:20] = 85.0  # -15% after the warning
        card = signal_log.scorecard(log, "ndx", prices)
        self.assertEqual(len(card["warnings"]), 1)
        w = card["warnings"][0]
        self.assertTrue(w["hit10"])
        self.assertTrue(w["complete"])
        self.assertIn("1 次之後跌 ≥10%", signal_log.scorecard_text(card, "那斯達克 100"))

    def test_empty_log_text(self):
        card = signal_log.scorecard(pd.DataFrame(columns=signal_log.COLUMNS), "ndx", _series("2026-10-19"))
        self.assertIn("還沒有紀錄", signal_log.scorecard_text(card, "那斯達克 100"))


if __name__ == "__main__":
    unittest.main()
