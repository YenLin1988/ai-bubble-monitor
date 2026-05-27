import tempfile
import unittest
from pathlib import Path

from bubble_monitor.storage import HistoryStore


class HistoryStoreTests(unittest.TestCase):
    def test_upsert_and_load_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = HistoryStore(Path(tmp) / "history.sqlite3")
            store.upsert_record({"time": "2026-05-27 12:00", "stress_score": 50})
            store.upsert_record({"time": "2026-05-27 12:00", "stress_score": 55})
            records = store.load_history()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["stress_score"], 55)


if __name__ == "__main__":
    unittest.main()
