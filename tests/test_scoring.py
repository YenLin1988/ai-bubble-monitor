import datetime
import unittest

from bubble_monitor.scoring import classify_regime, cluster_decomposition, normalize, stress_delta, weighted_stress_score


class ScoringTests(unittest.TestCase):
    def test_normalize_clamps_to_range(self):
        self.assertEqual(normalize(0, 10, 20), 0)
        self.assertEqual(normalize(30, 10, 20), 100)
        self.assertEqual(normalize(15, 10, 20), 50)

    def test_regime_classification(self):
        self.assertEqual(classify_regime(10, 40).name, "擴張期 Expansion")
        self.assertEqual(classify_regime(10, 60).name, "過熱期 Overheating")
        self.assertEqual(classify_regime(-10, 60).name, "泡沫破裂 Bust")
        self.assertEqual(classify_regime(-10, 40).name, "沉澱期 Contraction")

    def test_weighted_score_and_clusters(self):
        risk_scores = {"A": 50, "B": 100}
        weights = {"A": 0.25, "B": 0.75}
        clusters = {"A": "one", "B": "two"}
        self.assertEqual(weighted_stress_score(risk_scores, weights), 87.5)
        result = cluster_decomposition(risk_scores, weights, clusters)
        self.assertEqual(result["one"]["total"], 12.5)
        self.assertEqual(result["two"]["weight_sum"], 0.75)

    def test_stress_delta_uses_last_record_before_cutoff(self):
        now = datetime.datetime(2026, 5, 27, 12, 0)
        history = [
            {"time": "2026-05-26 10:00", "stress_score": 40},
            {"time": "2026-05-26 12:00", "stress_score": 45},
            {"time": "2026-05-27 12:00", "stress_score": 55},
        ]
        self.assertEqual(stress_delta(history, 55, 24, now), 10)


if __name__ == "__main__":
    unittest.main()
