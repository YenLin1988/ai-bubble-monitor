import unittest

from bubble_monitor.alerts import evaluate_alerts


class AlertRuleTests(unittest.TestCase):
    def test_critical_stress_alert(self):
        alerts = evaluate_alerts({"stress_score": 80})
        self.assertTrue(any(alert.key == "stress-danger" for alert in alerts))

    def test_telegram_is_not_a_channel(self):
        alerts = evaluate_alerts({"stress_score": 30, "regime": "擴張期 Expansion"})
        self.assertEqual(alerts, [])

    def test_macro_rule_alerts(self):
        alerts = evaluate_alerts({
            "nfci": 0.3,
            "m2_yoy": -3,
            "consumer_sentiment": 50,
            "buffett_indicator": 210,
        })
        keys = {alert.key for alert in alerts}
        self.assertIn("nfci-tightening", keys)
        self.assertIn("m2-contraction", keys)
        self.assertIn("consumer-sentiment-recession-watch", keys)
        self.assertIn("buffett-extreme", keys)


if __name__ == "__main__":
    unittest.main()
