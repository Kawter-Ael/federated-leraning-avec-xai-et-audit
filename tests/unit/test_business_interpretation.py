from __future__ import annotations

import unittest

from shared.business_interpretation import (
    build_client_result_payload,
    build_rule_sentence,
    feature_to_label,
    prediction_to_diagnosis,
    prediction_to_risk_level,
)


class BusinessInterpretationTests(unittest.TestCase):
    def test_prediction_labels_are_business_friendly(self) -> None:
        self.assertEqual(prediction_to_diagnosis(0), "Negative outcome")
        self.assertEqual(prediction_to_diagnosis(1), "Positive outcome")
        self.assertEqual(
            prediction_to_diagnosis(
                1,
                config={
                    "project": {
                        "positive_label": "TB positive",
                        "negative_label": "TB negative",
                    }
                },
            ),
            "TB positive",
        )
        self.assertEqual(prediction_to_risk_level(0.20), "Low risk")
        self.assertEqual(prediction_to_risk_level(0.50), "Medium risk")
        self.assertEqual(prediction_to_risk_level(0.90), "High risk")

    def test_build_client_result_payload_formats_if_then_rules(self) -> None:
        payload = build_client_result_payload(
            probability=0.81,
            threshold=0.50,
            explanation_rows=[
                {"feature": "Glucose", "feature_value": 182.0, "shap_value": 0.42},
                {"feature": "BMI", "feature_value": 34.5, "shap_value": 0.18},
            ],
        )
        self.assertIn("81.0%", payload["diagnosis_label"])
        self.assertIn("Positive outcome", payload["diagnosis_label"])
        self.assertIn("High risk", payload["risk_level"])
        self.assertIn("81.0%", payload["risk_level"])
        self.assertTrue(payload["if_then_rules"])
        self.assertIn("glucose", payload["if_then_rules"][0].lower())

    def test_rule_sentence_uses_business_labels(self) -> None:
        sentence = build_rule_sentence(
            {"feature": "BloodPressure", "operator": ">=", "value": 85}
        )
        self.assertIn(feature_to_label("BloodPressure"), sentence)


if __name__ == "__main__":
    unittest.main()
