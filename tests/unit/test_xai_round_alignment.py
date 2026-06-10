from __future__ import annotations

import unittest

from explainability.shap_explainer import EmbeddingShapExplainer


class XaiRoundAlignmentTests(unittest.TestCase):
    def test_collect_saved_round_summaries_uses_saved_model_round(self) -> None:
        explainer = EmbeddingShapExplainer.__new__(EmbeddingShapExplainer)
        metrics_payload = {
            "saved_model_round": 1,
            "rounds": [
                {
                    "round": 1,
                    "client_metrics": [
                        {
                            "client_id": "client_a",
                            "num_examples": 10,
                            "shap_summary": {
                                "feature_names": ["f1"],
                                "mean_abs_shap": {"f1": 0.25},
                            },
                            "rule_summary": {
                                "rules": [
                                    {
                                        "signature": "f1|>=|1",
                                        "feature": "f1",
                                        "operator": ">=",
                                        "value": 1,
                                        "description": "f1 >= 1",
                                        "importance": 0.25,
                                        "support": 0.5,
                                    }
                                ]
                            },
                        }
                    ],
                },
                {
                    "round": 2,
                    "client_metrics": [
                        {
                            "client_id": "client_a",
                            "num_examples": 10,
                            "shap_summary": {
                                "feature_names": ["f1"],
                                "mean_abs_shap": {"f1": 0.9},
                            },
                            "rule_summary": {
                                "rules": [
                                    {
                                        "signature": "f1|>=|9",
                                        "feature": "f1",
                                        "operator": ">=",
                                        "value": 9,
                                        "description": "f1 >= 9",
                                        "importance": 0.9,
                                        "support": 0.9,
                                    }
                                ]
                            },
                        }
                    ],
                },
            ],
        }

        local_summaries, local_rule_entries, source_round = (
            explainer._collect_saved_round_summaries(metrics_payload)
        )

        self.assertEqual(source_round, 1)
        self.assertEqual(local_summaries[0]["mean_abs_shap"]["f1"], 0.25)
        self.assertEqual(local_rule_entries[0]["rules"][0]["signature"], "f1|>=|1")


if __name__ == "__main__":
    unittest.main()
