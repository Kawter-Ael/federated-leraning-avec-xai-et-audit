from __future__ import annotations

import os
import unittest

from shared.auth_store import (
    authenticate_client_user,
    create_client_user,
    list_case_history,
    list_user_runs,
    reset_auth_store_cache,
    save_case_history,
    upsert_user_run,
)


class ClientAuthTests(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["MONGODB_URI"] = "mongomock://localhost"
        os.environ["MONGODB_DB_NAME"] = "ensaj_test_auth"
        reset_auth_store_cache()

    def tearDown(self) -> None:
        reset_auth_store_cache()

    def test_create_and_authenticate_client_user(self) -> None:
        ok, message = create_client_user("client_demo", "password123")
        self.assertTrue(ok, msg=message)

        authenticated, user, auth_message = authenticate_client_user("client_demo", "password123")
        self.assertTrue(authenticated, msg=auth_message)
        self.assertIsNotNone(user)
        self.assertEqual(user["username"], "client_demo")
        self.assertEqual(user["role"], "client")

    def test_duplicate_username_is_rejected(self) -> None:
        ok, _ = create_client_user("duplicate_user", "password123")
        self.assertTrue(ok)
        ok, message = create_client_user("duplicate_user", "password123")
        self.assertFalse(ok)
        self.assertIn("deja", message.lower())

    def test_wrong_password_is_rejected(self) -> None:
        ok, _ = create_client_user("client_demo", "password123")
        self.assertTrue(ok)
        authenticated, user, message = authenticate_client_user("client_demo", "wrongpass")
        self.assertFalse(authenticated)
        self.assertIsNone(user)
        self.assertIn("incorrect", message.lower())

    def test_user_runs_are_linked_to_username(self) -> None:
        ok, _ = create_client_user("client_demo", "password123")
        self.assertTrue(ok)
        runtime_config = {
            "run_id": "temporaire-123",
            "run_mode": "temporaire",
            "dataset_name": "demo.csv",
            "artifact_root": "artifacts/runs/temporaire-123",
            "timestamp_utc": "2026-04-10T00:00:00+00:00",
        }
        upsert_user_run("client_demo", runtime_config, "training_completed", "training")
        rows = list_user_runs("client_demo")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["run_id"], "temporaire-123")
        self.assertEqual(rows[0]["username"], "client_demo")

    def test_case_history_is_persisted_for_client(self) -> None:
        ok, _ = create_client_user("client_demo", "password123")
        self.assertTrue(ok)
        save_case_history(
            "client_demo",
            {
                "case_id": "case-001",
                "timestamp": "2026-04-11T00:00:00+00:00",
                "final_decision": "Patient diabetique - Risque eleve",
                "diagnosis_label": "Patient diabetique",
                "risk_level": "Risque eleve",
                "rule_explanations": ["Si taux de glucose = 180, alors cela augmente le risque de diabete."],
                "local_xai_summary": [{"feature": "taux de glucose", "impact": "hausse du risque", "value": "180"}],
                "model_version": "run-001",
            },
        )
        rows = list_case_history("client_demo")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["case_id"], "case-001")
        self.assertEqual(rows[0]["diagnosis_label"], "Patient diabetique")


if __name__ == "__main__":
    unittest.main()
