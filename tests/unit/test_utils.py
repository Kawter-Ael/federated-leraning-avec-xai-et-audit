from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from shared.utils import apply_config_override, contains_forbidden_keys, find_free_port, load_json, utc_now


class TestUtils(unittest.TestCase):
    def test_utc_now_returns_iso_string(self) -> None:
        result = utc_now()
        self.assertIsInstance(result, str)
        self.assertIn("T", result)
        self.assertTrue(result.endswith("Z") or "+" in result)

    def test_load_json_reads_file(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as fh:
            json.dump({"key": "value"}, fh)
            path = fh.name
        try:
            result = load_json(path)
            self.assertEqual(result, {"key": "value"})
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_json_raises_on_missing_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_json("/nonexistent/path.json")

    def test_find_free_port_returns_int_in_valid_range(self) -> None:
        port = find_free_port()
        self.assertIsInstance(port, int)
        self.assertGreater(port, 0)
        self.assertLess(port, 65536)

    def test_find_free_port_returns_different_ports(self) -> None:
        ports = {find_free_port() for _ in range(10)}
        self.assertGreater(len(ports), 1)

    def test_apply_config_override_deep_merge(self) -> None:
        target = {"a": {"b": 1, "c": 2}, "d": 3}
        override = {"a": {"b": 10}, "e": 5}
        apply_config_override(target, override)
        self.assertEqual(target, {"a": {"b": 10, "c": 2}, "d": 3, "e": 5})

    def test_apply_config_override_scalar_replaces_dict(self) -> None:
        target = {"a": {"b": 1}}
        apply_config_override(target, {"a": "scalar"})
        self.assertEqual(target, {"a": "scalar"})

    def test_apply_config_override_empty_does_nothing(self) -> None:
        target = {"a": 1}
        original = dict(target)
        apply_config_override(target, {})
        self.assertEqual(target, original)

    def test_contains_forbidden_keys_top_level(self) -> None:
        result = contains_forbidden_keys({"safe": 1, "forbidden_key": 2}, {"forbidden_key"})
        self.assertEqual(result, ["forbidden_key"])

    def test_contains_forbidden_keys_nested(self) -> None:
        obj = {"outer": {"inner": {"secret": "value"}}}
        result = contains_forbidden_keys(obj, {"secret"})
        self.assertEqual(result, ["secret"])

    def test_contains_forbidden_keys_in_list(self) -> None:
        obj = [{"a": 1}, {"b": "raw_data"}]
        result = contains_forbidden_keys(obj, {"raw_data"})
        self.assertEqual(result, ["raw_data"])

    def test_contains_forbidden_keys_no_match(self) -> None:
        result = contains_forbidden_keys({"a": 1, "b": {"c": 2}}, {"secret", "hidden"})
        self.assertEqual(result, [])

    def test_contains_forbidden_keys_deduplicates(self) -> None:
        obj = [{"x": {"secret": 1}}, {"y": {"secret": 2}}]
        result = contains_forbidden_keys(obj, {"secret"})
        self.assertEqual(result, ["secret"])


if __name__ == "__main__":
    unittest.main()
