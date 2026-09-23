"""Tests for mind/brain.py — decide function with mocked HTTP."""
import json
import unittest
from unittest.mock import patch, MagicMock


class TestBrainDecide(unittest.TestCase):
    """Test the decide() function without real API calls."""

    @patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key-123"})
    @patch("mind.brain.urllib.request.urlopen")
    def test_decide_returns_json(self, mock_urlopen):
        from mind.brain import decide

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "choices": [{"message": {"content": json.dumps({"target": "fridge", "reason": "hungry"})}}]
        }).encode()
        mock_urlopen.return_value = mock_resp

        result = decide("system", "user")
        self.assertEqual(result["target"], "fridge")
        self.assertEqual(result["reason"], "hungry")

    @patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key-123"})
    @patch("mind.brain.urllib.request.urlopen")
    def test_decide_returns_text(self, mock_urlopen):
        from mind.brain import decide

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "choices": [{"message": {"content": "I walk to the fridge."}}]
        }).encode()
        mock_urlopen.return_value = mock_resp

        result = decide("system", "user", as_json=False)
        self.assertEqual(result, "I walk to the fridge.")

    @patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key-123"})
    @patch("mind.brain.urllib.request.urlopen")
    def test_decide_fallback_json_parse(self, mock_urlopen):
        from mind.brain import decide

        # Simulate messy JSON output wrapped in markdown
        messy = '```json\n{"target": "sofa", "reason": "tired"}\n```'
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "choices": [{"message": {"content": messy}}]
        }).encode()
        mock_urlopen.return_value = mock_resp

        result = decide("system", "user")
        self.assertEqual(result["target"], "sofa")

    @patch.dict("os.environ", {}, clear=True)
    def test_decide_missing_key_raises(self):
        import os
        os.environ.pop("DEEPSEEK_API_KEY", None)
        from mind.brain import decide

        with self.assertRaises(KeyError):
            decide("system", "user")


if __name__ == "__main__":
    unittest.main()
