import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


CONFIG_PATH = Path(__file__).parents[1] / "src" / "docupipe" / "config.py"


def load_config(environment):
    spec = importlib.util.spec_from_file_location("docupipe_test_config", CONFIG_PATH)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(os.environ, environment, clear=True):
        spec.loader.exec_module(module)
    return module


class DocupipeConfigTests(unittest.TestCase):
    def test_explicit_tier_keys_take_precedence(self):
        with tempfile.TemporaryDirectory() as tempdir:
            config = load_config(
                {
                    "DOCUPIPE_DATA": tempdir,
                    "DOCUPIPE_BULK_KEY": "bulk-key",
                    "DOCUPIPE_WRITER_KEY": "writer-key",
                    "OPENCODE_GO_API_KEY": "generic-key",
                }
            )
        self.assertEqual(config.LLM["bulk"]["key"], "bulk-key")
        self.assertEqual(config.LLM["writer"]["key"], "writer-key")

    def test_explicit_token_file_supplies_both_tiers(self):
        with tempfile.TemporaryDirectory() as tempdir:
            token_file = Path(tempdir) / "opencode-go.token"
            token_file.write_text("token-file-key\n", encoding="utf-8")
            config = load_config(
                {
                    "DOCUPIPE_DATA": tempdir,
                    "DOCUPIPE_OPENCODE_GO_TOKEN_FILE": str(token_file),
                }
            )
        self.assertEqual(config.LLM["bulk"]["key"], "token-file-key")
        self.assertEqual(config.LLM["writer"]["key"], "token-file-key")

    def test_opencode_auth_file_is_the_default_fallback(self):
        with tempfile.TemporaryDirectory() as tempdir:
            auth_file = Path(tempdir) / "auth.json"
            auth_file.write_text(
                json.dumps({"opencode-go": {"key": "auth-file-key"}}),
                encoding="utf-8",
            )
            config = load_config(
                {
                    "DOCUPIPE_DATA": tempdir,
                    "DOCUPIPE_OPENCODE_AUTH_FILE": str(auth_file),
                }
            )
        self.assertEqual(config.LLM["writer"]["key"], "auth-file-key")

    def test_config_has_no_retired_containers_dependency(self):
        source = CONFIG_PATH.read_text(encoding="utf-8")
        self.assertNotIn("/srv/nvme-data/containers", source)
        self.assertNotIn("/home/josh/containers", source)


if __name__ == "__main__":
    unittest.main()
