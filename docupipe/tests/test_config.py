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
        source_root = CONFIG_PATH.parents[1]
        source = "\n".join(
            path.read_text(encoding="utf-8") for path in source_root.rglob("*.py")
        )
        self.assertNotIn("/srv/nvme-data/containers", source)
        self.assertNotIn("/home/josh/containers", source)

    def test_custom_url_does_not_receive_opencode_fallback_key(self):
        with tempfile.TemporaryDirectory() as tempdir:
            config = load_config(
                {
                    "DOCUPIPE_DATA": tempdir,
                    "DOCUPIPE_WRITER_URL": "https://provider.example/v1",
                    "OPENCODE_GO_API_KEY": "opencode-key",
                }
            )
        self.assertEqual(config.LLM["writer"]["key"], "")

    def test_custom_url_accepts_only_explicit_tier_key(self):
        with tempfile.TemporaryDirectory() as tempdir:
            config = load_config(
                {
                    "DOCUPIPE_DATA": tempdir,
                    "DOCUPIPE_WRITER_URL": "https://provider.example/v1",
                    "DOCUPIPE_WRITER_KEY": "provider-key",
                    "OPENCODE_GO_API_KEY": "opencode-key",
                }
            )
        self.assertEqual(config.LLM["writer"]["key"], "provider-key")

    def test_docgfx_manifest_is_minimal_and_has_no_unproven_assets(self):
        docupipe_root = CONFIG_PATH.parents[2]
        manifest = json.loads(
            (docupipe_root / "docgfx" / "package.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            manifest["dependencies"], {"@resvg/resvg-js": "2.6.2"}
        )
        prohibited = [
            path for path in (docupipe_root / "docgfx").rglob("*")
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".ttf", ".otf"}
        ]
        self.assertEqual(prohibited, [])


if __name__ == "__main__":
    unittest.main()
