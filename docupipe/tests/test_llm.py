import io
import json
import unittest
from unittest import mock

from docupipe.clients import llm


class LLMClientTests(unittest.TestCase):
    def test_chat_sends_bearer_token_and_declared_user_agent(self):
        response = io.BytesIO(
            b'{"choices":[{"message":{"content":"DOCUPIPE_OK"}}]}'
        )
        with (
            mock.patch.dict(
                llm.config.LLM,
                {
                    "writer": {
                        "base_url": "https://provider.example/v1",
                        "model": "deepseek-v4-flash",
                        "key": "provider-key",
                    }
                },
                clear=True,
            ),
            mock.patch.object(llm.config, "HTTP_UA", "docupipe/test"),
            mock.patch.object(llm.cache, "have", return_value=False),
            mock.patch.object(llm.cache, "save_json"),
            mock.patch.object(llm.urllib.request, "urlopen", return_value=response) as urlopen,
        ):
            result = llm.chat(
                "writer",
                "Return the token.",
                "DOCUPIPE_OK",
                use_cache=False,
                retries=1,
            )

        self.assertEqual(result, "DOCUPIPE_OK")
        request = urlopen.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer provider-key")
        self.assertEqual(request.get_header("User-agent"), "docupipe/test")

    def test_deepseek_v4_uses_high_effort_thinking_without_sampling(self):
        response = io.BytesIO(
            b'{"choices":[{"finish_reason":"stop","message":'
            b'{"content":"DOCUPIPE_OK","reasoning_content":"private"}}]}'
        )
        with (
            mock.patch.dict(
                llm.config.LLM,
                {
                    "writer": {
                        "base_url": "https://provider.example/v1",
                        "model": "deepseek-v4-flash",
                        "key": "provider-key",
                    }
                },
                clear=True,
            ),
            mock.patch.object(llm.cache, "have", return_value=False),
            mock.patch.object(llm.cache, "save_json"),
            mock.patch.object(llm.urllib.request, "urlopen", return_value=response) as urlopen,
        ):
            result = llm.chat(
                "writer", "system", "user", temperature=0.9,
                use_cache=False, retries=1,
            )

        self.assertEqual(result, "DOCUPIPE_OK")
        body = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(body["thinking"], {"type": "enabled"})
        self.assertEqual(body["reasoning_effort"], "high")
        self.assertNotIn("temperature", body)

    def test_reasoning_without_final_content_is_not_returned(self):
        response = io.BytesIO(
            b'{"choices":[{"finish_reason":"length","message":'
            b'{"content":"","reasoning_content":"unfinished reasoning"}}]}'
        )
        with (
            mock.patch.dict(
                llm.config.LLM,
                {
                    "writer": {
                        "base_url": "https://provider.example/v1",
                        "model": "deepseek-v4-flash",
                        "key": "provider-key",
                    }
                },
                clear=True,
            ),
            mock.patch.object(llm.cache, "have", return_value=False),
            mock.patch.object(
                llm.urllib.request, "urlopen", return_value=response
            ) as urlopen,
            mock.patch.object(llm.time, "sleep"),
        ):
            with self.assertRaisesRegex(llm.CompletionLengthError, "max_tokens=4000"):
                llm.chat("writer", "system", "user", use_cache=False, retries=3)

        urlopen.assert_called_once()

    def test_chat_json_does_not_retry_token_exhaustion(self):
        with mock.patch.object(
            llm,
            "chat",
            side_effect=llm.CompletionLengthError("budget exhausted"),
        ) as chat:
            with self.assertRaisesRegex(llm.CompletionLengthError, "budget exhausted"):
                llm.chat_json("writer", "system", "user")
        chat.assert_called_once()


if __name__ == "__main__":
    unittest.main()
