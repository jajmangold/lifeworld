import unittest
from unittest import mock

from docupipe import nodes, provenance
from docupipe.clients import archives, slides
from docupipe.run import _job_id


class SafetyTests(unittest.TestCase):
    def test_presenton_is_unavailable_without_explicit_auth(self):
        with mock.patch.object(slides, "PRESENTON_AUTH", ""):
            self.assertFalse(slides.available())
            self.assertEqual(slides.deck_pages("content"), [])

    def test_loc_rights_require_explicit_safe_advisory(self):
        safe, ok = archives._loc_rights(
            {"item": {"rights_advisory": ["No known restrictions on publication."]}}
        )
        self.assertTrue(ok)
        self.assertIn("No known restrictions", safe)

        _, ok = archives._loc_rights(
            {"item": {"rights_advisory": ["Permission may be required."]}}
        )
        self.assertFalse(ok)
        _, ok = archives._loc_rights({"item": {}})
        self.assertFalse(ok)

    def test_editor_preserves_structured_visual_fields(self):
        original = {
            "id": 0,
            "kind": "narration",
            "narration": "Original narration.",
            "image_query": "old query",
            "shot_type": "map",
            "quote": "",
            "speaker": "",
            "map_title": "ROUTE",
            "map_subtitle": "1912",
            "map_points": [{"name": "A", "lat": 1, "lng": 2}],
            "map_route": True,
            "gfx_title": "",
            "events": [],
            "nodes": [],
            "links": [],
            "stat_value": "",
            "stat_label": "",
            "enhanced": False,
        }
        response = {
            "notes": {},
            "segments": [{"narration": "Revised narration.", "image_query": "new query"}],
        }
        state = {
            "topic": "Topic",
            "segments": [original],
            "story_bible": {"cast": [], "threads": []},
        }
        with (
            mock.patch.object(nodes.cache, "have", return_value=False),
            mock.patch.object(nodes.cache, "save_json"),
            mock.patch.object(nodes.llm, "chat_json", return_value=response),
            mock.patch.object(nodes.config, "EDITOR_PASSES", 1),
            mock.patch.object(nodes, "_progress"),
        ):
            edited = nodes.script_editor_node(state)["segments"][0]

        self.assertEqual(edited["narration"], "Revised narration.")
        self.assertEqual(edited["image_query"], "new query")
        self.assertEqual(edited["shot_type"], "map")
        self.assertEqual(edited["map_title"], "ROUTE")
        self.assertEqual(edited["map_points"], original["map_points"])
        self.assertTrue(edited["map_route"])

    def test_evidence_images_are_gated_and_credited(self):
        unresolved = [{"id": 3, "nodes": [{"img": "/tmp/image.png"}]}]
        self.assertEqual(
            provenance.gate(unresolved),
            ["segment 3 evidence node 0: image has unresolved rights"],
        )

        asset = {
            "rights_ok": True,
            "title": "Archive image",
            "attribution": "Archive",
            "license": "Public domain",
            "page_url": "https://example.invalid/item",
        }
        resolved = [{"id": 3, "nodes": [{"img": "/tmp/image.png", "asset": asset}]}]
        self.assertEqual(provenance.gate(resolved), [])
        credits = provenance.credits_text(resolved, [])
        self.assertIn("Archive image", credits)
        self.assertIn("Public domain", credits)

    def test_job_ids_reject_path_components(self):
        self.assertEqual(_job_id("episode_01"), "episode_01")
        for value in ("../escape", "/absolute", "two/levels", "", "UPPER"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    _job_id(value)


if __name__ == "__main__":
    unittest.main()
