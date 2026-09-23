"""Tests for memory/graph.py — Memory class with mock Neo4j driver."""
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure neo4j is mockable before importing memory.graph
sys.modules.setdefault("neo4j", MagicMock())


class TestMemoryInit(unittest.TestCase):
    """Test Memory instantiation."""

    def test_instantiation_with_mock_driver(self):
        mock_gdb = MagicMock()
        with patch.dict("sys.modules", {"neo4j": mock_gdb}):
            # Re-import to pick up the patched module
            import importlib
            from memory import graph as graph_mod
            importlib.reload(graph_mod)

            m = graph_mod.Memory(uri="bolt://mock:7688", user="neo4j", pwd="test")
            mock_gdb.GraphDatabase.driver.assert_called_once_with(
                "bolt://mock:7688", auth=("neo4j", "test")
            )
            m.close()
            mock_gdb.GraphDatabase.driver.return_value.close.assert_called_once()

    def test_instantiation_default_params(self):
        import importlib
        from memory import graph as graph_mod
        importlib.reload(graph_mod)
        mock_driver = MagicMock()
        with patch.object(graph_mod.GraphDatabase, "driver", return_value=mock_driver):
            m = graph_mod.Memory()
            m.close()
            mock_driver.close.assert_called_once()


class TestMemoryOperations(unittest.TestCase):
    """Test Memory methods with a mocked driver."""

    def setUp(self):
        import importlib
        from memory import graph as graph_mod
        importlib.reload(graph_mod)

        self.mock_session = MagicMock()
        self.mock_driver = MagicMock()
        self.mock_driver.session.return_value.__enter__ = MagicMock(return_value=self.mock_session)
        self.mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        with patch.object(graph_mod.GraphDatabase, "driver", return_value=self.mock_driver):
            self.mem = graph_mod.Memory(uri="bolt://mock:7688", user="neo4j", pwd="test")

    def test_ensure_agent(self):
        self.mem.ensure_agent("Mara")
        self.mock_session.run.assert_called_once_with(
            "MERGE (a:Agent {name:$n})", n="Mara"
        )

    def test_log_step(self):
        self.mem.log_step("Mara", 0, "fridge, sofa", "I'm hungry", "fridge", "kitchen", "food")
        self.mock_session.run.assert_called_once()
        call_args = self.mock_session.run.call_args
        self.assertIn("agent", call_args[1])

    def test_add_utterance(self):
        self.mem.add_utterance("Mara", 1, "Hello!", "wave", "living_room")
        self.mock_session.run.assert_called_once()

    def test_set_feeling(self):
        self.mem.set_feeling("Mara", "Theo", "annoyed", 0.3)
        self.mock_session.run.assert_called_once()

    def test_set_feeling_self_skipped(self):
        self.mem.set_feeling("Mara", "Mara", "lonely", -0.5)
        self.mock_session.run.assert_not_called()

    def test_add_reflection(self):
        self.mem.add_reflection("Mara", 10, "I should check the fridge more often.")
        self.mock_session.run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
