import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class TestBlock2Overlay(unittest.TestCase):
    def setUp(self):
        from tools import delegate_tool_registry as reg
        with reg._active_subagents_lock:
            self._saved_active = dict(reg._active_subagents)
            reg._active_subagents.clear()

    def tearDown(self):
        from tools import delegate_tool_registry as reg
        with reg._active_subagents_lock:
            reg._active_subagents.clear()
            reg._active_subagents.update(self._saved_active)

    def test_1_list_payload_includes_delegation_id_and_elapsed_seconds(self):
        from tools import delegate_tool_registry as reg

        parent = SimpleNamespace(session_id="sess-parent-1")
        child = SimpleNamespace(_live_transcript_path="/tmp/live/task-0.log")
        reg._register_subagent({
            "subagent_id": "sub-123",
            "delegation_id": "deleg-abc",
            "parent_id": None,
            "goal": "Inspect logs",
            "model": "test-model",
            "status": "running",
            "started_at": time.time() - 12.5,
            "owner_agent_session_id": "sess-parent-1",
            "agent": child,
        })

        payload = reg._list_payload(parent)
        self.assertEqual(payload["count"], 1)
        entry = payload["subagents"][0]
        self.assertEqual(entry["subagent_id"], "sub-123")
        self.assertEqual(entry["delegation_id"], "deleg-abc")
        self.assertIsInstance(entry["elapsed_seconds"], float)
        self.assertGreaterEqual(entry["elapsed_seconds"], 12.0)
        self.assertEqual(entry["elapsed_seconds"], entry["running_seconds"])

    def test_2_delegate_task_deduplicates_active_same_goal_in_same_parent_session(self):
        from tools import delegate_tool, delegate_tool_registry as reg

        parent = SimpleNamespace(session_id="sess-parent-dup", _delegate_depth=0)
        other_parent = SimpleNamespace(session_id="sess-parent-other", _delegate_depth=0)
        child = SimpleNamespace(_live_transcript_path="/tmp/live/dup.log", _interrupt_requested=False)
        reg._register_subagent({
            "subagent_id": "sub-existing-1",
            "delegation_id": "deleg-existing-1",
            "parent_id": None,
            "goal": "Audit   overlay   files",
            "model": "test-model",
            "status": "running",
            "started_at": time.time() - 5.0,
            "owner_agent_session_id": "sess-parent-dup",
            "agent": child,
        })

        with patch.object(delegate_tool, "is_spawn_paused", return_value=False), \
             patch.object(delegate_tool, "_load_config", return_value={"max_iterations": 20}), \
             patch.object(delegate_tool, "_routes_for", side_effect=AssertionError("should not route duplicate")):
            raw = delegate_tool.delegate_task(
                goal="  audit overlay files ",
                background=True,
                parent_agent=parent,
            )
        data = json.loads(raw)
        self.assertTrue(data.get("deduplicated"))
        self.assertEqual(data.get("subagent_id"), "sub-existing-1")
        self.assertEqual(data.get("subagent_ids"), ["sub-existing-1"])
        self.assertEqual(data.get("delegation_id"), "deleg-existing-1")

        # Different parent session must NOT deduplicate against sess-parent-dup
        self.assertIsNone(reg._find_active_duplicate_subagent(other_parent, "audit overlay files"))

        # Interrupted child must NOT deduplicate
        child._interrupt_requested = True
        self.assertIsNone(reg._find_active_duplicate_subagent(parent, "audit overlay files"))

    def test_3_interrupted_async_delegation_header_says_stopped_by_parent(self):
        from tools.process_registry_notifications import _format_async_delegation

        single_interrupted = _format_async_delegation({
            "delegation_id": "deleg-stop-1",
            "status": "interrupted",
            "goal": "Long task",
            "summary": "Partial findings",
            "error": "Stopped by parent",
            "duration_seconds": 4.2,
        })
        self.assertIn("[ASYNC DELEGATION STOPPED BY PARENT — deleg-stop-1]", single_interrupted)
        self.assertNotIn("[ASYNC DELEGATION COMPLETE — deleg-stop-1]", single_interrupted)
        self.assertIn("Partial findings", single_interrupted)

        single_completed = _format_async_delegation({
            "delegation_id": "deleg-ok-1",
            "status": "completed",
            "goal": "Done task",
            "summary": "All good",
            "duration_seconds": 2.0,
        })
        self.assertIn("[ASYNC DELEGATION COMPLETE — deleg-ok-1]", single_completed)

        batch_interrupted = _format_async_delegation({
            "delegation_id": "deleg-batch-stop",
            "is_batch": True,
            "status": "interrupted",
            "goals": ["Task A", "Task B"],
            "results": [
                {"task_index": 0, "status": "interrupted", "summary": "Part A"},
                {"task_index": 1, "status": "interrupted", "error": "Cancelled"},
            ],
        })
        self.assertIn("[ASYNC DELEGATION BATCH STOPPED BY PARENT — deleg-batch-stop]", batch_interrupted)
        self.assertNotIn("[ASYNC DELEGATION BATCH COMPLETE — deleg-batch-stop]", batch_interrupted)

    def test_4_iteration_budget_warning_for_subagents_and_small_budgets(self):
        from agent.turn_iteration_prep import _maybe_inject_iteration_budget_warning

        # Subagent with max_total=20 should warn at used=15 (5 iterations before max)
        subagent = SimpleNamespace(
            platform="subagent",
            side_agent=False,
            _delegate_depth=1,
            _interrupt_requested=False,
            budget_warning_ratio=None,
            _iteration_budget_warning_injected=False,
            iteration_budget=SimpleNamespace(used=15, max_total=20),
            valid_tool_names=(),
        )
        msgs = [{"role": "tool", "content": "tool output"}]
        injected = _maybe_inject_iteration_budget_warning(subagent, msgs)
        self.assertTrue(injected)
        self.assertTrue(subagent._iteration_budget_warning_injected)
        self.assertIn("[SYSTEM NOTICE — iteration budget checkpoint]", msgs[-1]["content"])
        self.assertIn("15 of 20", msgs[-1]["content"])

        # Before threshold (used=14 of 20) should NOT warn
        subagent_early = SimpleNamespace(
            platform="subagent",
            side_agent=False,
            _delegate_depth=1,
            _interrupt_requested=False,
            budget_warning_ratio=None,
            _iteration_budget_warning_injected=False,
            iteration_budget=SimpleNamespace(used=14, max_total=20),
            valid_tool_names=(),
        )
        msgs_early = [{"role": "tool", "content": "tool output"}]
        self.assertFalse(_maybe_inject_iteration_budget_warning(subagent_early, msgs_early))

        # Main agent with large budget (max_total=100) and no ratio should NOT warn
        main_agent = SimpleNamespace(
            platform="cli",
            side_agent=False,
            _delegate_depth=0,
            _interrupt_requested=False,
            budget_warning_ratio=None,
            _iteration_budget_warning_injected=False,
            iteration_budget=SimpleNamespace(used=95, max_total=100),
            valid_tool_names=(),
        )
        msgs_main = [{"role": "tool", "content": "tool output"}]
        self.assertFalse(_maybe_inject_iteration_budget_warning(main_agent, msgs_main))

    def test_5_hermes_state_compaction_does_not_duplicate_tool_calls_tail(self):
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "state.db"
            db = SessionDB(db_path=db_path)
            try:
                sid = "sess-compact-tool-tail"
                db.create_session(session_id=sid, source="cli")
                m1 = {"role": "user", "content": "Run check"}
                m2 = {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "terminal", "arguments": "{\"command\":\"ls\"}"}}],
                }
                m3 = {"role": "tool", "tool_call_id": "call_1", "tool_name": "terminal", "content": "ok"}
                m4 = {"role": "assistant", "content": "Done"}
                db.append_messages_batch(sid, [m1, m2, m3, m4])

                # Capture watermark before m2/m3/m4? Simulate watermark at m1 so m2/m3/m4 sit above watermark
                # while also being carried in compacted_messages (the #137197 race).
                rows = db.get_messages(sid)
                self.assertEqual(len(rows), 4)
                watermark = m1["_row_id"]

                compacted = [
                    {"role": "user", "content": "[Summary of earlier context]", "_compressed_summary": True},
                    dict(m2),
                    dict(m3),
                    dict(m4),
                ]
                new_count = db.archive_and_compact(
                    sid,
                    compacted,
                    watermark=watermark,
                    tail_count=3,
                )
                self.assertEqual(new_count, 4)
                active_after = db.get_messages(sid)
                self.assertEqual(len(active_after), 4)
                tool_call_msgs = [m for m in active_after if m.get("tool_calls")]
                self.assertEqual(len(tool_call_msgs), 1)
                self.assertEqual(tool_call_msgs[0]["tool_calls"][0]["id"], "call_1")
            finally:
                db.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
