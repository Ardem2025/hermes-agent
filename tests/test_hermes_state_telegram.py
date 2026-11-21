"""Unit tests for hermes_state_telegram topic delivery, sync checkpoint, and unbind methods."""

import ast
from pathlib import Path

from hermes_state import SessionDB


def test_no_duplicate_methods_in_hermes_state_telegram():
    src_path = Path(__file__).resolve().parents[1] / "hermes_state_telegram.py"
    tree = ast.parse(src_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "SessionTelegramTopicsMixin":
            method_names = [
                sub.name
                for sub in node.body
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            assert len(method_names) == len(set(method_names)), (
                f"Duplicate methods found: {[m for m in method_names if method_names.count(m) > 1]}"
            )
            for expected in (
                "set_telegram_topic_delivery_enabled",
                "advance_telegram_topic_sync_checkpoint",
                "unbind_telegram_topic",
            ):
                assert method_names.count(expected) == 1
            return
    raise AssertionError("SessionTelegramTopicsMixin not found")


def test_set_telegram_topic_delivery_enabled(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("sess_deliv_1", source="telegram")
    assert db.bind_telegram_topic(
        chat_id="1001",
        thread_id="42",
        user_id="1001",
        session_key="telegram:dm:1001:42",
        session_id="sess_deliv_1",
    )

    assert db.set_telegram_topic_delivery_enabled("sess_deliv_1", False) is True
    binding = db.get_telegram_topic_binding_by_session(session_id="sess_deliv_1")
    assert binding is not None
    assert binding["delivery_enabled"] == 0

    assert db.set_telegram_topic_delivery_enabled("sess_deliv_1", True) is True
    binding = db.get_telegram_topic_binding_by_session(session_id="sess_deliv_1")
    assert binding["delivery_enabled"] == 1

    assert db.set_telegram_topic_delivery_enabled("nonexistent_sess", False) is False


def test_advance_telegram_topic_sync_checkpoint_monotonic(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("sess_sync_1", source="telegram")
    assert db.bind_telegram_topic(
        chat_id="1002",
        thread_id="77",
        user_id="1002",
        session_key="telegram:dm:1002:77",
        session_id="sess_sync_1",
    )

    assert db.advance_telegram_topic_sync_checkpoint("sess_sync_1", 150) is True
    binding = db.get_telegram_topic_binding_by_session(session_id="sess_sync_1")
    assert binding["last_synced_message_id"] == 150

    # Lower message_id must not regress the checkpoint
    assert db.advance_telegram_topic_sync_checkpoint("sess_sync_1", 90) is True
    binding = db.get_telegram_topic_binding_by_session(session_id="sess_sync_1")
    assert binding["last_synced_message_id"] == 150

    # Higher message_id advances it
    assert db.advance_telegram_topic_sync_checkpoint("sess_sync_1", 220) is True
    binding = db.get_telegram_topic_binding_by_session(session_id="sess_sync_1")
    assert binding["last_synced_message_id"] == 220

    assert db.advance_telegram_topic_sync_checkpoint("missing_sess", 10) is False


def test_unbind_telegram_topic_preserves_session(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("sess_unbind_1", source="telegram")
    assert db.bind_telegram_topic(
        chat_id="1003",
        thread_id="88",
        user_id="1003",
        session_key="telegram:dm:1003:88",
        session_id="sess_unbind_1",
        profile_name="bot_a",
    )

    # Wrong profile does not unbind
    assert db.unbind_telegram_topic(chat_id="1003", thread_id="88", profile_name="default") is False
    assert db.get_telegram_topic_binding_by_session(session_id="sess_unbind_1") is not None

    # Matching profile unbinds without deleting the session
    assert db.unbind_telegram_topic(chat_id="1003", thread_id="88", profile_name="bot_a") is True
    assert db.get_telegram_topic_binding_by_session(session_id="sess_unbind_1") is None
    assert db.get_session("sess_unbind_1") is not None
    assert db.unbind_telegram_topic(chat_id="1003", thread_id="88", profile_name="bot_a") is False


def test_pre_v3_topic_tables_migrate_without_delivery_enabled_column(tmp_path):
    import sqlite3

    db_path = tmp_path / "legacy_pre_v3.db"
    db = SessionDB(db_path)
    db.create_session(session_id="sess_legacy_1", source="telegram")
    db.close()

    # Simulate a pre-v3 database where telegram_dm_topic_mode and telegram_dm_topic_bindings
    # lack profile_name and the 5 newer binding columns (delivery_enabled, etc.).
    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP TABLE IF EXISTS telegram_dm_topic_bindings")
        conn.execute("DROP TABLE IF EXISTS telegram_dm_topic_mode")
        conn.execute(
            """
            CREATE TABLE telegram_dm_topic_mode (
                chat_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1,
                activated_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE telegram_dm_topic_bindings (
                chat_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                session_key TEXT NOT NULL,
                session_id TEXT NOT NULL,
                managed_mode TEXT NOT NULL DEFAULT 'auto',
                linked_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (chat_id, thread_id)
            )
            """
        )
        conn.execute(
            "INSERT INTO telegram_dm_topic_mode (chat_id, user_id, enabled, activated_at, updated_at) "
            "VALUES ('2001', '3001', 1, 100.0, 100.0)"
        )
        conn.execute(
            "INSERT INTO telegram_dm_topic_bindings "
            "(chat_id, thread_id, user_id, session_key, session_id, managed_mode, linked_at, updated_at) "
            "VALUES ('2001', '77', '3001', 'tg:2001:77', 'sess_legacy_1', 'auto', 100.0, 100.0)"
        )
        conn.commit()

    reopened = SessionDB(db_path)
    try:
        reopened.apply_telegram_topic_migration()
        assert reopened.is_telegram_topic_mode_enabled(chat_id="2001", user_id="3001", profile_name="default") is True

        binding = reopened.get_telegram_topic_binding(chat_id="2001", thread_id="77", profile_name="default")
        assert binding is not None
        assert binding["session_id"] == "sess_legacy_1"
        assert binding["delivery_enabled"] == 1
        assert binding["last_synced_message_id"] is None
        assert binding["role_id"] is None
        assert binding["prompt_revision"] is None
        assert binding["reply_anchor_message_id"] is None
    finally:
        reopened.close()
