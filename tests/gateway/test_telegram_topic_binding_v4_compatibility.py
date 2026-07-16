"""Regression coverage for the v4 Telegram topic-binding migration.

The v4 database can already contain nullable role columns.  Upgrading must keep
those columns and every binding row: role/profile routing is intentionally a
separate concern, but legacy data must not be discarded by a clean runtime.
"""
import sqlite3

from hermes_state import SessionDB


def _make_v4_database(path):
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        PRAGMA foreign_keys = ON;
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            user_id TEXT,
            model TEXT,
            model_config TEXT,
            system_prompt TEXT,
            parent_session_id TEXT,
            cwd TEXT,
            started_at REAL NOT NULL,
            updated_at REAL,
            ended_at REAL,
            end_reason TEXT,
            message_count INTEGER DEFAULT 0,
            tool_call_count INTEGER DEFAULT 0,
            total_tokens INTEGER DEFAULT 0,
            title TEXT,
            archived INTEGER DEFAULT 0
        );
        CREATE TABLE state_meta (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE telegram_dm_topic_mode (
            chat_id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
            enabled INTEGER NOT NULL DEFAULT 1, activated_at REAL NOT NULL,
            updated_at REAL NOT NULL, has_topics_enabled INTEGER,
            allows_users_to_create_topics INTEGER, capability_checked_at REAL,
            intro_message_id TEXT, pinned_message_id TEXT
        );
        CREATE TABLE telegram_dm_topic_bindings (
            chat_id TEXT NOT NULL, thread_id TEXT NOT NULL, user_id TEXT NOT NULL,
            session_key TEXT NOT NULL,
            session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            managed_mode TEXT NOT NULL DEFAULT 'auto',
            delivery_enabled INTEGER NOT NULL DEFAULT 1,
            last_synced_message_id INTEGER,
            role_id TEXT, prompt_revision TEXT,
            linked_at REAL NOT NULL, updated_at REAL NOT NULL,
            PRIMARY KEY (chat_id, thread_id)
        );
        INSERT INTO sessions (id, source, started_at) VALUES ('legacy-session', 'telegram', 1);
        INSERT INTO telegram_dm_topic_bindings
          VALUES ('chat', 'topic', 'owner', 'telegram:chat:topic', 'legacy-session',
                  'restored', 1, 42, 'life-coach', 'v1', 1, 2);
        INSERT INTO state_meta VALUES ('telegram_dm_topic_schema_version', '4');
        """
    )
    conn.commit()
    conn.close()


def test_v4_nullable_role_columns_and_binding_survive_clean_runtime(tmp_path):
    path = tmp_path / "v4-state.db"
    _make_v4_database(path)

    db = SessionDB(path)
    # Startup itself must not destructively touch the opt-in topic tables.
    before = db.get_telegram_topic_binding(chat_id="chat", thread_id="topic")
    assert before == {
        "chat_id": "chat", "thread_id": "topic", "user_id": "owner",
        "session_key": "telegram:chat:topic", "session_id": "legacy-session",
        "managed_mode": "restored", "delivery_enabled": 1,
        "last_synced_message_id": 42, "role_id": "life-coach",
        "prompt_revision": "v1", "linked_at": 1.0, "updated_at": 2.0,
    }

    db.apply_telegram_topic_migration()
    after = db.get_telegram_topic_binding(chat_id="chat", thread_id="topic")
    assert after == before
    db.close()

    conn = sqlite3.connect(path)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(telegram_dm_topic_bindings)")}
    version = conn.execute(
        "SELECT value FROM state_meta WHERE key='telegram_dm_topic_schema_version'"
    ).fetchone()[0]
    conn.close()
    assert {"role_id", "prompt_revision"}.issubset(columns)
    assert version == "4"
