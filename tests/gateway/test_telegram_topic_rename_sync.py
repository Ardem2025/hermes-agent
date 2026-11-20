"""Tests for Telegram DM topic rename/creation -> SessionDB title sync (Scenarios 1, 2, 3, 5)."""

import logging
import sys
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import Platform
from gateway.platforms.event import MessageType
from gateway.run_topics import GatewayTopicThreadsMixin
from gateway.session import SessionSource
from hermes_state import AsyncSessionDB, SessionDB
from tests.gateway.conftest import _ensure_telegram_mock

_ensure_telegram_mock()
sys.modules.pop("plugins.platforms.telegram.adapter", None)
import plugins.platforms.telegram.adapter as tg_adapter_mod
from plugins.platforms.telegram.adapter import TelegramAdapter

CHAT_ID = 139351986
USER_ID = 42
BOT_ID = 999000
THREAD_ID = 401485
T0 = datetime(2026, 11, 18, 10, 0, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _isolate_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))


def _make_db(tmp_path) -> SessionDB:
    db = SessionDB(db_path=tmp_path / "state.db")
    db.apply_telegram_topic_migration()
    return db


def _make_adapter(db: SessionDB | None = None, *, extra: dict | None = None, bot_id: int = BOT_ID) -> TelegramAdapter:
    cfg_extra = {"allow_from": [str(USER_ID)]}
    if extra:
        cfg_extra.update(extra)
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***", extra=cfg_extra))
    adapter._bot = SimpleNamespace(id=bot_id, username="hermes_test_bot")
    if db is not None:
        adapter._session_store = SimpleNamespace(_db=db)
    return adapter


def _edited_update(
    name: str | None,
    *,
    chat_id: int = CHAT_ID,
    chat_type: str = "private",
    thread_id: int | None = THREAD_ID,
    user_id: int | None = USER_ID,
    is_bot: bool = False,
    message_id: int = 5001,
):
    from_user = None if user_id is None else SimpleNamespace(id=user_id, full_name="User", is_bot=is_bot)
    msg = SimpleNamespace(
        chat=SimpleNamespace(id=chat_id, type=chat_type, title=None, full_name="User"),
        from_user=from_user,
        text=None,
        message_thread_id=thread_id,
        is_topic_message=thread_id is not None,
        message_id=message_id,
        reply_to_message=None,
        date=T0,
        edit_date=None,
        forum_topic_edited=SimpleNamespace(name=name),
        forum_topic_created=None,
    )
    return SimpleNamespace(effective_message=msg)


def _created_update(
    name: str | None,
    *,
    chat_id: int = CHAT_ID,
    chat_type: str = "private",
    thread_id: int | None = 402000,
    user_id: int | None = USER_ID,
    is_bot: bool = False,
    message_id: int = 6001,
):
    from_user = None if user_id is None else SimpleNamespace(id=user_id, full_name="User", is_bot=is_bot)
    msg = SimpleNamespace(
        chat=SimpleNamespace(id=chat_id, type=chat_type, title=None, full_name="User"),
        from_user=from_user,
        text=None,
        message_thread_id=thread_id,
        is_topic_message=thread_id is not None,
        message_id=message_id,
        reply_to_message=None,
        date=T0,
        edit_date=None,
        forum_topic_created=SimpleNamespace(name=name),
        forum_topic_edited=None,
    )
    return SimpleNamespace(effective_message=msg)


class _DummyRunner(GatewayTopicThreadsMixin):
    def __init__(self, session_db, adapter):
        self._session_db = session_db
        self._adapter = adapter

    def _delivery_adapter_for(self, source):
        return self._adapter

    def _is_telegram_topic_lane(self, source):
        return source.platform == Platform.TELEGRAM and source.chat_type == "dm" and bool(source.thread_id)


# ── Scenario 1: FORUM_TOPIC_EDITED registration, DB update, provenance, _dm_topics, no turn ──


def test_register_handlers_registers_forum_topic_edited(tmp_path, monkeypatch):
    db = _make_db(tmp_path)
    adapter = _make_adapter(db)
    registered = []
    monkeypatch.setattr(
        tg_adapter_mod,
        "TelegramMessageHandler",
        lambda flt, cb: SimpleNamespace(filters=flt, callback=cb),
    )
    app = SimpleNamespace(handlers={0: []}, add_handler=lambda h, group=0: registered.append((group, h)))
    adapter._register_handlers(app)
    callbacks = [getattr(h, "callback", None) for g, h in registered if g == 0]
    assert adapter._handle_forum_topic_edited in callbacks
    assert adapter._handle_forum_topic_created in callbacks
    edited_handlers = [h for g, h in registered if g == 0 and getattr(h, "callback", None) == adapter._handle_forum_topic_edited]
    assert edited_handlers[0].filters is tg_adapter_mod.filters.StatusUpdate.FORUM_TOPIC_EDITED


@pytest.mark.asyncio
async def test_scenario1_forum_topic_edited_updates_title_and_protects_static_dm_topics(tmp_path):
    db = _make_db(tmp_path)
    db.create_session("sess-1", source="telegram")
    db.set_auto_title("sess-1", "Старое авто-имя", source="llm")
    db.bind_telegram_topic(
        chat_id=str(CHAT_ID),
        thread_id=str(THREAD_ID),
        user_id=str(USER_ID),
        session_key="tg:139351986:401485",
        session_id="sess-1",
        profile_name="default",
    )
    # Also bind a static topic in the same chat to verify it is never mutated.
    static_tid = 401999
    db.create_session("sess-static", source="telegram")
    db.bind_telegram_topic(
        chat_id=str(CHAT_ID),
        thread_id=str(static_tid),
        user_id=str(USER_ID),
        session_key="tg:139351986:401999",
        session_id="sess-static",
        profile_name="default",
    )

    adapter = _make_adapter(
        db,
        extra={
            "dm_topics": [
                {"chat_id": str(CHAT_ID), "topics": [{"name": "Ops", "thread_id": static_tid, "skill": "devops"}]}
            ]
        },
    )
    adapter._dm_topics[f"{CHAT_ID}:Ops"] = static_tid
    adapter._dm_topics[f"{CHAT_ID}:Старое авто-имя"] = THREAD_ID
    adapter._enqueue_text_event = MagicMock()

    await adapter._handle_forum_topic_edited(_edited_update("Архитектура биллинга"), None)

    row = db.get_session("sess-1")
    assert row["title"] == "Архитектура биллинга"
    assert row["title_source"] == "user"

    # Subsequent LLM / derived auto-title must NOT overwrite user title.
    assert db.set_auto_title("sess-1", "Новое LLM имя", source="llm") is False
    assert db.set_auto_title("sess-1", "Derived имя", source="derived") is False
    assert db.get_session("sess-1")["title"] == "Архитектура биллинга"

    # Dynamic topic is NOT added to _dm_topics; stale non-static key is removed.
    assert f"{CHAT_ID}:Архитектура биллинга" not in adapter._dm_topics
    assert f"{CHAT_ID}:Старое авто-имя" not in adapter._dm_topics

    # Static topic remains intact with is_static=True and skill.
    assert adapter._dm_topics[f"{CHAT_ID}:Ops"] == static_tid
    static_info = adapter._get_dm_topic_info(str(CHAT_ID), str(static_tid))
    assert static_info == {"name": "Ops", "thread_id": static_tid, "skill": "devops", "is_static": True}

    # Service message never triggers an agent turn.
    adapter._enqueue_text_event.assert_not_called()


# ── Scenario 2: Uniqueness conflict fallback ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_scenario2_title_conflict_uses_deterministic_thread_fallback(tmp_path, caplog):
    db = _make_db(tmp_path)
    db.create_session("sess-other", source="telegram")
    db.append_message("sess-other", "user", "hello")
    db.set_session_title("sess-other", "Финансы")

    db.create_session("sess-target", source="telegram")
    db.append_message("sess-target", "user", "hi")
    db.bind_telegram_topic(
        chat_id=str(CHAT_ID),
        thread_id="401500",
        user_id=str(USER_ID),
        session_key="tg:139351986:401500",
        session_id="sess-target",
        profile_name="default",
    )

    adapter = _make_adapter(db)
    await adapter._handle_forum_topic_edited(_edited_update("Финансы", thread_id=401500), None)

    row = db.get_session("sess-target")
    assert row["title"] == "Финансы (#401500)"
    assert row["title_source"] == "user"

    # Long title conflict respects MAX_TITLE_LENGTH.
    long_title = "А" * db.MAX_TITLE_LENGTH
    db.create_session("sess-long-other", source="telegram")
    db.append_message("sess-long-other", "user", "x")
    db.set_session_title("sess-long-other", long_title)

    db.create_session("sess-long-target", source="telegram")
    db.append_message("sess-long-target", "user", "y")
    db.bind_telegram_topic(
        chat_id=str(CHAT_ID),
        thread_id="401501",
        user_id=str(USER_ID),
        session_key="tg:139351986:401501",
        session_id="sess-long-target",
        profile_name="default",
    )
    await adapter._handle_forum_topic_edited(_edited_update(long_title, thread_id=401501), None)
    long_row = db.get_session("sess-long-target")
    assert len(long_row["title"]) <= db.MAX_TITLE_LENGTH
    assert long_row["title"].endswith(" (#401501)")

    # Double conflict logs WARNING and does not raise.
    db.create_session("sess-dup", source="telegram")
    db.append_message("sess-dup", "user", "z")
    db.set_session_title("sess-dup", "Дубль")
    db.create_session("sess-dup-fb", source="telegram")
    db.append_message("sess-dup-fb", "user", "w")
    db.set_session_title("sess-dup-fb", "Дубль (#401502)")
    db.create_session("sess-dup-target", source="telegram")
    db.append_message("sess-dup-target", "user", "q")
    db.set_auto_title("sess-dup-target", "Исходное", source="llm")
    db.bind_telegram_topic(
        chat_id=str(CHAT_ID),
        thread_id="401502",
        user_id=str(USER_ID),
        session_key="tg:139351986:401502",
        session_id="sess-dup-target",
        profile_name="default",
    )
    with caplog.at_level(logging.WARNING):
        await adapter._handle_forum_topic_edited(_edited_update("Дубль", thread_id=401502), None)
    assert db.get_session("sess-dup-target")["title"] == "Исходное"
    assert any("Failed to set topic title" in rec.message for rec in caplog.records)


# ── Scenario 3: FORUM_TOPIC_CREATED -> _created_topic_names -> _record_telegram_topic_binding ──


@pytest.mark.asyncio
async def test_scenario3_created_topic_name_persisted_on_first_binding_and_opener_preserved(tmp_path):
    db = _make_db(tmp_path)
    adapter = _make_adapter(db)

    await adapter._handle_forum_topic_created(_created_update("Разбор логов", thread_id=402000), None)

    # Not cached in _dm_topics, stored in _created_topic_names, opener follow preserved.
    assert f"{CHAT_ID}:Разбор логов" not in adapter._dm_topics
    assert (str(CHAT_ID), "402000") in adapter._created_topic_names

    opener_msg = SimpleNamespace(
        chat=SimpleNamespace(id=CHAT_ID, type="private", title=None, full_name="User"),
        from_user=SimpleNamespace(id=USER_ID, full_name="User", is_bot=False),
        text="первое сообщение",
        message_thread_id=None,
        is_topic_message=False,
        message_id=6002,
        reply_to_message=None,
        date=T0,
        edit_date=None,
        forum_topic_created=None,
    )
    assert adapter._build_message_event(opener_msg, MessageType.TEXT).source.thread_id == "402000"

    # Rename before first message updates _created_topic_names.
    await adapter._handle_forum_topic_edited(_edited_update("Разбор логов v2", thread_id=402000), None)
    assert adapter._created_topic_names[(str(CHAT_ID), "402000")][0] == "Разбор логов v2"

    # First message binds session -> consumes pop_created_topic_name -> title_source='user'.
    db.create_session("sess-new", source="telegram")
    runner = _DummyRunner(db, adapter)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=str(CHAT_ID),
        thread_id="402000",
        user_id=str(USER_ID),
        chat_type="dm",
    )
    runner._record_telegram_topic_binding(
        source, SimpleNamespace(session_key="tg:139351986:402000", session_id="sess-new")
    )

    row = db.get_session("sess-new")
    assert row["title"] == "Разбор логов v2"
    assert row["title_source"] == "user"
    assert (str(CHAT_ID), "402000") not in adapter._created_topic_names


@pytest.mark.asyncio
async def test_scenario3_created_topic_existing_binding_and_ttl_cap(tmp_path, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(tg_adapter_mod.time, "monotonic", lambda: clock[0])

    db = _make_db(tmp_path)
    db.create_session("sess-bound", source="telegram")
    db.set_auto_title("sess-bound", "Авто", source="llm")
    db.bind_telegram_topic(
        chat_id=str(CHAT_ID),
        thread_id="402010",
        user_id=str(USER_ID),
        session_key="tg:139351986:402010",
        session_id="sess-bound",
        profile_name="default",
    )
    adapter = _make_adapter(db)

    await adapter._handle_forum_topic_created(_created_update("Сразу привязан", thread_id=402010), None)
    row = db.get_session("sess-bound")
    assert row["title"] == "Сразу привязан"
    assert row["title_source"] == "user"

    # Cap 256 and TTL 600s checks.
    for i in range(260):
        clock[0] += 0.01
        adapter._store_created_topic_name(str(CHAT_ID), str(500000 + i), f"Topic {i}")
    assert len(adapter._created_topic_names) == 256
    assert (str(CHAT_ID), "500000") not in adapter._created_topic_names

    clock[0] += 601.0
    assert adapter.pop_created_topic_name(str(CHAT_ID), "500259") is None
    assert len(adapter._created_topic_names) == 0


# ── Scenario 5: Negative, edge cases, and race protection (5.1 - 5.7) ───────────────────────


@pytest.mark.asyncio
async def test_scenario5_edge_cases_and_guards(tmp_path, caplog):
    db = _make_db(tmp_path)
    db.create_session("sess-5", source="telegram")
    db.set_auto_title("sess-5", "LLM Заголовок", source="llm")
    db.bind_telegram_topic(
        chat_id=str(CHAT_ID),
        thread_id=str(THREAD_ID),
        user_id=str(USER_ID),
        session_key="tg:139351986:401485",
        session_id="sess-5",
        profile_name="default",
    )
    adapter = _make_adapter(db)

    # 5.1 Icon-only change (None / empty / whitespace) -> no-op.
    for empty_name in (None, "", "   \t\n  "):
        await adapter._handle_forum_topic_edited(_edited_update(empty_name), None)
    assert db.get_session("sess-5")["title"] == "LLM Заголовок"
    assert db.get_session("sess-5")["title_source"] == "llm"

    # 5.2 Unknown thread_id -> no crash, no fake session created.
    before_count = db._read_one("SELECT COUNT(*) AS c FROM sessions")["c"]
    await adapter._handle_forum_topic_edited(_edited_update("Неизвестный", thread_id=999888), None)
    after_count = db._read_one("SELECT COUNT(*) AS c FROM sessions")["c"]
    assert after_count == before_count

    # 5.3 Bot echo (is_bot=True, own bot id) and from_user=None -> rejected before DB.
    await adapter._handle_forum_topic_edited(_edited_update("Эхо бота", user_id=BOT_ID, is_bot=True), None)
    await adapter._handle_forum_topic_edited(_edited_update("Другой бот", user_id=777, is_bot=True), None)
    await adapter._handle_forum_topic_edited(_edited_update("Без отправителя", user_id=None), None)
    assert db.get_session("sess-5")["title"] == "LLM Заголовок"
    assert db.get_session("sess-5")["title_source"] == "llm"

    # 5.5 Unauthorized user and non-private chat -> ignored.
    await adapter._handle_forum_topic_edited(_edited_update("Чужой", user_id=123456), None)
    await adapter._handle_forum_topic_edited(_edited_update("Группа", chat_type="supergroup"), None)
    assert db.get_session("sess-5")["title"] == "LLM Заголовок"

    # 5.6 _session_store / _db is None -> safe return.
    adapter_no_db = _make_adapter(None)
    await adapter_no_db._handle_forum_topic_edited(_edited_update("Без БД"), None)

    # 5.7 Canonical hidden Bot Chat protection -> catches ValueError, logs WARNING, no crash.
    db.create_session("sess-bot-chat", source="telegram")
    db._execute_write(
        lambda conn: conn.execute(
            "UPDATE sessions SET title = ?, hidden = 1 WHERE id = ?",
            (db.CANONICAL_BOT_CHAT_TITLE, "sess-bot-chat"),
        )
    )
    db.bind_telegram_topic(
        chat_id=str(CHAT_ID),
        thread_id="401777",
        user_id=str(USER_ID),
        session_key="tg:139351986:401777",
        session_id="sess-bot-chat",
        profile_name="default",
    )
    with caplog.at_level(logging.WARNING):
        await adapter._handle_forum_topic_edited(_edited_update("Попытка переименовать", thread_id=401777), None)
    assert db.get_session("sess-bot-chat")["title"] == db.CANONICAL_BOT_CHAT_TITLE
    assert any("canonical Bot Chat" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_scenario5_4_rename_telegram_topic_race_guard(tmp_path):
    db = _make_db(tmp_path)
    db.create_session("sess-race", source="telegram")
    db.set_session_title("sess-race", "Архитектура биллинга")
    db.bind_telegram_topic(
        chat_id=str(CHAT_ID),
        thread_id=str(THREAD_ID),
        user_id=str(USER_ID),
        session_key="tg:139351986:401485",
        session_id="sess-race",
        profile_name="default",
    )

    async_db = AsyncSessionDB(db)
    adapter = _make_adapter(db)
    adapter.rename_dm_topic = AsyncMock()
    adapter._bot = SimpleNamespace(id=BOT_ID, edit_forum_topic=AsyncMock())
    runner = _DummyRunner(async_db, adapter)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=str(CHAT_ID),
        thread_id=str(THREAD_ID),
        user_id=str(USER_ID),
        chat_type="dm",
    )

    # User title already set -> stale LLM auto-title must NOT call rename_dm_topic / edit_forum_topic.
    await runner._rename_telegram_topic_for_session_title(source, "sess-race", "Старое LLM имя")
    adapter.rename_dm_topic.assert_not_called()
    adapter._bot.edit_forum_topic.assert_not_called()

    # Also if title_source='llm' but sessions.title already changed to a newer title -> no-op.
    db.create_session("sess-race-llm", source="telegram")
    db.set_auto_title("sess-race-llm", "Новый LLM заголовок", source="llm")
    db.bind_telegram_topic(
        chat_id=str(CHAT_ID),
        thread_id="401888",
        user_id=str(USER_ID),
        session_key="tg:139351986:401888",
        session_id="sess-race-llm",
        profile_name="default",
    )
    source2 = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=str(CHAT_ID),
        thread_id="401888",
        user_id=str(USER_ID),
        chat_type="dm",
    )
    await runner._rename_telegram_topic_for_session_title(source2, "sess-race-llm", "Устаревший LLM заголовок")
    adapter.rename_dm_topic.assert_not_called()

    # Matching current LLM title -> proceeds to rename_dm_topic.
    await runner._rename_telegram_topic_for_session_title(source2, "sess-race-llm", "Новый LLM заголовок")
    adapter.rename_dm_topic.assert_awaited_once_with(
        chat_id=str(CHAT_ID), thread_id="401888", name="Новый LLM заголовок"
    )


# ── Scenario 6: RCA fix — from_user=None / is_bot=True in DM, TTL echo guard, unchanged_title, INFO logs ──


@pytest.mark.asyncio
async def test_scenario6_from_user_none_and_other_bot_in_authorized_private_chat(tmp_path, caplog):
    db = _make_db(tmp_path)
    db.create_session("sess-none", source="telegram")
    db.set_auto_title("sess-none", "Старое имя", source="llm")
    db.bind_telegram_topic(
        chat_id=str(USER_ID),
        thread_id=str(THREAD_ID),
        user_id=str(USER_ID),
        session_key=f"tg:{USER_ID}:{THREAD_ID}",
        session_id="sess-none",
        profile_name="default",
    )
    adapter = _make_adapter(db)

    with caplog.at_level(logging.INFO):
        # 1) from_user=None in authorized private chat (chat_id == USER_ID in allow_from) -> updates DB.
        await adapter._handle_forum_topic_edited(
            _edited_update("Имя без отправителя", chat_id=USER_ID, user_id=None),
            None,
        )
        assert db.get_session("sess-none")["title"] == "Имя без отправителя"
        assert db.get_session("sess-none")["title_source"] == "user"

        # 2) from_user.is_bot=True (not own bot, authorized) -> updates DB.
        await adapter._handle_forum_topic_edited(
            _edited_update("Имя от сервисного бота", chat_id=USER_ID, user_id=USER_ID, is_bot=True),
            None,
        )
        assert db.get_session("sess-none")["title"] == "Имя от сервисного бота"
        assert db.get_session("sess-none")["title_source"] == "user"

    info_msgs = [r.message for r in caplog.records if r.levelno == logging.INFO]
    assert any("Received forum_topic_edited:" in m for m in info_msgs)
    assert any("Synced forum_topic_edited to session:" in m for m in info_msgs)


@pytest.mark.asyncio
async def test_scenario6_from_user_none_unauthorized_private_chat_rejected(tmp_path, caplog):
    db = _make_db(tmp_path)
    unauth_chat_id = 888777666
    db.create_session("sess-unauth", source="telegram")
    db.set_auto_title("sess-unauth", "Исходное", source="llm")
    db.bind_telegram_topic(
        chat_id=str(unauth_chat_id),
        thread_id=str(THREAD_ID),
        user_id=str(unauth_chat_id),
        session_key=f"tg:{unauth_chat_id}:{THREAD_ID}",
        session_id="sess-unauth",
        profile_name="default",
    )
    adapter = _make_adapter(db)

    with caplog.at_level(logging.INFO):
        await adapter._handle_forum_topic_edited(
            _edited_update("Взлом", chat_id=unauth_chat_id, user_id=None),
            None,
        )

    assert db.get_session("sess-unauth")["title"] == "Исходное"
    assert db.get_session("sess-unauth")["title_source"] == "llm"
    assert any("reason=unauthorized" in r.message for r in caplog.records if r.levelno == logging.INFO)


@pytest.mark.asyncio
async def test_scenario6_bot_rename_echo_ttl_and_own_bot_echo_and_unchanged_title(tmp_path, monkeypatch, caplog):
    clock = [2000.0]
    monkeypatch.setattr(tg_adapter_mod.time, "monotonic", lambda: clock[0])

    db = _make_db(tmp_path)
    db.create_session("sess-echo", source="telegram")
    db.set_auto_title("sess-echo", "Начальное", source="llm")
    db.bind_telegram_topic(
        chat_id=str(USER_ID),
        thread_id=str(THREAD_ID),
        user_id=str(USER_ID),
        session_key=f"tg:{USER_ID}:{THREAD_ID}",
        session_id="sess-echo",
        profile_name="default",
    )
    adapter = _make_adapter(db)
    adapter._bot = SimpleNamespace(id=BOT_ID, username="hermes_test_bot", edit_forum_topic=AsyncMock())

    set_title_spy = MagicMock(wraps=db.set_session_title)
    monkeypatch.setattr(db, "set_session_title", set_title_spy)

    with caplog.at_level(logging.INFO):
        # 1) Bot renames topic via rename_dm_topic -> records in _recent_bot_topic_renames.
        await adapter.rename_dm_topic(USER_ID, THREAD_ID, "Авто-имя от бота")
        assert (str(USER_ID), str(THREAD_ID), "Авто-имя от бота") in adapter._recent_bot_topic_renames

        # 2) Echo arrives with from_user=None within 30s TTL -> ignored as bot_rename_echo, DB untouched.
        await adapter._handle_forum_topic_edited(
            _edited_update("Авто-имя от бота", chat_id=USER_ID, user_id=None),
            None,
        )
        assert db.get_session("sess-echo")["title"] == "Начальное"
        assert db.get_session("sess-echo")["title_source"] == "llm"
        set_title_spy.assert_not_called()

        # 3) Echo arrives with own bot id -> ignored as own_bot_echo, DB untouched.
        await adapter._handle_forum_topic_edited(
            _edited_update("Собственное эхо", chat_id=USER_ID, user_id=BOT_ID, is_bot=True),
            None,
        )
        assert db.get_session("sess-echo")["title"] == "Начальное"
        set_title_spy.assert_not_called()

        # 4) After TTL expires (>30s), user rename to same title succeeds; repeating same title hits unchanged_title.
        clock[0] += 31.0
        await adapter._handle_forum_topic_edited(
            _edited_update("Авто-имя от бота", chat_id=USER_ID, user_id=USER_ID),
            None,
        )
        assert db.get_session("sess-echo")["title"] == "Авто-имя от бота"
        assert db.get_session("sess-echo")["title_source"] == "user"
        assert set_title_spy.call_count == 1

        await adapter._handle_forum_topic_edited(
            _edited_update("Авто-имя от бота", chat_id=USER_ID, user_id=USER_ID),
            None,
        )
        assert set_title_spy.call_count == 1

    info_msgs = [r.message for r in caplog.records if r.levelno == logging.INFO]
    assert any("reason=bot_rename_echo" in m for m in info_msgs)
    assert any("reason=own_bot_echo" in m for m in info_msgs)
    assert any("reason=unchanged_title" in m for m in info_msgs)

    # 5) Cap 256 check on _recent_bot_topic_renames.
    for i in range(260):
        clock[0] += 0.01
        adapter._record_bot_topic_rename(str(USER_ID), str(700000 + i), f"Title {i}")
    assert len(adapter._recent_bot_topic_renames) == 256
    assert (str(USER_ID), "700000", "Title 0") not in adapter._recent_bot_topic_renames


@pytest.mark.asyncio
async def test_scenario6_early_returns_info_logging(tmp_path, caplog):
    db = _make_db(tmp_path)
    adapter = _make_adapter(db)

    with caplog.at_level(logging.INFO):
        await adapter._handle_forum_topic_edited(SimpleNamespace(effective_message=None), None)
        await adapter._handle_forum_topic_edited(_edited_update("Группа", chat_type="supergroup"), None)
        await adapter._handle_forum_topic_edited(_edited_update("Без треда", chat_id=USER_ID, user_id=USER_ID, thread_id=None), None)
        await adapter._handle_forum_topic_edited(_edited_update(None, chat_id=USER_ID, user_id=USER_ID), None)
        await adapter._handle_forum_topic_edited(_edited_update("Нет привязки", chat_id=USER_ID, user_id=USER_ID, thread_id=999111), None)

        adapter_no_db = _make_adapter(None)
        await adapter_no_db._handle_forum_topic_edited(_edited_update("Нет БД", chat_id=USER_ID, user_id=USER_ID), None)

        bad_db = SimpleNamespace(get_telegram_topic_binding=MagicMock(side_effect=RuntimeError("boom")))
        adapter_bad_db = _make_adapter(bad_db)
        await adapter_bad_db._handle_forum_topic_edited(_edited_update("Ошибка БД", chat_id=USER_ID, user_id=USER_ID), None)

    info_msgs = [r.message for r in caplog.records if r.levelno == logging.INFO]
    for reason in (
        "reason=missing_msg_or_edited",
        "reason=non_private_chat",
        "reason=missing_thread_id",
        "reason=empty_or_icon_only_name",
        "reason=no_binding_cached_pending",
        "reason=db_unavailable",
        "reason=binding_lookup_error",
    ):
        assert any(reason in m for m in info_msgs), f"Missing INFO log for {reason}"


# ── BDD Scenarios 1-6: Default placeholder auto-title & WebUI PATCH -> Telegram sync ─────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "placeholder",
    ["New Chat", "New Chat (#402000)", "Hermes Chat", "Новый чат"],
)
async def test_bdd1_default_topic_placeholder_uses_derived_and_allows_llm_auto_title(tmp_path, placeholder):
    from agent.title_generator import _has_upgraded_title

    db = _make_db(tmp_path)
    adapter = _make_adapter(db)
    adapter.rename_dm_topic = AsyncMock()

    # Case A: topic created before binding -> stored in _created_topic_names -> _record_telegram_topic_binding
    thread_a = 402100
    await adapter._handle_forum_topic_created(_created_update(placeholder, thread_id=thread_a), None)
    db.create_session("sess-bdd1-a", source="telegram")
    runner = _DummyRunner(AsyncSessionDB(db), adapter)
    source_a = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=str(CHAT_ID),
        thread_id=str(thread_a),
        user_id=str(USER_ID),
        chat_type="dm",
    )
    # _record_telegram_topic_binding expects sync SessionDB
    sync_runner = _DummyRunner(db, adapter)
    sync_runner._record_telegram_topic_binding(
        source_a, SimpleNamespace(session_key=f"tg:{CHAT_ID}:{thread_a}", session_id="sess-bdd1-a")
    )

    row_a = db.get_session("sess-bdd1-a")
    assert row_a["title"] == placeholder
    assert row_a["title_source"] == "derived"
    assert _has_upgraded_title(db, "sess-bdd1-a") is False

    assert db.set_auto_title("sess-bdd1-a", "Настройка деплоя", source="llm") is True
    assert db.get_session("sess-bdd1-a")["title"] == "Настройка деплоя"
    assert db.get_session("sess-bdd1-a")["title_source"] == "llm"

    await runner._rename_telegram_topic_for_session_title(source_a, "sess-bdd1-a", "Настройка деплоя")
    adapter.rename_dm_topic.assert_awaited_once_with(
        chat_id=str(CHAT_ID), thread_id=str(thread_a), name="Настройка деплоя"
    )

    # Case B: binding already exists when _handle_forum_topic_created fires
    thread_b = 402101
    db.create_session("sess-bdd1-b", source="telegram")
    db.bind_telegram_topic(
        chat_id=str(CHAT_ID),
        thread_id=str(thread_b),
        user_id=str(USER_ID),
        session_key=f"tg:{CHAT_ID}:{thread_b}",
        session_id="sess-bdd1-b",
        profile_name="default",
    )
    await adapter._handle_forum_topic_created(_created_update(placeholder, thread_id=thread_b), None)
    row_b = db.get_session("sess-bdd1-b")
    assert row_b["title_source"] == "derived"
    assert _has_upgraded_title(db, "sess-bdd1-b") is False


@pytest.mark.asyncio
async def test_bdd2_custom_created_topic_name_stays_user_and_blocks_llm_auto_title(tmp_path):
    from agent.title_generator import _has_upgraded_title

    db = _make_db(tmp_path)
    adapter = _make_adapter(db)

    await adapter._handle_forum_topic_created(_created_update("Разбор логов", thread_id=402200), None)
    db.create_session("sess-bdd2", source="telegram")
    runner = _DummyRunner(db, adapter)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=str(CHAT_ID),
        thread_id="402200",
        user_id=str(USER_ID),
        chat_type="dm",
    )
    runner._record_telegram_topic_binding(
        source, SimpleNamespace(session_key=f"tg:{CHAT_ID}:402200", session_id="sess-bdd2")
    )

    row = db.get_session("sess-bdd2")
    assert row["title"] == "Разбор логов"
    assert row["title_source"] == "user"
    assert _has_upgraded_title(db, "sess-bdd2") is True
    assert db.set_auto_title("sess-bdd2", "Другое имя", source="llm") is False
    assert db.get_session("sess-bdd2")["title"] == "Разбор логов"


def _build_api_test_app(monkeypatch, tmp_path, db: SessionDB, telegram_adapter):
    from aiohttp import web
    from gateway.platforms.api_server import APIServerAdapter

    monkeypatch.delenv("API_SERVER_KEY", raising=False)
    api_adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    api_adapter._session_db = db
    monkeypatch.setattr(api_adapter, "_telegram_runner_and_adapter", lambda: (None, telegram_adapter))
    app = web.Application()
    for method, path, handler in api_adapter._http_route_table():
        app.router.add_route(method, path, handler)
    return app


@pytest.mark.asyncio
async def test_bdd3_patch_session_title_syncs_to_bound_telegram_topic(tmp_path, monkeypatch):
    from aiohttp.test_utils import TestClient, TestServer

    db = _make_db(tmp_path)
    db.create_session("sess-bdd3", source="telegram")
    db.bind_telegram_topic(
        chat_id="139351986",
        thread_id="401485",
        user_id=str(USER_ID),
        session_key="tg:139351986:401485",
        session_id="sess-bdd3",
        profile_name="default",
    )
    tg_adapter = _make_adapter(db)
    tg_adapter.rename_dm_topic = AsyncMock()

    app = _build_api_test_app(monkeypatch, tmp_path, db, tg_adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.patch("/api/sessions/sess-bdd3", json={"title": "Новое имя из WebUI"})
        assert resp.status == 200
        payload = await resp.json()
        assert payload["session"]["title"] == "Новое имя из WebUI"

    row = db.get_session("sess-bdd3")
    assert row["title"] == "Новое имя из WebUI"
    assert row["title_source"] == "user"
    tg_adapter.rename_dm_topic.assert_awaited_once_with(
        chat_id="139351986", thread_id="401485", name="Новое имя из WebUI"
    )


@pytest.mark.asyncio
async def test_bdd4_patch_session_title_survives_telegram_api_error(tmp_path, monkeypatch, caplog):
    from aiohttp.test_utils import TestClient, TestServer

    db = _make_db(tmp_path)
    db.create_session("sess-bdd4", source="telegram")
    db.bind_telegram_topic(
        chat_id="139351986",
        thread_id="401486",
        user_id=str(USER_ID),
        session_key="tg:139351986:401486",
        session_id="sess-bdd4",
        profile_name="default",
    )
    tg_adapter = _make_adapter(db)
    tg_adapter.rename_dm_topic = AsyncMock(side_effect=RuntimeError("Flood control"))

    app = _build_api_test_app(monkeypatch, tmp_path, db, tg_adapter)
    with caplog.at_level(logging.WARNING):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.patch("/api/sessions/sess-bdd4", json={"title": "Имя при сбое TG"})
            assert resp.status == 200

    assert db.get_session("sess-bdd4")["title"] == "Имя при сбое TG"
    assert any("Failed to sync WebUI rename to Telegram topic" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_bdd5_patch_session_echo_guard_prevents_loop(tmp_path, monkeypatch, caplog):
    from aiohttp.test_utils import TestClient, TestServer

    db = _make_db(tmp_path)
    db.create_session("sess-bdd5", source="telegram")
    db.bind_telegram_topic(
        chat_id=str(USER_ID),
        thread_id="401487",
        user_id=str(USER_ID),
        session_key=f"tg:{USER_ID}:401487",
        session_id="sess-bdd5",
        profile_name="default",
    )
    tg_adapter = _make_adapter(db)
    tg_adapter._bot = SimpleNamespace(id=BOT_ID, username="hermes_test_bot", edit_forum_topic=AsyncMock())

    app = _build_api_test_app(monkeypatch, tmp_path, db, tg_adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.patch("/api/sessions/sess-bdd5", json={"title": "Синхронизированное имя"})
        assert resp.status == 200

    tg_adapter._bot.edit_forum_topic.assert_awaited_once_with(
        chat_id=USER_ID, message_thread_id=401487, name="Синхронизированное имя"
    )
    assert (str(USER_ID), "401487", "Синхронизированное имя") in tg_adapter._recent_bot_topic_renames

    set_title_spy = MagicMock(wraps=db.set_session_title)
    monkeypatch.setattr(db, "set_session_title", set_title_spy)
    with caplog.at_level(logging.INFO):
        await tg_adapter._handle_forum_topic_edited(
            _edited_update("Синхронизированное имя", chat_id=USER_ID, thread_id=401487, user_id=None),
            None,
        )
    set_title_spy.assert_not_called()
    assert any("reason=bot_rename_echo" in r.message for r in caplog.records if r.levelno == logging.INFO)


@pytest.mark.asyncio
async def test_bdd6_patch_unbound_or_empty_title_skips_telegram_rename(tmp_path, monkeypatch):
    from aiohttp.test_utils import TestClient, TestServer

    db = _make_db(tmp_path)
    db.create_session("sess-bdd6-unbound", source="webui")
    db.create_session("sess-bdd6-bound", source="telegram")
    db.set_session_title("sess-bdd6-bound", "Было имя")
    db.bind_telegram_topic(
        chat_id="139351986",
        thread_id="401488",
        user_id=str(USER_ID),
        session_key="tg:139351986:401488",
        session_id="sess-bdd6-bound",
        profile_name="default",
    )
    tg_adapter = _make_adapter(db)
    tg_adapter.rename_dm_topic = AsyncMock()

    app = _build_api_test_app(monkeypatch, tmp_path, db, tg_adapter)
    async with TestClient(TestServer(app)) as cli:
        r1 = await cli.patch("/api/sessions/sess-bdd6-unbound", json={"title": "Локальное имя"})
        assert r1.status == 200
        r2 = await cli.patch("/api/sessions/sess-bdd6-bound", json={"title": ""})
        assert r2.status == 200

    tg_adapter.rename_dm_topic.assert_not_called()

