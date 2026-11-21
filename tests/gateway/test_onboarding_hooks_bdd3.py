"""Tests for BDD-3: onboarding plugin hooks, session_envelope round-trip, and first-contact suppression."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import hermes_yaml as yaml
import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource, SessionStore
from gateway.session_authority import initialize_session_authority
from gateway.session_envelope import restore_native, snapshot_native
from hermes_cli.plugins import VALID_HOOKS
from hermes_state_runtime import RuntimeStoreError


def _make_event(text: str = "hello", platform: Platform = Platform.TELEGRAM) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_id="m-bdd3",
        source=SessionSource(
            platform=platform,
            user_id="1001",
            chat_id="1001",
            user_name="tester",
            chat_type="dm",
        ),
    )


def _make_runner(platform: Platform = Platform.TELEGRAM):
    config = GatewayConfig(platforms={platform: PlatformConfig(enabled=True)})
    runner = object.__new__(GatewayRunner)
    runner.config = config
    adapter = SimpleNamespace(send=AsyncMock())
    runner.adapters = {platform: adapter}
    runner.session_store = MagicMock()
    runner._running_agents = {}
    return runner, adapter


def test_post_gateway_session_bound_in_valid_hooks():
    assert "post_gateway_session_bound" in VALID_HOOKS


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "hook_payload",
    [
        {"action": "allow", "workflow": "active"},
        {"action": "allow", "workflow": "resume"},
        {"action": "allow", "storage_namespace": "workflow:onboarding"},
        {"action": "allow", "onboarding_active": True},
        {"action": "allow", "deferred_activation": {"kind": "activate_onboarding", "topic_id": 42}},
    ],
)
async def test_pre_gateway_dispatch_captures_onboarding_metadata(monkeypatch, hook_payload):
    async def _fake_hook(name, **_kwargs):
        if name == "pre_gateway_dispatch":
            return [
                {**hook_payload, "system_context": "  [Onboarding step 1]  "},
                {"action": "rewrite", "text": "rewritten text", "system_context": "[Step 2]"},
            ]
        return []

    monkeypatch.setattr("hermes_cli.lifecycle.ainvoke_hook", _fake_hook)
    runner, _ = _make_runner()
    event = _make_event("hi")

    result = await runner._hm_pre_gateway_dispatch_hook(event, event.source)
    assert result is not None
    assert result.text == "rewritten text"
    assert result.plugin_onboarding_active is True
    assert result.plugin_system_context == "[Onboarding step 1]\n\n[Step 2]"
    if "deferred_activation" in hook_payload:
        assert result.plugin_deferred_intents == [hook_payload["deferred_activation"]]
    else:
        assert result.plugin_deferred_intents == []


@pytest.mark.asyncio
async def test_session_envelope_round_trip_and_default_fingerprint(tmp_path, monkeypatch):
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "15551234567")
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={"session_path": str(tmp_path / "wa")}))
    runner.adapters = {Platform.WHATSAPP: adapter}
    authority = await initialize_session_authority(runner, profile_id="default", instance_id="bdd3-codec")
    try:
        chat = "15551234567@s.whatsapp.net"
        event = MessageEvent(
            text="hello",
            source=adapter.build_source(chat_id=chat, user_id=chat, chat_type="dm"),
            message_id="c-bdd3",
        )
        default_snap = snapshot_native(runner, event)
        for key in ("plugin_deferred_intents", "plugin_system_context", "plugin_onboarding_active"):
            assert key not in default_snap["native_text_v1"]["event"]

        enriched = replace(
            event,
            plugin_deferred_intents=[{"kind": "activate_onboarding", "chat_id": chat}],
            plugin_system_context="[Onboarding active]",
            plugin_onboarding_active=True,
        )
        snap = snapshot_native(runner, enriched)
        restored = restore_native(snap, runner)
        assert restored.plugin_deferred_intents == [{"kind": "activate_onboarding", "chat_id": chat}]
        assert restored.plugin_system_context == "[Onboarding active]"
        assert restored.plugin_onboarding_active is True

        for bad in (
            replace(event, plugin_system_context=123),
            replace(event, plugin_onboarding_active="yes"),
            replace(event, plugin_deferred_intents="bad"),
            replace(event, plugin_deferred_intents=[{"bad": {1, 2}}]),
        ):
            with pytest.raises(RuntimeStoreError, match="invalid_params"):
                snapshot_native(runner, bad)
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_post_gateway_session_bound_runs_after_topic_heal_and_updates_event(monkeypatch):
    runner, _ = _make_runner()
    source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="1001",
        chat_id="1001",
        thread_id="77",
        chat_type="dm",
    )
    event = MessageEvent(
        text="hi",
        message_id="m-bound",
        source=source,
        plugin_deferred_intents=[{"kind": "activate_onboarding"}],
        plugin_system_context="pre-ctx",
    )
    initial_entry = SimpleNamespace(session_id="sess-initial", session_key="k1")
    healed_entry = SimpleNamespace(session_id="sess-healed", session_key="k1")

    monkeypatch.setattr(
        GatewayRunner,
        "async_session_store",
        property(lambda _self: SimpleNamespace(get_or_create_session=AsyncMock(return_value=initial_entry))),
    )
    runner._cache_session_source = MagicMock()
    runner._is_telegram_topic_lane = MagicMock(return_value=True)
    runner._hmwa_heal_telegram_topic_binding = AsyncMock(return_value=healed_entry)

    monkeypatch.setattr("gateway.run_heartbeat_acceptance.resolve_heartbeat_owner", AsyncMock(return_value=True))

    seen = {}

    async def _fake_hook(name, **kwargs):
        if name == "post_gateway_session_bound":
            seen.update(kwargs)
            return [{"system_context": "bound-ctx", "onboarding_active": True}]
        return []

    monkeypatch.setattr("hermes_cli.lifecycle.ainvoke_hook", _fake_hook)

    resolved = await runner._hmwa_resolve_session(event, source)
    assert resolved == (source, healed_entry, "k1")
    assert seen["session_id"] == "sess-healed"
    assert seen["session_entry"] is healed_entry
    assert seen["deferred_intents"] == [{"kind": "activate_onboarding"}]
    assert event.plugin_system_context == "pre-ctx\n\nbound-ctx"
    assert event.plugin_onboarding_active is True


@pytest.mark.asyncio
async def test_post_gateway_session_bound_skips_internal_and_is_fail_open(monkeypatch):
    runner, _ = _make_runner()
    source = SessionSource(platform=Platform.TELEGRAM, user_id="1001", chat_id="1001", chat_type="dm")
    entry = SimpleNamespace(session_id="sess-1", session_key="k1")
    monkeypatch.setattr(
        GatewayRunner,
        "async_session_store",
        property(lambda _self: SimpleNamespace(get_or_create_session=AsyncMock(return_value=entry))),
    )
    runner._cache_session_source = MagicMock()
    runner._is_telegram_topic_lane = MagicMock(return_value=False)
    monkeypatch.setattr("gateway.run_heartbeat_acceptance.resolve_heartbeat_owner", AsyncMock(return_value=True))

    calls = []

    async def _boom(name, **kwargs):
        calls.append(name)
        raise RuntimeError("plugin boom")

    monkeypatch.setattr("hermes_cli.lifecycle.ainvoke_hook", _boom)

    internal_event = MessageEvent(text="wake", message_id="m-int", source=source, internal=True)
    assert await runner._hmwa_resolve_session(internal_event, source) == (source, entry, "k1")
    assert calls == []

    user_event = MessageEvent(text="hi", message_id="m-usr", source=source, internal=False)
    assert await runner._hmwa_resolve_session(user_event, source) == (source, entry, "k1")
    assert calls == ["post_gateway_session_bound"]


@pytest.mark.asyncio
async def test_first_contact_turn_note_suppressed_when_onboarding_active(tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "home")
    (tmp_path / "config.yaml").write_text("model: test\n", encoding="utf-8")

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.session_store = SessionStore(sessions_dir=tmp_path / "sessions", config=runner.config)
    runner._deliver_platform_notice = AsyncMock()

    source = SessionSource(platform=Platform.TELEGRAM, chat_id="dm", user_id="u", chat_type="dm")

    notes_active: list[str] = []
    ev_active = MessageEvent(text="hi", message_id="m1", source=source, plugin_onboarding_active=True)
    await runner._hmwa_first_contact_notes(source, [], notes_active, event=ev_active)
    assert notes_active == []

    notes_deferred: list[str] = []
    ev_deferred = MessageEvent(
        text="hi",
        message_id="m2",
        source=source,
        plugin_deferred_intents=[{"kind": "activate_onboarding"}],
    )
    await runner._hmwa_first_contact_notes(source, [], notes_deferred, event=ev_deferred)
    assert notes_deferred == []

    # Config must not have latched profile_build_offered while onboarding suppressed it.
    cfg_after_suppressed = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert "onboarding" not in cfg_after_suppressed

    # Normal first contact still emits the note when onboarding is not active.
    notes_normal: list[str] = []
    ev_normal = MessageEvent(text="hi", message_id="m3", source=source)
    await runner._hmwa_first_contact_notes(source, [], notes_normal, event=ev_normal)
    assert len(notes_normal) == 1


@pytest.mark.asyncio
async def test_prepare_turn_injects_plugin_system_context_into_sidecar_not_pinned_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "home")
    (tmp_path / "config.yaml").write_text("model: test\n", encoding="utf-8")

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.session_store = SessionStore(sessions_dir=tmp_path / "sessions", config=runner.config)
    runner._deliver_platform_notice = AsyncMock()
    runner._session_state = MagicMock(return_value=SimpleNamespace(turn=SimpleNamespace(lease_tokens={})))
    runner._hmwa_open_session = AsyncMock(return_value=(False, False))
    runner._set_session_env = MagicMock(return_value=[])
    runner._pinned_session_context_prompt = MagicMock(return_value="PINNED_PROMPT")
    runner._hmwa_acquire_turn_lease = AsyncMock()
    runner._mark_durable_active_turn = AsyncMock(return_value=True)
    runner._hmwa_run_session_hygiene = AsyncMock(return_value=[])
    runner._voice_channel_sidecar_note = MagicMock(return_value=None)
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(return_value="hello")
    runner._hmwa_apply_message_timestamp = MagicMock(return_value=("hello", "hello", None))
    runner._delivery_adapter_for = MagicMock(return_value=None)
    runner._bind_adapter_run_generation = MagicMock()

    monkeypatch.setattr(
        GatewayRunner,
        "async_session_store",
        property(
            lambda _self: SimpleNamespace(
                load_transcript=AsyncMock(return_value=[]),
                has_any_sessions=AsyncMock(return_value=False),
            )
        ),
    )

    captured_sidecar = {}

    def _capture_sidecar(skey, notes):
        captured_sidecar[skey] = list(notes)

    runner._set_pending_turn_sidecar_notes = _capture_sidecar

    source = SessionSource(platform=Platform.TELEGRAM, chat_id="dm", user_id="u", chat_type="dm")
    entry = SimpleNamespace(
        session_id="s1",
        session_key="k1",
        model_override=None,
        reasoning_effort=None,
        yolo=None,
        created_at=1,
        updated_at=2,
    )

    event = MessageEvent(
        text="hello",
        message_id="m-prep",
        source=source,
        plugin_system_context="[Onboarding step context]",
        plugin_onboarding_active=True,
    )

    prepared, env_tokens = await runner._hmwa_prepare_turn(event, source, entry, "k1", "k1", 1)
    assert isinstance(prepared, GatewayRunner._PreparedTurn)
    assert env_tokens == []
    assert prepared.context_prompt == "PINNED_PROMPT"
    assert "[Onboarding step context]" not in prepared.context_prompt
    runner._pinned_session_context_prompt.assert_called_once()
    assert captured_sidecar == {"k1": ["[Onboarding step context]"]}
