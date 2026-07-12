"""FriendsHost compatibility contracts kept across the v2026.7.7.2 upgrade."""
from types import SimpleNamespace

from gateway.config import Platform
from gateway.platforms.api_server import APIServerAdapter
from gateway.run import GatewayRunner


def _api_adapter():
    adapter = object.__new__(APIServerAdapter)
    adapter.gateway_runner = SimpleNamespace(adapters={})
    return adapter


def test_personal_dm_topic_uses_bot_deep_link(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_USERNAME", "@FriendsBot")
    binding = _api_adapter()._build_telegram_binding_dict("139351986", "287067")
    assert binding["bot_username"] == "friendsbot"
    assert binding["link"] == "https://t.me/friendsbot?start=topic_287067"


def test_personal_dm_topic_uses_tg_uri_without_public_username(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_USERNAME", raising=False)
    monkeypatch.delenv("BOT_USERNAME", raising=False)
    binding = _api_adapter()._build_telegram_binding_dict("139351986", "287067")
    assert binding["link"] == "tg://openmessage?chat_id=139351986&message_thread_id=287067"


def test_isolated_smoke_mode_is_explicit_and_opt_in(monkeypatch):
    from gateway.run import _isolated_gateway_smoke_mode
    monkeypatch.delenv("HERMES_ISOLATED_GATEWAY_SMOKE", raising=False)
    assert _isolated_gateway_smoke_mode() is False
    monkeypatch.setenv("HERMES_ISOLATED_GATEWAY_SMOKE", "1")
    assert _isolated_gateway_smoke_mode() is True


def test_russian_lifecycle_contract_is_present_in_runner_source():
    # Keep this lightweight and no-network: explicit source contracts guard
    # FriendsHost wording while upstream lifecycle control flow remains tested.
    import inspect
    source = inspect.getsource(GatewayRunner._notify_active_sessions_of_shutdown)
    assert "система перезапустила ассистента" in source
    assert "Система ассистента выключается для технических работ" in source
    source = inspect.getsource(GatewayRunner._send_restart_notification)
    assert "Ассистент успешно перезапущен" in source
