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


def test_cli_duplicate_preflight_is_unchanged_without_smoke(monkeypatch):
    import pytest
    from hermes_cli import gateway as gateway_cli
    import gateway.status as status

    monkeypatch.delenv("HERMES_ISOLATED_GATEWAY_SMOKE", raising=False)
    monkeypatch.setattr(status, "get_running_pid", lambda: 424242)
    with pytest.raises(SystemExit) as exc:
        gateway_cli._guard_existing_gateway_process_conflict()
    assert exc.value.code == 1


def test_cli_smoke_preflight_never_queries_or_replaces_live_pid(monkeypatch):
    from hermes_cli import gateway as gateway_cli
    import gateway.status as status

    monkeypatch.setenv("HERMES_ISOLATED_GATEWAY_SMOKE", "1")
    def forbidden_pid_probe():
        raise AssertionError("smoke preflight must not inspect a live PID")
    monkeypatch.setattr(status, "get_running_pid", forbidden_pid_probe)
    gateway_cli._guard_existing_gateway_process_conflict()


def test_smoke_runtime_identity_paths_are_under_explicit_home(monkeypatch, tmp_path):
    from gateway import status
    isolated_home = tmp_path / "copied-home" / ".hermes"
    monkeypatch.setenv("HOME", str(tmp_path / "copied-home"))
    monkeypatch.setenv("HERMES_HOME", str(isolated_home))
    monkeypatch.setenv("XDG_STATE_HOME", str(isolated_home / ".state"))
    assert status._get_pid_path() == isolated_home / "gateway.pid"
    assert status._get_gateway_lock_path() == isolated_home / "gateway.lock"
    assert status._get_runtime_status_path() == isolated_home / "gateway_state.json"
    assert status._get_lock_dir() == isolated_home / ".state" / "hermes" / "gateway-locks"


def test_smoke_side_effect_guards_cover_adapters_scheduler_and_hooks():
    import inspect
    runner_source = inspect.getsource(GatewayRunner)
    import gateway.run as gateway_run
    module_source = inspect.getsource(gateway_run)
    assert "ISOLATED_SMOKE: platform adapters are hard-disabled" in runner_source
    assert "ISOLATED_SMOKE: secondary-profile adapters are hard-disabled" in runner_source
    assert "ISOLATED_SMOKE: shell hooks, event hooks, and process recovery are hard-disabled" in runner_source
    assert "ISOLATED_SMOKE: cron scheduler and housekeeping are hard-disabled" in module_source
