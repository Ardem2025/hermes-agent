"""Tests for cron-flood-fallback-fix (BDD-1, BDD-2, BDD-6, BDD-7)."""
from __future__ import annotations

import ast
import asyncio
import logging
import subprocess
import sys
import threading
from pathlib import Path

import pytest

import cron.scheduler_delivery as sd
from gateway import delivery_ledger as dl
from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import SendResult


@pytest.fixture(autouse=True)
def _fresh_ledger(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setattr(dl, "_db_path", lambda: home / "state.db")
    monkeypatch.setattr(dl, "ledger_enabled", lambda config=None: True)
    monkeypatch.setattr(sd, "_maybe_mirror_cron_delivery", lambda *a, **k: None)


@pytest.fixture
def gateway_loop():
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)


def _make_target(
    loop,
    *,
    platform: Platform = Platform.TELEGRAM,
    platform_name: str = "telegram",
    chat_id: str = "139351986",
    thread_id: str | None = None,
    send_result=None,
):
    class Transport:
        adapter = type("Adapter", (), {"_owner_profile": "satellite"})()
        is_relay = False

        async def send(self, _platform, _chat_id, _content, metadata=None):
            return send_result

    fields = {name: None for name in sd._TargetDelivery.__dataclass_fields__}
    fields.update(
        job={"id": "job-flood-1"},
        platform=platform,
        platform_name=platform_name,
        chat_id=chat_id,
        thread_id=thread_id,
        transport=Transport(),
        config=GatewayConfig(),
        loop=loop,
        target_adapters={},
        mirror_text="",
        origin={},
    )
    return sd._TargetDelivery(**fields)


def test_bdd1_flood_control_exception_arm_blocks_standalone_and_suppresses_log(monkeypatch, gateway_loop, caplog):
    t = _make_target(
        gateway_loop,
        send_result=SendResult(success=False, error="flood_control:16428.0", retry_after=16428.0),
    )
    standalone_calls = []
    reconnect_calls = []
    monkeypatch.setattr(
        sd,
        "_standalone_send",
        lambda target, content, media: standalone_calls.append(content) or ({"success": True}, None),
    )
    orig_queue = sd._queue_for_live_reconnect
    monkeypatch.setattr(
        sd,
        "_queue_for_live_reconnect",
        lambda *a, **k: reconnect_calls.append(a) or orig_queue(*a, **k),
    )

    target_errors, delivery_errors = [], []
    with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
        delivered = sd._deliver_via_live_adapter(
            t,
            "cron payload",
            [],
            target_errors=target_errors,
            delivery_errors=delivery_errors,
            unverified_targets=[],
        )
        assert delivered is False
        sd._deliver_standalone(t, "cron payload", [], target_errors, delivery_errors)

    assert t.live_error == "flood_control:16428.0"
    assert standalone_calls == []
    assert reconnect_calls == []
    assert ", falling back to standalone" not in caplog.text
    assert any("flood_control:16428.0" in err for err in delivery_errors)
    assert dl.sweep_failed_for_runtime("telegram", profile="satellite") == []


def test_bdd1_flood_control_unconfirmed_result_arm_blocks_standalone_and_suppresses_log(
    monkeypatch, gateway_loop, caplog
):
    t = _make_target(
        gateway_loop,
        send_result={"success": False, "error": "flood_control:16428.0"},
    )
    standalone_calls = []
    reconnect_calls = []
    monkeypatch.setattr(
        sd,
        "_standalone_send",
        lambda target, content, media: standalone_calls.append(content) or ({"success": True}, None),
    )
    monkeypatch.setattr(
        sd,
        "_queue_for_live_reconnect",
        lambda *a, **k: reconnect_calls.append(a),
    )

    target_errors, delivery_errors = [], []
    with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
        delivered = sd._deliver_via_live_adapter(
            t,
            "cron payload",
            [],
            target_errors=target_errors,
            delivery_errors=delivery_errors,
            unverified_targets=[],
        )
        assert delivered is False
        sd._deliver_standalone(t, "cron payload", [], target_errors, delivery_errors)

    assert t.live_error == "flood_control:16428.0"
    assert standalone_calls == []
    assert reconnect_calls == []
    assert ", falling back to standalone" not in caplog.text
    assert any("flood_control:16428.0" in err for err in delivery_errors)


@pytest.mark.parametrize(
    ("live_error", "standalone_ret", "expect_queued"),
    [
        ("chat not found", ({"success": True}, None), False),
        ("send_path_degraded", (None, "You must pass the token from BotFather"), True),
        ("Not connected", (None, "You must pass the token from BotFather"), False),
    ],
)
def test_bdd2_non_flood_telegram_errors_preserve_standalone_fallback(
    monkeypatch, gateway_loop, caplog, live_error, standalone_ret, expect_queued
):
    t = _make_target(
        gateway_loop,
        chat_id="-100",
        thread_id="42",
        send_result=SendResult(success=False, error=live_error, retryable=(live_error == "send_path_degraded")),
    )
    standalone_calls = []
    monkeypatch.setattr(
        sd,
        "_standalone_send",
        lambda target, content, media: standalone_calls.append(content) or standalone_ret,
    )

    target_errors, delivery_errors = [], []
    with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
        delivered = sd._deliver_via_live_adapter(
            t,
            "the report",
            [],
            target_errors=target_errors,
            delivery_errors=delivery_errors,
            unverified_targets=[],
        )
        assert delivered is False
        sd._deliver_standalone(t, "the report", [], target_errors, delivery_errors)

    assert standalone_calls == ["the report"]
    assert ", falling back to standalone" in caplog.text
    if standalone_ret[1] is None:
        assert delivery_errors == []
    else:
        assert delivery_errors
    claimed = dl.sweep_failed_for_runtime("telegram", profile="satellite")
    if expect_queued:
        assert [(row["chat_id"], row["thread_id"], row["content"]) for row in claimed] == [
            ("-100", "42", "the report")
        ]
    else:
        assert claimed == []


def test_bdd6_no_cron_to_plugins_import_and_clean_import_orders():
    repo_root = Path(__file__).resolve().parents[2]
    sched_delivery = repo_root / "cron" / "scheduler_delivery.py"
    tree = ast.parse(sched_delivery.read_text(encoding="utf-8"), filename=str(sched_delivery))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("plugins.platforms"), f"Forbidden import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert not mod.startswith("plugins.platforms"), f"Forbidden import-from: {mod}"

    orders = [
        [
            "cron.scheduler",
            "cron.scheduler_delivery",
            "plugins.platforms.telegram.standalone",
            "plugins.platforms.telegram.telegram_outbound",
            "plugins.platforms.telegram.adapter",
        ],
        [
            "plugins.platforms.telegram.adapter",
            "plugins.platforms.telegram.telegram_outbound",
            "plugins.platforms.telegram.standalone",
            "cron.scheduler_delivery",
            "cron.scheduler",
        ],
    ]
    for order in orders:
        code = "; ".join(f"import {m}" for m in order)
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 0, f"Import order {order} failed: {proc.stderr}"


@pytest.mark.parametrize(
    ("platform", "platform_name", "chat_id"),
    [
        (Platform.DISCORD, "discord", "123456789"),
        (Platform.SLACK, "slack", "C123456"),
    ],
)
def test_bdd7_non_telegram_platforms_preserve_standalone_fallback(
    monkeypatch, gateway_loop, caplog, platform, platform_name, chat_id
):
    t = _make_target(
        gateway_loop,
        platform=platform,
        platform_name=platform_name,
        chat_id=chat_id,
        send_result=SendResult(success=False, error="Not connected"),
    )
    standalone_calls = []
    monkeypatch.setattr(
        sd,
        "_standalone_send",
        lambda target, content, media: standalone_calls.append(content) or ({"success": True}, None),
    )

    target_errors, delivery_errors = [], []
    with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
        delivered = sd._deliver_via_live_adapter(
            t,
            "cross-platform report",
            [],
            target_errors=target_errors,
            delivery_errors=delivery_errors,
            unverified_targets=[],
        )
        assert delivered is False
        sd._deliver_standalone(t, "cross-platform report", [], target_errors, delivery_errors)

    assert standalone_calls == ["cross-platform report"]
    assert ", falling back to standalone" in caplog.text
    assert delivery_errors == []
