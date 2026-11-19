"""Unit tests for coordinator auto-wake passport scoping in gateway/wake.py."""

import json
from pathlib import Path
from unittest.mock import MagicMock

from gateway.wake import check_coordinator_wake_eligibility


def _make_idle_adapter():
    db = MagicMock()
    db._read_one.return_value = None
    db.get_session.return_value = None
    adapter = MagicMock()
    adapter._ensure_session_db.return_value = db
    return adapter


def _write_passport(home: Path, task_name: str, payload: dict) -> None:
    task_dir = home / ".hermes" / "tasks" / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")


def test_active_passport_for_session_a_does_not_wake_session_b(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    _write_passport(tmp_path, "task-a", {"status": "in_progress", "session_id": "session_A"})

    adapter = _make_idle_adapter()
    assert check_coordinator_wake_eligibility("session_A", adapter, evt={}) is True
    assert check_coordinator_wake_eligibility("session_B", adapter, evt={}) is False


def test_ownerless_active_passport_does_not_wake_unrelated_session(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    _write_passport(tmp_path, "task-ownerless", {"status": "active"})
    _write_passport(tmp_path, "task-empty-sid", {"status": "in_progress", "session_id": ""})

    adapter = _make_idle_adapter()
    assert check_coordinator_wake_eligibility("session_B", adapter, evt={}) is False


def test_completed_passport_for_same_session_does_not_wake(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    _write_passport(tmp_path, "task-done", {"status": "completed", "session_id": "session_A"})

    adapter = _make_idle_adapter()
    assert check_coordinator_wake_eligibility("session_A", adapter, evt={}) is False
