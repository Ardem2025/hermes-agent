"""Unit tests for exponential backoff in _async_delegation_watcher / _route_worker."""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest

import gateway.run_notifications as rn
from gateway.run_notifications import GatewayNotificationsMixin, _back_off_refused_completions


def test_back_off_refused_completions_exponential_steps(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(rn.time, "monotonic", lambda: now[0])

    group = [{"task_id": "t1"}, {"task_id": "t2"}]
    expected_delays = [4.0, 8.0, 16.0, 32.0, 60.0, 60.0]
    for idx, expected_delay in enumerate(expected_delays, start=1):
        _back_off_refused_completions(group)
        for evt in group:
            assert evt["_watch_retry_step"] == idx
            assert evt["_watch_next_retry_at"] == pytest.approx(now[0] + expected_delay)


class _FakeWatcherRunner(GatewayNotificationsMixin):
    _ASYNC_GROUP_KEY_FIELDS = ("session_key",)

    def __init__(self, outcomes):
        self._running = True
        self._outcomes = list(outcomes)
        self.delivery_times = []
        self._served_profile_homes = {}

    def _sweep_orphaned_completion_ledgers(self):
        return 0

    async def _drain_watch_notifications(self, queue):
        return None

    def _enrich_async_delegation_routing(self, evt):
        return None

    def _event_route_key(self, evt, fields):
        return tuple(str(evt.get(f) or "") for f in fields)

    @asynccontextmanager
    async def _completion_event_scope(self, evt):
        yield

    async def _deliver_async_delegation_group(self, group):
        self.delivery_times.append(rn.time.monotonic())
        if self._outcomes:
            return self._outcomes.pop(0)
        return True


@pytest.mark.asyncio
async def test_async_delegation_watcher_respects_backoff_between_ticks(monkeypatch):
    from tools.process_registry import process_registry as _pr

    # Drain any leftover items in shared completion_queue
    while not _pr.completion_queue.empty():
        try:
            _pr.completion_queue.get_nowait()
        except Exception:
            break

    now = [100.0]
    monkeypatch.setattr(rn.time, "monotonic", lambda: now[0])

    runner = _FakeWatcherRunner(outcomes=[False, False, True])
    evt = {"type": "async_delegation", "session_key": "telegram:dm:1:2", "task_id": "t-backoff"}
    _pr.completion_queue.put(evt)

    orig_sleep = asyncio.sleep
    tick_times = [100.0, 102.0, 103.9, 104.1, 108.0, 112.2]
    tick_idx = {"i": 0}

    async def fake_sleep(delay, *args, **kwargs):
        await orig_sleep(0)
        if delay == 3:
            return
        tick_idx["i"] += 1
        if tick_idx["i"] < len(tick_times):
            now[0] = tick_times[tick_idx["i"]]
        else:
            runner._running = False

    monkeypatch.setattr(rn.asyncio, "sleep", fake_sleep)

    await runner._async_delegation_watcher(interval=2.0)

    # Attempt 1 at t=100.0 -> False -> backoff +4.0s (next_retry_at=104.0)
    # Ticks at 102.0 and 103.9 must NOT spawn _route_worker
    # Attempt 2 at t=104.1 -> False -> backoff +8.0s (next_retry_at=112.1)
    # Tick at 108.0 must NOT spawn _route_worker
    # Attempt 3 at t=112.2 -> True -> delivered
    assert runner.delivery_times == [100.0, 104.1, 112.2]
    assert evt["_watch_retry_step"] == 2
    assert _pr.completion_queue.empty()
