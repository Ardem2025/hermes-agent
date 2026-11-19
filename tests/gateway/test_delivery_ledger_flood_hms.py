"""Unit tests for H:MM:SS and timedelta flood handling across gateway modules."""

from datetime import timedelta
from types import SimpleNamespace
import time

import pytest

from gateway.delivery_ledger import _raw_flood_wait, flood_wait_seconds, is_flood_error
from gateway.run_turn_progress import _note_flood
from gateway.stream_consumer_fallback import StreamFallbackMixin
from gateway.stream_consumer_transport import StreamTransportMixin


def test_delivery_ledger_parses_hms_flood_strings():
    msg = "Flood control exceeded. Retry in 0:24:46"
    assert _raw_flood_wait(msg) == pytest.approx(1486.0)
    assert is_flood_error(msg) is True
    assert flood_wait_seconds(msg) == pytest.approx(1486.0)

    assert _raw_flood_wait("Flood control exceeded. Retry in 185 seconds") == pytest.approx(185.0)
    assert _raw_flood_wait("Please retry in 0:24:46") is None
    assert is_flood_error("Please retry in 0:24:46") is False

    assert flood_wait_seconds("flood_control:0:24:46") == pytest.approx(1486.0)
    assert flood_wait_seconds("flood_control:nan", default=60.0) == pytest.approx(60.0)
    assert flood_wait_seconds("flood_control:-5", default=60.0) == pytest.approx(60.0)


def test_run_turn_progress_note_flood_handles_timedelta_and_hms():
    st = SimpleNamespace(flood_until=0.0, flood_pending=False)
    before = time.monotonic()
    ok = _note_flood(st, SimpleNamespace(error="flood_control:1488.0", retry_after=timedelta(seconds=1488)))
    assert ok is True
    assert st.flood_pending is True
    assert st.flood_until >= before + 1487.0

    st2 = SimpleNamespace(flood_until=0.0, flood_pending=False)
    before2 = time.monotonic()
    ok2 = _note_flood(st2, SimpleNamespace(error="Flood control exceeded. Retry in 0:24:46", retry_after=None))
    assert ok2 is True
    assert st2.flood_pending is True
    assert st2.flood_until >= before2 + 1485.0

    st3 = SimpleNamespace(flood_until=0.0, flood_pending=False)
    before3 = time.monotonic()
    ok3 = _note_flood(st3, SimpleNamespace(error="flood_control:bad", retry_after=True))
    assert ok3 is True
    assert before3 + 29.0 <= st3.flood_until <= before3 + 35.0


class _DummyFallback(StreamFallbackMixin):
    _max_fallback_flood_retry_seconds = 10.0


def test_stream_consumer_fallback_flood_retry_delay_handles_timedelta_and_hms():
    dummy = _DummyFallback()
    assert dummy._fallback_flood_retry_delay(
        SimpleNamespace(error="flood_control:1488.0", retry_after=timedelta(seconds=1488))
    ) is None
    assert dummy._fallback_flood_retry_delay(
        SimpleNamespace(error="flood_control:4.0", retry_after=timedelta(seconds=4))
    ) == pytest.approx(4.0)
    assert dummy._fallback_flood_retry_delay(
        SimpleNamespace(error="Flood control exceeded. Retry in 0:24:46", retry_after=None)
    ) is None
    assert dummy._fallback_flood_retry_delay(
        SimpleNamespace(error="flood_control:bad", retry_after=True)
    ) == pytest.approx(3.0)


class _DummyTransport(StreamTransportMixin):
    _MAX_FLOOD_STRIKES = 3

    def __init__(self):
        self.adapter = SimpleNamespace(FALLBACK_ON_FINAL_EDIT_FLOOD=False)
        self.cfg = SimpleNamespace(cursor="")
        self._last_sent_text = ""
        self._flood_strikes = 0
        self._current_edit_interval = 1.0
        self._last_edit_time = 0.0

    def _is_flood_error(self, result):
        return "flood" in (getattr(result, "error", "") or "").lower()


@pytest.mark.asyncio
async def test_stream_consumer_transport_edit_backoff_handles_timedelta():
    dummy = _DummyTransport()
    res = await dummy._on_edit_failure(
        SimpleNamespace(error="flood_control:15.0", retry_after=timedelta(seconds=15), raw_response=None),
        text="hello",
        finalize=False,
        is_turn_final=False,
    )
    assert res is False
    assert dummy._current_edit_interval == pytest.approx(15.0)

    dummy2 = _DummyTransport()
    res2 = await dummy2._on_edit_failure(
        SimpleNamespace(error="flood_control:bad", retry_after=True, raw_response=None),
        text="hello",
        finalize=False,
        is_turn_final=False,
    )
    assert res2 is False
    assert dummy2._current_edit_interval == pytest.approx(2.0)
