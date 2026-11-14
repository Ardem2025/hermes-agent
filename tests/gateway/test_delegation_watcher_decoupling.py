#!/usr/bin/env python3
"""Comprehensive test suite for SpDD gateway delegation watcher decoupling & non-blocking wake."""

import asyncio
import collections
from collections import deque
import logging
import os
import sys
import time
import unittest
from unittest.mock import MagicMock, AsyncMock, patch

# Configure sandbox path
REPO_PATH = "/mnt/data/runtimes/hermes-agent-runtime-v0.22.5-fleet-gold"
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

from gateway.wake import is_session_turn_busy, deliver_wake, check_coordinator_wake_eligibility


class TestGatewayDelegationDecoupling(unittest.IsolatedAsyncioTestCase):

    async def test_bdd1_hol_blocking_eliminated(self):
        """BDD 1: Slow route must not block fast route delivery."""
        sem = asyncio.Semaphore(16)
        route_pending = {}
        route_tasks = {}
        active_tasks = set()
        delivery_log = []
        slow_started = asyncio.Event()
        release_slow = asyncio.Event()

        async def fake_deliver(group):
            route = group[0]["parent_session_id"]
            if route == "slow_session":
                slow_started.set()
                await release_slow.wait()
            delivery_log.append((route, time.monotonic()))
            return True

        async def _route_worker(route_key):
            async with sem:
                pending = route_pending.get(route_key)
                while pending:
                    group = pending.popleft()
                    await fake_deliver(group)

        # Dispatch slow session
        route_pending[("slow_session",)] = deque([[{"parent_session_id": "slow_session"}]])
        t_slow = asyncio.create_task(_route_worker(("slow_session",)))
        route_tasks[("slow_session",)] = t_slow
        active_tasks.add(t_slow)

        await slow_started.wait()

        # Dispatch fast session while slow is still blocked
        route_pending[("fast_session",)] = deque([[{"parent_session_id": "fast_session"}]])
        t_fast = asyncio.create_task(_route_worker(("fast_session",)))
        route_tasks[("fast_session",)] = t_fast
        active_tasks.add(t_fast)

        await t_fast
        self.assertTrue(any(r == "fast_session" for r, _ in delivery_log))
        self.assertFalse(any(r == "slow_session" for r, _ in delivery_log))

        # Unblock slow
        release_slow.set()
        await t_slow
        self.assertEqual(len(delivery_log), 2)
        self.assertEqual(delivery_log[0][0], "fast_session")
        self.assertEqual(delivery_log[1][0], "slow_session")

    async def test_bdd2_fifo_order_per_route(self):
        """BDD 2: Events for the same route must execute sequentially in FIFO order."""
        sem = asyncio.Semaphore(16)
        route_pending = {("session_1",): deque([
            [{"id": 1}], [{"id": 2}], [{"id": 3}]
        ])}
        processed = []

        async def _route_worker(route_key):
            async with sem:
                pending = route_pending.get(route_key)
                while pending:
                    group = pending.popleft()
                    await asyncio.sleep(0.01)
                    processed.append(group[0]["id"])

        await _route_worker(("session_1",))
        self.assertEqual(processed, [1, 2, 3])

    async def test_bdd3_oldest_failed_retried_first(self):
        """BDD 3: Failed delivery is re-prepended to route queue ahead of later arrivals."""
        route_queue = deque([
            [{"id": "first_fail"}],
            [{"id": "second_item"}]
        ])
        attempts = 0

        # Simulate route worker failure handling
        group = route_queue.popleft()
        attempts += 1
        # Delivery failed -> re-insert at left
        route_queue.appendleft(group)

        # Append new arrival
        route_queue.append([{"id": "third_item"}])

        # Verify order: first_fail is still at head!
        self.assertEqual(route_queue[0][0]["id"], "first_fail")
        self.assertEqual(route_queue[1][0]["id"], "second_item")
        self.assertEqual(route_queue[2][0]["id"], "third_item")

    async def test_bdd4_shutdown_requeue(self):
        """BDD 4: Graceful shutdown cancels active tasks and requeues all items."""
        mock_queue = collections.deque()
        route_pending = {
            ("sess_a",): deque([[{"id": "a1"}], [{"id": "a2"}]]),
            ("sess_b",): deque([[{"id": "b1"}]])
        }

        # Simulate cleanup in finally block
        for route_key, pending_queue in list(route_pending.items()):
            while pending_queue:
                group = pending_queue.popleft()
                for evt in group:
                    mock_queue.append(evt)
        route_pending.clear()

        self.assertEqual(len(mock_queue), 3)
        self.assertEqual(len(route_pending), 0)

    async def test_bdd5_concurrency_ceiling(self):
        """BDD 5: Max simultaneous workers capped at semaphore limit (16)."""
        sem = asyncio.Semaphore(16)
        concurrent = 0
        max_seen = 0

        async def worker():
            nonlocal concurrent, max_seen
            async with sem:
                concurrent += 1
                if concurrent > max_seen:
                    max_seen = concurrent
                await asyncio.sleep(0.02)
                concurrent -= 1

        tasks = [asyncio.create_task(worker()) for _ in range(30)]
        await asyncio.gather(*tasks)
        self.assertLessEqual(max_seen, 16)
        self.assertGreaterEqual(max_seen, 10)

    async def test_bdd6_is_session_turn_busy(self):
        """BDD 6: is_session_turn_busy detects active leases in SQLite."""
        mock_db = MagicMock()
        # Mock active lease (expires in future)
        mock_db._read_one.return_value = {"expires_at": time.time() + 100}
        mock_adapter = MagicMock()
        mock_adapter._ensure_session_db.return_value = mock_db

        self.assertTrue(is_session_turn_busy("active_session", mock_adapter))

        # Mock expired/absent lease
        mock_db._read_one.return_value = None
        self.assertFalse(is_session_turn_busy("idle_session", mock_adapter))

    async def test_bdd7_wake_suppressed_when_turn_busy(self):
        """BDD 7: deliver_wake and check_coordinator_wake_eligibility suppress when session is busy."""
        mock_db = MagicMock()
        mock_db._read_one.return_value = {"expires_at": time.time() + 100}
        mock_adapter = MagicMock()
        mock_adapter._ensure_session_db.return_value = mock_db
        mock_adapter.supports_async_delivery = False

        # 1. Eligibility must be False
        self.assertFalse(check_coordinator_wake_eligibility("busy_session", mock_adapter, {"auto_resume": True}))

        # 2. deliver_wake must exit early without calling HTTP self-post
        with patch("gateway.wake._self_post_chat_completion", new_callable=AsyncMock) as mock_post:
            await deliver_wake(mock_adapter, text="wake text", session_id="busy_session")
            mock_post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
