"""Telegram Subagent Live Pulse Ticker.

Provides real-time dynamic card updates in Telegram during subagent execution:
- Elapsed timer (MM:SS)
- Safe transcript tail reading (~4KB) with tool verb translation
- Batch subagent aggregation (tasks: dict[int, dict])
- Telegram Flood Control (429) & 'message is not modified' handling
- Bounded 3.0s cadence and cancellation/stop cleanup
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from agent.display import _TOOL_VERBS_RU

logger = logging.getLogger("gateway.subagent_pulse_ticker")


class SubagentPulseTicker:
    """Manages background pulse ticker updates for subagents in Telegram."""

    def __init__(
        self,
        adapter: Any,
        chat_id: str | int,
        thread_id: Optional[int | str] = None,
        interval_seconds: float = 3.0,
        turn_context: Optional[Any] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self.adapter = adapter
        self.chat_id = str(chat_id)
        self.thread_id = thread_id
        self.interval = max(3.0, float(interval_seconds))
        self.turn_context = turn_context
        self.metadata = dict(metadata or {})
        if thread_id and "message_thread_id" not in self.metadata:
            self.metadata["message_thread_id"] = thread_id

        self.status_key = f"subagent_pulse_{uuid.uuid4().hex[:8]}"
        self.started_at: float = time.time()
        self.active: bool = False
        self.ticker_task: Optional[asyncio.Task] = None
        self.current_message_id: Optional[str] = None
        self._last_content: str = ""
        self._lock = asyncio.Lock()

        # Registry of subagent tasks: task_index -> dict
        # {task_index: {"goal": str, "path": Optional[Path], "started_at": float,
        #               "last_action": str, "status": str, "summary": Optional[str]}}
        self.tasks: dict[int, dict[str, Any]] = {}

    def on_subagent_start(
        self,
        task_index: int = 0,
        goal: str = "",
        transcript_path: Optional[str | Path] = None,
    ) -> None:
        """Register subagent task start."""
        path_obj = Path(transcript_path) if transcript_path else None
        self.tasks[task_index] = {
            "goal": goal or f"Задача #{task_index + 1}",
            "path": path_obj,
            "started_at": time.time(),
            "last_action": "🚀 Инициализация окружения...",
            "status": "running",
            "summary": None,
        }

    register_task = on_subagent_start

    def on_subagent_progress(self, task_index: int = 0, action: str = "") -> None:
        """Update current action of subagent."""
        if task_index in self.tasks:
            self.tasks[task_index]["last_action"] = action
        else:
            self.tasks[task_index] = {
                "goal": f"Задача #{task_index + 1}",
                "path": None,
                "started_at": time.time(),
                "last_action": action,
                "status": "running",
                "summary": None,
            }

    update_task_progress = on_subagent_progress

    def on_subagent_complete(
        self,
        task_index: int = 0,
        status: str = "completed",
        summary: Optional[str] = None,
    ) -> None:
        """Record subagent completion."""
        if task_index in self.tasks:
            self.tasks[task_index]["status"] = status
            self.tasks[task_index]["summary"] = summary
            if status == "completed":
                self.tasks[task_index]["last_action"] = "✅ Завершен"
            else:
                self.tasks[task_index]["last_action"] = f"❌ Ошибка ({status})"
        else:
            self.tasks[task_index] = {
                "goal": f"Задача #{task_index + 1}",
                "path": None,
                "started_at": time.time(),
                "last_action": "✅ Завершен" if status == "completed" else f"❌ {status}",
                "status": status,
                "summary": summary,
            }

    async def start(self) -> None:
        """Start background ticker loop."""
        if self.active:
            return
        self.active = True
        self.started_at = time.time()
        loop = asyncio.get_running_loop()
        self.ticker_task = loop.create_task(self.run_loop())

        if self.turn_context is not None:
            bg_tasks = getattr(self.turn_context, "background_tasks", None)
            if isinstance(bg_tasks, set):
                bg_tasks.add(self.ticker_task)
                self.ticker_task.add_done_callback(bg_tasks.discard)

    @staticmethod
    def safe_read_transcript(path: Optional[Path | str]) -> str:
        """Safely read tail of live transcript log and map to Russian tool action."""
        if not path:
            return "🚀 Инициализация окружения..."
        path_obj = Path(path)
        if not path_obj.exists():
            return "🚀 Инициализация окружения..."
        try:
            stat = path_obj.stat()
            if stat.st_size == 0:
                return "🚀 Инициализация окружения..."
            read_size = min(4096, stat.st_size)
            with path_obj.open("r", encoding="utf-8", errors="replace") as f:
                if stat.st_size > read_size:
                    f.seek(stat.st_size - read_size)
                content = f.read()
            return SubagentPulseTicker._parse_action_from_content(content)
        except (FileNotFoundError, OSError, UnicodeDecodeError):
            return "⚙️ Выполнение..."

    @staticmethod
    def _parse_action_from_content(content: str) -> str:
        """Extract tool action line from transcript tail."""
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        for line in reversed(lines):
            if " tool " in line and "->" in line:
                action_part = line.split("->", 1)[-1].strip()
                # Parse tool name and arg preview
                match = re.match(r"^([a-zA-Z0-9_-]+)\((.*)\)$", action_part)
                if match:
                    tool_name, args = match.groups()
                    tool_ru = _TOOL_VERBS_RU.get(tool_name, tool_name)
                    arg_preview = args.strip()[:35]
                    if arg_preview:
                        return f"⚙️ {tool_ru} ({arg_preview})"
                    return f"⚙️ {tool_ru}"
                return f"⚙️ {action_part[:40]}"
            if " result " in line:
                result_part = line.split(" result ", 1)[-1].strip()
                return f"⚙️ Обработка: {result_part[:35]}"
            if " start " in line:
                return "🚀 Запуск шага..."
        return "⚙️ Выполнение..."

    def _format_time(self, seconds: float) -> str:
        """Format seconds into MM:SS."""
        secs = max(0, int(seconds))
        mins, s = divmod(secs, 60)
        return f"{mins:02d}:{s:02d}"

    def _render_card(self, cancelled: bool = False, final_status: Optional[str] = None) -> str:
        """Render text card according to BDD specifications."""
        now = time.time()
        elapsed = now - self.started_at
        time_str = self._format_time(elapsed)

        if cancelled:
            return f"⏹️ [{time_str}] 🔀 Выполнение отменено пользователем"

        # Check global timeout
        if elapsed >= 900:
            return "⏱️ [15:00] 🔀 Превышен таймаут выполнения (900s)"

        total_tasks = len(self.tasks)
        if total_tasks == 0:
            return f"⏳ [{time_str}] 🔀 Субагент: подготовка..."

        active_tasks = [t for t in self.tasks.values() if t.get("status") == "running"]
        active_count = len(active_tasks)

        # Single subagent mode
        if total_tasks == 1:
            task = next(iter(self.tasks.values()))
            goal = task.get("goal", "")
            status = task.get("status", "running")

            if final_status or status != "running":
                st = final_status or status
                summary = task.get("summary") or ""
                summary_snippet = f": {summary[:60]}" if summary else ""
                if st == "completed":
                    return f"✅ [{time_str}] 🔀 Субагент 1/1 завершен{summary_snippet}"
                return f"❌ [{time_str}] 🔀 Субагент 1/1 ошибка ({st}){summary_snippet}"

            # Running
            action = task.get("last_action") or ""
            if task.get("path"):
                action = self.safe_read_transcript(task.get("path"))
            goal_short = goal[:45] if goal else "Выполнение задачи"
            newline = '\n'
            return f"⏳ [{time_str}] 🔀 Субагент 1/1: {goal_short}{newline}↳ {action}"

        # Batch mode (N > 1)
        if final_status == "completed" or (active_count == 0 and not final_status):
            return f"✅ [{time_str}] 🔀 Все субагенты ({total_tasks}) завершены"

        if final_status and final_status != "completed":
            return f"❌ [{time_str}] 🔀 Субагенты остановлены ({final_status})"

        lines = [f"⏳ [{time_str}] 🔀 Субагенты ({active_count} активны):"]
        for idx in sorted(self.tasks.keys()):
            task = self.tasks[idx]
            task_status = task.get("status", "running")
            task_elapsed = self._format_time(now - task.get("started_at", self.started_at))
            if task_status == "completed":
                lines.append(f" • #{idx + 1} [{task_elapsed}] ✅ Завершен")
            elif task_status in ("failed", "error"):
                lines.append(f" • #{idx + 1} [{task_elapsed}] ❌ Ошибка")
            else:
                action = task.get("last_action") or ""
                if task.get("path"):
                    action = self.safe_read_transcript(task.get("path"))
                lines.append(f" • #{idx + 1} [{task_elapsed}] {action}")

        return "\n".join(lines)

    async def _send_or_edit(self, text: str) -> None:
        """Send or update status message safely with flood & not-modified handling."""
        if not text or text == self._last_content:
            return

        async with self._lock:
            # Check flood window cooldown if adapter supports it
            cooldown_fn = getattr(self.adapter, "_send_flood_cooldown_remaining", None)
            if cooldown_fn and cooldown_fn(self.chat_id) and cooldown_fn(self.chat_id) > 0:
                logger.debug("Pulse ticker edit skipped due to flood window cooldown")
                return

            try:
                from plugins.platforms.telegram.telegram_status import send_or_update_status
                res = await send_or_update_status(
                    self.adapter,
                    self.chat_id,
                    self.status_key,
                    text,
                    metadata=self.metadata,
                )
                if getattr(res, "success", False):
                    self._last_content = text
                    if getattr(res, "message_id", None):
                        self.current_message_id = str(res.message_id)
                        if self.turn_context is not None and getattr(self.turn_context, "_cleanup_progress", False):
                            msg_ids = getattr(self.turn_context, "_cleanup_msg_ids", None)
                            if isinstance(msg_ids, list) and self.current_message_id not in msg_ids:
                                msg_ids.append(self.current_message_id)
            except Exception as exc:
                err_msg = str(exc).lower()
                # Suppress "message is not modified"
                if "message is not modified" in err_msg:
                    self._last_content = text
                    return

                # Handle flood 429
                try:
                    from plugins.platforms.telegram.telegram_flood_window import note_flood
                    if note_flood(self.adapter, self.chat_id, exc):
                        logger.warning("Telegram flood encountered in pulse ticker: %s", exc)
                        return
                except Exception:
                    pass

                logger.debug("Pulse ticker send_or_edit error (non-fatal): %s", exc)

    async def run_loop(self) -> None:
        """Main periodic refresh loop."""
        try:
            while self.active:
                text = self._render_card()
                await self._send_or_edit(text)

                # Stop if elapsed >= 900
                if time.time() - self.started_at >= 900:
                    self.active = False
                    break

                await asyncio.sleep(self.interval)
        except asyncio.CancelledError:
            # User cancellation or turn cleanup
            cancel_card = self._render_card(cancelled=True)
            try:
                await self._send_or_edit(cancel_card)
            except Exception:
                pass
            raise
        except Exception as exc:
            logger.debug("Unexpected error in SubagentPulseTicker loop: %s", exc)

    async def stop(self, status: str = "completed", summary: str = "") -> None:
        """Stop ticker and render final state card."""
        self.active = False
        if summary and 0 in self.tasks:
            self.tasks[0]["summary"] = summary
        if self.ticker_task and not self.ticker_task.done():
            self.ticker_task.cancel()
            try:
                await self.ticker_task
            except (asyncio.CancelledError, Exception):
                pass
            self.ticker_task = None

        final_card = self._render_card(final_status=status)
        try:
            await self._send_or_edit(final_card)
        except Exception as exc:
            logger.debug("Final pulse render error (non-fatal): %s", exc)
