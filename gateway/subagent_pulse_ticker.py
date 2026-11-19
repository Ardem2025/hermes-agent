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
import inspect
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from agent.display import get_tool_verb

logger = logging.getLogger("gateway.subagent_pulse_ticker")


def _is_ru_language() -> bool:
    """Return True when active UI language is Russian."""
    try:
        from agent.i18n import get_language
        return get_language() == "ru"
    except Exception:
        return False


class SubagentPulseTicker:
    """Manages background pulse ticker updates for subagents in Telegram."""

    def __init__(
        self,
        adapter: Any,
        chat_id: str | int,
        thread_id: Optional[int | str] = None,
        interval_seconds: float = 10.0,
        turn_context: Optional[Any] = None,
        metadata: Optional[dict[str, Any]] = None,
        cleanup_progress: Optional[bool] = None,
    ) -> None:
        self.adapter = adapter
        self.chat_id = str(chat_id)
        self.thread_id = thread_id
        self.interval = max(5.0, float(interval_seconds))
        self.turn_context = turn_context
        if cleanup_progress is None:
            self.cleanup_progress = bool(getattr(turn_context, "_cleanup_progress", False))
        else:
            self.cleanup_progress = bool(cleanup_progress)
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
        is_ru = _is_ru_language()
        default_goal = f"Задача #{task_index + 1}" if is_ru else f"Task #{task_index + 1}"
        init_action = "🚀 Инициализация окружения..." if is_ru else "🚀 Initializing environment..."
        self.tasks[task_index] = {
            "goal": goal or default_goal,
            "path": path_obj,
            "started_at": time.time(),
            "last_action": init_action,
            "status": "running",
            "summary": None,
        }

    register_task = on_subagent_start

    def on_subagent_progress(self, task_index: int = 0, action: str = "") -> None:
        """Update current action of subagent."""
        if task_index in self.tasks:
            self.tasks[task_index]["last_action"] = action
        else:
            default_goal = f"Задача #{task_index + 1}" if _is_ru_language() else f"Task #{task_index + 1}"
            self.tasks[task_index] = {
                "goal": default_goal,
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
        is_ru = _is_ru_language()
        done_label = "✅ Завершен" if is_ru else "✅ Completed"
        err_label = f"❌ Ошибка ({status})" if is_ru else f"❌ Error ({status})"
        if task_index in self.tasks:
            self.tasks[task_index]["status"] = status
            self.tasks[task_index]["summary"] = summary
            self.tasks[task_index]["last_action"] = done_label if status == "completed" else err_label
        else:
            default_goal = f"Задача #{task_index + 1}" if is_ru else f"Task #{task_index + 1}"
            self.tasks[task_index] = {
                "goal": default_goal,
                "path": None,
                "started_at": time.time(),
                "last_action": done_label if status == "completed" else f"❌ {status}",
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
        """Safely read tail of live transcript log and map to localized tool action."""
        is_ru = _is_ru_language()
        init_msg = "🚀 Инициализация окружения..." if is_ru else "🚀 Initializing environment..."
        running_msg = "⚙️ Выполнение..." if is_ru else "⚙️ Running..."
        if not path:
            return init_msg
        path_obj = Path(path)
        if not path_obj.exists():
            return init_msg
        try:
            stat = path_obj.stat()
            if stat.st_size == 0:
                return init_msg
            read_size = min(4096, stat.st_size)
            with path_obj.open("r", encoding="utf-8", errors="replace") as f:
                if stat.st_size > read_size:
                    f.seek(stat.st_size - read_size)
                content = f.read()
            return SubagentPulseTicker._parse_action_from_content(content)
        except (FileNotFoundError, OSError, UnicodeDecodeError):
            return running_msg

    @staticmethod
    def _parse_action_from_content(content: str) -> str:
        """Extract tool action line from transcript tail."""
        is_ru = _is_ru_language()
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        for line in reversed(lines):
            if " tool " in line and "->" in line:
                action_part = line.split("->", 1)[-1].strip()
                # Parse tool name and arg preview
                match = re.match(r"^([a-zA-Z0-9_-]+)\((.*)\)$", action_part)
                if match:
                    tool_name, args = match.groups()
                    tool_label = get_tool_verb(tool_name) or tool_name
                    arg_preview = args.strip()[:35]
                    if arg_preview:
                        return f"⚙️ {tool_label} ({arg_preview})"
                    return f"⚙️ {tool_label}"
                return f"⚙️ {action_part[:40]}"
            if " result " in line:
                result_part = line.split(" result ", 1)[-1].strip()
                prefix = "Обработка" if is_ru else "Processing"
                return f"⚙️ {prefix}: {result_part[:35]}"
            if " start " in line:
                return "🚀 Запуск шага..." if is_ru else "🚀 Starting step..."
        return "⚙️ Выполнение..." if is_ru else "⚙️ Running..."

    def _format_time(self, seconds: float, *, quantize_secs: int = 0) -> str:
        """Format seconds into MM:SS (optionally quantized down to quantize_secs)."""
        secs = max(0, int(seconds))
        if quantize_secs > 1:
            secs = (secs // quantize_secs) * quantize_secs
        mins, s = divmod(secs, 60)
        return f"{mins:02d}:{s:02d}"

    def _render_card(self, cancelled: bool = False, final_status: Optional[str] = None) -> str:
        """Render text card according to BDD specifications and active language."""
        now = time.time()
        elapsed = now - self.started_at
        time_str = self._format_time(elapsed)
        interim_time_str = self._format_time(elapsed, quantize_secs=30)
        is_ru = _is_ru_language()

        if cancelled:
            if is_ru:
                return f"⏹️ [{time_str}] 🔀 Выполнение отменено пользователем"
            return f"⏹️ [{time_str}] 🔀 Execution cancelled by user"

        # Check global timeout
        if elapsed >= 900:
            if is_ru:
                return "⏱️ [15:00] 🔀 Превышен таймаут выполнения (900s)"
            return "⏱️ [15:00] 🔀 Execution timeout exceeded (900s)"

        total_tasks = len(self.tasks)
        if total_tasks == 0:
            prep_time = time_str if final_status else interim_time_str
            if is_ru:
                return f"⏳ [{prep_time}] 🔀 Субагент: подготовка..."
            return f"⏳ [{prep_time}] 🔀 Subagent: preparing..."

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
                    if is_ru:
                        return f"✅ [{time_str}] 🔀 Субагент 1/1 завершен{summary_snippet}"
                    return f"✅ [{time_str}] 🔀 Subagent 1/1 completed{summary_snippet}"
                if is_ru:
                    return f"❌ [{time_str}] 🔀 Субагент 1/1 ошибка ({st}){summary_snippet}"
                return f"❌ [{time_str}] 🔀 Subagent 1/1 error ({st}){summary_snippet}"

            # Running
            action = task.get("last_action") or ""
            if task.get("path"):
                action = self.safe_read_transcript(task.get("path"))
            default_goal = "Выполнение задачи" if is_ru else "Running task"
            goal_short = goal[:45] if goal else default_goal
            newline = '\n'
            label = "Субагент 1/1" if is_ru else "Subagent 1/1"
            return f"⏳ [{interim_time_str}] 🔀 {label}: {goal_short}{newline}↳ {action}"

        # Batch mode (N > 1)
        if final_status == "completed" or (active_count == 0 and not final_status):
            if is_ru:
                return f"✅ [{time_str}] 🔀 Все субагенты ({total_tasks}) завершены"
            return f"✅ [{time_str}] 🔀 All subagents ({total_tasks}) completed"

        if final_status and final_status != "completed":
            if is_ru:
                return f"❌ [{time_str}] 🔀 Субагенты остановлены ({final_status})"
            return f"❌ [{time_str}] 🔀 Subagents stopped ({final_status})"

        header = (
            f"⏳ [{interim_time_str}] 🔀 Субагенты ({active_count} активны):"
            if is_ru
            else f"⏳ [{interim_time_str}] 🔀 Subagents ({active_count} active):"
        )
        lines = [header]
        for idx in sorted(self.tasks.keys()):
            task = self.tasks[idx]
            task_status = task.get("status", "running")
            task_elapsed = self._format_time(now - task.get("started_at", self.started_at), quantize_secs=30)
            if task_status == "completed":
                done_txt = "✅ Завершен" if is_ru else "✅ Completed"
                lines.append(f" • #{idx + 1} [{task_elapsed}] {done_txt}")
            elif task_status in ("failed", "error"):
                err_txt = "❌ Ошибка" if is_ru else "❌ Error"
                lines.append(f" • #{idx + 1} [{task_elapsed}] {err_txt}")
            else:
                action = task.get("last_action") or ""
                if task.get("path"):
                    action = self.safe_read_transcript(task.get("path"))
                lines.append(f" • #{idx + 1} [{task_elapsed}] {action}")

        return "\n".join(lines)

    async def _send_or_edit(self, text: str, *, finalize: bool = False) -> None:
        """Send or update status message safely with flood & not-modified handling."""
        if not text or text == self._last_content:
            return

        async with self._lock:
            if not finalize:
                # Check flood window cooldown if adapter supports it
                cooldown_fn = getattr(self.adapter, "_send_flood_cooldown_remaining", None)
                if callable(cooldown_fn):
                    rem = cooldown_fn(self.chat_id)
                    if isinstance(rem, (int, float)) and rem > 0:
                        logger.debug("Pulse ticker edit skipped due to flood window cooldown")
                        return
                # Check shared per-chat outbound slot budget for interim edits
                if self.current_message_id is not None:
                    slot_fn = getattr(self.adapter, "_chat_outbound_slot_remaining", None)
                    if callable(slot_fn):
                        rem = slot_fn(self.chat_id)
                        if isinstance(rem, (int, float)) and rem > 0:
                            logger.debug("Pulse ticker interim edit skipped due to outbound slot budget")
                            return

            try:
                from plugins.platforms.telegram.telegram_status import send_or_update_status
                res = await send_or_update_status(
                    self.adapter,
                    self.chat_id,
                    self.status_key,
                    text,
                    metadata=self.metadata,
                    finalize=finalize,
                )
                if getattr(res, "success", False):
                    raw = getattr(res, "raw_response", None)
                    if not (isinstance(raw, dict) and raw.get("skipped")):
                        self._last_content = text
                    if getattr(res, "message_id", None):
                        self.current_message_id = str(res.message_id)
                        # Delegation pulse cards stay visible in chat across parent turn completion
                        # (even when cleanup_progress=True) so background subagents remain observable.
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
                is_timeout = time.time() - self.started_at >= 900
                text = self._render_card()
                await self._send_or_edit(text, finalize=is_timeout)

                # Stop if elapsed >= 900
                if is_timeout:
                    self.active = False
                    break

                await asyncio.sleep(self.interval)
        except asyncio.CancelledError:
            # User cancellation or external task abort (skip when stop() already set active=False)
            if self.active:
                cancel_card = self._render_card(cancelled=True)
                try:
                    await self._send_or_edit(cancel_card, finalize=True)
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
            await self._send_or_edit(final_card, finalize=True)
        except Exception as exc:
            logger.debug("Final pulse render error (non-fatal): %s", exc)

        if status == "completed" and self.cleanup_progress and self.current_message_id:
            delete_fn = getattr(self.adapter, "delete_message", None)
            if callable(delete_fn):
                try:
                    res = delete_fn(self.chat_id, self.current_message_id)
                    if inspect.isawaitable(res):
                        await res
                except Exception as exc:
                    logger.debug("Pulse card cleanup delete_message error (non-fatal): %s", exc)
            status_ids = getattr(self.adapter, "_status_message_ids", None)
            if isinstance(status_ids, dict):
                status_ids.pop((str(self.chat_id), str(self.status_key)), None)
