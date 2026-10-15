"""
Secretary Plugin — Hardened Telegram Business API message interceptor and assistant.

Features:
- Intercepts incoming business messages from customers and generates AI response options.
- Fail-Safe Prompt Leak Guard (regex & heuristic pattern filtering, prevents prompt leakage).
- Telegram Inline Keyboard (Click-to-Send buttons: [1️⃣], [2️⃣], [3️⃣], [🚫 Спам], [❌ Игнор]).
- Callback Query handler via `telegram_callback_query` hook.
- Configurable dynamic Obsidian paths (inbox_subpath, log_subpath, contacts_subpath).
- SQLite Chronicle Event Logging (`chronicle_events` table).
- Spam blacklist persistence and Dmitry / Admin pre-notification hooks.
- Strict authorization gates and safe sticky context (with prompt leak protection).
"""

import os
import re
import asyncio
import datetime
import logging
import json
import sqlite3
import uuid
import yaml
import time as _time
from pathlib import Path
from collections import OrderedDict
from typing import Tuple, Optional, Dict, Any, List

from gateway.config import Platform

# Configure logger
logger = logging.getLogger("hermes.plugin.secretary")
logger.setLevel(logging.INFO)

_LAST_ACTIVE_CLIENT = {}
STICKY_CONTEXT_TTL = 900  # 15 minutes
_PENDING_CONFIRMATIONS: Dict[str, Dict[str, Any]] = {}
CONFIRMATION_TTL = 300  # 5 minutes

_OBSERVED_BUSINESS_MESSAGE_IDS = OrderedDict()
_OBSERVED_BUSINESS_MESSAGE_ID_LIMIT = 2048
_OBSERVED_BUSINESS_MESSAGE_TTL = 900
_OBSERVATION_NAMESPACE = "observation:telegram-business"

# Prompt Leak Guard regex patterns
LEAK_PATTERNS = [
    re.compile(r"^\s*(?:промпт|prompt|нарисуй|сделай|сгенерируй|напиши|придумай|создай|код|запрос|midjourney|nanabanana|flux|dall-?e)\b", re.IGNORECASE),
    re.compile(r"(?:шпаргалка|заметка|todo|отдать завтра|срочно ушко|брак литья|клиенты список|чек-лист)", re.IGNORECASE),
    re.compile(r"^\s*[/!]", re.IGNORECASE),
    re.compile(r"(?:system prompt|instruction|assistant rules|context window|llm|gpt-)", re.IGNORECASE),
]


def is_prompt_or_internal_command(text: str) -> Tuple[bool, str]:
    """Check whether the provided text appears to be an internal AI command or note."""
    if not text:
        return False, ""
    clean = text.strip()
    for pat in LEAK_PATTERNS:
        match = pat.search(clean)
        if match:
            return True, match.group(0)
    return False, ""


def _business_observation(*, reason):
    """Fail closed before gateway auth/session/reply for Business updates."""
    return {
        "action": "skip",
        "reason": reason,
        "storage_namespace": _OBSERVATION_NAMESPACE,
        "suppress_session": True,
        "suppress_reply": True,
        "metadata": {"kind": "telegram_business", "redacted": True},
    }


def _remember_business_update(event, now=None):
    """Bounded TTL/LRU replay cache scoped to Telegram Business conversation."""
    source = getattr(event, "source", None)
    message_id = getattr(event, "message_id", None)
    if not source or message_id is None:
        return True
    key = (
        str(getattr(source, "telegram_business_connection_id", "")),
        str(getattr(source, "chat_id", "")),
        str(message_id),
        str(getattr(event, "update_type", type(event).__name__)),
    )
    now = _time.monotonic() if now is None else now
    while _OBSERVED_BUSINESS_MESSAGE_IDS:
        _, seen = next(iter(_OBSERVED_BUSINESS_MESSAGE_IDS.items()))
        if now - seen <= _OBSERVED_BUSINESS_MESSAGE_TTL:
            break
        _OBSERVED_BUSINESS_MESSAGE_IDS.popitem(last=False)
    if key in _OBSERVED_BUSINESS_MESSAGE_IDS:
        _OBSERVED_BUSINESS_MESSAGE_IDS.move_to_end(key)
        return False
    _OBSERVED_BUSINESS_MESSAGE_IDS[key] = now
    while len(_OBSERVED_BUSINESS_MESSAGE_IDS) > _OBSERVED_BUSINESS_MESSAGE_ID_LIMIT:
        _OBSERVED_BUSINESS_MESSAGE_IDS.popitem(last=False)
    return True


# ============================================================================
# CONFIGURATION LOADER
# ============================================================================
CONFIG_PATH = Path(os.path.expanduser("~/.hermes/config.yaml"))
settings = {}

if CONFIG_PATH.exists():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            if cfg and isinstance(cfg, dict):
                settings = cfg.get("plugins", {}).get("settings", {}).get("secretary", {}) or {}
    except Exception as e:
        logger.error(f"[Secretary] Failed to read settings from config.yaml: {e}")

OWNER_CHAT_ID = str(settings.get("owner_chat_id", os.environ.get("SECRETARY_OWNER_CHAT_ID", "REPLACE_WITH_OWNER_CHAT_ID")))
OWNER_NAME = settings.get("owner_name", os.environ.get("OWNER_NAME", "Владелец"))
BUSINESS_DESC = settings.get("business_desc", os.environ.get("BUSINESS_DESC", "бизнес"))
OBSIDIAN_VAULT = settings.get("obsidian_vault", os.environ.get("SECRETARY_VAULT_PATH", "/var/www/obsidian-vault/Knowledge"))
raw_thread_id = settings.get("owner_thread_id", os.environ.get("SECRETARY_OWNER_THREAD_ID"))
OWNER_THREAD_ID = str(raw_thread_id) if raw_thread_id is not None else None

# Dynamic subpaths
INBOX_SUBPATH = settings.get("inbox_subpath", "Secretary/Inbox")
LOG_SUBPATH = settings.get("log_subpath", "Secretary/Log")
CONTACTS_SUBPATH = settings.get("contacts_subpath", "Secretary/Contacts")
LEAK_GUARD_ENABLED = bool(settings.get("leak_guard_enabled", True))
SQLITE_CHRONICLE_PATH = settings.get("sqlite_chronicle_path", os.environ.get("SECRETARY_CHRONICLE_DB", os.path.expanduser("~/.hermes/secretary_chronicle.db")))

# Trusted admin IDs for callbacks and management
TRUSTED_ADMIN_IDS = {"139351986", OWNER_CHAT_ID}
raw_trusted = settings.get("trusted_admin_ids", [])
if isinstance(raw_trusted, list):
    for tid in raw_trusted:
        TRUSTED_ADMIN_IDS.add(str(tid))


def resolve_vault_path(subpath: str) -> Path:
    """Resolve subpath against OBSIDIAN_VAULT if relative, or return absolute path."""
    p = Path(subpath)
    if p.is_absolute():
        return p
    return Path(OBSIDIAN_VAULT) / p


SECRETARY_DIR = resolve_vault_path("Secretary")
INBOX_DIR = resolve_vault_path(INBOX_SUBPATH)
LOG_DIR = resolve_vault_path(LOG_SUBPATH)
CONTACTS_DIR = resolve_vault_path(CONTACTS_SUBPATH)

# Ensure directories exist
for d in [SECRETARY_DIR, INBOX_DIR, LOG_DIR, CONTACTS_DIR]:
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"[Secretary] Failed to create directory {d}: {e}")


def _fix_permissions(path):
    """Set permissions to 0o664 and owner www-data if running as root."""
    try:
        p = Path(path)
        if not p.exists():
            return
        if os.geteuid() == 0:
            import pwd
            import grp
            try:
                uid = pwd.getpwnam("www-data").pw_uid
                gid = grp.getgrnam("www-data").gr_gid
                os.chown(str(p), uid, gid)
            except KeyError:
                pass
        if p.is_dir():
            os.chmod(str(p), 0o775)
        else:
            os.chmod(str(p), 0o664)
    except Exception as e:
        logger.error(f"[Secretary] Failed to fix permissions on {path}: {e}")


# ============================================================================
# SQLITE CHRONICLE LOGGER
# ============================================================================
class SecretaryChronicle:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        p = Path(self.db_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(p), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        try:
            with self._get_conn() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS chronicle_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_uuid TEXT NOT NULL UNIQUE,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        event_type TEXT NOT NULL,
                        client_id TEXT NOT NULL,
                        client_name TEXT,
                        business_connection_id TEXT,
                        incoming_text TEXT,
                        classification TEXT,
                        summary TEXT,
                        options_json TEXT,
                        selected_option INTEGER,
                        outbound_text TEXT,
                        delivery_status TEXT,
                        error_message TEXT
                    );
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_chronicle_client ON chronicle_events(client_id);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_chronicle_ts ON chronicle_events(timestamp);")
                conn.commit()
            _fix_permissions(self.db_path)
        except Exception as e:
            logger.error(f"[SecretaryChronicle] Database init failed: {e}")

    def log_event(
        self,
        event_type: str,
        client_id: str,
        client_name: Optional[str] = None,
        business_connection_id: Optional[str] = None,
        incoming_text: Optional[str] = None,
        classification: Optional[str] = None,
        summary: Optional[str] = None,
        options: Optional[List[str]] = None,
        selected_option: Optional[int] = None,
        outbound_text: Optional[str] = None,
        delivery_status: str = "success",
        error_message: Optional[str] = None,
    ) -> str:
        event_uuid = str(uuid.uuid4())
        options_json = json.dumps(options, ensure_ascii=False) if options is not None else None
        try:
            with self._get_conn() as conn:
                conn.execute("""
                    INSERT INTO chronicle_events (
                        event_uuid, event_type, client_id, client_name,
                        business_connection_id, incoming_text, classification,
                        summary, options_json, selected_option, outbound_text,
                        delivery_status, error_message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event_uuid, event_type, str(client_id), client_name,
                    business_connection_id, incoming_text, classification,
                    summary, options_json, selected_option, outbound_text,
                    delivery_status, error_message
                ))
                conn.commit()
            _fix_permissions(self.db_path)
        except Exception as e:
            logger.error(f"[SecretaryChronicle] Failed to log event {event_type}: {e}")
        return event_uuid


chronicle = SecretaryChronicle(SQLITE_CHRONICLE_PATH)


def _schedule_async(coro):
    """Schedule an async coroutine from synchronous context within a running event loop."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(coro, loop=loop)
        else:
            loop.run_until_complete(coro)
    except RuntimeError:
        logger.warning("[Secretary] No event loop available for async operation")


def _generate_options(client_name, client_id, client_text):
    """Generate response options using synchronous PluginLlm.complete()."""
    try:
        from agent.plugin_llm import PluginLlm
        llm = PluginLlm(plugin_id="secretary")

        system_prompt = (
            f"Ты — профессиональный AI-секретарь {OWNER_NAME}, помогающий в ведении {BUSINESS_DESC}. "
            f"{OWNER_NAME} общается с клиентами. Твоя роль — составить краткое резюме и 3 подходящих варианта "
            f"ответа для {OWNER_NAME}. Общайся вежливо, дружелюбно, как высококлассный партнер. "
            f"Не используй технический жаргон. "
            f"Верни ответ строго в формате JSON с ключами: 'summary', 'classification', 'options' (массив из 3 строк ответы от лица {OWNER_NAME})."
        )

        user_msg = (
            f"Входящее сообщение от клиента {client_name} (ID: {client_id}):\n"
            f"\"{client_text}\"\n\n"
            f"Верни JSON с summary, classification (выбери один смайлик: 🔴 Срочное / 🟡 Важное / 🟢 Информационное / ⚪ Спам) и options."
        )

        result = llm.complete(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg}
            ]
        )
        raw_text = result.text.strip()
        if raw_text.startswith("```"):
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_text)
            if match:
                raw_text = match.group(1).strip()
        if not raw_text.startswith("{"):
            start_idx = raw_text.find("{")
            end_idx = raw_text.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                raw_text = raw_text[start_idx:end_idx+1]
        data = json.loads(raw_text)
        summary = data.get("summary", "Без резюме")
        classification = data.get("classification", "🟡 Важное")
        options = data.get("options", ["Вариант 1", "Вариант 2", "Вариант 3"])
        return summary, classification, options
    except Exception as e:
        logger.error(f"[Secretary] LLM call failed: {e}")
        return (
            "Не удалось сгенерировать резюме",
            "🟡 Важное",
            [
                "Добрый день! Спасибо за сообщение, скоро отвечу.",
                "Приветствую! Подскажите, пожалуйста, какой именно вопрос вас интересует?",
                "Здравствуйте! Принял ваш запрос, уточню детали."
            ]
        )


def build_secretary_inline_keyboard(client_id: str) -> Dict[str, Any]:
    """Build Telegram InlineKeyboardMarkup representation for the card."""
    return {
        "inline_keyboard": [
            [
                {"text": "1️⃣ Вариант 1", "callback_data": f"sec:opt:{client_id}:1"},
                {"text": "2️⃣ Вариант 2", "callback_data": f"sec:opt:{client_id}:2"},
                {"text": "3️⃣ Вариант 3", "callback_data": f"sec:opt:{client_id}:3"},
            ],
            [
                {"text": "🚫 Спам / Бан", "callback_data": f"sec:spam:{client_id}"},
                {"text": "❌ Игнор", "callback_data": f"sec:dismiss:{client_id}"},
            ]
        ]
    }


def build_confirmation_inline_keyboard(confirm_id: str) -> Dict[str, Any]:
    """Build InlineKeyboardMarkup representation for leak guard confirmation."""
    return {
        "inline_keyboard": [
            [
                {"text": "✅ Все равно отправить", "callback_data": f"sec:conf:{confirm_id}"},
                {"text": "❌ Отмена", "callback_data": f"sec:cancel:{confirm_id}"},
            ]
        ]
    }


# ============================================================================
# DISPATCH HOOKS
# ============================================================================
def pre_gateway_dispatch(event, gateway, session_store, **kwargs):
    """Synchronous hook — called by invoke_hook() in gateway dispatch pipeline."""
    source = getattr(event, "source", None)
    if not source or source.platform != Platform.TELEGRAM:
        return None

    business_conn_id = getattr(source, "telegram_business_connection_id", None)
    if business_conn_id and not _remember_business_update(event):
        logger.info("[SecretaryDecision] decision=deny namespace=%s reason=replay", _OBSERVATION_NAMESPACE)
        return _business_observation(reason="replay")

    # Load spam blacklist
    blacklist_path = SECRETARY_DIR / "spam_blacklist.json"
    blacklist = set()
    if blacklist_path.exists():
        try:
            with open(blacklist_path, "r", encoding="utf-8") as f:
                blacklist = set(json.load(f))
        except Exception:
            pass

    # Check if the message is from a bot
    is_bot = getattr(source, "is_bot", False)
    if not is_bot:
        raw_msg = getattr(event, "raw_message", None)
        if raw_msg:
            from_user = getattr(raw_msg, "from_user", None)
            if from_user:
                is_bot = getattr(from_user, "is_bot", False)
            elif isinstance(raw_msg, dict):
                from_user_dict = raw_msg.get("from_user")
                if isinstance(from_user_dict, dict):
                    is_bot = from_user_dict.get("is_bot", False)
                else:
                    is_bot = raw_msg.get("is_bot", False)

    if is_bot:
        logger.info("[SecretaryDecision] decision=deny namespace=%s reason=bot", _OBSERVATION_NAMESPACE)
        return _business_observation(reason="bot") if business_conn_id else {"action": "skip", "reason": "Message from bot"}

    telegram_adapter = gateway.adapters.get(Platform.TELEGRAM)
    bot_id = None
    if telegram_adapter and getattr(telegram_adapter, "_bot", None):
        bot_id = getattr(telegram_adapter._bot, "id", None)

    bot_id_str = str(bot_id) if bot_id is not None else None
    client_id_str = str(source.user_id) if source.user_id else ""
    chat_id_str = str(source.chat_id) if source.chat_id else ""
    owner_id_str = str(OWNER_CHAT_ID)
    dmitry_id_str = "139351986"

    ignore_ids = {owner_id_str, dmitry_id_str, "8482395251"}
    if bot_id_str:
        ignore_ids.add(bot_id_str)

    # 1. Filter business messages
    if business_conn_id:
        is_ignored_chat = (
            client_id_str in ignore_ids or
            chat_id_str in ignore_ids or
            ":" in chat_id_str or
            chat_id_str.startswith("-")
        )
        if is_ignored_chat:
            logger.info("[SecretaryDecision] decision=deny namespace=%s reason=control_or_group", _OBSERVATION_NAMESPACE)
            return _business_observation(reason="control_or_group")
    else:
        # Normal non-business messages
        is_ignored_chat = (
            client_id_str in ignore_ids or
            chat_id_str in ignore_ids or
            ":" in chat_id_str or
            chat_id_str.startswith("-")
        )
        is_owner_card_reply = (
            str(source.chat_id) == str(OWNER_CHAT_ID) and
            bool(event.reply_to_message_id)
        )
        if is_ignored_chat and not is_owner_card_reply:
            return None

    client_id = source.user_id
    client_name = source.user_name or f"Client_{client_id}"
    client_text = event.text or ""

    # 3. Handle incoming business message from customer
    if business_conn_id:
        if str(source.user_id) != str(source.chat_id):
            logger.info("[SecretaryDecision] decision=deny namespace=%s reason=outgoing", _OBSERVATION_NAMESPACE)
            return _business_observation(reason="outgoing")

        if client_id_str in blacklist:
            logger.info("[SecretaryDecision] decision=deny namespace=%s reason=blacklisted", _OBSERVATION_NAMESPACE)
            return _business_observation(reason="blacklisted")

        logger.info("[SecretaryDecision] decision=allow namespace=%s reason=business_inbound", _OBSERVATION_NAMESPACE)

        summary, classification, options = _generate_options(client_name, client_id, client_text)
        is_spam = "Спам" in classification or classification == "⚪ Спам"

        # Log inbound event to SQLite Chronicle
        chronicle.log_event(
            event_type="inbound_message",
            client_id=str(client_id),
            client_name=client_name,
            business_connection_id=str(business_conn_id),
            incoming_text=client_text,
            classification=classification,
            summary=summary,
            options=options,
            delivery_status="spam_detected" if is_spam else "received"
        )

        # Write to Obsidian Log & Inbox ONLY if it's not Spam
        if not is_spam:
            try:
                today_str = datetime.date.today().strftime("%Y-%m-%d")
                log_file = LOG_DIR / f"{today_str}.md"
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")

                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"\n- **{timestamp}** | Client: {client_name} ({client_id}) | Msg: {client_text}\n")
                _fix_permissions(log_file)

                inbox_file = INBOX_DIR / f"client_{client_id}.md"
                file_exists = inbox_file.exists()
                with open(inbox_file, "a", encoding="utf-8") as f:
                    if not file_exists or inbox_file.stat().st_size == 0:
                        f.write(f"# Chat History with {client_name} ({client_id})\n\n")
                    f.write(f"- [{timestamp}] CLIENT: {client_text}\n")
                _fix_permissions(inbox_file)
            except Exception as e:
                logger.error(f"[Secretary] Failed to write to Obsidian: {e}")

        # Escape brackets
        opt_1 = options[0].replace("[", "\\[").replace("]", "\\]")
        opt_2 = options[1].replace("[", "\\[").replace("]", "\\]")
        opt_3 = options[2].replace("[", "\\[").replace("]", "\\]")

        reply_markup = build_secretary_inline_keyboard(str(client_id))

        if is_spam:
            formatted_msg = (
                f"📨 **Спам-сигнал от:** {client_name}\n"
                f"📋 **Категория:** ⚪ Спам\n"
                f"💬 **Суть:** {summary}\n"
                f"✉️ **Текст:**\n> {client_text}\n\n"
                f"**Я распознал это как спам.** Сообщение НЕ добавлено в Obsidian CRM.\n"
                f"Нажмите кнопку ниже или ответьте **спам** / **бан**.\n\n"
                f"|| [client_id:{client_id}][conn_id:{business_conn_id}][v1:{opt_1}][v2:{opt_2}][v3:{opt_3}] ||"
            )
        else:
            formatted_msg = (
                f"📨 **Входящее от:** {client_name}\n"
                f"📋 **Категория:** {classification}\n"
                f"💬 **Суть:** {summary}\n"
                f"✉️ **Текст:**\n> {client_text}\n\n"
                f"**Варианты ответов:**\n"
                f"1️⃣ {opt_1}\n"
                f"2️⃣ {opt_2}\n"
                f"3️⃣ {opt_3}\n\n"
                f"**Как ответить клиенту:**\n"
                f"Нажмите одну из кнопок ниже для отправки в 1 клик,\n"
                f"или сделайте **Reply** с номером (`1`, `2`, `3`) или текстом ответа.\n\n"
                f"|| [client_id:{client_id}][conn_id:{business_conn_id}][v1:{opt_1}][v2:{opt_2}][v3:{opt_3}] ||"
            )

        # Send notification to Owner with inline keyboard
        if telegram_adapter:
            send_metadata = {"reply_markup": reply_markup}
            if OWNER_THREAD_ID:
                send_metadata["thread_id"] = OWNER_THREAD_ID
            _schedule_async(telegram_adapter.send(chat_id=OWNER_CHAT_ID, content=formatted_msg, metadata=send_metadata))
            logger.info(f"[Secretary] Forwarded customer {client_id} message with buttons to owner ({OWNER_CHAT_ID})")
        else:
            logger.error("[Secretary] Telegram adapter not found in gateway!")

        _sticky_key = str(OWNER_THREAD_ID or OWNER_CHAT_ID)
        _LAST_ACTIVE_CLIENT[_sticky_key] = {
            "client_id": str(client_id),
            "conn_id": str(business_conn_id),
            "client_name": client_name,
            "options": [options[0], options[1], options[2]] if (options and len(options) >= 3) else [],
            "timestamp": _time.time(),
        }

        return _business_observation(reason="business_inbound")

    # 4. Handle Owner's reply to client card
    if str(source.chat_id) == str(OWNER_CHAT_ID) and event.reply_to_message_id:
        reply_to_text = event.reply_to_text or ""

        meta_match = re.search(
            r"\[client_id:(\d+)\]\[conn_id:([^\]]+)\]\[v1:([^\]]*)\]\[v2:([^\]]*)\]\[v3:([^\]]*)\]",
            reply_to_text
        )
        if meta_match:
            logger.info("[Secretary] Detected owner's reply to client card")

            client_id = meta_match.group(1)
            conn_id = meta_match.group(2)
            v1 = meta_match.group(3)
            v2 = meta_match.group(4)
            v3 = meta_match.group(5)

            owner_text = (event.text or "").strip()

            # Check if this is a spam/block request
            if owner_text.lower() in ["спам", "spam", "бан", "ban"]:
                _handle_spam_block(client_id, client_name, telegram_adapter, event.message_id)
                return {"action": "skip", "reason": "Manually blacklisted spammer"}

            # Prompt Leak Guard verification
            if LEAK_GUARD_ENABLED and owner_text not in ["1", "2", "3"]:
                is_leak, leak_term = is_prompt_or_internal_command(owner_text)
                if is_leak:
                    logger.warning("[PromptLeakGuard] Blocked prompt leak to customer %s: '%s' (matched '%s')", client_id, owner_text, leak_term)
                    chronicle.log_event(
                        event_type="prompt_leak_prevented",
                        client_id=str(client_id),
                        business_connection_id=str(conn_id),
                        outbound_text=owner_text,
                        delivery_status="blocked",
                        error_message=f"Matched leak pattern: {leak_term}"
                    )
                    confirm_id = str(uuid.uuid4())[:8]
                    _PENDING_CONFIRMATIONS[confirm_id] = {
                        "client_id": str(client_id),
                        "conn_id": str(conn_id),
                        "text": owner_text,
                        "timestamp": _time.time(),
                    }

                    warn_msg = (
                        f"⚠️ **Внимание! Защита от утечки промптов!**\n\n"
                        f"Ваш ответ клиенту `{client_id}` похож на команду для ИИ или внутреннюю заметку:\n"
                        f"> {owner_text}\n\n"
                        f"Отправка клиенту заблокирована для безопасности. Если вы действительно хотите отправить этот текст клиенту, нажмите подтверждение:"
                    )
                    if telegram_adapter:
                        send_metadata = {"reply_markup": build_confirmation_inline_keyboard(confirm_id)}
                        if OWNER_THREAD_ID:
                            send_metadata["thread_id"] = OWNER_THREAD_ID
                        _schedule_async(telegram_adapter.send(
                            chat_id=OWNER_CHAT_ID,
                            content=warn_msg,
                            reply_to=event.message_id,
                            metadata=send_metadata
                        ))
                    return {"action": "skip", "reason": "Prompt leak prevented"}

            # Select option text
            selected_opt_num = None
            if owner_text == "1":
                text_to_send = v1
                selected_opt_num = 1
            elif owner_text == "2":
                text_to_send = v2
                selected_opt_num = 2
            elif owner_text == "3":
                text_to_send = v3
                selected_opt_num = 3
            else:
                text_to_send = owner_text

            if not text_to_send:
                text_to_send = owner_text

            _dispatch_reply_to_customer(
                client_id=str(client_id),
                conn_id=str(conn_id),
                text_to_send=text_to_send,
                selected_opt_num=selected_opt_num,
                telegram_adapter=telegram_adapter,
                reply_to_msg_id=event.message_id
            )
            return {"action": "skip", "reason": "Processed owner's decision"}

    # 5. Sticky context fallback logic (STRICTLY GATED: numbers only or guarded)
    if str(source.chat_id) == str(OWNER_CHAT_ID) and str(source.thread_id or "") == str(OWNER_THREAD_ID or ""):
        _sticky_key = str(OWNER_THREAD_ID or OWNER_CHAT_ID)
        last_active = _LAST_ACTIVE_CLIENT.get(_sticky_key)
        if last_active:
            age = _time.time() - last_active["timestamp"]
            if age < STICKY_CONTEXT_TTL:
                owner_text = (event.text or "").strip()

                # ONLY accept '1', '2', '3', or explicit 'спам' in sticky context!
                # Arbitrary text fallthrough in sticky context is disabled to prevent catastrophic leaks!
                if owner_text in ["1", "2", "3"] or owner_text.lower() in ["спам", "spam", "бан", "ban"]:
                    client_id = last_active["client_id"]
                    conn_id = last_active["conn_id"]
                    client_name = last_active["client_name"]
                    options = last_active["options"]

                    if owner_text.lower() in ["спам", "spam", "бан", "ban"]:
                        _handle_spam_block(client_id, client_name, telegram_adapter, event.message_id)
                        return {"action": "skip", "reason": "Sticky context spam block"}

                    opt_idx = int(owner_text) - 1
                    if 0 <= opt_idx < len(options):
                        text_to_send = options[opt_idx]
                        _dispatch_reply_to_customer(
                            client_id=str(client_id),
                            conn_id=str(conn_id),
                            text_to_send=text_to_send,
                            selected_opt_num=int(owner_text),
                            telegram_adapter=telegram_adapter,
                            reply_to_msg_id=event.message_id
                        )
                        return {"action": "skip", "reason": "Sticky context routed"}

    return None


def _handle_spam_block(client_id: str, client_name: str, telegram_adapter, reply_to_msg_id: Optional[int]):
    """Persist spammer to blacklist and remove Obsidian history."""
    try:
        blacklist_path = SECRETARY_DIR / "spam_blacklist.json"
        current_blacklist = []
        if blacklist_path.exists():
            try:
                with open(blacklist_path, "r", encoding="utf-8") as f:
                    current_blacklist = json.load(f)
            except Exception:
                current_blacklist = []
        if str(client_id) not in current_blacklist:
            current_blacklist.append(str(client_id))
            with open(blacklist_path, "w", encoding="utf-8") as f:
                json.dump(current_blacklist, f, indent=2)
            _fix_permissions(blacklist_path)
    except Exception as e:
        logger.error(f"[Secretary] Failed to update blacklist: {e}")

    try:
        inbox_file = INBOX_DIR / f"client_{client_id}.md"
        if inbox_file.exists():
            inbox_file.unlink()
    except Exception as e:
        logger.error(f"[Secretary] Failed to delete spam file: {e}")

    chronicle.log_event(
        event_type="spam_blocked",
        client_id=str(client_id),
        client_name=client_name,
        delivery_status="blacklisted"
    )

    if telegram_adapter:
        async def _send_confirm():
            send_metadata = {"thread_id": OWNER_THREAD_ID} if OWNER_THREAD_ID else None
            await telegram_adapter.send(
                chat_id=OWNER_CHAT_ID,
                content=f"🤘 **Враг повержен!** Отправитель `{client_name}` ({client_id}) заблокирован и добавлен в чёрный список. CRM-логи стёрты.",
                reply_to=reply_to_msg_id,
                metadata=send_metadata
            )
        _schedule_async(_send_confirm())


def _dispatch_reply_to_customer(
    client_id: str,
    conn_id: str,
    text_to_send: str,
    selected_opt_num: Optional[int],
    telegram_adapter,
    reply_to_msg_id: Optional[int]
):
    """Send message to customer via Telegram Business API and log to Obsidian & Chronicle."""
    if not telegram_adapter:
        logger.error("[Secretary] Telegram adapter not found for sending!")
        return

    # Log Owner's reply to Obsidian
    try:
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        inbox_file = INBOX_DIR / f"client_{client_id}.md"
        with open(inbox_file, "a", encoding="utf-8") as f:
            f.write(f"- [{timestamp}] OWNER: {text_to_send}\n")
        _fix_permissions(inbox_file)
    except Exception as e:
        logger.error(f"[Secretary] Failed to log reply to Obsidian: {e}")

    async def _send_and_confirm():
        send_res = await telegram_adapter.send(
            chat_id=client_id,
            content=text_to_send,
            metadata={"business_connection_id": conn_id}
        )
        send_metadata = {"thread_id": OWNER_THREAD_ID} if OWNER_THREAD_ID else None
        if send_res.success:
            chronicle.log_event(
                event_type="reply_sent",
                client_id=str(client_id),
                business_connection_id=str(conn_id),
                selected_option=selected_opt_num,
                outbound_text=text_to_send,
                delivery_status="success"
            )
            opt_tag = f"(Вариант {selected_opt_num})" if selected_opt_num else ""
            await telegram_adapter.send(
                chat_id=OWNER_CHAT_ID,
                content=f"✅ **Отправлено клиенту {opt_tag}:**\n> {text_to_send}",
                reply_to=reply_to_msg_id,
                metadata=send_metadata
            )
        else:
            chronicle.log_event(
                event_type="reply_sent",
                client_id=str(client_id),
                business_connection_id=str(conn_id),
                selected_option=selected_opt_num,
                outbound_text=text_to_send,
                delivery_status="failed",
                error_message=send_res.error
            )
            await telegram_adapter.send(
                chat_id=OWNER_CHAT_ID,
                content=f"❌ **Не удалось отправить:** {send_res.error}",
                reply_to=reply_to_msg_id,
                metadata=send_metadata
            )

    _schedule_async(_send_and_confirm())


# ============================================================================
# CALLBACK QUERY HOOK HANDLER
# ============================================================================
async def handle_telegram_callback_query(query, data: str, cb: Dict[str, Any], adapter, **kwargs) -> bool:
    """Handle incoming inline button clicks with prefix sec:"""
    if not data or not data.startswith("sec:"):
        return False

    user_id = str(getattr(query.from_user, "id", ""))
    if user_id not in TRUSTED_ADMIN_IDS:
        logger.warning("[SecretaryCallback] Unauthorized callback attempt from user %s (data=%s)", user_id, data)
        try:
            await query.answer("⛔ Доступ запрещен.", show_alert=True)
        except Exception:
            pass
        return True

    parts = data.split(":")
    action = parts[1] if len(parts) > 1 else ""

    # 1. Option selection: sec:opt:<client_id>:<opt_idx>
    if action == "opt" and len(parts) >= 4:
        client_id = parts[2]
        try:
            opt_idx = int(parts[3])
        except ValueError:
            await query.answer("Неверный номер варианта.")
            return True

        # Extract options and conn_id from card message text or sticky
        card_text = getattr(getattr(query, "message", None), "text", "") or ""
        meta_match = re.search(
            r"\[client_id:(\d+)\]\[conn_id:([^\]]+)\]\[v1:([^\]]*)\]\[v2:([^\]]*)\]\[v3:([^\]]*)\]",
            card_text
        )

        conn_id = ""
        options = []
        if meta_match:
            conn_id = meta_match.group(2)
            options = [meta_match.group(3), meta_match.group(4), meta_match.group(5)]
        else:
            _sticky_key = str(OWNER_THREAD_ID or OWNER_CHAT_ID)
            last = _LAST_ACTIVE_CLIENT.get(_sticky_key)
            if last and str(last.get("client_id")) == str(client_id):
                conn_id = last.get("conn_id", "")
                options = last.get("options", [])

        if not options or opt_idx < 1 or opt_idx > len(options):
            await query.answer("Вариант не найден или устарел.", show_alert=True)
            return True

        selected_text = options[opt_idx - 1]

        # Send to customer
        send_res = await adapter.send(
            chat_id=client_id,
            content=selected_text,
            metadata={"business_connection_id": conn_id}
        )

        if send_res.success:
            await query.answer(f"✅ Вариант {opt_idx} отправлен!")
            chronicle.log_event(
                event_type="reply_sent",
                client_id=str(client_id),
                business_connection_id=str(conn_id),
                selected_option=opt_idx,
                outbound_text=selected_text,
                delivery_status="success"
            )

            # Log to Obsidian
            try:
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                inbox_file = INBOX_DIR / f"client_{client_id}.md"
                with open(inbox_file, "a", encoding="utf-8") as f:
                    f.write(f"- [{timestamp}] OWNER (button {opt_idx}): {selected_text}\n")
                _fix_permissions(inbox_file)
            except Exception as e:
                logger.error(f"[Secretary] Failed to write reply to Obsidian: {e}")

            # Update card message: remove inline keyboard and append confirmation
            if query.message:
                try:
                    cur_text = query.message.text or ""
                    clean_text = cur_text.split("||")[0].strip()
                    time_now = datetime.datetime.now().strftime("%H:%M:%S")
                    updated_text = (
                        f"{clean_text}\n\n"
                        f"━━━━━━━━━━━━━━━━━━━━\n"
                        f"✅ **Отправлено клиенту (Вариант {opt_idx}) в {time_now}:**\n"
                        f"> {selected_text}"
                    )
                    await query.edit_message_text(text=updated_text, reply_markup=None)
                except Exception as e:
                    logger.error(f"[Secretary] Failed to update card message: {e}")
        else:
            await query.answer(f"❌ Ошибка отправки: {send_res.error}", show_alert=True)
            chronicle.log_event(
                event_type="reply_sent",
                client_id=str(client_id),
                business_connection_id=str(conn_id),
                selected_option=opt_idx,
                outbound_text=selected_text,
                delivery_status="failed",
                error_message=send_res.error
            )
        return True

    # 2. Spam button: sec:spam:<client_id>
    elif action == "spam" and len(parts) >= 3:
        client_id = parts[2]
        _handle_spam_block(client_id, f"Client_{client_id}", adapter, None)
        await query.answer("🚫 Отправитель заблокирован и добавлен в спам-лист.")
        if query.message:
            try:
                await query.edit_message_text(
                    text=f"🚫 **Отправитель {client_id} заблокирован.** Карточка перемещена в спам.",
                    reply_markup=None
                )
            except Exception:
                pass
        return True

    # 3. Dismiss button: sec:dismiss:<client_id>
    elif action == "dismiss":
        await query.answer("Карточка закрыта без ответа.")
        if query.message:
            try:
                cur_text = (query.message.text or "").split("||")[0].strip()
                await query.edit_message_text(
                    text=f"{cur_text}\n\n*Карточка отклонена владельцем.*",
                    reply_markup=None
                )
            except Exception:
                pass
        return True

    # 4. Confirm prompt leak send: sec:conf:<confirm_id>
    elif action == "conf" and len(parts) >= 3:
        confirm_id = parts[2]
        item = _PENDING_CONFIRMATIONS.pop(confirm_id, None)
        if not item or (_time.time() - item["timestamp"]) > CONFIRMATION_TTL:
            await query.answer("Срок действия подтверждения истек.", show_alert=True)
            return True

        client_id = item["client_id"]
        conn_id = item["conn_id"]
        text_to_send = item["text"]

        send_res = await adapter.send(
            chat_id=client_id,
            content=text_to_send,
            metadata={"business_connection_id": conn_id}
        )
        if send_res.success:
            await query.answer("✅ Отправлено клиенту!")
            chronicle.log_event(
                event_type="reply_sent",
                client_id=str(client_id),
                business_connection_id=str(conn_id),
                outbound_text=text_to_send,
                delivery_status="success_after_leak_warning"
            )
            if query.message:
                try:
                    await query.edit_message_text(
                        text=f"✅ **Сообщение подтверждено и отправлено клиенту:**\n> {text_to_send}",
                        reply_markup=None
                    )
                except Exception:
                    pass
        else:
            await query.answer(f"❌ Ошибка отправки: {send_res.error}", show_alert=True)
        return True

    # 5. Cancel prompt leak send: sec:cancel:<confirm_id>
    elif action == "cancel" and len(parts) >= 3:
        confirm_id = parts[2]
        _PENDING_CONFIRMATIONS.pop(confirm_id, None)
        await query.answer("Отправка отменена.")
        if query.message:
            try:
                await query.edit_message_text(
                    text="❌ **Отправка отменена владельцем.** Команда не попала к клиенту.",
                    reply_markup=None
                )
            except Exception:
                pass
        return True

    return False


def register(ctx):
    ctx.register_hook("pre_gateway_dispatch", pre_gateway_dispatch)
    ctx.register_hook("telegram_callback_query", handle_telegram_callback_query)
