#!/usr/bin/env python3
import os
import re
import sys
import time
import sqlite3
import subprocess
from datetime import datetime, timedelta

DB_PATH = "/root/.hermes/state.db"
OBSIDIAN_DIR = "/var/www/obsidian-vault"
CLIENT_METRICS_PATH = f"{OBSIDIAN_DIR}/Secretary/Клиенты.md"
TASKS_PATH = f"{OBSIDIAN_DIR}/Inbox/Текущие_задачи.md"
CONTACTS_DIR = f"{OBSIDIAN_DIR}/Secretary/Contacts"
LOGS_ERRORS_PATH = "/root/.hermes/logs/errors.log"

OLGA_TG_ID = "1192001069"

def run_command(cmd):
    try:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        return res.stdout.strip(), res.stderr.strip(), res.returncode
    except Exception as e:
        return "", str(e), -1

def get_db_metrics():
    if not os.path.exists(DB_PATH):
        return {
            "error": "State DB not found",
            "sessions_count": 0, "msg_user": 0, "msg_ai": 0,
            "wad": 0, "input_tokens": 0, "output_tokens": 0,
            "roi_usd": 0.0, "avg_trajectory": 0.0, "tool_error_rate": 0.0,
            "user_corrections": 0, "top_tools": []
        }
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    # Calculate timestamp boundary (last 7 days)
    seven_days_ago = time.time() - 7 * 24 * 3600
    
    # Traverse all sessions in last 7 days to find those belonging to Olga (explicitly or via parent trail)
    cur.execute("SELECT id, parent_session_id, user_id FROM sessions WHERE started_at >= ?", (seven_days_ago,))
    sessions_cache = {row["id"]: {"parent": row["parent_session_id"], "user": row["user_id"]} for row in cur.fetchall()}
    
    olga_session_ids = set()
    for s_id, info in sessions_cache.items():
        curr_id = s_id
        user_id = info["user"]
        parent = info["parent"]
        
        visited = set()
        while not user_id and parent and parent not in visited:
            visited.add(parent)
            if parent in sessions_cache:
                user_id = sessions_cache[parent]["user"]
                parent = sessions_cache[parent]["parent"]
            else:
                # Query DB for parent if it's not in the 7-day cache
                parent_row = cur.execute("SELECT parent_session_id, user_id FROM sessions WHERE id = ?", (parent,)).fetchone()
                if parent_row:
                    user_id = parent_row["user_id"]
                    parent = parent_row["parent_session_id"]
                else:
                    break
                    
        if user_id == OLGA_TG_ID:
            olga_session_ids.add(s_id)
            
    session_ids = list(olga_session_ids)
    
    if not session_ids:
        conn.close()
        return {
            "sessions_count": 0, "msg_user": 0, "msg_ai": 0,
            "wad": 0, "input_tokens": 0, "output_tokens": 0,
            "roi_usd": 0.0, "avg_trajectory": 0.0, "tool_error_rate": 0.0,
            "user_corrections": 0, "top_tools": []
        }
    
    placeholders = ",".join("?" for _ in session_ids)
    
    # Msg count (User/AI)
    msg_query = f"""
        SELECT role, count(*), group_concat(content, '|||') as contents 
        FROM messages 
        WHERE session_id IN ({placeholders}) AND timestamp >= ?
        GROUP BY role
    """
    params = session_ids + [seven_days_ago]
    msg_rows = cur.execute(msg_query, params).fetchall()
    
    msg_user = 0
    msg_ai = 0
    msg_tool = 0
    user_contents = []
    
    for row in msg_rows:
        role = row["role"]
        count = row["count(*)"]
        if role == "user":
            msg_user = count
            if row["contents"]:
                user_contents = row["contents"].split("|||")
        elif role == "assistant":
            msg_ai = count
        elif role == "tool":
            msg_tool = count
            
    # WAD (Weekly Active Days)
    wad_query = f"""
        SELECT count(DISTINCT date(timestamp, 'unixepoch', 'localtime')) as active_days
        FROM messages
        WHERE session_id IN ({placeholders}) AND role = 'user' AND timestamp >= ?
    """
    wad_row = cur.execute(wad_query, params).fetchone()
    wad = wad_row["active_days"] if wad_row else 0
    
    # Tokens & Cost ROI
    token_query = f"""
        SELECT sum(input_tokens) as in_t, sum(output_tokens) as out_t
        FROM sessions
        WHERE id IN ({placeholders})
    """
    token_row = cur.execute(token_query, session_ids).fetchone()
    in_t = token_row["in_t"] or 0
    out_t = token_row["out_t"] or 0
    
    # Equivalent raw API cost for Claude 3.5 Sonnet ($3 / 1M input, $15 / 1M output)
    roi_usd = (in_t / 1_000_000.0) * 3.0 + (out_t / 1_000_000.0) * 15.0
    
    # Avg Trajectory Length (tool calls per assistant response)
    avg_trajectory = (msg_tool / float(msg_ai)) if msg_ai > 0 else 0.0
    
    # Tool Error Rate
    tool_error_query = f"""
        SELECT count(*) as err_count
        FROM messages
        WHERE session_id IN ({placeholders}) 
          AND role = 'tool' 
          AND timestamp >= ?
          AND (content LIKE '%Error%' OR content LIKE '%Traceback%' OR content LIKE '%Failed%')
    """
    err_row = cur.execute(tool_error_query, params).fetchone()
    err_count = err_row["err_count"] if err_row else 0
    tool_error_rate = (err_count / float(msg_tool) * 100) if msg_tool > 0 else 0.0
    
    # Top 5 tools used
    tools_query = f"""
        SELECT tool_name, count(*) as c
        FROM messages
        WHERE session_id IN ({placeholders}) AND role = 'tool' AND timestamp >= ?
        GROUP BY tool_name
        ORDER BY c DESC
        LIMIT 5
    """
    top_tools = [(row["tool_name"], row["c"]) for row in cur.execute(tools_query, params).fetchall()]
    
    # User Corrections (Regex pattern for Russian stop words with word boundaries)
    correction_pat = re.compile(r"(?i)\b(нет|не\s+то|ошибка|исправь|удали|не\s+так|отмени)\b")
    user_corrections = 0
    for content in user_contents:
        if content and correction_pat.search(content):
            user_corrections += 1
            
    conn.close()
    
    return {
        "sessions_count": len(session_ids),
        "msg_user": msg_user,
        "msg_ai": msg_ai,
        "wad": wad,
        "input_tokens": in_t,
        "output_tokens": out_t,
        "roi_usd": roi_usd,
        "avg_trajectory": avg_trajectory,
        "tool_error_rate": tool_error_rate,
        "user_corrections": user_corrections,
        "top_tools": top_tools
    }

def get_obsidian_metrics():
    crm_total = 0
    crm_new = 0
    
    # 1. CRM Contacts
    if os.path.exists(CONTACTS_DIR):
        files = [f for f in os.listdir(CONTACTS_DIR) if f.endswith(".md")]
        crm_total = len(files)
        seven_days_ago = time.time() - 7 * 24 * 3600
        for f in files:
            p = os.path.join(CONTACTS_DIR, f)
            try:
                # If file was modified in last 7 days
                if os.path.getmtime(p) >= seven_days_ago:
                    crm_new += 1
            except:
                pass
                
    # 2. Production Orders
    orders_by_status = {}
    total_orders = 0
    if os.path.exists(CLIENT_METRICS_PATH):
        try:
            with open(CLIENT_METRICS_PATH, "r", encoding="utf-8") as f:
                content = f.read()
            # Find bullet points starting with * **[[Name]]**
            order_blocks = re.split(r'\n\s*\*\s+\*\*\[\[', content)
            # The first block is header, ignore it
            if len(order_blocks) > 1:
                total_orders = len(order_blocks) - 1
                for block in order_blocks[1:]:
                    # Extract status
                    status_match = re.search(r'\*\*\u0422\u0435\u043a\u0443\u0449\u0438\u0439 \u0441\u0442\u0430\u0442\u0443\u0441:\*\* ([^\n]+)', block)
                    if status_match:
                        status = status_match.group(1).strip()
                        # Strip emojis from status for cleaner report
                        status_clean = re.sub(r'[\u2600-\u27BF\U0001f300-\U0001f64f\U0001f680-\U0001f6ff]', '', status).strip()
                        orders_by_status[status_clean] = orders_by_status.get(status_clean, 0) + 1
                    else:
                        orders_by_status["Не указан"] = orders_by_status.get("Не указан", 0) + 1
        except Exception as e:
            orders_by_status["Error parsing"] = str(e)

    # 3. Tasks Status (from Текущие_задачи.md)
    tasks_pending = 0
    tasks_completed = 0
    lately_pending = 0
    lately_completed = 0
    
    if os.path.exists(TASKS_PATH):
        try:
            with open(TASKS_PATH, "r", encoding="utf-8") as f:
                lines = f.readlines()
                
            in_recent_section = False
            seven_days_ago = datetime.now() - timedelta(days=7)
            
            for line in lines:
                header_match = re.match(r'^## \u0414\u043e\u0431\u0430\u0432\u043b\u0435\u043d\u043e\s+(\d{2})\.(\d{2})\.(\d{4})', line)
                if header_match:
                    day, month, year = map(int, header_match.groups())
                    header_date = datetime(year, month, day)
                    if header_date >= seven_days_ago:
                        in_recent_section = True
                    else:
                        in_recent_section = False
                
                if line.strip().startswith("- [ ]"):
                    tasks_pending += 1
                    if in_recent_section:
                        lately_pending += 1
                elif line.strip().startswith("- [x]") or line.strip().startswith("- [X]"):
                    tasks_completed += 1
                    if in_recent_section:
                        lately_completed += 1
        except:
            pass

    return {
        "crm_total": crm_total,
        "crm_new": crm_new,
        "total_orders": total_orders,
        "orders_by_status": orders_by_status,
        "tasks_pending": tasks_pending,
        "tasks_completed": tasks_completed,
        "lately_pending": lately_pending,
        "lately_completed": lately_completed
    }

def get_system_metrics():
    # Services status
    services = ["hermes-gateway", "omniroute", "qdrant", "searxng"]
    status_map = {}
    for s in services:
        out, _, _ = run_command(f"systemctl is-active {s}")
        if out == "active":
            status_map[s] = "🟢 Активен"
        else:
            status_map[s] = f"🔴 Неактивен ({out or 'failed'})"
            
    # System RAM & Swap & Disk
    ram_raw, _, _ = run_command("free -m | grep Mem:")
    swap_raw, _, _ = run_command("free -m | grep Swap:")
    disk_raw, _, _ = run_command("df -h / | tail -n 1")
    
    ram_text = "N/A"
    swap_text = "N/A"
    disk_text = "N/A"
    
    if ram_raw:
        parts = ram_raw.split()
        if len(parts) >= 3:
            total_m = float(parts[1])
            used_m = float(parts[2])
            ram_text = f"{used_m/1024.0:.1f}G / {total_m/1024.0:.1f}G ({used_m/total_m*100.0:.0f}%)"
            
    if swap_raw:
        parts = swap_raw.split()
        if len(parts) >= 3:
            total_m = float(parts[1])
            used_m = float(parts[2])
            if total_m > 0:
                swap_text = f"{used_m:.0f}M / {total_m:.0f}M ({used_m/total_m*100.0:.0f}%)"
            else:
                swap_text = "0M / 0M (0%)"
                
    if disk_raw:
        parts = disk_raw.split()
        if len(parts) >= 5:
            disk_text = f"{parts[2]} / {parts[1]} ({parts[4]})"
            
    # Log files errors count
    errors_count = 0
    if os.path.exists(LOGS_ERRORS_PATH):
        try:
            seven_days_ago = datetime.now() - timedelta(days=7)
            with open(LOGS_ERRORS_PATH, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    m = re.match(r'^(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})', line)
                    if m:
                        try:
                            dt = datetime.strptime(f"{m.group(1)} {m.group(2)}", "%Y-%m-%d %H:%M:%S")
                            if dt >= seven_days_ago:
                                if "error" in line.lower() or "warning" in line.lower() or "exception" in line.lower() or "traceback" in line.lower():
                                    errors_count += 1
                        except:
                            pass
        except:
            pass

    return {
        "services": status_map,
        "ram": ram_text,
        "swap": swap_text,
        "disk": disk_text,
        "errors_count": errors_count
    }

def main():
    db = get_db_metrics()
    obsidian = get_obsidian_metrics()
    sys_metrics = get_system_metrics()
    
    # Build markdown report
    today_str = datetime.now().strftime("%d.%m.%Y")
    seven_days_ago_str = (datetime.now() - timedelta(days=7)).strftime("%d.%m.%Y")
    
    report = []
    report.append(f"📊 **ЕЖЕНЕДЕЛЬНЫЙ ОТЧЕТ ПИЛОТА: OlyaServer ({seven_days_ago_str} - {today_str})**")
    report.append(f"Проект: `AI-Business-Assistant` | Клиент: `Ольга` (ID: `{OLGA_TG_ID}`)")
    report.append("")
    
    # 1. Activity Section
    report.append("⚡ **АКТИВНОСТЬ:**")
    msg_user_target = " (целевой: >35) ✅" if db["msg_user"] >= 35 else " (целевой: >35) ⚠️"
    wad_target = " (целевой: >=4) ✅" if db["wad"] >= 4 else " (целевой: >=4) ⚠️"
    report.append(f"• Запросов от Оли: **{db['msg_user']}** сообщений{msg_user_target}")
    report.append(f"• Ответов ассистента: **{db['msg_ai']}** (соотношение ~1:{msg_ai_ratio(db)})")
    report.append(f"• Активных дней (WAD): **{db['wad']} из 7**{wad_target}")
    report.append(f"• Создано/обновлено CRM карточек контактов: **+{obsidian['crm_new']}** (всего в базе: {obsidian['crm_total']})")
    report.append("")
    
    # 2. Business Value Section
    report.append("💼 **БИЗНЕС-ЦЕННОСТЬ:**")
    report.append(f"• Всего заказов в каталоге `Клиенты.md`: **{obsidian['total_orders']}**")
    for status, count in obsidian["orders_by_status"].items():
        report.append(f"  - {status}: **{count}**")
        
    report.append(f"• Задачи (Текущие_задачи.md) за последнюю неделю:")
    report.append(f"  - Новых задач создано: **{obsidian['lately_pending'] + obsidian['lately_completed']}**")
    report.append(f"  - Из них выполнено: **{obsidian['lately_completed']}**")
    report.append(f"  - Всего нерешенных задач в списке: **{obsidian['tasks_pending']}**")
    report.append(f"• 💵 **ROI ДЛЯ КЛИЕНТА (Savings):** Сэкономлено на API с начала недели **~${db['roi_usd']:.2f} USD**")
    report.append("  *(В эквиваленте тарифов Claude 3.5 Sonnet при аналогичном объеме токенов)*")
    report.append("")
    
    # 3. Agent Quality Metrics
    report.append("🎯 **КАЧЕСТВО АГЕНТА:**")
    correction_icon = "🟢" if db["user_corrections"] <= 5 else "🟡"
    report.append(f"• {correction_icon} **Индекс коррекций:** **{db['user_corrections']}** правок от пользователя за неделю")
    trajectory_icon = "🟢" if db["avg_trajectory"] < 3.0 else "🟡"
    report.append(f"• {trajectory_icon} **Средняя траектория:** **{db['avg_trajectory']:.1f}** шагов (тулколлов) на запрос")
    tools_err_icon = "🟢" if db["tool_error_rate"] < 5.0 else "🟡"
    report.append(f"• {tools_err_icon} **Сбои инструментов:** **{db['tool_error_rate']:.1f}%** ({int(db['tool_error_rate']*db['msg_user']/100)} ошибок)")
    
    if db["top_tools"]:
        tools_str = ", ".join(f"`{t[0]}` ({t[1]} раз)" for t in db["top_tools"])
        report.append(f"• Топ инструментов: {tools_str}")
    report.append("")
    
    # 4. System Health Section
    report.append("🛠️ **СИСТЕМНОЕ ЗДОРОВЬЕ:**")
    for s, st in sys_metrics["services"].items():
        report.append(f"• {st} : `{s}`")
    report.append(f"• Ресурсы VPS: RAM: **{sys_metrics['ram']}** | Swap: **{sys_metrics['swap']}** | Disk: **{sys_metrics['disk']}**")
    
    err_icon = "🟢" if sys_metrics["errors_count"] == 0 else ("🟡" if sys_metrics["errors_count"] < 100 else "🔴")
    report.append(f"• {err_icon} Ошибок в `errors.log` за неделю: **{sys_metrics['errors_count']}**")
    report.append("")
    
    # 5. Recommendation
    report.append("💡 **АНАЛИЗ И РЕКОМЕНДАЦИЯ:**")
    rec = generate_recommendation(db, obsidian, sys_metrics)
    report.append(rec)
    
    print("\n".join(report))

def msg_ai_ratio(db):
    if db["msg_user"] > 0:
        return f"{db['msg_ai']/float(db['msg_user']):.1f}"
    return "0"

def generate_recommendation(db, obsidian, sys_metrics):
    recs = []
    
    # Activity recommendation
    if db["msg_user"] < 35:
        recs.append("Вовлеченность пользователя ниже целевой (>35 сообщений/неделю). Рекомендуется настроить мягкий утренний дайджест с фокусом на текущие дела и интересные идеи дизайна.")
    else:
        recs.append("Пользователь демонстрирует отличную вовлеченность во взаимодействие с ассистентом (>35 сообщений/неделю).")
        
    # Correction rate
    if db["user_corrections"] > 5:
        recs.append("Высокий индекс коррекции (пользователь часто правит бота). Возможно, требуется аудит системных промптов в `config.yaml` или донастройка спецификаций бренда.")
        
    # Tool failures
    if db["tool_error_rate"] > 5.0:
        recs.append("Сбои инструментов превышают 5%. Проверьте логи на предмет таймаутов или конфликтов при работе с файлами Obsidian.")
        
    # Resource metrics
    if "90%" in sys_metrics["ram"] or "95%" in sys_metrics["ram"]:
        recs.append("Внимание: Высокое потребление оперативной памяти на сервере. Проконтролируйте Swap-файл.")
        
    if not recs:
        recs.append("Все показатели в норме. Сервер стабилен, клиент активно адаптирует ИИ-ассистента под бизнес-задачи.")
        
    return " ".join(recs)

if __name__ == "__main__":
    main()
