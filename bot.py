"""TeamTrustGate — AI Agent for Sales Request Processing.
Telegram → LLM → Jira
"""
import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, filters, ContextTypes, Defaults
)

from config import CONFIG, Config
from state_manager import STATE_MANAGER
from llm_adapter import get_llm_provider
from jira_client import JIRA_CLIENT
from deduplicator import Deduplicator
from scorer import Scorer
from parse_utils import parse_llm_number

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, CONFIG.LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("teamtrustgate")

# ── Init ──────────────────────────────────────────────────────────────────
Config.validate()
llm = get_llm_provider()
deduplicator = Deduplicator(llm)
scorer = Scorer(llm)

STATUS_POLL_INTERVAL = 300  # 5 минут

# ── Load prompts ──────────────────────────────────────────────────────────
def _load_prompt(name: str) -> str:
    for ext in ("md", "txt"):
        path = f"prompts/{name}.{ext}"
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                logger.info(f"Промпт загружен: {path}")
                return content
        except FileNotFoundError:
            continue
    raise FileNotFoundError(
        f"Промпт '{name}' не найден. Ожидается prompts/{name}.md или prompts/{name}.txt"
    )

EXTRACTION_PROMPT = _load_prompt("extraction")

# ── Роли и права доступа ──────────────────────────────────────────────────
def _norm(username: str) -> str:
    """Нормализует username: убирает @ и приводит к lowercase."""
    return (username or "").lstrip("@").lower()

def _is_allowed(username: str) -> bool:
    """Сейлз или админ — есть доступ к боту вообще."""
    u = _norm(username)
    if not CONFIG.ALLOWED_USERNAMES and not CONFIG.ADMIN_USERNAMES:
        return True  # списки пусты — открытый доступ (dev режим)
    return u in CONFIG.ALLOWED_USERNAMES or u in CONFIG.ADMIN_USERNAMES

def _is_admin(username: str) -> bool:
    """Админ/менеджер — полный доступ."""
    if not CONFIG.ADMIN_USERNAMES:
        return False  # список пуст — никто не админ
    return _norm(username) in CONFIG.ADMIN_USERNAMES

def _can_manage_ticket(username: str, issue_key: str) -> bool:
    """
    Может ли пользователь управлять тикетом (удалять, менять статус)?
    - Админ может управлять любым тикетом
    - Сейлз — только своим (созданным им)
    - Если владелец неизвестен (старый тикет) — только админ
    """
    if _is_admin(username):
        return True
    owner = STATE_MANAGER.get_ticket_owner(issue_key)
    if owner is None:
        return False  # неизвестный владелец — только админ может
    return owner == _norm(username)

# ── Keyboard builders ─────────────────────────────────────────────────────
def _ticket_keyboard(issue_key: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔗 Открыть в Jira", url=f"{CONFIG.JIRA_URL}/browse/{issue_key}")],
        [
            InlineKeyboardButton("📊 Статус",   callback_data=f"status:{issue_key}"),
            InlineKeyboardButton("🗑 Удалить",  callback_data=f"delete_confirm:{issue_key}"),
        ],
        [
            InlineKeyboardButton("▶️ В работу", callback_data=f"transition:in_progress:{issue_key}"),
            InlineKeyboardButton("✅ Готово",    callback_data=f"transition:done:{issue_key}"),
        ],
    ])

def _status_keyboard(issue_key: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔗 Открыть в Jira", url=f"{CONFIG.JIRA_URL}/browse/{issue_key}")],
        [
            InlineKeyboardButton("▶️ В работу",    callback_data=f"transition:in_progress:{issue_key}"),
            InlineKeyboardButton("✅ Готово",       callback_data=f"transition:done:{issue_key}"),
        ],
        [
            InlineKeyboardButton("🔄 На проверку", callback_data=f"transition:review:{issue_key}"),
            InlineKeyboardButton("🚫 Отклонить",   callback_data=f"transition:reject:{issue_key}"),
        ],
        [InlineKeyboardButton("🗑 Удалить",        callback_data=f"delete_confirm:{issue_key}")],
    ])

def _confirm_delete_keyboard(issue_key: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("✅ Да, удалить", callback_data=f"delete_yes:{issue_key}"),
        InlineKeyboardButton("❌ Отмена",      callback_data=f"delete_no:{issue_key}"),
    ]])

def _dup_keyboard(issue_key: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("🔗 Открыть дубликат", url=f"{CONFIG.JIRA_URL}/browse/{issue_key}"),
        InlineKeyboardButton("📊 Статус", callback_data=f"status:{issue_key}"),
    ]])

def _preview_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("✅ Создать тикет", callback_data="preview:confirm")],
        [
            InlineKeyboardButton("✏️ Уточнить", callback_data="preview:clarify"),
            InlineKeyboardButton("❌ Отменить",  callback_data="preview:cancel"),
        ],
    ])

# ── Helpers ───────────────────────────────────────────────────────────────
def _build_jira_description(analysis: dict, scoring: dict, raw_text: str, requested_by: str) -> str:
    return (
        f"*Проблема:* {analysis.get('problem_statement', '')}\n"
        f"*Контекст клиента:* {analysis.get('client_context', '')}\n"
        f"*Потенциальный ROI:* {analysis.get('revenue_at_risk', 0)}/10\n"
        f"*Охват:* {analysis.get('reach', 'unknown')}\n"
        f"*Инициатор:* {requested_by}\n"
        f"*Оригинальный запрос:* {raw_text}\n\n"
        f"*AI-скоринг:* {scoring.get('total_score', 0)} ({scoring.get('priority', 'Low')})\n"
        f"*Обоснование:* {scoring.get('justification', '')}"
    )

def _safe_confidence(analysis: dict) -> float:
    return parse_llm_number(analysis.get("confidence", 0))

def _escape_md(text: str) -> str:
    return str(text) if text else ""

def _priority_emoji(priority: str) -> str:
    return {"Highest": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}.get(priority, "⚪")

def _reach_label(reach: str) -> str:
    return {"one_client": "Один клиент", "segment": "Сегмент", "all_clients": "Все клиенты"}.get(reach, reach)

def _build_preview(analysis: dict, scoring: dict) -> str:
    priority = scoring.get("priority", "Low")
    reach    = analysis.get("reach", "unknown")
    return (
        "🤖 *Вот что я понял из запроса:*\n\n"
        f"📋 *Проблема:*\n{_escape_md(analysis.get('problem_statement', ''))}\n\n"
        f"👤 *Клиент:* {_escape_md(analysis.get('client_context', 'не указан'))}\n"
        f"💰 *Риск потери выручки:* {analysis.get('revenue_at_risk', 0)}/10\n"
        f"📊 *Охват:* {_reach_label(reach)}\n"
        f"🎯 *Приоритет:* {_priority_emoji(priority)} {_escape_md(priority)} "
        f"({scoring.get('total_score', 0)}/400)\n\n"
        "Всё верно? Создаём тикет?"
    )

def _build_stats_message(local: dict, jira: dict | None) -> str:
    parts = ["📈 *Статистика TeamTrustGate*\n"]
    parts.append(
        f"*За последние 30 дней:*\n"
        f"📋 Создано тикетов: *{local['this_month']}*\n"
        f"📦 Всего через бота: *{local['total']}*"
    )
    if jira:
        parts.append(f"✅ Закрыто: *{jira['done']}*\n▶️ В работе: *{jira['in_progress']}*")
    if jira and jira.get("by_priority"):
        emoji_map = {"Highest": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}
        rows = ["\n*Распределение по приоритетам:*"]
        for p in ["Highest", "High", "Medium", "Low"]:
            count = jira["by_priority"].get(p, 0)
            if count:
                bar = "█" * min(count, 10) + ("+" if count > 10 else "")
                rows.append(f"{emoji_map.get(p, '⚪')} {p}: {bar} *{count}*")
        parts.append("\n".join(rows))
    if jira and jira.get("by_status"):
        rows = ["\n*По статусам:*"]
        for status, count in sorted(jira["by_status"].items(), key=lambda x: -x[1]):
            rows.append(f"  _{_escape_md(status)}_: {count}")
        parts.append("\n".join(rows))
    if local.get("top_users"):
        medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
        rows = ["\n*🏆 Топ сейлзов:*"]
        for idx, u in enumerate(local["top_users"]):
            medal = medals[idx] if idx < len(medals) else "  "
            rows.append(f"{medal} @{_escape_md(u['username'])}: *{u['count']}* тикетов")
        parts.append("\n".join(rows))
    if local.get("failed", 0) > 0:
        parts.append(
            f"\n⚠️ Упавших запросов: *{local['failed_month']}* за месяц "
            f"(*{local['failed']}* всего)"
        )
    return "\n".join(parts)

# ── Transition helpers ────────────────────────────────────────────────────
_TRANSITION_NAMES = {
    "in_progress": ["in progress", "в работе", "start progress", "начать работу"],
    "done":        ["done", "готово", "закрыть", "close", "resolved"],
    "review":      ["review", "на проверке", "in review", "code review"],
    "reject":      ["reject", "отклонить", "won't do", "won't fix", "отменить"],
}

_TRANSITION_LABELS = {
    "in_progress": "▶️ В работу",
    "done":        "✅ Готово",
    "review":      "🔄 На проверку",
    "reject":      "🚫 Отклонено",
}

# ── Background job ─────────────────────────────────────────────────────────
async def _poll_status_changes(context: ContextTypes.DEFAULT_TYPE):
    tracked = STATE_MANAGER.get_tracked_tickets()
    if not tracked:
        return

    logger.info(f"poll: проверяю {len(tracked)} отслеживаемых тикетов")

    for ticket in tracked:
        issue_key   = ticket["issue_key"]
        chat_id     = ticket["chat_id"]
        last_status = ticket["last_status"]

        try:
            _, text = await JIRA_CLIENT._request(
                "GET", f"/issue/{issue_key}?fields=status,summary"
            )
            d              = json.loads(text)
            current_status = d["fields"]["status"]["name"]
            summary        = d["fields"]["summary"]
        except Exception as e:
            if "404" in str(e):
                logger.info(f"poll: тикет {issue_key} не найден, перестаём отслеживать")
                STATE_MANAGER.untrack_ticket(issue_key)
            else:
                logger.warning(f"poll: ошибка проверки {issue_key}: {e}")
            continue

        if last_status and current_status != last_status:
            try:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=(
                        f"📬 *Обновление по тикету [{issue_key}]"
                        f"({CONFIG.JIRA_URL}/browse/{issue_key})*\n\n"
                        f"Статус изменился:\n"
                        f"_{_escape_md(last_status)}_ → *{_escape_md(current_status)}*\n\n"
                        f"📋 {_escape_md(summary[:80])}"
                    ),
                    parse_mode="Markdown",
                )
                logger.info(f"poll: уведомление отправлено {issue_key} {last_status}→{current_status}")
            except Exception as e:
                logger.warning(f"poll: не удалось отправить уведомление для {issue_key}: {e}")

            if current_status.lower() in ("done", "готово", "closed", "resolved", "отклонено"):
                STATE_MANAGER.untrack_ticket(issue_key)
            else:
                STATE_MANAGER.update_ticket_status(issue_key, current_status)
        else:
            STATE_MANAGER.update_ticket_status(issue_key, current_status)

# ── Callback query handler ─────────────────────────────────────────────────
async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query    = update.callback_query
    await query.answer()

    data     = query.data or ""
    chat_id  = update.effective_chat.id
    user     = update.effective_user
    username = user.username or user.first_name or str(user.id)

    if data == "preview:confirm":
        session = STATE_MANAGER.get_session(chat_id)
        if not session or session.get("state") != "awaiting_confirmation":
            await query.message.reply_text("⚠️ Сессия устарела. Отправь запрос заново.")
            return
        await query.message.edit_reply_markup(reply_markup=None)
        await query.message.reply_text("📝 Создаю тикет в Jira...")
        await _create_confirmed_ticket(query.message, chat_id, session, username)

    elif data == "preview:clarify":
        session = STATE_MANAGER.get_session(chat_id)
        if not session:
            await query.message.reply_text("⚠️ Сессия устарела. Отправь запрос заново.")
            return
        STATE_MANAGER.update_session(
            chat_id, "clarifying",
            session.get("collected_answers", []),
            session.get("round", 0),
            session.get("analysis_json"),
        )
        await query.message.edit_reply_markup(reply_markup=None)
        await query.message.reply_text(
            "✏️ Что именно нужно уточнить или исправить? Напиши — я пересоздам превью."
        )

    elif data == "preview:cancel":
        STATE_MANAGER.clear_session(chat_id)
        await query.message.edit_reply_markup(reply_markup=None)
        await query.message.reply_text("🚫 Создание тикета отменено.")

    elif data.startswith("transition:"):
        _, transition_type, issue_key = data.split(":", 2)
        # Проверка прав: управлять статусом может владелец или админ
        if not _can_manage_ticket(username, issue_key):
            await query.message.reply_text(
                "❌ Недостаточно прав. Менять статус может только создатель тикета или администратор."
            )
            return
        await _handle_transition(query, issue_key, transition_type, username)

    elif data.startswith("status:"):
        issue_key = data.split(":", 1)[1]
        try:
            _, text = await JIRA_CLIENT._request(
                "GET", f"/issue/{issue_key}?fields=status,summary,priority"
            )
            d           = json.loads(text)
            status_name = d["fields"]["status"]["name"]
            summary     = d["fields"]["summary"]
            priority    = d["fields"].get("priority", {}).get("name", "—")
            # Кнопки управления показываем только если есть права
            kb = _status_keyboard(issue_key) if _can_manage_ticket(username, issue_key) else None
            await query.message.reply_text(
                f"📋 *{issue_key}*\n"
                f"*Статус:* {_escape_md(status_name)}\n"
                f"*Приоритет:* {_priority_emoji(priority)} {_escape_md(priority)}\n"
                f"*Тема:* {_escape_md(summary)}",
                reply_markup=kb,
            )
        except Exception as e:
            await query.message.reply_text(
                f"❌ Не удалось получить статус: {_escape_md(str(e)[:200])}"
            )

    elif data.startswith("delete_confirm:"):
        issue_key = data.split(":", 1)[1]
        # Проверка прав на удаление
        if not _can_manage_ticket(username, issue_key):
            await query.message.reply_text(
                "❌ Недостаточно прав. Удалить тикет может только его создатель или администратор."
            )
            return
        await query.message.reply_text(
            f"⚠️ Удалить тикет *{issue_key}*?\nЭто действие необратимо.",
            reply_markup=_confirm_delete_keyboard(issue_key),
        )

    elif data.startswith("delete_yes:"):
        issue_key = data.split(":", 1)[1]
        # Повторная проверка прав (защита от устаревших кнопок)
        if not _can_manage_ticket(username, issue_key):
            await query.message.reply_text("❌ Недостаточно прав.")
            return
        try:
            await JIRA_CLIENT.delete_issue(issue_key)
            STATE_MANAGER.untrack_ticket(issue_key)
            logger.info(f"ticket_deleted key={issue_key} by={username}")
            await query.message.edit_text(f"🗑 Тикет *{issue_key}* удалён.")
        except Exception as e:
            await query.message.reply_text(
                f"❌ Не удалось удалить: {_escape_md(str(e)[:200])}"
            )

    elif data.startswith("delete_no:"):
        await query.message.edit_text("✅ Удаление отменено.")

async def _handle_transition(query, issue_key: str, transition_type: str, username: str):
    try:
        _, text     = await JIRA_CLIENT._request("GET", f"/issue/{issue_key}/transitions")
        data        = json.loads(text)
        transitions = data.get("transitions", [])

        target_names  = _TRANSITION_NAMES.get(transition_type, [])
        transition_id = None
        matched_name  = None

        for t in transitions:
            if any(target in t.get("name", "").lower() for target in target_names):
                transition_id = t["id"]
                matched_name  = t["name"]
                break

        if not transition_id:
            available = ", ".join(t["name"] for t in transitions)
            await query.message.reply_text(
                f"⚠️ Переход *{_TRANSITION_LABELS.get(transition_type, transition_type)}* "
                f"недоступен для *{issue_key}*.\n"
                f"Доступные переходы: {_escape_md(available)}"
            )
            return

        await JIRA_CLIENT._request(
            "POST", f"/issue/{issue_key}/transitions",
            {"transition": {"id": transition_id}},
        )
        STATE_MANAGER.update_ticket_status(issue_key, matched_name)

        label = _TRANSITION_LABELS.get(transition_type, transition_type)
        logger.info(f"transition applied key={issue_key} type={transition_type} by={username}")
        await query.message.reply_text(
            f"✅ Статус *{issue_key}* изменён: *{_escape_md(matched_name)}*\n"
            f"Действие: {label}",
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("🔗 Открыть в Jira", url=f"{CONFIG.JIRA_URL}/browse/{issue_key}"),
                InlineKeyboardButton("📊 Статус", callback_data=f"status:{issue_key}"),
            ]])
        )
    except Exception as e:
        await query.message.reply_text(
            f"❌ Не удалось изменить статус: {_escape_md(str(e)[:200])}"
        )

# ── Command handlers ───────────────────────────────────────────────────────
async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not _is_allowed(user.username or ""):
        await update.message.reply_text("❌ Доступ ограничен.")
        return

    is_admin = _is_admin(user.username or "")
    role_line = "👑 Роль: *Администратор*" if is_admin else "👤 Роль: *Сейлз*"

    base_commands = (
        "/start — начать\n"
        "/cancel — отменить текущую сессию\n"
        "/status TT-XX — статус тикета\n"
        "/list — мои последние 5 тикетов\n"
        "/delete TT-XX — удалить тикет (только свой)"
    )
    admin_commands = "\n/stats — аналитика за 30 дней 👑" if is_admin else ""

    await update.message.reply_text(
        "👋 Привет! Я TeamTrustGate — агент обработки клиентских запросов.\n\n"
        f"{role_line}\n\n"
        "Просто опиши запрос клиента своими словами, и я создам тикет для продуктовой команды.\n\n"
        "Я также буду присылать уведомления когда статус твоего тикета изменится.\n\n"
        "Команды:\n"
        f"{base_commands}{admin_commands}"
    )

async def cancel_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    STATE_MANAGER.clear_session(chat_id)
    await update.message.reply_text("🚫 Сессия отменена. Отправь новый запрос.")

async def status_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not _is_allowed(user.username or ""):
        await update.message.reply_text("❌ Доступ ограничен.")
        return
    username = user.username or user.first_name or str(user.id)
    args = context.args
    if not args:
        await update.message.reply_text("Использование: /status TT-42")
        return
    issue_key = args[0].upper()
    try:
        _, text = await JIRA_CLIENT._request(
            "GET", f"/issue/{issue_key}?fields=status,summary,priority"
        )
        d           = json.loads(text)
        status_name = d["fields"]["status"]["name"]
        summary     = d["fields"]["summary"]
        priority    = d["fields"].get("priority", {}).get("name", "—")
        kb = _status_keyboard(issue_key) if _can_manage_ticket(username, issue_key) else None
        await update.message.reply_text(
            f"📋 *{issue_key}*\n"
            f"*Статус:* {_escape_md(status_name)}\n"
            f"*Приоритет:* {_priority_emoji(priority)} {_escape_md(priority)}\n"
            f"*Тема:* {_escape_md(summary)}",
            reply_markup=kb,
        )
    except Exception as e:
        await update.message.reply_text(
            f"❌ Не удалось получить статус: {_escape_md(str(e)[:200])}"
        )

async def list_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not _is_allowed(user.username or ""):
        await update.message.reply_text("❌ Доступ ограничен.")
        return
    username = user.username or user.first_name or str(user.id)
    await update.message.reply_text("🔍 Ищу твои последние тикеты...")
    try:
        issues = await JIRA_CLIENT.search_user_issues(username, max_results=5)
    except Exception as e:
        await update.message.reply_text(
            f"❌ Не удалось получить тикеты: {_escape_md(str(e)[:200])}"
        )
        return
    if not issues:
        await update.message.reply_text("📭 Тикетов не найдено.")
        return
    lines = ["📋 *Твои последние тикеты:*\n"]
    for i in issues:
        emoji = _priority_emoji(i.get("priority", "Low"))
        lines.append(
            f"{emoji} [{i['key']}]({CONFIG.JIRA_URL}/browse/{i['key']}) — "
            f"{_escape_md(i['summary'][:60])}\n"
            f"   Статус: _{_escape_md(i.get('status', '—'))}_"
        )
    await update.message.reply_text(
        "\n\n".join(lines),
        reply_markup=InlineKeyboardMarkup([[
            InlineKeyboardButton(
                "🔗 Открыть проект в Jira",
                url=f"{CONFIG.JIRA_URL}/projects/{CONFIG.JIRA_PROJECT_KEY}"
            )
        ]])
    )

async def delete_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not _is_allowed(user.username or ""):
        await update.message.reply_text("❌ Доступ ограничен.")
        return
    username = user.username or user.first_name or str(user.id)
    args = context.args
    if not args:
        await update.message.reply_text("Использование: /delete TT-42")
        return
    issue_key = args[0].upper()
    # Проверка прав
    if not _can_manage_ticket(username, issue_key):
        await update.message.reply_text(
            "❌ Недостаточно прав. Удалить тикет может только его создатель или администратор."
        )
        return
    await update.message.reply_text(
        f"⚠️ Удалить тикет *{issue_key}*?\nЭто действие необратимо.",
        reply_markup=_confirm_delete_keyboard(issue_key),
    )

async def stats_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Аналитика — только для админов."""
    user = update.effective_user
    if not _is_admin(user.username or ""):
        await update.message.reply_text(
            "❌ Команда /stats доступна только администраторам."
        )
        return

    await update.message.reply_text("📊 Собираю статистику...")

    local = STATE_MANAGER.get_local_stats()
    try:
        jira = await JIRA_CLIENT.get_project_stats(days=30)
    except Exception as e:
        logger.warning(f"stats: не удалось получить данные из Jira: {e}")
        jira = None

    await update.message.reply_text(
        _build_stats_message(local, jira),
        reply_markup=InlineKeyboardMarkup([[
            InlineKeyboardButton(
                "🔗 Открыть проект в Jira",
                url=f"{CONFIG.JIRA_URL}/projects/{CONFIG.JIRA_PROJECT_KEY}"
            )
        ]])
    )

# ── Message handler ────────────────────────────────────────────────────────
async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user     = update.effective_user
    chat_id  = update.effective_chat.id
    text     = update.message.text or ""
    username = user.username or user.first_name or str(user.id)

    if not _is_allowed(user.username or ""):
        await update.message.reply_text("❌ Доступ ограничен.")
        return
    if not text.strip():
        await update.message.reply_text("Пожалуйста, отправьте текстовое описание запроса.")
        return

    session = STATE_MANAGER.get_session(chat_id)

    if session and session["state"] == "clarifying":
        await _handle_clarification(update, chat_id, text, username, session)
        return

    logger.info(f"request_received chat_id={chat_id} msg_len={len(text)}")
    STATE_MANAGER.create_session(chat_id, text)
    await _process_request(update, chat_id, text, username, collected_answers=[])

# ── Core logic ─────────────────────────────────────────────────────────────
async def _handle_clarification(update, chat_id, text, username, session):
    collected = session["collected_answers"] + [text]
    round_num = session["round"] + 1
    original  = session["original_message"]

    await update.message.reply_text("⏳ Анализирую уточнения...")

    try:
        analysis = await llm.analyze(original, collected, EXTRACTION_PROMPT)
    except Exception as e:
        logger.error(f"ai_clarification_error: {e}")
        full_raw = f"Оригинальный запрос: {original}\nУточнения: " + " | ".join(collected)
        await _create_raw_ticket(update, chat_id, full_raw, username, f"Ошибка: {e}")
        return

    confidence    = _safe_confidence(analysis)
    should_reject = analysis.get("should_reject", False)

    if should_reject:
        STATE_MANAGER.clear_session(chat_id)
        reason = analysis.get("reject_reason", "Запрос отклонён.")
        await update.message.reply_text(f"❌ *Запрос отклонён.*\nПричина: {_escape_md(reason)}")
        return

    max_rounds_reached = round_num >= CONFIG.MAX_CLARIFICATION_ROUNDS
    if confidence >= CONFIG.CONFIDENCE_THRESHOLD or max_rounds_reached:
        await _show_preview(update, chat_id, analysis, username, original, collected, max_rounds_reached, round_num)
    else:
        missing = analysis.get("missing_info", [])
        if missing:
            STATE_MANAGER.update_session(chat_id, "clarifying", collected, round_num, analysis)
            await update.message.reply_text(f"❓ {_escape_md(missing[0])}")
        else:
            await _show_preview(update, chat_id, analysis, username, original, collected, False, round_num)

async def _process_request(update, chat_id, text, username, collected_answers):
    await update.message.reply_text("⏳ Анализирую запрос...")

    try:
        analysis = await llm.analyze(text, collected_answers, EXTRACTION_PROMPT)
    except Exception as e:
        logger.error(f"ai_analysis_error: {e}")
        await _create_raw_ticket(update, chat_id, text, username, str(e))
        return

    logger.info(f"ai_analysis_completed confidence={analysis.get('confidence', 0)}")

    confidence    = _safe_confidence(analysis)
    should_reject = analysis.get("should_reject", False)

    if should_reject:
        STATE_MANAGER.clear_session(chat_id)
        reason = analysis.get("reject_reason", "Запрос отклонён.")
        await update.message.reply_text(f"❌ *Запрос отклонён.*\nПричина: {_escape_md(reason)}")
        return

    if confidence < CONFIG.CONFIDENCE_THRESHOLD:
        missing = analysis.get("missing_info", [])
        if missing:
            STATE_MANAGER.update_session(chat_id, "clarifying", collected_answers, 1, analysis)
            await update.message.reply_text(f"❓ {_escape_md(missing[0])}")
            return

    await _show_preview(update, chat_id, analysis, username, text, collected_answers, False, 1)

async def _show_preview(update, chat_id, analysis, username, raw_text, collected_answers, forced, round_num):
    await update.message.reply_text("🔍 Проверяю на дубликаты...")

    try:
        dup = await deduplicator.check_duplicate(analysis.get("problem_statement", ""))
    except Exception as e:
        logger.error(f"dedup_error: {e}")
        dup = None

    if dup:
        try:
            await JIRA_CLIENT.add_comment(
                dup["key"],
                f"Дополнительное обращение от @{username}: {raw_text}"
            )
        except Exception as e:
            logger.error(f"comment_error: {e}")
        STATE_MANAGER.clear_session(chat_id)
        await update.message.reply_text(
            f"⚠️ *Этот запрос уже есть в бэклоге:* "
            f"[{dup['key']}]({CONFIG.JIRA_URL}/browse/{dup['key']})\n"
            f"Ваше обращение учтено (добавлен комментарий).",
            reply_markup=_dup_keyboard(dup["key"]),
        )
        return

    await update.message.reply_text("📊 Оцениваю приоритет...")
    try:
        scoring = await scorer.score(analysis)
    except Exception as e:
        logger.error(f"scoring_error: {e}")
        scoring = {"total_score": 0, "priority": "Low", "justification": "Scoring failed"}

    STATE_MANAGER.update_session(
        chat_id, "awaiting_confirmation", collected_answers, round_num, {
            "analysis": analysis,
            "scoring":  scoring,
            "raw_text": raw_text,
            "username": username,
            "forced":   forced,
        }
    )

    await update.message.reply_text(
        _build_preview(analysis, scoring),
        reply_markup=_preview_keyboard(),
    )

async def _create_confirmed_ticket(message, chat_id, session, username):
    data     = session.get("analysis_json", {})
    analysis = data.get("analysis", {})
    scoring  = data.get("scoring", {})
    raw_text = data.get("raw_text", "")
    forced   = data.get("forced", False)

    summary     = analysis.get("problem_statement", "Sales request")[:255]
    description = _build_jira_description(analysis, scoring, raw_text, username)
    priority    = scoring.get("priority", "Low")
    labels      = ["teamtrustgate", "ai-generated", "sales-request"]
    if forced:
        labels.append("insufficient-data")

    try:
        issue = await JIRA_CLIENT.create_issue(summary, description, priority, labels)
    except Exception as e:
        logger.error(f"jira_create_error: {e}")
        STATE_MANAGER.save_failed_request(chat_id, username, raw_text, analysis, str(e))
        STATE_MANAGER.clear_session(chat_id)
        await message.reply_text(
            f"⚠️ Jira временно недоступна. Запрос сохранён, повторим позже.\n"
            f"Ошибка: {_escape_md(str(e)[:200])}"
        )
        return

    STATE_MANAGER.clear_session(chat_id)
    # Сохраняем владельца (нормализованный username) для проверки прав
    STATE_MANAGER.track_ticket(issue["key"], chat_id, _norm(username), status="")
    logger.info(f"ticket_created key={issue['key']} priority={priority} score={scoring.get('total_score', 0)}")

    flag = "⚠️ Данных было недостаточно, тикет создан с пометкой.\n\n" if forced else ""
    await message.reply_text(
        f"{flag}"
        f"✅ *Тикет создан:* [{issue['key']}]({issue['url']})\n"
        f"*Приоритет:* {_priority_emoji(priority)} {_escape_md(priority)}\n"
        f"*Скоринг:* {scoring.get('total_score', 0)}/400\n"
        f"*Охват:* {_escape_md(analysis.get('reach', 'unknown'))}\n"
        f"*Я уведомлю тебя когда статус изменится.*",
        reply_markup=_ticket_keyboard(issue["key"]),
    )

async def _create_raw_ticket(update, chat_id, text, username, error_msg):
    await update.message.reply_text("⚠️ AI-анализ недоступен. Создаю сырой тикет...")
    try:
        issue = await JIRA_CLIENT.create_issue(
            f"[RAW] {text[:200]}",
            f"*Инициатор:* @{username}\n*Оригинал:* {text}\n*Ошибка AI:* {error_msg}",
            "Low",
            ["teamtrustgate", "raw-request", "sales-request"],
        )
        STATE_MANAGER.clear_session(chat_id)
        STATE_MANAGER.track_ticket(issue["key"], chat_id, _norm(username), status="")
        await update.message.reply_text(
            f"✅ *Сырой тикет создан:* [{issue['key']}]({issue['url']})\n"
            f"Продуктовая команда рассмотрит запрос вручную.",
            reply_markup=_ticket_keyboard(issue["key"]),
        )
    except Exception as e:
        STATE_MANAGER.save_failed_request(chat_id, username, text, None, str(e))
        STATE_MANAGER.clear_session(chat_id)
        await update.message.reply_text(
            f"❌ Не удалось создать тикет. Админ уведомлён.\n{_escape_md(str(e)[:200])}"
        )

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    app = (
        Application.builder()
        .token(CONFIG.TELEGRAM_TOKEN)
        .defaults(Defaults(parse_mode="Markdown"))
        .concurrent_updates(True)
        .build()
    )

    app.add_handler(CommandHandler("start",  start_handler))
    app.add_handler(CommandHandler("cancel", cancel_handler))
    app.add_handler(CommandHandler("status", status_handler))
    app.add_handler(CommandHandler("list",   list_handler))
    app.add_handler(CommandHandler("delete", delete_handler))
    app.add_handler(CommandHandler("stats",  stats_handler))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))

    job_queue = app.job_queue
    if job_queue:
        job_queue.run_repeating(_poll_status_changes, interval=STATUS_POLL_INTERVAL, first=60)
        logger.info(f"Status polling запущен (каждые {STATUS_POLL_INTERVAL}с)")
    else:
        logger.warning("JobQueue недоступна. Установи: pip install 'python-telegram-bot[job-queue]'")

    logger.info("TeamTrustGate bot starting...")
    app.run_polling()

if __name__ == "__main__":
    main()
