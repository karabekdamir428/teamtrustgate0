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

# ── Keyboard builders ─────────────────────────────────────────────────────
def _ticket_keyboard(issue_key: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔗 Открыть в Jira", url=f"{CONFIG.JIRA_URL}/browse/{issue_key}")],
        [
            InlineKeyboardButton("📊 Статус", callback_data=f"status:{issue_key}"),
            InlineKeyboardButton("🗑 Удалить", callback_data=f"delete_confirm:{issue_key}"),
        ],
    ])

def _confirm_delete_keyboard(issue_key: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("✅ Да, удалить", callback_data=f"delete_yes:{issue_key}"),
        InlineKeyboardButton("❌ Отмена",       callback_data=f"delete_no:{issue_key}"),
    ]])

def _dup_keyboard(issue_key: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("🔗 Открыть дубликат", url=f"{CONFIG.JIRA_URL}/browse/{issue_key}"),
        InlineKeyboardButton("📊 Статус", callback_data=f"status:{issue_key}"),
    ]])

def _preview_keyboard() -> InlineKeyboardMarkup:
    """Кнопки под превью тикета перед созданием."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("✅ Создать тикет",  callback_data="preview:confirm")],
        [
            InlineKeyboardButton("✏️ Уточнить",   callback_data="preview:clarify"),
            InlineKeyboardButton("❌ Отменить",    callback_data="preview:cancel"),
        ],
    ])

# ── Helpers ───────────────────────────────────────────────────────────────
def _is_allowed(username: str) -> bool:
    if not CONFIG.ALLOWED_USERNAMES:
        return True
    return username in CONFIG.ALLOWED_USERNAMES

def _build_jira_description(
    analysis: dict, scoring: dict, raw_text: str, requested_by: str
) -> str:
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
    """Формирует текст превью тикета для подтверждения сейлзом."""
    priority = scoring.get("priority", "Low")
    reach    = analysis.get("reach", "unknown")
    return (
        f"🤖 *Вот что я понял из запроса:*\n\n"
        f"📋 *Проблема:*\n{_escape_md(analysis.get('problem_statement', ''))}\n\n"
        f"👤 *Клиент:* {_escape_md(analysis.get('client_context', 'не указан'))}\n"
        f"💰 *Риск потери выручки:* {analysis.get('revenue_at_risk', 0)}/10\n"
        f"📊 *Охват:* {_reach_label(reach)}\n"
        f"🎯 *Приоритет:* {_priority_emoji(priority)} {_escape_md(priority)} "
        f"({scoring.get('total_score', 0)}/400)\n\n"
        f"Всё верно? Создаём тикет?"
    )

# ── Callback query handler ─────────────────────────────────────────────────
async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    data     = query.data or ""
    chat_id  = update.effective_chat.id
    user     = update.effective_user
    username = user.username or user.first_name or str(user.id)

    # ── Превью: подтвердить создание ──────────────────────────────────────
    if data == "preview:confirm":
        session = STATE_MANAGER.get_session(chat_id)
        if not session or session.get("state") != "awaiting_confirmation":
            await query.message.reply_text("⚠️ Сессия устарела. Отправь запрос заново.")
            return
        await query.message.edit_reply_markup(reply_markup=None)
        await query.message.reply_text("📝 Создаю тикет в Jira...")
        await _create_confirmed_ticket(query.message, chat_id, session, username, forced=False)

    # ── Превью: уточнить ──────────────────────────────────────────────────
    elif data == "preview:clarify":
        session = STATE_MANAGER.get_session(chat_id)
        if not session:
            await query.message.reply_text("⚠️ Сессия устарела. Отправь запрос заново.")
            return
        # Переводим в режим уточнения
        STATE_MANAGER.update_session(
            chat_id, "clarifying",
            session.get("collected_answers", []),
            session.get("round", 0),
            session.get("analysis_json"),
        )
        await query.message.edit_reply_markup(reply_markup=None)
        await query.message.reply_text(
            "✏️ Что именно нужно уточнить или исправить? Напиши и я пересоздам превью."
        )

    # ── Превью: отменить ──────────────────────────────────────────────────
    elif data == "preview:cancel":
        STATE_MANAGER.clear_session(chat_id)
        await query.message.edit_reply_markup(reply_markup=None)
        await query.message.reply_text("🚫 Создание тикета отменено.")

    # ── Статус тикета ─────────────────────────────────────────────────────
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
            await query.message.reply_text(
                f"📋 *{issue_key}*\n"
                f"*Статус:* {_escape_md(status_name)}\n"
                f"*Приоритет:* {_priority_emoji(priority)} {_escape_md(priority)}\n"
                f"*Тема:* {_escape_md(summary)}"
            )
        except Exception as e:
            await query.message.reply_text(
                f"❌ Не удалось получить статус: {_escape_md(str(e)[:200])}"
            )

    # ── Подтверждение удаления ────────────────────────────────────────────
    elif data.startswith("delete_confirm:"):
        issue_key = data.split(":", 1)[1]
        await query.message.reply_text(
            f"⚠️ Удалить тикет *{issue_key}*?\nЭто действие необратимо.",
            reply_markup=_confirm_delete_keyboard(issue_key),
        )

    elif data.startswith("delete_yes:"):
        issue_key = data.split(":", 1)[1]
        try:
            await JIRA_CLIENT.delete_issue(issue_key)
            logger.info(f"ticket_deleted key={issue_key} by={username}")
            await query.message.edit_text(f"🗑 Тикет *{issue_key}* удалён.")
        except Exception as e:
            await query.message.reply_text(
                f"❌ Не удалось удалить: {_escape_md(str(e)[:200])}"
            )

    elif data.startswith("delete_no:"):
        await query.message.edit_text("✅ Удаление отменено.")

# ── Command handlers ───────────────────────────────────────────────────────
async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not _is_allowed(user.username or ""):
        await update.message.reply_text("❌ Доступ ограничен.")
        return
    await update.message.reply_text(
        "👋 Привет! Я TeamTrustGate — агент обработки клиентских запросов.\n\n"
        "Просто опиши запрос клиента своими словами, и я создам тикет для продуктовой команды.\n\n"
        "Команды:\n"
        "/start — начать\n"
        "/cancel — отменить текущую сессию\n"
        "/status TT-XX — статус тикета\n"
        "/list — мои последние 5 тикетов\n"
        "/delete TT-XX — удалить тикет"
    )

async def cancel_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    STATE_MANAGER.clear_session(chat_id)
    await update.message.reply_text("🚫 Сессия отменена. Отправь новый запрос.")

async def status_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
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
        await update.message.reply_text(
            f"📋 *{issue_key}*\n"
            f"*Статус:* {_escape_md(status_name)}\n"
            f"*Приоритет:* {_priority_emoji(priority)} {_escape_md(priority)}\n"
            f"*Тема:* {_escape_md(summary)}",
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("🔗 Открыть в Jira", url=f"{CONFIG.JIRA_URL}/browse/{issue_key}"),
                InlineKeyboardButton("🗑 Удалить", callback_data=f"delete_confirm:{issue_key}"),
            ]])
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
    args = context.args
    if not args:
        await update.message.reply_text("Использование: /delete TT-42")
        return
    issue_key = args[0].upper()
    await update.message.reply_text(
        f"⚠️ Удалить тикет *{issue_key}*?\nЭто действие необратимо.",
        reply_markup=_confirm_delete_keyboard(issue_key),
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

    # Если ждём уточнения после нажатия "Уточнить" в превью
    if session and session["state"] == "clarifying":
        await _handle_clarification(update, chat_id, text, username, session)
        return

    # Новый запрос
    logger.info(f"request_received chat_id={chat_id} msg_len={len(text)}")
    STATE_MANAGER.create_session(chat_id, text)
    await _process_request(update, chat_id, text, username, collected_answers=[])

# ── Core logic ─────────────────────────────────────────────────────────────
async def _handle_clarification(
    update: Update, chat_id: int, text: str, username: str, session: dict
):
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
        # Достаточно данных — показываем превью
        await _show_preview(update, chat_id, analysis, username, original, collected, max_rounds_reached, round_num)
    else:
        missing = analysis.get("missing_info", [])
        if missing:
            STATE_MANAGER.update_session(chat_id, "clarifying", collected, round_num, analysis)
            await update.message.reply_text(f"❓ {_escape_md(missing[0])}")
        else:
            await _show_preview(update, chat_id, analysis, username, original, collected, False, round_num)

async def _process_request(
    update: Update, chat_id: int, text: str, username: str, collected_answers: list
):
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

    # Данных достаточно — показываем превью
    await _show_preview(update, chat_id, analysis, username, text, collected_answers, False, 1)

async def _show_preview(
    update: Update, chat_id: int, analysis: dict, username: str,
    raw_text: str, collected_answers: list, forced: bool, round_num: int
):
    """Показывает превью тикета сейлзу для подтверждения перед созданием."""
    await update.message.reply_text("🔍 Проверяю на дубликаты...")

    # Дедупликация ДО показа превью — нет смысла показывать превью если это дубликат
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
        logger.info(f"dedup_found key={dup['key']}")
        return

    # Скоринг
    await update.message.reply_text("📊 Оцениваю приоритет...")
    try:
        scoring = await scorer.score(analysis)
    except Exception as e:
        logger.error(f"scoring_error: {e}")
        scoring = {"total_score": 0, "priority": "Low", "justification": "Scoring failed"}

    # Сохраняем всё необходимое в сессии для создания тикета после подтверждения
    STATE_MANAGER.update_session(
        chat_id, "awaiting_confirmation", collected_answers, round_num, {
            "analysis": analysis,
            "scoring":  scoring,
            "raw_text": raw_text,
            "username": username,
            "forced":   forced,
        }
    )

    # Показываем превью с кнопками
    await update.message.reply_text(
        _build_preview(analysis, scoring),
        reply_markup=_preview_keyboard(),
    )

async def _create_confirmed_ticket(message, chat_id: int, session: dict, username: str, forced: bool):
    """Создаёт тикет после подтверждения сейлзом."""
    data     = session.get("analysis_json", {})
    analysis = data.get("analysis", {})
    scoring  = data.get("scoring", {})
    raw_text = data.get("raw_text", "")
    forced   = data.get("forced", forced)

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
    logger.info(f"ticket_created key={issue['key']} priority={priority} score={scoring.get('total_score', 0)}")

    flag = "⚠️ Данных было недостаточно, тикет создан с пометкой.\n\n" if forced else ""
    await message.reply_text(
        f"{flag}"
        f"✅ *Тикет создан:* [{issue['key']}]({issue['url']})\n"
        f"*Приоритет:* {_priority_emoji(priority)} {_escape_md(priority)}\n"
        f"*Скоринг:* {scoring.get('total_score', 0)}/400\n"
        f"*Охват:* {_escape_md(analysis.get('reach', 'unknown'))}\n"
        f"*Продуктовая команда получила запрос.*",
        reply_markup=_ticket_keyboard(issue["key"]),
    )

async def _create_raw_ticket(
    update: Update, chat_id: int, text: str, username: str, error_msg: str
):
    await update.message.reply_text("⚠️ AI-анализ недоступен. Создаю сырой тикет...")
    try:
        issue = await JIRA_CLIENT.create_issue(
            f"[RAW] {text[:200]}",
            f"*Инициатор:* @{username}\n*Оригинал:* {text}\n*Ошибка AI:* {error_msg}",
            "Low",
            ["teamtrustgate", "raw-request", "sales-request"],
        )
        STATE_MANAGER.clear_session(chat_id)
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
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))

    logger.info("TeamTrustGate bot starting...")
    app.run_polling()

if __name__ == "__main__":
    main()
