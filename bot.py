"""TeamTrustGate — AI Agent for Sales Request Processing.
Telegram → LLM → Jira
"""
import asyncio
import json
import logging
from datetime import datetime, timezone

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from config import CONFIG, Config
from state_manager import STATE_MANAGER
from llm_adapter import get_llm_provider
from jira_client import JIRA_CLIENT
from deduplicator import Deduplicator
from scorer import Scorer

# ── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, CONFIG.LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("teamtrustgate")

# ── Init ─────────────────────────────────────────────────────────────────
Config.validate()
llm = get_llm_provider()
deduplicator = Deduplicator(llm)
scorer = Scorer(llm)

with open("prompts/extraction.txt", "r", encoding="utf-8") as f:
    EXTRACTION_PROMPT = f.read()

# ── Helpers ──────────────────────────────────────────────────────────────
def _is_allowed(username: str) -> bool:
    if not CONFIG.ALLOWED_USERNAMES:
        return True
    return username in CONFIG.ALLOWED_USERNAMES

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

# ── Handlers ─────────────────────────────────────────────────────────────
async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not _is_allowed(user.username or ""):
        await update.message.reply_text("❌ Доступ ограничен.")
        return
    await update.message.reply_text(
        "👋 Привет! Я TeamTrustGate — агент обработки клиентских запросов.\n\n"
        "Просто опиши запрос клиента своими словами, и я создам тикет для продуктовой команды.\n"
        "Команды:\n"
        "/start — начать\n"
        "/cancel — отменить текущую сессию\n"
        "/status [TT-XX] — проверить статус тикета"
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
        status, text = await JIRA_CLIENT._request("GET", f"/issue/{issue_key}?fields=status,summary")
        data = json.loads(text)
        status_name = data["fields"]["status"]["name"]
        summary = data["fields"]["summary"]
        await update.message.reply_text(
            f"📋 *{issue_key}*\n"
            f"*Статус:* {status_name}\n"
            f"*Тема:* {summary}"
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Не удалось получить статус: {str(e)[:200]}")

async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    chat_id = update.effective_chat.id
    text = update.message.text or ""
    username = user.username or user.first_name or str(user.id)

    if not _is_allowed(user.username or ""):
        await update.message.reply_text("❌ Доступ ограничен.")
        return

    if not text.strip():
        await update.message.reply_text("Пожалуйста, отправьте текстовое описание запроса.")
        return

    session = STATE_MANAGER.get_session(chat_id)

    # ── Active clarification session ─────────────────────────────────────
    if session and session["state"] == "clarifying":
        await _handle_clarification(update, chat_id, text, username, session)
        return

    # ── New request ──────────────────────────────────────────────────────
    logger.info(f"request_received chat_id={chat_id} msg_len={len(text)}")
    STATE_MANAGER.create_session(chat_id, text)
    await _process_request(update, chat_id, text, username, collected_answers=[])

async def _handle_clarification(update: Update, chat_id: int, text: str, username: str, session: dict):
    collected = session["collected_answers"] + [text]
    round_num = session["round"] + 1
    original = session["original_message"]

    await update.message.reply_text("⏳ Анализирую уточнения...")

    try:
        analysis = await llm.analyze(original, collected, EXTRACTION_PROMPT)
    except Exception as e:
        logger.error(f"ai_error during clarification: {e}")
        await update.message.reply_text("⚠️ Ошибка анализа. Попробуйте еще раз или отправьте /cancel.")
        return

    confidence = float(analysis.get("confidence", 0))
    should_reject = analysis.get("should_reject", False)

    if should_reject:
        STATE_MANAGER.clear_session(chat_id)
        reason = analysis.get("reject_reason", "Запрос отклонен.")
        await update.message.reply_text(f"❌ *Запрос отклонен.*\nПричина: {reason}")
        logger.info(f"ticket_rejected reason={reason}")
        return

    if confidence >= CONFIG.CONFIDENCE_THRESHOLD or round_num >= CONFIG.MAX_CLARIFICATION_ROUNDS:
        STATE_MANAGER.update_session(chat_id, "scored", collected, round_num, analysis)
        await _finalize_ticket(update, chat_id, analysis, username, original, collected, round_num >= CONFIG.MAX_CLARIFICATION_ROUNDS)
    else:
        missing = analysis.get("missing_info", [])
        if missing:
            STATE_MANAGER.update_session(chat_id, "clarifying", collected, round_num, analysis)
            await update.message.reply_text(f"❓ {missing[0]}")
        else:
            STATE_MANAGER.update_session(chat_id, "scored", collected, round_num, analysis)
            await _finalize_ticket(update, chat_id, analysis, username, original, collected, False)

async def _process_request(update: Update, chat_id: int, text: str, username: str, collected_answers: list):
    await update.message.reply_text("⏳ Анализирую запрос...")

    try:
        analysis = await llm.analyze(text, collected_answers, EXTRACTION_PROMPT)
    except Exception as e:
        logger.error(f"ai_analysis_error: {e}")
        # Fallback: create raw ticket
        await _create_raw_ticket(update, chat_id, text, username, str(e))
        return

    logger.info(f"ai_analysis_completed confidence={analysis.get('confidence', 0)}")

    confidence = float(analysis.get("confidence", 0))
    should_reject = analysis.get("should_reject", False)

    if should_reject:
        STATE_MANAGER.clear_session(chat_id)
        reason = analysis.get("reject_reason", "Запрос отклонен.")
        await update.message.reply_text(f"❌ *Запрос отклонен.*\nПричина: {reason}")
        logger.info(f"ticket_rejected reason={reason}")
        return

    if confidence < CONFIG.CONFIDENCE_THRESHOLD:
        missing = analysis.get("missing_info", [])
        if missing:
            STATE_MANAGER.update_session(chat_id, "clarifying", collected_answers, 1, analysis)
            await update.message.reply_text(f"❓ {missing[0]}")
            return

    await _finalize_ticket(update, chat_id, analysis, username, text, collected_answers, False)

async def _finalize_ticket(update: Update, chat_id: int, analysis: dict, username: str, raw_text: str, collected_answers: list, forced: bool):
    # Deduplication
    await update.message.reply_text("🔍 Проверяю на дубликаты...")
    try:
        dup = await deduplicator.check_duplicate(analysis.get("problem_statement", ""))
    except Exception as e:
        logger.error(f"dedup_error: {e}")
        dup = None

    if dup:
        try:
            comment = f"Дополнительное обращение от @{username}: {raw_text}"
            await JIRA_CLIENT.add_comment(dup["key"], comment)
        except Exception as e:
            logger.error(f"comment_error: {e}")

        STATE_MANAGER.clear_session(chat_id)
        await update.message.reply_text(
            f"⚠️ *Этот запрос уже есть в бэклоге:* [{dup['key']}]({CONFIG.JIRA_URL}/browse/{dup['key']})\n"
            f"Ваше обращение учтено (добавлен комментарий)."
        )
        logger.info(f"dedup_found key={dup['key']}")
        return

    logger.info("dedup_check_completed duplicate_found=False")

    # Scoring
    await update.message.reply_text("📊 Оцениваю приоритет...")
    try:
        scoring = await scorer.score(analysis)
    except Exception as e:
        logger.error(f"scoring_error: {e}")
        scoring = {"total_score": 0, "priority": "Low", "justification": "Scoring failed"}

    # Create Jira ticket
    await update.message.reply_text("📝 Создаю тикет в Jira...")

    summary = analysis.get("problem_statement", "Sales request")[:255]
    description = _build_jira_description(analysis, scoring, raw_text, username)
    priority = scoring.get("priority", "Low")
    labels = ["teamtrustgate", "ai-generated", "sales-request"]
    if forced:
        labels.append("insufficient-data")

    try:
        issue = await JIRA_CLIENT.create_issue(summary, description, priority, labels)
    except Exception as e:
        logger.error(f"jira_create_error: {e}")
        STATE_MANAGER.save_failed_request(chat_id, username, raw_text, analysis, str(e))
        STATE_MANAGER.clear_session(chat_id)
        await update.message.reply_text(
            f"⚠️ Jira временно недоступна. Запрос сохранен, повторим позже.\n"
            f"Ошибка: {str(e)[:200]}"
        )
        return

    STATE_MANAGER.clear_session(chat_id)
    logger.info(f"ticket_created key={issue['key']} priority={priority} score={scoring.get('total_score', 0)}")

    # Feedback
    flag = "⚠️ Данных было недостаточно, тикет создан с пометкой.\n\n" if forced else ""
    await update.message.reply_text(
        f"{flag}"
        f"✅ *Тикет создан:* [{issue['key']}]({issue['url']})\n"
        f"*Приоритет:* {priority}\n"
        f"*Скоринг:* {scoring.get('total_score', 0)}/400\n"
        f"*Охват:* {analysis.get('reach', 'unknown')}\n"
        f"*Продуктовая команда получила запрос.*"
    )

async def _create_raw_ticket(update: Update, chat_id: int, text: str, username: str, error_msg: str):
    await update.message.reply_text("⚠️ AI-анализ недоступен. Создаю сырой тикет...")
    summary = f"[RAW] {text[:200]}"
    description = f"*Инициатор:* @{username}\n*Оригинал:* {text}\n*Ошибка AI:* {error_msg}"
    try:
        issue = await JIRA_CLIENT.create_issue(summary, description, "Low", ["teamtrustgate", "raw-request", "sales-request"])
        STATE_MANAGER.clear_session(chat_id)
        await update.message.reply_text(
            f"✅ *Сырой тикет создан:* [{issue['key']}]({issue['url']})\n"
            f"Продуктовая команда рассмотрит запрос вручную."
        )
    except Exception as e:
        STATE_MANAGER.save_failed_request(chat_id, username, text, None, str(e))
        STATE_MANAGER.clear_session(chat_id)
        await update.message.reply_text(f"❌ Не удалось создать тикет. Админ уведомлен.\n{str(e)[:200]}")

# ── Main ─────────────────────────────────────────────────────────────────
def main():
    app = Application.builder().token(CONFIG.TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("cancel", cancel_handler))
    app.add_handler(CommandHandler("status", status_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))

    logger.info("🚀 TeamTrustGate bot starting...")
    app.run_polling()

if __name__ == "__main__":
    main()
