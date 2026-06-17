"""Configuration and environment validation."""
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("teamtrustgate")


class Config:
    # ── Telegram ──────────────────────────────────────────────────────────
    TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_TOKEN", "")

    # ── LLM провайдеры ────────────────────────────────────────────────────
    LLM_PROVIDER:    str = os.getenv("LLM_PROVIDER", "gemini")
    LLM_API_KEY:     str = os.getenv("LLM_API_KEY", "")
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")

    # Названия моделей — берутся из Railway Variables, дефолты на случай отсутствия
    GEMINI_MODEL:   str = os.getenv("GEMINI_MODEL",   "gemini-2.5-flash")
    DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    # ── Jira ──────────────────────────────────────────────────────────────
    JIRA_URL:         str = os.getenv("JIRA_URL", "").rstrip("/")
    JIRA_EMAIL:       str = os.getenv("JIRA_EMAIL", "")
    JIRA_API_TOKEN:   str = os.getenv("JIRA_API_TOKEN", "")
    JIRA_PROJECT_KEY: str = os.getenv("JIRA_PROJECT_KEY", "")

    # ── Продуктовый контекст ──────────────────────────────────────────────
    PRODUCT_STRATEGY: str = os.getenv("PRODUCT_STRATEGY", "")

    # ── Доступ ────────────────────────────────────────────────────────────
    ALLOWED_USERNAMES: list = [
        u.strip()
        for u in os.getenv("ALLOWED_USERNAMES", "").split(",")
        if u.strip()
    ]

    # ── База данных ───────────────────────────────────────────────────────
    DB_PATH: str = os.getenv("DB_PATH", "teamtrustgate.db")

    # ── Логика агента ─────────────────────────────────────────────────────
    MAX_CLARIFICATION_ROUNDS: int   = int(os.getenv("MAX_CLARIFICATION_ROUNDS", "3"))
    CONFIDENCE_THRESHOLD:     float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
    DEDUP_DAYS:               int   = int(os.getenv("DEDUP_DAYS", "90"))

    # ── Таймауты ──────────────────────────────────────────────────────────
    JIRA_TIMEOUT: int = int(os.getenv("JIRA_TIMEOUT", "10"))
    LLM_TIMEOUT:  int = int(os.getenv("LLM_TIMEOUT",  "30"))

    # ── Логирование ───────────────────────────────────────────────────────
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    @classmethod
    def validate(cls):
        """
        Проверяет наличие обязательных переменных окружения.
        Вызывается при старте бота — падает с понятной ошибкой если что-то не заполнено.
        """
        required = [
            ("TELEGRAM_TOKEN",   cls.TELEGRAM_TOKEN),
            ("LLM_API_KEY",      cls.LLM_API_KEY),
            ("JIRA_URL",         cls.JIRA_URL),
            ("JIRA_EMAIL",       cls.JIRA_EMAIL),
            ("JIRA_API_TOKEN",   cls.JIRA_API_TOKEN),
            ("JIRA_PROJECT_KEY", cls.JIRA_PROJECT_KEY),
        ]

        missing = [name for name, val in required if not val]

        # Если основной провайдер DeepSeek — нужен хотя бы один из ключей
        if (
            cls.LLM_PROVIDER.lower() == "deepseek"
            and not cls.DEEPSEEK_API_KEY
            and not cls.LLM_API_KEY
        ):
            missing.append("DEEPSEEK_API_KEY или LLM_API_KEY")

        if missing:
            raise ValueError(
                f"❌ Отсутствуют обязательные переменные окружения: {', '.join(missing)}\n"
                f"Заполни их в Railway → Variables."
            )

        # Предупреждения о необязательных но важных переменных
        if not cls.PRODUCT_STRATEGY:
            logger.warning(
                "⚠️ PRODUCT_STRATEGY не задана — скоринг стратегического соответствия будет неточным"
            )
        if not cls.ALLOWED_USERNAMES:
            logger.warning(
                "⚠️ ALLOWED_USERNAMES не задан — бот доступен всем пользователям Telegram"
            )

        logger.info("✅ Конфигурация прошла валидацию")
        logger.info(f"   LLM_PROVIDER  = {cls.LLM_PROVIDER}")
        logger.info(f"   GEMINI_MODEL  = {cls.GEMINI_MODEL}")
        logger.info(f"   DEEPSEEK_MODEL = {cls.DEEPSEEK_MODEL}")
        logger.info(f"   JIRA_PROJECT  = {cls.JIRA_PROJECT_KEY}")
        logger.info(f"   DEDUP_DAYS    = {cls.DEDUP_DAYS}")
        logger.info(f"   CONFIDENCE    = {cls.CONFIDENCE_THRESHOLD}")


CONFIG = Config()
