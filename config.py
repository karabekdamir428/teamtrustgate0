"""Configuration and environment validation."""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_TOKEN", "")
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    JIRA_URL: str = os.getenv("JIRA_URL", "").rstrip("/")
    JIRA_EMAIL: str = os.getenv("JIRA_EMAIL", "")
    JIRA_API_TOKEN: str = os.getenv("JIRA_API_TOKEN", "")
    JIRA_PROJECT_KEY: str = os.getenv("JIRA_PROJECT_KEY", "")
    PRODUCT_STRATEGY: str = os.getenv("PRODUCT_STRATEGY", "")
    ALLOWED_USERNAMES: list = [u.strip() for u in os.getenv("ALLOWED_USERNAMES", "").split(",") if u.strip()]
    DB_PATH: str = os.getenv("DB_PATH", "teamtrustgate.db")
    MAX_CLARIFICATION_ROUNDS: int = int(os.getenv("MAX_CLARIFICATION_ROUNDS", "3"))
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
    DEDUP_DAYS: int = int(os.getenv("DEDUP_DAYS", "90"))
    JIRA_TIMEOUT: int = int(os.getenv("JIRA_TIMEOUT", "10"))
    LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "30"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    @classmethod
    def validate(cls):
        required = [
            ("TELEGRAM_TOKEN", cls.TELEGRAM_TOKEN),
            ("LLM_API_KEY", cls.LLM_API_KEY),
            ("JIRA_URL", cls.JIRA_URL),
            ("JIRA_EMAIL", cls.JIRA_EMAIL),
            ("JIRA_API_TOKEN", cls.JIRA_API_TOKEN),
            ("JIRA_PROJECT_KEY", cls.JIRA_PROJECT_KEY),
        ]
        missing = [name for name, val in required if not val]
        if missing:
            raise ValueError(f"Missing required env vars: {', '.join(missing)}")

CONFIG = Config()
