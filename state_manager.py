"""SQLite state management for clarification sessions and ticket tracking."""
import sqlite3
import json
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from config import CONFIG


class StateManager:
    def __init__(self, db_path: str = CONFIG.DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    chat_id INTEGER PRIMARY KEY,
                    state TEXT NOT NULL DEFAULT 'idle',
                    original_message TEXT,
                    collected_answers TEXT DEFAULT '[]',
                    round INTEGER DEFAULT 0,
                    analysis_json TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS failed_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER,
                    username TEXT,
                    message TEXT,
                    analysis_json TEXT,
                    error TEXT,
                    created_at TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tracked_tickets (
                    issue_key TEXT PRIMARY KEY,
                    chat_id INTEGER NOT NULL,
                    username TEXT,
                    last_status TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sent_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    issue_key TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    sent_at TEXT,
                    UNIQUE(issue_key, alert_type)
                )
            """)
            conn.commit()

    # ── Sessions ──────────────────────────────────────────────────────────
    def get_session(self, chat_id: int) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM sessions WHERE chat_id = ?", (chat_id,)
            ).fetchone()
            if not row:
                return None
            return {
                "chat_id": row["chat_id"],
                "state": row["state"],
                "original_message": row["original_message"] or "",
                "collected_answers": json.loads(row["collected_answers"] or "[]"),
                "round": row["round"] or 0,
                "analysis_json": json.loads(row["analysis_json"]) if row["analysis_json"] else None,
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }

    def create_session(self, chat_id: int, original_message: str):
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO sessions
                (chat_id, state, original_message, collected_answers, round, analysis_json, created_at, updated_at)
                VALUES (?, 'analyzing', ?, '[]', 0, NULL, ?, ?)""",
                (chat_id, original_message, now, now)
            )
            conn.commit()

    def update_session(self, chat_id: int, state: str, collected_answers: list, round_num: int, analysis_json: Optional[dict] = None):
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """UPDATE sessions SET state=?, collected_answers=?, round=?, analysis_json=?, updated_at=?
                WHERE chat_id=?""",
                (state, json.dumps(collected_answers, ensure_ascii=False), round_num,
                 json.dumps(analysis_json, ensure_ascii=False) if analysis_json else None, now, chat_id)
            )
            conn.commit()

    def clear_session(self, chat_id: int):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM sessions WHERE chat_id = ?", (chat_id,))
            conn.commit()

    def save_failed_request(self, chat_id: int, username: str, message: str, analysis: Optional[dict], error: str):
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO failed_requests (chat_id, username, message, analysis_json, error, created_at)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (chat_id, username, message,
                 json.dumps(analysis, ensure_ascii=False) if analysis else None, error, now)
            )
            conn.commit()

    # ── Ticket tracking + ownership ───────────────────────────────────────
    def track_ticket(self, issue_key: str, chat_id: int, username: str, status: str = ""):
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO tracked_tickets
                (issue_key, chat_id, username, last_status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (issue_key, chat_id, username, status, now, now)
            )
            conn.commit()

    def get_ticket_owner(self, issue_key: str) -> Optional[str]:
        """
        Возвращает username создателя тикета (нормализованный, lowercase).
        None если тикет не отслеживается (создан не через этот бот или старый).
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT username FROM tracked_tickets WHERE issue_key = ?",
                (issue_key,)
            ).fetchone()
            if not row or not row["username"]:
                return None
            return row["username"].lstrip("@").lower()

    def get_tracked_tickets(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM tracked_tickets").fetchall()
            return [
                {
                    "issue_key": r["issue_key"],
                    "chat_id": r["chat_id"],
                    "username": r["username"],
                    "last_status": r["last_status"] or "",
                }
                for r in rows
            ]

    def update_ticket_status(self, issue_key: str, new_status: str):
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE tracked_tickets SET last_status=?, updated_at=? WHERE issue_key=?",
                (new_status, now, issue_key)
            )
            conn.commit()

    def untrack_ticket(self, issue_key: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM tracked_tickets WHERE issue_key = ?", (issue_key,))
            conn.execute("DELETE FROM sent_alerts WHERE issue_key = ?", (issue_key,))
            conn.commit()

    # ── SLA alerts ─────────────────────────────────────────────────────────
    def was_alert_sent(self, issue_key: str, alert_type: str) -> bool:
        """Проверяет был ли уже отправлен алерт данного типа для тикета."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT 1 FROM sent_alerts WHERE issue_key = ? AND alert_type = ?",
                (issue_key, alert_type)
            ).fetchone()
            return row is not None

    def mark_alert_sent(self, issue_key: str, alert_type: str):
        """Помечает что алерт отправлен, чтобы не повторять."""
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR IGNORE INTO sent_alerts (issue_key, alert_type, sent_at)
                VALUES (?, ?, ?)""",
                (issue_key, alert_type, now)
            )
            conn.commit()

    def clear_alert(self, issue_key: str, alert_type: str):
        """Сбрасывает отметку об алерте (если тикет сдвинулся — можно алертить заново)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM sent_alerts WHERE issue_key = ? AND alert_type = ?",
                (issue_key, alert_type)
            )
            conn.commit()

    # ── Stats ──────────────────────────────────────────────────────────────
    def get_local_stats(self) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            total = conn.execute(
                "SELECT COUNT(*) as cnt FROM tracked_tickets"
            ).fetchone()["cnt"]

            this_month = conn.execute(
                "SELECT COUNT(*) as cnt FROM tracked_tickets WHERE created_at >= ?",
                (month_start,)
            ).fetchone()["cnt"]

            top_users = conn.execute(
                """SELECT username, COUNT(*) as cnt
                FROM tracked_tickets
                WHERE username IS NOT NULL AND username != ''
                GROUP BY username
                ORDER BY cnt DESC
                LIMIT 5"""
            ).fetchall()

            failed = conn.execute(
                "SELECT COUNT(*) as cnt FROM failed_requests"
            ).fetchone()["cnt"]

            failed_month = conn.execute(
                "SELECT COUNT(*) as cnt FROM failed_requests WHERE created_at >= ?",
                (month_start,)
            ).fetchone()["cnt"]

        return {
            "total":       total,
            "this_month":  this_month,
            "top_users":   [{"username": r["username"], "count": r["cnt"]} for r in top_users],
            "failed":      failed,
            "failed_month": failed_month,
        }


STATE_MANAGER = StateManager()
