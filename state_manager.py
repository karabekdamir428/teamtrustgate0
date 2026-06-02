"""SQLite state management for clarification sessions."""
import sqlite3
import json
from datetime import datetime, timezone
from typing import Optional, Dict, Any
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
            conn.commit()

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

STATE_MANAGER = StateManager()
