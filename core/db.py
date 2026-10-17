"""Persistence layer.

DataNova stores users, activity history, feedback, issue reports and shared
reports. By default everything goes into a local SQLite file
(``data/datanova.db``) so the app runs with zero setup. If the environment
variable ``DATANOVA_MONGO_URI`` is set and ``pymongo`` is installed, user
accounts are stored in MongoDB instead.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("DATANOVA_DATA_DIR", BASE_DIR / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "datanova.db"

_LOCK = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    username      TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL,
    email         TEXT,
    full_name     TEXT,
    role          TEXT DEFAULT 'user',
    created_at    TEXT,
    last_login    TEXT,
    preferences   TEXT DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS activity (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    username   TEXT,
    action     TEXT,
    detail     TEXT,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS feedback (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    username   TEXT,
    rating     INTEGER,
    category   TEXT,
    message    TEXT,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS issues (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    username   TEXT,
    severity   TEXT,
    page       TEXT,
    message    TEXT,
    status     TEXT DEFAULT 'open',
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS shares (
    share_id      TEXT PRIMARY KEY,
    owner         TEXT,
    title         TEXT,
    visibility    TEXT,
    password_hash TEXT,
    expires_at    TEXT,
    views         INTEGER DEFAULT 0,
    created_at    TEXT
);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@contextmanager
def connect():
    with _LOCK:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


def execute(sql: str, params: tuple = ()) -> None:
    with connect() as conn:
        conn.execute(sql, params)


def query(sql: str, params: tuple = ()) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------- activity
def log_activity(username: str, action: str, detail: str | dict = "") -> None:
    if isinstance(detail, dict):
        detail = json.dumps(detail, default=str)
    try:
        execute(
            "INSERT INTO activity(username, action, detail, created_at) VALUES (?,?,?,?)",
            (username or "guest", action, str(detail)[:2000], now_iso()),
        )
    except sqlite3.Error:
        pass  # activity logging must never break the app


def recent_activity(username: str | None = None, limit: int = 50) -> list[dict]:
    if username:
        return query(
            "SELECT * FROM activity WHERE username=? ORDER BY id DESC LIMIT ?", (username, limit)
        )
    return query("SELECT * FROM activity ORDER BY id DESC LIMIT ?", (limit,))


# ---------------------------------------------------------------- feedback
def add_feedback(username: str, rating: int, category: str, message: str) -> None:
    execute(
        "INSERT INTO feedback(username, rating, category, message, created_at) VALUES (?,?,?,?,?)",
        (username, int(rating), category, message, now_iso()),
    )


def add_issue(username: str, severity: str, page: str, message: str) -> None:
    execute(
        "INSERT INTO issues(username, severity, page, message, created_at) VALUES (?,?,?,?,?)",
        (username, severity, page, message, now_iso()),
    )


init_db()
