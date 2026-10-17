"""Shareable report links.

A shared report is a frozen HTML snapshot saved under ``data/shares``. The
app serves it at ``?share=<id>`` (see app.py). Links can be public or
password-protected and can expire.
"""
from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone

from . import auth, db

SHARE_DIR = db.DATA_DIR / "shares"
SHARE_DIR.mkdir(parents=True, exist_ok=True)
EXPIRY = {"1 day": 1, "1 week": 7, "1 month": 30, "Never": None}


def create_share(html: str, owner: str, title: str, visibility: str = "Public", password: str = "",
                 expiry: str = "1 week") -> str:
    share_id = secrets.token_urlsafe(8)
    (SHARE_DIR / f"{share_id}.html").write_text(html, encoding="utf-8")
    days = EXPIRY.get(expiry)
    exp = (datetime.now(timezone.utc) + timedelta(days=days)).replace(microsecond=0).isoformat() if days else None
    db.execute(
        "INSERT INTO shares(share_id, owner, title, visibility, password_hash, expires_at, created_at) VALUES (?,?,?,?,?,?,?)",
        (share_id, owner, title, visibility, auth.hash_password(password) if password else None, exp, db.now_iso()),
    )
    db.log_activity(owner, "share_report", {"share_id": share_id, "visibility": visibility, "expiry": expiry})
    return share_id


def get_share(share_id: str) -> dict | None:
    rows = db.query("SELECT * FROM shares WHERE share_id=?", (share_id,))
    if not rows:
        return None
    row = rows[0]
    if row["expires_at"] and datetime.fromisoformat(row["expires_at"]) < datetime.now(timezone.utc):
        row["expired"] = True
        return row
    path = SHARE_DIR / f"{share_id}.html"
    row["expired"] = False
    row["html"] = path.read_text(encoding="utf-8") if path.exists() else None
    return row


def check_password(row: dict, password: str) -> bool:
    return not row.get("password_hash") or auth.verify_password(password, row["password_hash"])


def register_view(share_id: str) -> None:
    db.execute("UPDATE shares SET views = views + 1 WHERE share_id=?", (share_id,))


def list_shares(owner: str) -> list[dict]:
    return db.query("SELECT share_id, title, visibility, expires_at, views, created_at FROM shares WHERE owner=? "
                    "ORDER BY created_at DESC", (owner,))


def delete_share(share_id: str, owner: str) -> None:
    db.execute("DELETE FROM shares WHERE share_id=? AND owner=?", (share_id, owner))
    (SHARE_DIR / f"{share_id}.html").unlink(missing_ok=True)
