"""User authentication.

* Passwords are hashed with PBKDF2-HMAC-SHA256 (200k iterations, random salt)
  from the Python standard library — no external crypto dependency.
* bcrypt hashes (``$2b$...``) imported from other systems are also accepted
  if ``passlib`` is installed.
* Backend: SQLite by default, MongoDB when ``DATANOVA_MONGO_URI`` is set.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets

from . import db

ITERATIONS = 200_000
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")


# ---------------------------------------------------------------- hashing
def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, ITERATIONS)
    return "pbkdf2_sha256${}${}${}".format(
        ITERATIONS, base64.b64encode(salt).decode(), base64.b64encode(dk).decode()
    )


def verify_password(password: str, stored: str) -> bool:
    if not stored:
        return False
    if stored.startswith("pbkdf2_sha256$"):
        try:
            _, iters, salt_b64, dk_b64 = stored.split("$")
            dk = hashlib.pbkdf2_hmac(
                "sha256", password.encode(), base64.b64decode(salt_b64), int(iters)
            )
            return hmac.compare_digest(dk, base64.b64decode(dk_b64))
        except (ValueError, TypeError):
            return False
    if stored.startswith("$2"):  # bcrypt hash (needs passlib)
        try:
            from passlib.hash import bcrypt  # type: ignore

            return bcrypt.verify(password, stored)
        except Exception:
            return False
    return False


def password_strength(password: str) -> tuple[int, str]:
    """Return a 0-4 score and a label."""
    score = 0
    if len(password) >= 8:
        score += 1
    if re.search(r"[A-Z]", password) and re.search(r"[a-z]", password):
        score += 1
    if re.search(r"\d", password):
        score += 1
    if re.search(r"[^A-Za-z0-9]", password):
        score += 1
    return score, ["Very weak", "Weak", "Fair", "Good", "Strong"][score]


# ---------------------------------------------------------------- backends
class _SQLiteUsers:
    def get(self, username: str) -> dict | None:
        rows = db.query("SELECT * FROM users WHERE username=?", (username,))
        return rows[0] if rows else None

    def create(self, doc: dict) -> None:
        db.execute(
            "INSERT INTO users(username, password_hash, email, full_name, role, created_at, preferences)"
            " VALUES (?,?,?,?,?,?,?)",
            (
                doc["username"], doc["password_hash"], doc.get("email", ""),
                doc.get("full_name", ""), doc.get("role", "user"), db.now_iso(), "{}",
            ),
        )

    def update(self, username: str, fields: dict) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        db.execute(f"UPDATE users SET {cols} WHERE username=?", (*fields.values(), username))

    def count(self) -> int:
        return db.query("SELECT COUNT(*) AS n FROM users")[0]["n"]

    def all(self) -> list[dict]:
        return db.query(
            "SELECT username, email, full_name, role, created_at, last_login FROM users ORDER BY created_at"
        )


class _MongoUsers:
    def __init__(self, uri: str):
        from pymongo import MongoClient  # type: ignore

        self.col = MongoClient(uri, serverSelectionTimeoutMS=3000)["DataNova"]["Users"]
        self.col.database.client.admin.command("ping")

    def get(self, username):
        doc = self.col.find_one({"username": username}, {"_id": 0})
        return doc

    def create(self, doc):
        doc = {**doc, "created_at": db.now_iso(), "preferences": "{}"}
        self.col.insert_one(doc)

    def update(self, username, fields):
        self.col.update_one({"username": username}, {"$set": fields})

    def count(self):
        return self.col.count_documents({})

    def all(self):
        return list(self.col.find({}, {"_id": 0, "password_hash": 0}))


def _make_backend():
    uri = os.environ.get("DATANOVA_MONGO_URI")
    if uri:
        try:
            return _MongoUsers(uri), "MongoDB"
        except Exception:
            pass
    return _SQLiteUsers(), "SQLite"


USERS, BACKEND_NAME = _make_backend()


# ---------------------------------------------------------------- API
def register_user(username: str, password: str, email: str = "", full_name: str = "") -> tuple[bool, str]:
    username = (username or "").strip()
    if not USERNAME_RE.match(username):
        return False, "Username must be 3-32 characters: letters, digits, _ . -"
    if len(password or "") < 6:
        return False, "Password must be at least 6 characters."
    if email and not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        return False, "Email address looks invalid."
    if USERS.get(username):
        return False, "That username is already taken."
    # Admins: usernames listed in DATANOVA_ADMINS (comma-separated, e.g. set in Streamlit secrets).
    # If that is not set, the first account ever created becomes the admin.
    admins = [a.strip().lower() for a in os.environ.get("DATANOVA_ADMINS", "").split(",") if a.strip()]
    if admins:
        role = "admin" if username.lower() in admins else "user"
    else:
        role = "admin" if USERS.count() == 0 else "user"
    USERS.create(
        {"username": username, "password_hash": hash_password(password),
         "email": email, "full_name": full_name, "role": role}
    )
    db.log_activity(username, "register", {"role": role})
    return True, "Account created" + (" (you are the admin)." if role == "admin" else ".")


def authenticate_user(username: str, password: str) -> dict | None:
    user = USERS.get((username or "").strip())
    if not user or not verify_password(password or "", user.get("password_hash", "")):
        return None
    USERS.update(user["username"], {"last_login": db.now_iso()})
    db.log_activity(user["username"], "login")
    user = dict(user)
    user.pop("password_hash", None)
    return user


def change_password(username: str, old: str, new: str) -> tuple[bool, str]:
    user = USERS.get(username)
    if not user or not verify_password(old, user["password_hash"]):
        return False, "Current password is incorrect."
    if len(new) < 6:
        return False, "New password must be at least 6 characters."
    USERS.update(username, {"password_hash": hash_password(new)})
    db.log_activity(username, "change_password")
    return True, "Password updated."


def get_preferences(username: str) -> dict:
    user = USERS.get(username) or {}
    try:
        return json.loads(user.get("preferences") or "{}")
    except (TypeError, ValueError):
        return {}


def save_preferences(username: str, prefs: dict) -> None:
    if USERS.get(username):
        USERS.update(username, {"preferences": json.dumps(prefs, default=str)})


def list_users() -> list[dict]:
    return USERS.all()
