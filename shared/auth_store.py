from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from typing import Any

import mongomock
from pymongo import ASCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError

from shared.utils import utc_now

_CLIENT: MongoClient | mongomock.MongoClient | None = None
_DB: Database | None = None


def _get_mongo_settings() -> tuple[str, str]:
    uri = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
    db_name = os.environ.get("MONGODB_DB_NAME", "ensaj_client_portal")
    return uri, db_name


def reset_auth_store_cache() -> None:
    global _CLIENT, _DB
    if _CLIENT is not None and hasattr(_CLIENT, "close"):
        _CLIENT.close()
    _CLIENT = None
    _DB = None


def get_auth_db() -> Database:
    global _CLIENT, _DB
    if _DB is not None:
        return _DB

    uri, db_name = _get_mongo_settings()
    if uri.startswith("mongomock://"):
        _CLIENT = mongomock.MongoClient()
    else:
        _CLIENT = MongoClient(uri, serverSelectionTimeoutMS=3000)
        _CLIENT.admin.command("ping")
    _DB = _CLIENT[db_name]
    _ensure_indexes(_DB)
    return _DB


def _ensure_indexes(db: Database) -> None:
    db["users"].create_index([("username", ASCENDING)], unique=True)
    db["user_runs"].create_index([("username", ASCENDING), ("run_id", ASCENDING)], unique=True)
    db["case_history"].create_index([("username", ASCENDING), ("case_id", ASCENDING)], unique=True)
    db["case_history"].create_index([("username", ASCENDING), ("timestamp", ASCENDING)])


def _hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    password_salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), password_salt.encode("utf-8"), 120000)
    return digest.hex(), password_salt


def _verify_password(password: str, expected_hash: str, salt: str) -> bool:
    digest, _ = _hash_password(password, salt=salt)
    return hmac.compare_digest(digest, expected_hash)


def get_users_collection() -> Collection:
    return get_auth_db()["users"]


def get_user_runs_collection() -> Collection:
    return get_auth_db()["user_runs"]


def get_case_history_collection() -> Collection:
    return get_auth_db()["case_history"]


def create_client_user(username: str, password: str) -> tuple[bool, str]:
    normalized = username.strip().lower()
    if not normalized:
        return False, "Username requis."
    if len(password) < 8:
        return False, "Le mot de passe doit contenir au moins 8 caracteres."

    password_hash, password_salt = _hash_password(password)
    document = {
        "username": normalized,
        "password_hash": password_hash,
        "password_salt": password_salt,
        "role": "client",
        "created_at": utc_now(),
        "is_active": True,
        "last_login_at": None,
    }
    try:
        get_users_collection().insert_one(document)
    except DuplicateKeyError:
        return False, "Username deja utilise."
    return True, "Compte client cree avec succes."


def authenticate_client_user(username: str, password: str) -> tuple[bool, dict[str, Any] | None, str]:
    normalized = username.strip().lower()
    user = get_users_collection().find_one({"username": normalized})
    if user is None:
        return False, None, "Utilisateur introuvable."
    if user.get("role") != "client":
        return False, None, "Seuls les comptes client sont autorises ici."
    if not user.get("is_active", True):
        return False, None, "Compte desactive."
    if not _verify_password(password, str(user["password_hash"]), str(user["password_salt"])):
        return False, None, "Mot de passe incorrect."

    get_users_collection().update_one(
        {"_id": user["_id"]},
        {"$set": {"last_login_at": utc_now()}},
    )
    return True, {"username": user["username"], "role": user["role"]}, "Connexion reussie."


def list_user_runs(username: str) -> list[dict[str, Any]]:
    rows = list(
        get_user_runs_collection()
        .find({"username": username}, {"_id": 0})
        .sort("created_at", -1)
    )
    return rows


def upsert_user_run(username: str, runtime_config: dict[str, Any], status: str, stage: str) -> None:
    payload = {
        "username": username.strip().lower(),
        "run_id": runtime_config["run_id"],
        "run_mode": runtime_config["run_mode"],
        "dataset_name": runtime_config["dataset_name"],
        "artifact_root": runtime_config["artifact_root"],
        "status": status,
        "stage": stage,
        "created_at": runtime_config["timestamp_utc"],
        "updated_at": utc_now(),
    }
    get_user_runs_collection().update_one(
        {"username": payload["username"], "run_id": payload["run_id"]},
        {"$set": payload},
        upsert=True,
    )


def save_case_history(username: str, case_payload: dict[str, Any]) -> None:
    normalized = username.strip().lower()
    payload = {
        **case_payload,
        "username": normalized,
        "updated_at": utc_now(),
    }
    get_case_history_collection().update_one(
        {"username": normalized, "case_id": payload["case_id"]},
        {"$set": payload},
        upsert=True,
    )


def list_case_history(username: str, limit: int = 50) -> list[dict[str, Any]]:
    rows = list(
        get_case_history_collection()
        .find({"username": username.strip().lower()}, {"_id": 0})
        .sort("timestamp", -1)
        .limit(int(limit))
    )
    return rows


def list_all_case_history(limit: int = 200) -> list[dict[str, Any]]:
    rows = list(
        get_case_history_collection()
        .find({}, {"_id": 0})
        .sort("timestamp", -1)
        .limit(int(limit))
    )
    return rows
