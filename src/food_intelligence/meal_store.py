"""SQLite persistence for the food log (database/meal_history.db).

Opt-in: the UI saves/loads explicitly, so session-state behaviour and tests are unchanged.
Single-user local file; not secure multi-user storage.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from src.food_intelligence.food_log import FoodLog

DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "database" / "meal_history.db"

_SCHEMA = """CREATE TABLE IF NOT EXISTS meal_entries (
    entry_id TEXT PRIMARY KEY, log_date TEXT NOT NULL, meal TEXT NOT NULL, food_id TEXT NOT NULL,
    portion_g REAL NOT NULL, source TEXT NOT NULL, user_b12_ug_per_100g REAL)"""
_COLUMNS = ("entry_id", "log_date", "meal", "food_id", "portion_g", "source", "user_b12_ug_per_100g")


def _connect(path: Path) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(_SCHEMA)
    return conn


def save_log(log: FoodLog, path: Path = DEFAULT_DB_PATH) -> int:
    """Upsert every entry in ``log`` (idempotent on entry_id). Returns the number written."""
    records = log.to_records()
    with _connect(path) as conn:
        conn.executemany(
            f"INSERT OR REPLACE INTO meal_entries ({','.join(_COLUMNS)}) VALUES ({','.join('?' * len(_COLUMNS))})",
            [tuple(r[c] for c in _COLUMNS) for r in records])
    return len(records)


def load_records(path: Path = DEFAULT_DB_PATH) -> list[dict]:
    if not Path(path).exists():
        return []
    with _connect(path) as conn:
        rows = conn.execute(f"SELECT {','.join(_COLUMNS)} FROM meal_entries ORDER BY log_date, rowid").fetchall()
    return [dict(zip(_COLUMNS, row)) for row in rows]


def delete_entry(entry_id: str, path: Path = DEFAULT_DB_PATH) -> None:
    with _connect(path) as conn:
        conn.execute("DELETE FROM meal_entries WHERE entry_id = ?", (entry_id,))
