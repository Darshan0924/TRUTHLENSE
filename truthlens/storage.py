"""Small SQLite result store. Uploaded originals are never retained."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any
from uuid import UUID


class AnalysisStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / "analyses.sqlite3"
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS analyses ("
                "id TEXT PRIMARY KEY, created_at TEXT NOT NULL, result TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS analyses_created ON analyses(created_at DESC)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database, timeout=10)

    def directory(self, analysis_id: str) -> Path:
        # Normalizing UUIDs keeps every generated path inside this store.
        return self.root / "results" / str(UUID(analysis_id))

    def save(self, result: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO analyses (id, created_at, result) VALUES (?, ?, ?)",
                (result["id"], result["created_at"], json.dumps(result, allow_nan=False)),
            )

    def get(self, analysis_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT result FROM analyses WHERE id = ?", (str(UUID(analysis_id)),)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def recent(self, limit: int = 20) -> dict[str, Any]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT result FROM analyses ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            total = connection.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
        summaries = []
        for row in rows:
            result = json.loads(row[0])
            result.pop("frames", None)
            summaries.append(result)
        return {"analyses": summaries, "total": total}
