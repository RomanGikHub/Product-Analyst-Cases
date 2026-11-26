"""SQLite-хранилище истории прослушиваний треков, сессий и профилей калибровки."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import List, Optional

from core.emotion_engine import aggregate_track_history
from core.models import CalibrationProfile, TrackAggregateProfile, TrackListenRecord


class TrackDatabase:
    """Управляет постоянной базой данных треков и профилей калибровки."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                started_at REAL NOT NULL,
                ended_at REAL,
                device_name TEXT,
                bipolar INTEGER DEFAULT 1,
                hdf5_path TEXT
            );

            CREATE TABLE IF NOT EXISTS track_listens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                track_key TEXT NOT NULL,
                artist TEXT NOT NULL,
                title TEXT NOT NULL,
                source_app TEXT,
                start_time REAL NOT NULL,
                end_time REAL NOT NULL,
                duration_sec REAL NOT NULL,
                quality_score REAL NOT NULL,
                scores_json TEXT NOT NULL,
                dominant_emotion TEXT NOT NULL,
                avg_heart_rate REAL,
                avg_stress_index REAL,
                peak_emotion TEXT,
                peak_offset_sec REAL DEFAULT 0.0
            );

            CREATE INDEX IF NOT EXISTS idx_track_listens_key ON track_listens(track_key);
            CREATE INDEX IF NOT EXISTS idx_track_listens_session ON track_listens(session_id);

            CREATE TABLE IF NOT EXISTS calibration_profiles (
                profile_name TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            """
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # --- Сессии ---
    def register_session(
        self,
        session_id: str,
        device_name: str,
        bipolar: bool,
        hdf5_path: str,
    ) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO sessions (session_id, started_at, device_name, bipolar, hdf5_path)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, time.time(), device_name, int(bipolar), hdf5_path),
        )
        self._conn.commit()

    def finish_session(self, session_id: str) -> None:
        self._conn.execute(
            "UPDATE sessions SET ended_at = ? WHERE session_id = ?",
            (time.time(), session_id),
        )
        self._conn.commit()

    # --- Прослушивания треков ---
    def add_listen(self, rec: TrackListenRecord) -> int:
        cur = self._conn.execute(
            """
            INSERT INTO track_listens (
                session_id, track_key, artist, title, source_app,
                start_time, end_time, duration_sec, quality_score,
                scores_json, dominant_emotion, avg_heart_rate,
                avg_stress_index, peak_emotion, peak_offset_sec
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rec.session_id,
                rec.track_key,
                rec.artist,
                rec.title,
                rec.source_app,
                rec.start_time,
                rec.end_time,
                rec.duration_sec,
                rec.quality_score,
                json.dumps(rec.scores, ensure_ascii=False),
                rec.dominant_emotion,
                rec.avg_heart_rate,
                rec.avg_stress_index,
                rec.peak_emotion,
                rec.peak_offset_sec,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def _row_to_record(self, row: sqlite3.Row) -> TrackListenRecord:
        return TrackListenRecord(
            listen_id=row["id"],
            session_id=row["session_id"],
            track_key=row["track_key"],
            artist=row["artist"],
            title=row["title"],
            source_app=row["source_app"] or "",
            start_time=row["start_time"],
            end_time=row["end_time"],
            duration_sec=row["duration_sec"],
            quality_score=row["quality_score"],
            scores=json.loads(row["scores_json"]),
            dominant_emotion=row["dominant_emotion"],
            avg_heart_rate=row["avg_heart_rate"],
            avg_stress_index=row["avg_stress_index"],
            peak_emotion=row["peak_emotion"] or "neutral",
            peak_offset_sec=row["peak_offset_sec"] or 0.0,
        )

    def get_track_history(self, track_key: str) -> List[TrackListenRecord]:
        rows = self._conn.execute(
            "SELECT * FROM track_listens WHERE track_key = ? ORDER BY start_time ASC",
            (track_key,),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_session_listens(self, session_id: str) -> List[TrackListenRecord]:
        rows = self._conn.execute(
            "SELECT * FROM track_listens WHERE session_id = ? ORDER BY start_time ASC",
            (session_id,),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_all_track_profiles(self, emotion_filter: Optional[str] = None) -> List[TrackAggregateProfile]:
        keys = [
            r["track_key"]
            for r in self._conn.execute(
                "SELECT DISTINCT track_key FROM track_listens"
            ).fetchall()
        ]
        profiles: List[TrackAggregateProfile] = []
        for k in keys:
            history = self.get_track_history(k)
            if history:
                prof = aggregate_track_history(history)
                if emotion_filter and emotion_filter != "all" and prof.dominant_emotion != emotion_filter:
                    continue
                profiles.append(prof)
        profiles.sort(key=lambda p: (p.last_listened_ts, p.listen_count), reverse=True)
        return profiles

    # --- Профили калибровки ---
    def save_calibration_profile(self, profile: CalibrationProfile) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO calibration_profiles (profile_name, created_at, payload_json)
            VALUES (?, ?, ?)
            """,
            (
                profile.profile_name,
                profile.created_at,
                json.dumps(profile.to_dict(), ensure_ascii=False),
            ),
        )
        self._conn.commit()

    def load_latest_calibration(self) -> Optional[CalibrationProfile]:
        row = self._conn.execute(
            "SELECT payload_json FROM calibration_profiles ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        if not row:
            return None
        data = json.loads(row["payload_json"])
        return CalibrationProfile(**data)

    def list_calibration_profiles(self) -> List[CalibrationProfile]:
        rows = self._conn.execute(
            "SELECT payload_json FROM calibration_profiles ORDER BY created_at DESC"
        ).fetchall()
        return [CalibrationProfile(**json.loads(r["payload_json"])) for r in rows]
