"""Автоматические тесты ядра: ребалансировка весов треков, SQLite-база и Demo-симулятор."""
from pathlib import Path
import tempfile
import time
import unittest

from core.capsule_bridge import CapsuleEventCallbacks, DemoCapsuleController
from core.emotion_engine import aggregate_track_history, summarize_track_segment
from core.models import CalibrationProfile, MetricSnapshot, TrackListenRecord
from core.track_db import TrackDatabase


class TestEmotionTrackerCore(unittest.TestCase):
    def test_rebalancing_relax_then_three_stress_sessions(self) -> None:
        """Сценарий пользователя: 1-я сессия — расслабление, следующие 3 — стресс."""
        base_ts = 1_700_000_000.0
        records = [
            TrackListenRecord(
                session_id="s1",
                track_key="artist — track a",
                artist="Artist",
                title="Track A",
                source_app="Spotify",
                start_time=base_ts,
                end_time=base_ts + 180,
                duration_sec=180.0,
                quality_score=0.95,
                scores={"relaxation": 0.65, "focus": 0.15, "involvement": 0.10, "stress": 0.05, "fatigue": 0.05},
                dominant_emotion="relaxation",
                avg_heart_rate=64.0,
            ),
            TrackListenRecord(
                session_id="s2",
                track_key="artist — track a",
                artist="Artist",
                title="Track A",
                source_app="Spotify",
                start_time=base_ts + 86400,
                end_time=base_ts + 86580,
                duration_sec=180.0,
                quality_score=0.90,
                scores={"relaxation": 0.10, "focus": 0.15, "involvement": 0.15, "stress": 0.55, "fatigue": 0.05},
                dominant_emotion="stress",
                avg_heart_rate=82.0,
            ),
            TrackListenRecord(
                session_id="s3",
                track_key="artist — track a",
                artist="Artist",
                title="Track A",
                source_app="Spotify",
                start_time=base_ts + 172800,
                end_time=base_ts + 172980,
                duration_sec=180.0,
                quality_score=0.92,
                scores={"relaxation": 0.08, "focus": 0.12, "involvement": 0.15, "stress": 0.60, "fatigue": 0.05},
                dominant_emotion="stress",
                avg_heart_rate=85.0,
            ),
            TrackListenRecord(
                session_id="s4",
                track_key="artist — track a",
                artist="Artist",
                title="Track A",
                source_app="Spotify",
                start_time=base_ts + 259200,
                end_time=base_ts + 259380,
                duration_sec=180.0,
                quality_score=0.95,
                scores={"relaxation": 0.05, "focus": 0.15, "involvement": 0.10, "stress": 0.65, "fatigue": 0.05},
                dominant_emotion="stress",
                avg_heart_rate=88.0,
            ),
        ]

        profile = aggregate_track_history(records)
        self.assertEqual(profile.dominant_emotion, "stress")
        self.assertGreater(profile.weighted_scores["stress"], 0.45)
        self.assertGreater(profile.confidence, 0.75)
        self.assertIn("Смена реакции", profile.trend_note)

    def test_sqlite_persistence_and_calibration_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = TrackDatabase(Path(tmp) / "test.db")
            try:
                cal = CalibrationProfile(
                    profile_name="roman_calib",
                    created_at="2026-11-25T20:00:00",
                    individual_peak_frequency=10.4,
                    prod_concentration=1.2,
                    prod_relaxation=2.5,
                    phy_alpha=0.31,
                    phy_beta=0.49,
                )
                db.save_calibration_profile(cal)
                loaded = db.load_latest_calibration()
                self.assertIsNotNone(loaded)
                self.assertAlmostEqual(loaded.individual_peak_frequency, 10.4)
                self.assertTrue(loaded.has_productivity_baselines())
                self.assertTrue(loaded.has_phy_baselines())

                snaps = [
                    MetricSnapshot(
                        timestamp=1000.0 + i,
                        attention=25.0,
                        relaxation_emo=82.0,
                        cognitive_load=15.0,
                        self_control=80.0,
                        prod_relaxation=3.4,
                        prod_concentration=0.6,
                        prod_fatigue=1.2,
                        prod_stress_enum=0,
                        heart_rate=62.0,
                        stress_index=85.0,
                        max_resistance_ohm=220_000.0,
                    )
                    for i in range(30)
                ]
                rec = summarize_track_segment(
                    session_id="sess_1",
                    track_key="ambient — rain",
                    artist="Ambient",
                    title="Rain",
                    source_app="Spotify",
                    start_time=1000.0,
                    end_time=1030.0,
                    snapshots=snaps,
                )
                db.add_listen(rec)
                profiles = db.get_all_track_profiles(emotion_filter="relaxation")
                self.assertEqual(len(profiles), 1)
                self.assertEqual(profiles[0].dominant_emotion, "relaxation")
            finally:
                db.close()

    def test_demo_controller_emits_snapshots(self) -> None:
        received = []
        cb = CapsuleEventCallbacks()
        cb.on_snapshot = lambda s: received.append(s)
        demo = DemoCapsuleController(cb)
        demo.connect_device("Demo", bipolar=True)
        demo._last_metric_t = 0.0
        demo.update()
        self.assertGreaterEqual(len(received), 1)
        demo.stop_and_disconnect()


if __name__ == "__main__":
    unittest.main()
