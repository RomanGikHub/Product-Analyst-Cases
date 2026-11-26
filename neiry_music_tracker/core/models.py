"""Структуры данных для метрик Capsule API, треков, калибровок и отчётов."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


@dataclass
class TrackInfo:
    """Информация о текущем треке из Windows SMTC (Spotify и др.)."""
    title: str
    artist: str
    album: str = ""
    source_app: str = ""
    is_playing: bool = False
    position_sec: float = 0.0
    duration_sec: float = 0.0

    @property
    def track_key(self) -> str:
        return f"{self.artist.strip().lower()} — {self.title.strip().lower()}"

    @property
    def display_name(self) -> str:
        if self.artist and self.title:
            return f"{self.artist} — {self.title}"
        return self.title or self.artist or "Неизвестный трек"


@dataclass
class MetricSnapshot:
    """Единый срез биосигналов и метрик в момент времени (Unix сек)."""
    timestamp: float
    # Emotions_States (Emotions.py)
    attention: Optional[float] = None
    relaxation_emo: Optional[float] = None
    cognitive_load: Optional[float] = None
    cognitive_control: Optional[float] = None
    self_control: Optional[float] = None
    # Productivity_Metrics (Productivity.py)
    prod_relaxation: Optional[float] = None
    prod_concentration: Optional[float] = None
    prod_fatigue: Optional[float] = None
    prod_reverse_fatigue: Optional[float] = None
    prod_gravity: Optional[float] = None
    prod_score: Optional[float] = None
    prod_alpha: Optional[float] = None
    prod_stress_enum: Optional[int] = None  # 0=NoStress, 1=Anxiety, 2=Stress
    # PhysiologicalStates_Value (PhysiologicalStates.py) — вероятности 0..1 / 0..100
    phy_relaxation: Optional[float] = None
    phy_fatigue: Optional[float] = None
    phy_concentration: Optional[float] = None
    phy_involvement: Optional[float] = None
    phy_stress: Optional[float] = None
    # Cardio_Data (Cardio.py)
    heart_rate: Optional[float] = None
    stress_index: Optional[float] = None
    kaplan_index: Optional[float] = None
    cardio_has_artifacts: bool = False
    # Качество сигнала
    max_resistance_ohm: Optional[float] = None
    eeg_artifact_ratio: float = 0.0
    mems_motion_g: float = 0.0


@dataclass
class CalibrationProfile:
    """Сохранённый профиль калибровки (все 3 этапа Capsule API)."""
    profile_name: str
    created_at: str
    # 1. IndividualNFBData
    individual_frequency: float = 10.0
    individual_peak_frequency: float = 10.0
    individual_peak_frequency_power: float = 10.0
    individual_peak_frequency_suppression: float = 2.0
    individual_bandwidth: float = 6.0
    individual_normalized_power: float = 0.5
    lower_frequency: float = 7.0
    upper_frequency: float = 13.0
    # 2. Productivity_Baselines
    prod_concentration: float = -1.0
    prod_fatigue: float = -1.0
    prod_gravity: float = -1.0
    prod_productivity: float = -1.0
    prod_relaxation: float = -1.0
    prod_reverse_fatigue: float = -1.0
    # 3. PhysiologicalStates_Baselines
    phy_alpha: float = -1.0
    phy_beta: float = -1.0
    phy_alpha_gravity: float = -1.0
    phy_beta_gravity: float = -1.0
    phy_concentration: float = -1.0

    def has_productivity_baselines(self) -> bool:
        return self.prod_concentration >= 0 and self.prod_relaxation >= 0

    def has_phy_baselines(self) -> bool:
        return self.phy_alpha >= 0 and self.phy_beta >= 0

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TrackListenRecord:
    """Результат одного прослушивания трека в рамках конкретной сессии."""
    session_id: str
    track_key: str
    artist: str
    title: str
    source_app: str
    start_time: float
    end_time: float
    duration_sec: float
    quality_score: float  # 0.0 .. 1.0 (учёт импеданса, артефактов ЭЭГ/ФПГ и MEMS)
    # Нормализованный вектор состояний (0..1, сумма ~ 1)
    scores: Dict[str, float] = field(default_factory=dict)
    dominant_emotion: str = "neutral"
    avg_heart_rate: Optional[float] = None
    avg_stress_index: Optional[float] = None
    peak_emotion: str = "neutral"
    peak_offset_sec: float = 0.0
    listen_id: Optional[int] = None


@dataclass
class TrackAggregateProfile:
    """Агрегированный профиль трека с ребалансировкой весов по всем сессиям."""
    track_key: str
    artist: str
    title: str
    listen_count: int
    total_duration_sec: float
    avg_quality: float
    weighted_scores: Dict[str, float]
    dominant_emotion: str
    confidence: float            # 0.0 .. 1.0
    recommendation_text: str     # Человекочитаемый вердикт
    trend_note: str              # Заметка о смене восприятия между сессиями
    last_listened_ts: float
    avg_heart_rate: Optional[float] = None
