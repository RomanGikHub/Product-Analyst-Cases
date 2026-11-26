"""Движок оценки эмоций по срезам Capsule API и адаптивной ребалансировки весов треков."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import (
    EMOTION_LABELS_RU,
    MAX_GOOD_RESISTANCE_OHM,
    MEMS_MOTION_THRESHOLD_G,
    MIN_LISTENS_HIGH_CONFIDENCE,
    RECENCY_DECAY_LAMBDA,
)
from core.models import (
    MetricSnapshot,
    TrackAggregateProfile,
    TrackListenRecord,
)

CORE_EMOTIONS = ("relaxation", "focus", "involvement", "stress", "fatigue")


def _clamp01(val: float) -> float:
    return max(0.0, min(1.0, float(val)))


def _sigmoid_norm(val: float, center: float, scale: float) -> float:
    """Мягкая нормализация произвольной величины в диапазон [0, 1]."""
    z = (val - center) / max(scale, 1e-6)
    z = max(-10.0, min(10.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def compute_snapshot_quality(snap: MetricSnapshot) -> float:
    """Оценивает качество одного среза сигнала в диапазоне [0.1, 1.0]."""
    q = 1.0
    if snap.max_resistance_ohm is not None and snap.max_resistance_ohm > MAX_GOOD_RESISTANCE_OHM:
        # Плохой контакт электродов снижает вес, но не обнуляет полностью если артефакты низкие
        ratio = snap.max_resistance_ohm / MAX_GOOD_RESISTANCE_OHM
        q *= max(0.2, 1.0 / ratio)
    if snap.eeg_artifact_ratio > 0:
        q *= max(0.2, 1.0 - 0.8 * _clamp01(snap.eeg_artifact_ratio))
    if snap.cardio_has_artifacts:
        q *= 0.85
    if snap.mems_motion_g > MEMS_MOTION_THRESHOLD_G:
        q *= 0.65
    return max(0.1, min(1.0, q))


def snapshot_to_emotion_vector(
    snap: MetricSnapshot,
    session_baseline: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Преобразует срез Capsule API (Emotions + Productivity + PhysiologicalStates + Cardio)
    в единый нормализованный вектор из 5 состояний:
      relaxation, focus, involvement, stress, fatigue.
    """
    base = session_baseline or {}
    buckets: Dict[str, List[Tuple[float, float]]] = {k: [] for k in CORE_EMOTIONS}

    # 1. Emotions_States (быстрые метрики, доступны сразу без калибровки)
    if snap.relaxation_emo is not None:
        buckets["relaxation"].append((_clamp01(snap.relaxation_emo / 100.0 if snap.relaxation_emo > 1.5 else snap.relaxation_emo), 1.0))
    if snap.attention is not None:
        att = _clamp01(snap.attention / 100.0 if snap.attention > 1.5 else snap.attention)
        buckets["focus"].append((att, 1.0))
        buckets["involvement"].append((att, 0.5))
    if snap.cognitive_load is not None:
        cl = _clamp01(snap.cognitive_load / 100.0 if snap.cognitive_load > 1.5 else snap.cognitive_load)
        buckets["stress"].append((cl, 0.7))
        buckets["involvement"].append((cl, 0.4))
    if snap.self_control is not None:
        sc = _clamp01(snap.self_control / 100.0 if snap.self_control > 1.5 else snap.self_control)
        buckets["focus"].append((sc, 0.5))
        buckets["stress"].append((1.0 - sc, 0.4))

    # 2. Productivity_Metrics (относительно бейслайна сессии или шкалы)
    if snap.prod_relaxation is not None:
        ref = base.get("prod_relaxation", 2.0)
        buckets["relaxation"].append((_sigmoid_norm(snap.prod_relaxation, ref, max(abs(ref) * 0.5, 0.5)), 1.3))
    if snap.prod_concentration is not None:
        ref = base.get("prod_concentration", 1.0)
        buckets["focus"].append((_sigmoid_norm(snap.prod_concentration, ref, max(abs(ref) * 0.5, 0.5)), 1.3))
    if snap.prod_fatigue is not None:
        ref = base.get("prod_fatigue", 2.0)
        buckets["fatigue"].append((_sigmoid_norm(snap.prod_fatigue, ref, max(abs(ref) * 0.5, 0.5)), 1.4))
    if snap.prod_gravity is not None:
        ref = base.get("prod_gravity", 0.6)
        buckets["involvement"].append((_sigmoid_norm(snap.prod_gravity, ref, max(abs(ref) * 0.5, 0.3)), 0.9))
    if snap.prod_stress_enum is not None:
        # 0=NoStress, 1=Anxiety, 2=Stress
        stress_enum_score = {0: 0.15, 1: 0.65, 2: 0.95}.get(int(snap.prod_stress_enum), 0.3)
        buckets["stress"].append((stress_enum_score, 1.4))

    # 3. PhysiologicalStates_Value (редкие, но высокоуровневые вероятности — даём высокий вес)
    phy_map = [
        ("relaxation", snap.phy_relaxation),
        ("fatigue", snap.phy_fatigue),
        ("focus", snap.phy_concentration),
        ("involvement", snap.phy_involvement),
        ("stress", snap.phy_stress),
    ]
    for key, val in phy_map:
        if val is not None:
            v = val / 100.0 if val > 1.5 else val
            buckets[key].append((_clamp01(v), 2.0))

    # 4. Cardio_Data (индекс напряжения Баевского и ЧСС)
    if snap.stress_index is not None and not snap.cardio_has_artifacts:
        ref_si = base.get("stress_index", 150.0)
        si_norm = _sigmoid_norm(snap.stress_index, ref_si, max(ref_si * 0.4, 60.0))
        buckets["stress"].append((si_norm, 1.2))
        buckets["relaxation"].append((1.0 - si_norm, 0.6))

    raw_vec: Dict[str, float] = {}
    for k in CORE_EMOTIONS:
        if buckets[k]:
            num = sum(v * w for v, w in buckets[k])
            den = sum(w for _, w in buckets[k])
            raw_vec[k] = num / den
        else:
            raw_vec[k] = 0.0

    total = sum(raw_vec.values())
    if total <= 1e-6:
        return {k: 1.0 / len(CORE_EMOTIONS) for k in CORE_EMOTIONS}
    return {k: raw_vec[k] / total for k in CORE_EMOTIONS}


def summarize_track_segment(
    session_id: str,
    track_key: str,
    artist: str,
    title: str,
    source_app: str,
    start_time: float,
    end_time: float,
    snapshots: List[MetricSnapshot],
    session_baseline: Optional[Dict[str, float]] = None,
) -> TrackListenRecord:
    """Агрегирует все срезы за время одного прослушивания трека в TrackListenRecord."""
    duration = max(0.0, end_time - start_time)
    if not snapshots:
        empty_scores = {k: 0.2 for k in CORE_EMOTIONS}
        return TrackListenRecord(
            session_id=session_id,
            track_key=track_key,
            artist=artist,
            title=title,
            source_app=source_app,
            start_time=start_time,
            end_time=end_time,
            duration_sec=duration,
            quality_score=0.0,
            scores=empty_scores,
            dominant_emotion="neutral",
        )

    weighted_sum = {k: 0.0 for k in CORE_EMOTIONS}
    total_q = 0.0
    hrs: List[float] = []
    sis: List[float] = []

    peak_val = -1.0
    peak_emo = "neutral"
    peak_offset = 0.0

    for snap in snapshots:
        q = compute_snapshot_quality(snap)
        vec = snapshot_to_emotion_vector(snap, session_baseline)
        for k in CORE_EMOTIONS:
            weighted_sum[k] += vec[k] * q
            if vec[k] > peak_val:
                peak_val = vec[k]
                peak_emo = k
                peak_offset = max(0.0, snap.timestamp - start_time)
        total_q += q
        if snap.heart_rate is not None and snap.heart_rate > 30 and not snap.cardio_has_artifacts:
            hrs.append(snap.heart_rate)
        if snap.stress_index is not None and not snap.cardio_has_artifacts:
            sis.append(snap.stress_index)

    avg_scores = {k: weighted_sum[k] / max(total_q, 1e-6) for k in CORE_EMOTIONS}
    avg_q = total_q / len(snapshots)

    dominant = max(avg_scores, key=avg_scores.get)
    # Если распределение почти плоское, помечаем как нейтральное
    sorted_vals = sorted(avg_scores.values(), reverse=True)
    if len(sorted_vals) >= 2 and (sorted_vals[0] - sorted_vals[1]) < 0.04:
        dominant = "neutral"

    return TrackListenRecord(
        session_id=session_id,
        track_key=track_key,
        artist=artist,
        title=title,
        source_app=source_app,
        start_time=start_time,
        end_time=end_time,
        duration_sec=duration,
        quality_score=round(avg_q, 3),
        scores={k: round(v, 4) for k, v in avg_scores.items()},
        dominant_emotion=dominant,
        avg_heart_rate=round(float(np.mean(hrs)), 1) if hrs else None,
        avg_stress_index=round(float(np.mean(sis)), 1) if sis else None,
        peak_emotion=peak_emo,
        peak_offset_sec=round(peak_offset, 1),
    )


def aggregate_track_history(
    records: List[TrackListenRecord],
    decay_lambda: float = RECENCY_DECAY_LAMBDA,
) -> TrackAggregateProfile:
    """
    Объединяет все прослушивания трека из разных сессий с адаптивной ребалансировкой весов:
      вес_i = (decay_lambda ** age_rank) * quality_i * duration_factor_i
    где age_rank = 0 для самого свежего прослушивания, 1 для предыдущего и т.д.

    Пример: 1-я сессия — расслабление, следующие 3 — стресс.
    Свежие 3 сессии получают значительно больший суммарный вес, и итоговый профиль
    перебалансируется в сторону стресса с пометкой о смене тренда.
    """
    ordered = sorted(records, key=lambda r: r.start_time)
    n = len(ordered)
    latest = ordered[-1]

    weighted_sum = {k: 0.0 for k in CORE_EMOTIONS}
    total_w = 0.0
    weights: List[float] = []

    for idx, rec in enumerate(ordered):
        age_rank = (n - 1) - idx  # 0 для самого нового
        recency_w = decay_lambda ** age_rank
        duration_w = min(1.0, max(0.3, rec.duration_sec / 60.0))
        quality_w = max(0.15, rec.quality_score)
        w = recency_w * duration_w * quality_w
        weights.append(w)
        total_w += w
        for k in CORE_EMOTIONS:
            weighted_sum[k] += rec.scores.get(k, 0.0) * w

    weighted_scores = {
        k: round(weighted_sum[k] / max(total_w, 1e-6), 4) for k in CORE_EMOTIONS
    }

    sorted_items = sorted(weighted_scores.items(), key=lambda kv: kv[1], reverse=True)
    top_emo, top_val = sorted_items[0]
    second_val = sorted_items[1][1] if len(sorted_items) > 1 else 0.0

    dominant = top_emo if (top_val - second_val) >= 0.04 else "neutral"

    # Расчёт уверенности: зависит от числа прослушиваний, отрыва лидера и согласованности последних сессий
    count_factor = min(1.0, n / float(MIN_LISTENS_HIGH_CONFIDENCE))
    margin_factor = min(1.0, (top_val - second_val) / 0.25)

    # Согласованность последних прослушиваний (до 3)
    recent_slice = ordered[-min(3, n):]
    recent_matches = sum(1 for r in recent_slice if r.dominant_emotion == dominant)
    consistency_factor = recent_matches / len(recent_slice)

    confidence = round(_clamp01(0.45 * count_factor + 0.35 * consistency_factor + 0.20 * margin_factor), 2)

    # Анализ смены восприятия (тренд между ранними и свежими прослушиваниями)
    trend_note = "Стабильное восприятие" if n >= 2 else "Первое прослушивание"
    if n >= 2:
        first_emo = ordered[0].dominant_emotion
        recent_dom = [r.dominant_emotion for r in recent_slice]
        if first_emo != dominant and recent_dom.count(dominant) >= max(2, len(recent_slice) - 1):
            trend_note = (
                f"Смена реакции: раньше — {EMOTION_LABELS_RU.get(first_emo, first_emo).lower()}, "
                f"в последних сессиях ({recent_dom.count(dominant)} из {len(recent_slice)}) — "
                f"{EMOTION_LABELS_RU.get(dominant, dominant).lower()}"
            )
        elif len(set(r.dominant_emotion for r in ordered)) > 1 and consistency_factor < 0.67:
            trend_note = "Смешанная реакция между сессиями — нужно больше прослушиваний"

    label_ru = EMOTION_LABELS_RU.get(dominant, dominant)
    conf_pct = int(confidence * 100)
    if dominant == "neutral":
        rec_text = f"Нейтральное влияние (уверенность {conf_pct}%)"
    else:
        rec_text = f"Скорее всего триггерит: {label_ru} (уверенность {conf_pct}%)"

    hrs = [r.avg_heart_rate for r in ordered if r.avg_heart_rate is not None]
    avg_hr = round(float(np.mean(hrs)), 1) if hrs else None

    return TrackAggregateProfile(
        track_key=latest.track_key,
        artist=latest.artist,
        title=latest.title,
        listen_count=n,
        total_duration_sec=round(sum(r.duration_sec for r in ordered), 1),
        avg_quality=round(float(np.mean([r.quality_score for r in ordered])), 2),
        weighted_scores=weighted_scores,
        dominant_emotion=dominant,
        confidence=confidence,
        recommendation_text=rec_text,
        trend_note=trend_note,
        last_listened_ts=latest.start_time,
        avg_heart_rate=avg_hr,
    )
