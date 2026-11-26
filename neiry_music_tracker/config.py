"""Конфигурация приложения Neiry Music Emotion Tracker."""
from pathlib import Path

APP_NAME = "Neiry Music Emotion Tracker"
APP_VERSION = "1.1.0"

# Базовые директории (рядом с app.py)
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SESSIONS_DIR = DATA_DIR / "sessions"
LOGS_DIR = DATA_DIR / "logs"
DB_PATH = DATA_DIR / "music_emotions.db"

# Поиск папки CapsuleClientPython и нативной библиотеки
DEFAULT_CAPSULE_PY_DIR = BASE_DIR / "CapsuleClientPython"
DEFAULT_CAPSULE_DLL_PATH = BASE_DIR / "CapsuleClient.dll"

# Цикл обновления Capsule API (в мс, рекомендуется 40–50 мс)
CAPSULE_UPDATE_INTERVAL_MS = 50

# Частота перерисовки живых графиков (в мс, ~3 Гц — экономит CPU без потери плавности)
CHART_UPDATE_INTERVAL_MS = 333

# Опрос системного медиаплеера Windows SMTC (в мс)
MEDIA_POLL_INTERVAL_MS = 1500

# Параметры фильтрации и анализа треков
MIN_TRACK_LISTEN_SEC = 20.0          # Минимальная длительность прослушивания для учёта в статистике
MAX_GOOD_RESISTANCE_OHM = 1_000_000  # 1000 кОм = 1 000 000 Ом (порог хорошего контакта)
MEMS_MOTION_THRESHOLD_G = 1.8        # Порог резкого движения головы (отклонение модуля ускорения)

# Параметры адаптивной ребалансировки весов между сессиями
RECENCY_DECAY_LAMBDA = 0.78          # Экспоненциальное затухание веса более старых прослушиваний
MIN_LISTENS_HIGH_CONFIDENCE = 4      # От скольких прослушиваний уверенность может считаться высокой
EMA_SMOOTHING_ALPHA = 0.25           # Сглаживание метрик на живом графике

# Категории итоговых эмоциональных триггеров
EMOTION_LABELS_RU = {
    "relaxation": "Расслабление",
    "focus": "Фокус и концентрация",
    "involvement": "Вовлечённость / драйв",
    "stress": "Стресс / напряжение",
    "fatigue": "Усталость / спад",
    "neutral": "Нейтральный фон",
}

EMOTION_COLORS = {
    "relaxation": "#22c55e",
    "focus": "#3b82f6",
    "involvement": "#a855f7",
    "stress": "#ef4444",
    "fatigue": "#f59e0b",
    "neutral": "#94a3b8",
}


def ensure_directories() -> None:
    """Создаёт рабочие директории при запуске."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
