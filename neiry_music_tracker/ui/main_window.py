"""Главное окно десктопного приложения (PySide6 + pyqtgraph, оптимизировано по CPU)."""
from __future__ import annotations

from collections import deque
from datetime import datetime
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from config import (
    APP_NAME,
    APP_VERSION,
    CAPSULE_UPDATE_INTERVAL_MS,
    CHART_UPDATE_INTERVAL_MS,
    DB_PATH,
    DEFAULT_CAPSULE_DLL_PATH,
    DEFAULT_CAPSULE_PY_DIR,
    EMA_SMOOTHING_ALPHA,
    EMOTION_COLORS,
    EMOTION_LABELS_RU,
    LOGS_DIR,
    MAX_GOOD_RESISTANCE_OHM,
    MEDIA_POLL_INTERVAL_MS,
    MIN_TRACK_LISTEN_SEC,
    SESSIONS_DIR,
)
from core.capsule_bridge import (
    CapsuleEventCallbacks,
    DemoCapsuleController,
    RealCapsuleController,
)
from core.emotion_engine import (
    CORE_EMOTIONS,
    compute_snapshot_quality,
    snapshot_to_emotion_vector,
    summarize_track_segment,
)
from core.hdf5_store import SessionHDF5Writer, load_session_hdf5
from core.media_monitor import WindowsMediaMonitor
from core.models import CalibrationProfile, MetricSnapshot, TrackInfo, TrackListenRecord
from core.track_db import TrackDatabase

DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #0f172a;
    color: #e2e8f0;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 13px;
}
QTabWidget::pane {
    border: 1px solid #1e293b;
    border-radius: 6px;
    background-color: #0f172a;
}
QTabBar::tab {
    background-color: #1e293b;
    color: #94a3b8;
    padding: 8px 18px;
    margin-right: 4px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    font-weight: 600;
}
QTabBar::tab:selected {
    background-color: #334155;
    color: #f8fafc;
}
QFrame#card {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 10px;
}
QPushButton {
    background-color: #2563eb;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 7px 14px;
    font-weight: 600;
}
QPushButton:hover { background-color: #1d4ed8; }
QPushButton:disabled { background-color: #334155; color: #64748b; }
QPushButton#danger { background-color: #dc2626; }
QPushButton#danger:hover { background-color: #b91c1c; }
QPushButton#secondary { background-color: #334155; }
QPushButton#secondary:hover { background-color: #475569; }
QComboBox, QLineEdit, QTextEdit {
    background-color: #0f172a;
    color: #f1f5f9;
    border: 1px solid #334155;
    border-radius: 5px;
    padding: 5px 8px;
}
QTableWidget {
    background-color: #0f172a;
    alternate-background-color: #162033;
    gridline-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 6px;
}
QHeaderView::section {
    background-color: #1e293b;
    color: #cbd5e1;
    padding: 6px;
    border: none;
    font-weight: 600;
}
QProgressBar {
    border: 1px solid #334155;
    border-radius: 5px;
    text-align: center;
    background-color: #0f172a;
}
QProgressBar::chunk {
    background-color: #22c55e;
    border-radius: 4px;
}
"""


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.resize(1340, 860)
        self.setStyleSheet(DARK_STYLESHEET)

        self.db = TrackDatabase(DB_PATH)
        self.media_monitor = WindowsMediaMonitor(prefer_spotify=True)
        self.controller = None
        self.hdf5_writer: Optional[SessionHDF5Writer] = None

        self.session_id: Optional[str] = None
        self.session_start_ts: float = 0.0
        self.session_baseline: Dict[str, float] = {}
        self.session_listens: List[TrackListenRecord] = []

        self._active_track: Optional[TrackInfo] = None
        self._active_track_start: float = 0.0
        self._active_track_snaps: List[MetricSnapshot] = []

        # Буферы для графиков + флаг «нужна перерисовка»
        self._plot_max_len = 180
        self._plot_t = deque(maxlen=self._plot_max_len)
        self._plot_series = {k: deque(maxlen=self._plot_max_len) for k in CORE_EMOTIONS}
        self._plot_hr = deque(maxlen=self._plot_max_len)
        self._ema_vec: Optional[Dict[str, float]] = None
        self._charts_dirty = False
        self._last_verdict_color: Optional[str] = None
        self._last_imp_color: Optional[str] = None
        self._latest_verdict_text: str = "Текущий фон: —"
        self._latest_verdict_color: str = "#e2e8f0"

        self._build_ui()
        self._setup_timers()
        self._refresh_library_table()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.tab_live = QWidget()
        self.tab_report = QWidget()
        self.tab_library = QWidget()

        self.tabs.addTab(self.tab_live, "Сессия (Live)")
        self.tabs.addTab(self.tab_report, "Отчёт / HDF5")
        self.tabs.addTab(self.tab_library, "Библиотека треков")

        self._build_live_tab()
        self._build_report_tab()
        self._build_library_tab()

    def _make_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        return card

    def _build_live_tab(self) -> None:
        root = QVBoxLayout(self.tab_live)
        root.setSpacing(10)

        top_card = self._make_card()
        top_lay = QHBoxLayout(top_card)

        self.combo_device = QComboBox()
        self.combo_device.addItem("Any (все Neiry / BrainBit)", "Any")
        self.combo_device.addItem("Headband", "Headband")
        self.combo_device.addItem("BrainBit", "BrainBit")
        self.combo_device.addItem("SinWave (виртуальный)", "SinWave")
        self.combo_device.addItem("Noise (виртуальный)", "Noise")
        self.combo_device.addItem("Demo (без повязки)", "Demo")

        self.chk_bipolar = QCheckBox("Биполярный")
        self.chk_bipolar.setChecked(True)
        self.chk_bipolar.setToolTip("Уменьшает артефакты морганий (рекомендуется для Neiry Headband)")

        self.chk_raw_hdf5 = QCheckBox("Писать сырые ЭЭГ/MEMS")
        self.chk_raw_hdf5.setChecked(True)
        self.chk_raw_hdf5.setToolTip("Отключите для минимальной нагрузки на CPU (эмоции и треки всё равно запишутся в HDF5)")

        self.btn_connect = QPushButton("Сканировать и подключить")
        self.btn_connect.clicked.connect(self._on_connect_clicked)

        self.btn_disconnect = QPushButton("Завершить и сохранить")
        self.btn_disconnect.setObjectName("danger")
        self.btn_disconnect.setEnabled(False)
        self.btn_disconnect.clicked.connect(self._on_disconnect_clicked)

        self.lbl_status = QLabel("Статус: не подключено")
        self.lbl_battery = QLabel("Батарея: —")
        self.lbl_impedance = QLabel("Контакт: —")

        top_lay.addWidget(QLabel("Поиск:"))
        top_lay.addWidget(self.combo_device)
        top_lay.addWidget(self.chk_bipolar)
        top_lay.addWidget(self.chk_raw_hdf5)
        top_lay.addWidget(self.btn_connect)
        top_lay.addWidget(self.btn_disconnect)
        top_lay.addStretch()
        top_lay.addWidget(self.lbl_status)
        top_lay.addWidget(self.lbl_battery)
        top_lay.addWidget(self.lbl_impedance)
        root.addWidget(top_card)

        mid_row = QHBoxLayout()

        calib_card = self._make_card()
        calib_lay = QVBoxLayout(calib_card)
        calib_hdr = QLabel("Калибровка (все метрики)")
        calib_hdr.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        calib_lay.addWidget(calib_hdr)

        calib_btns = QHBoxLayout()
        self.btn_calib_quick = QPushButton("Быстрая калибровка (30 с + бейслайны)")
        self.btn_calib_quick.setEnabled(False)
        self.btn_calib_quick.clicked.connect(self._on_calibrate_clicked)

        self.btn_calib_import = QPushButton("Импортировать прошлую")
        self.btn_calib_import.setObjectName("secondary")
        self.btn_calib_import.setEnabled(False)
        self.btn_calib_import.clicked.connect(self._on_import_calib_clicked)

        calib_btns.addWidget(self.btn_calib_quick)
        calib_btns.addWidget(self.btn_calib_import)
        calib_lay.addLayout(calib_btns)

        self.pb_calib = QProgressBar()
        self.pb_calib.setRange(0, 100)
        self.pb_calib.setValue(0)
        self.lbl_calib_stage = QLabel("Калибровка не пройдена (доступны Emotions и Cardio)")
        self.lbl_calib_stage.setStyleSheet("color: #94a3b8;")
        calib_lay.addWidget(self.pb_calib)
        calib_lay.addWidget(self.lbl_calib_stage)
        mid_row.addWidget(calib_card, 1)

        track_card = self._make_card()
        track_lay = QVBoxLayout(track_card)
        smtc_status = (
            "Windows SMTC активен (Spotify / плееры)"
            if self.media_monitor.is_smtc_available
            else "SMTC недоступен (ручной ввод трека)"
        )
        track_hdr = QLabel(f"Текущий трек — {smtc_status}")
        track_hdr.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        track_lay.addWidget(track_hdr)

        self.lbl_track_name = QLabel("Ожидание воспроизведения...")
        self.lbl_track_name.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        self.lbl_track_name.setStyleSheet("color: #38bdf8;")
        self.lbl_track_meta = QLabel("Плеер: — | Длительность сегмента: 0 с")
        self.lbl_live_verdict = QLabel("Текущий фон: —")
        self.lbl_live_verdict.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))

        track_lay.addWidget(self.lbl_track_name)
        track_lay.addWidget(self.lbl_track_meta)
        track_lay.addWidget(self.lbl_live_verdict)

        manual_row = QHBoxLayout()
        self.inp_manual_track = QLineEdit()
        self.inp_manual_track.setPlaceholderText("Ручной тест трека: Исполнитель - Название")
        self.btn_manual_set = QPushButton("Задать")
        self.btn_manual_set.setObjectName("secondary")
        self.btn_manual_set.clicked.connect(self._on_manual_track_set)
        self.btn_manual_clear = QPushButton("Авто (SMTC)")
        self.btn_manual_clear.setObjectName("secondary")
        self.btn_manual_clear.clicked.connect(lambda: self.media_monitor.set_manual_track(None))
        manual_row.addWidget(self.inp_manual_track)
        manual_row.addWidget(self.btn_manual_set)
        manual_row.addWidget(self.btn_manual_clear)
        track_lay.addLayout(manual_row)

        mid_row.addWidget(track_card, 1)
        root.addLayout(mid_row)

        # antialias=False резко снижает CPU при программной отрисовке Qt
        pg.setConfigOptions(antialias=False)
        charts_split = QSplitter(Qt.Orientation.Horizontal)

        self.plot_emotions = pg.PlotWidget(title="Эмоциональный профиль в реальном времени (сглаженный)")
        self.plot_emotions.setBackground("#0f172a")
        self.plot_emotions.showGrid(x=True, y=True, alpha=0.2)
        self.plot_emotions.setYRange(0, 1.0)
        self.plot_emotions.addLegend(offset=(10, 10))
        self._curves: Dict[str, pg.PlotDataItem] = {}
        for emo in CORE_EMOTIONS:
            pen = pg.mkPen(color=EMOTION_COLORS[emo], width=2)
            self._curves[emo] = self.plot_emotions.plot([], [], name=EMOTION_LABELS_RU[emo], pen=pen)

        self.plot_cardio = pg.PlotWidget(title="Пульс (ЧСС, уд/мин)")
        self.plot_cardio.setBackground("#0f172a")
        self.plot_cardio.showGrid(x=True, y=True, alpha=0.2)
        self._curve_hr = self.plot_cardio.plot([], [], pen=pg.mkPen(color="#f43f5e", width=2))

        charts_split.addWidget(self.plot_emotions)
        charts_split.addWidget(self.plot_cardio)
        charts_split.setSizes([850, 400])
        root.addWidget(charts_split, 3)

        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMaximumHeight(130)
        root.addWidget(self.txt_log, 1)

    def _build_report_tab(self) -> None:
        lay = QVBoxLayout(self.tab_report)
        bar = QHBoxLayout()
        btn_open_h5 = QPushButton("Открыть HDF5-файл сессии (.h5)")
        btn_open_h5.clicked.connect(self._on_open_hdf5_clicked)
        self.lbl_h5_info = QLabel("Файл не выбран — показаны треки текущей сессии")
        bar.addWidget(btn_open_h5)
        bar.addWidget(self.lbl_h5_info)
        bar.addStretch()
        lay.addLayout(bar)

        self.plot_report = pg.PlotWidget(title="Таймлайн эмоций сессии с разметкой треков")
        self.plot_report.setBackground("#0f172a")
        self.plot_report.showGrid(x=True, y=True, alpha=0.2)
        self.plot_report.addLegend(offset=(10, 10))
        lay.addWidget(self.plot_report, 2)

        self.tbl_session = QTableWidget(0, 7)
        self.tbl_session.setHorizontalHeaderLabels([
            "Трек", "Длительность", "Доминирующая эмоция", "Пик (сек)", "Ср. ЧСС", "Качество", "Плеер"
        ])
        self.tbl_session.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tbl_session.setAlternatingRowColors(True)
        lay.addWidget(self.tbl_session, 2)

    def _build_library_tab(self) -> None:
        lay = QVBoxLayout(self.tab_library)

        top = QHBoxLayout()
        top.addWidget(QLabel("Фильтр по триггеру:"))
        self.combo_filter = QComboBox()
        self.combo_filter.addItem("Все треки", "all")
        for key, label in EMOTION_LABELS_RU.items():
            self.combo_filter.addItem(label, key)
        self.combo_filter.currentIndexChanged.connect(self._refresh_library_table)

        btn_refresh = QPushButton("Обновить")
        btn_refresh.setObjectName("secondary")
        btn_refresh.clicked.connect(self._refresh_library_table)

        top.addWidget(self.combo_filter)
        top.addWidget(btn_refresh)
        top.addStretch()
        lay.addLayout(top)

        split = QSplitter(Qt.Orientation.Vertical)

        self.tbl_library = QTableWidget(0, 7)
        self.tbl_library.setHorizontalHeaderLabels([
            "Трек", "Прослушиваний", "Рекомендация (ребалансированная)",
            "Уверенность", "Тренд между сессиями", "Ср. ЧСС", "Качество"
        ])
        self.tbl_library.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tbl_library.setAlternatingRowColors(True)
        self.tbl_library.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.tbl_library.itemSelectionChanged.connect(self._on_library_track_selected)
        split.addWidget(self.tbl_library)

        hist_box = self._make_card()
        hist_lay = QVBoxLayout(hist_box)
        self.lbl_hist_title = QLabel("Выберите трек выше, чтобы увидеть историю прослушиваний по сессиям")
        self.lbl_hist_title.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.tbl_history = QTableWidget(0, 6)
        self.tbl_history.setHorizontalHeaderLabels([
            "Дата / время", "Сессия", "Эмоция в этой сессии", "Длительность", "Качество", "Вектор (Rel / Foc / Inv / Str / Fat)"
        ])
        self.tbl_history.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        hist_lay.addWidget(self.lbl_hist_title)
        hist_lay.addWidget(self.tbl_history)
        split.addWidget(hist_box)

        split.setSizes([450, 260])
        lay.addWidget(split)

    # ------------------------------------------------------------------ Timers
    def _setup_timers(self) -> None:
        # 1. Capsule API event loop (~20 Гц)
        self.timer_capsule = QTimer(self)
        self.timer_capsule.setInterval(CAPSULE_UPDATE_INTERVAL_MS)
        self.timer_capsule.timeout.connect(self._on_capsule_tick)

        # 2. Перерисовка графиков отдельно от входящих пакетов (~3 Гц)
        self.timer_charts = QTimer(self)
        self.timer_charts.setInterval(CHART_UPDATE_INTERVAL_MS)
        self.timer_charts.timeout.connect(self._on_charts_tick)
        self.timer_charts.start()

        # 3. Опрос Windows SMTC (раз в 1.5 сек)
        self.timer_media = QTimer(self)
        self.timer_media.setInterval(MEDIA_POLL_INTERVAL_MS)
        self.timer_media.timeout.connect(self._on_media_tick)
        self.timer_media.start()

    def _log(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self.txt_log.append(f"[{ts}] {msg}")

    # ------------------------------------------------------------------ Device & Session lifecycle
    def _on_connect_clicked(self) -> None:
        dev_type = self.combo_device.currentData()
        bipolar = self.chk_bipolar.isChecked()
        record_raw = self.chk_raw_hdf5.isChecked()

        cb = CapsuleEventCallbacks()
        cb.on_log = self._log
        cb.on_status = lambda s: self.lbl_status.setText(f"Статус: {s}")
        cb.on_connected = self._handle_connected
        cb.on_scan_finished = self._handle_scan_finished
        cb.on_battery = lambda b: self.lbl_battery.setText(f"Батарея: {b}%")
        cb.on_resistances = self._handle_resistances
        cb.on_snapshot = self._handle_snapshot
        cb.on_raw_eeg_batch = self._handle_raw_eeg_batch
        cb.on_raw_ppg_batch = self._handle_raw_ppg_batch
        cb.on_raw_mems_batch = self._handle_raw_mems_batch
        cb.on_calibration_progress = self._handle_calib_progress
        cb.on_calibration_ready = self._handle_calib_ready

        try:
            if dev_type == "Demo":
                self.controller = DemoCapsuleController(cb, record_raw_signals=record_raw)
                self.timer_capsule.start()
                self.controller.start_scan("Demo", bipolar=bipolar)
            else:
                self.controller = RealCapsuleController(
                    capsule_py_dir=DEFAULT_CAPSULE_PY_DIR,
                    dll_path=DEFAULT_CAPSULE_DLL_PATH,
                    logs_dir=LOGS_DIR,
                    callbacks=cb,
                    record_raw_signals=record_raw,
                )
                self.timer_capsule.start()
                self.controller.start_scan(dev_type, bipolar=bipolar, auto_connect=True, max_attempts=3, scan_seconds=12)
        except Exception as exc:
            self.timer_capsule.stop()
            QMessageBox.critical(self, "Ошибка запуска Capsule API", str(exc))
            self._log(f"Ошибка: {exc}")
            return

        self.btn_connect.setEnabled(False)
        self.btn_disconnect.setEnabled(True)
        self.chk_raw_hdf5.setEnabled(False)
        self.btn_calib_quick.setEnabled(False)
        self.btn_calib_import.setEnabled(False)

    def _handle_scan_finished(self, found: bool) -> None:
        if not found:
            self.timer_capsule.stop()
            self.btn_connect.setEnabled(True)
            self.btn_disconnect.setEnabled(False)
            self.chk_raw_hdf5.setEnabled(True)
            self.btn_calib_quick.setEnabled(False)
            self.btn_calib_import.setEnabled(False)

    def _handle_connected(self, ok: bool) -> None:
        if ok:
            self.session_id = datetime.now().strftime("session_%Y%m%d_%H%M%S")
            self.session_start_ts = time.time()
            self.session_listens.clear()
            self._plot_t.clear()
            for q in self._plot_series.values():
                q.clear()
            self._plot_hr.clear()
            self._ema_vec = None

            bipolar = self.chk_bipolar.isChecked()
            h5_path = SESSIONS_DIR / f"{self.session_id}.h5"
            try:
                self.hdf5_writer = SessionHDF5Writer(
                    filepath=h5_path,
                    session_id=self.session_id,
                    channel_names=self.controller.channel_names if self.controller else ["Ch0", "Ch1"],
                    bipolar=bipolar,
                )
                self._log(f"Запись сессии HDF5 открыта: {h5_path.name}")
            except Exception as exc:
                self.hdf5_writer = None
                self._log(f"Предупреждение: запись HDF5 отключена ({exc})")

            self.db.register_session(
                session_id=self.session_id,
                device_name=self.controller.device_name if self.controller else "Neiry",
                bipolar=bipolar,
                hdf5_path=str(h5_path),
            )
            self.btn_calib_quick.setEnabled(True)
            self.btn_calib_import.setEnabled(self.db.load_latest_calibration() is not None)
        else:
            self.btn_calib_quick.setEnabled(False)
            self.btn_calib_import.setEnabled(False)

    def _on_disconnect_clicked(self) -> None:
        self._close_active_track_segment()
        self.timer_capsule.stop()
        if self.controller is not None:
            self.controller.stop_and_disconnect()
            self.controller = None
        if self.hdf5_writer is not None:
            self.hdf5_writer.close()
            self._log(f"HDF5-файл сессии сохранён: {self.hdf5_writer.filepath.name}")
            self.hdf5_writer = None
        if self.session_id is not None:
            self.db.finish_session(self.session_id)
            self.session_id = None

        self.btn_connect.setEnabled(True)
        self.btn_disconnect.setEnabled(False)
        self.chk_raw_hdf5.setEnabled(True)
        self.btn_calib_quick.setEnabled(False)
        self.btn_calib_import.setEnabled(False)
        self._refresh_library_table()
        self._refresh_session_report_table()

    def _on_capsule_tick(self) -> None:
        if self.controller is not None:
            self.controller.update()

    def _on_charts_tick(self) -> None:
        # Рисуем только если вкладка Live открыта и есть новые данные
        if not self._charts_dirty or self.tabs.currentWidget() is not self.tab_live:
            return
        self._charts_dirty = False
        xs = list(self._plot_t)
        for k in CORE_EMOTIONS:
            self._curves[k].setData(xs, list(self._plot_series[k]))
        self._curve_hr.setData(xs, list(self._plot_hr))
        self.lbl_live_verdict.setText(self._latest_verdict_text)
        if self._latest_verdict_color != self._last_verdict_color:
            self._last_verdict_color = self._latest_verdict_color
            self.lbl_live_verdict.setStyleSheet(f"color: {self._latest_verdict_color}; font-weight: 600;")

    # ------------------------------------------------------------------ Calibration
    def _on_calibrate_clicked(self) -> None:
        if self.controller is not None:
            self.controller.start_quick_calibration()

    def _on_import_calib_clicked(self) -> None:
        prof = self.db.load_latest_calibration()
        if prof is None:
            QMessageBox.information(self, "Калибровка", "Сохранённых профилей калибровки пока нет.")
            return
        if self.controller is not None:
            self.controller.import_calibration_profile(prof)

    def _handle_calib_progress(self, stage: str, progress: float) -> None:
        self.pb_calib.setValue(int(max(0.0, min(1.0, progress)) * 100))
        self.lbl_calib_stage.setText(f"Этап: {stage} ({int(progress * 100)}%)")

    def _handle_calib_ready(self, profile: CalibrationProfile) -> None:
        self.db.save_calibration_profile(profile)
        self.session_baseline = {
            "prod_relaxation": profile.prod_relaxation if profile.prod_relaxation > 0 else 2.0,
            "prod_concentration": profile.prod_concentration if profile.prod_concentration > 0 else 1.0,
            "prod_fatigue": profile.prod_fatigue if profile.prod_fatigue > 0 else 2.0,
            "prod_gravity": profile.prod_gravity if profile.prod_gravity > 0 else 0.6,
        }
        self.btn_calib_import.setEnabled(True)
        self.lbl_calib_stage.setText(f"Калибровка активна ({profile.profile_name})")
        self._log("Все калибровки готовы и сохранены в базу.")

    # ------------------------------------------------------------------ Incoming bio-signals
    def _handle_resistances(self, res: Dict[str, float]) -> None:
        if not res:
            return
        worst_kohm = max(res.values()) / 1000.0
        ok = max(res.values()) <= MAX_GOOD_RESISTANCE_OHM
        color = "#22c55e" if ok else "#ef4444"
        self.lbl_impedance.setText(f"Контакт: макс {worst_kohm:.0f} кОм")
        if color != self._last_imp_color:
            self._last_imp_color = color
            self.lbl_impedance.setStyleSheet(f"color: {color}; font-weight: 600;")

    def _handle_raw_eeg_batch(self, rows: List[Tuple]) -> None:
        if self.hdf5_writer is not None:
            self.hdf5_writer.append_eeg_batch(rows)

    def _handle_raw_ppg_batch(self, rows: List[Tuple]) -> None:
        if self.hdf5_writer is not None:
            self.hdf5_writer.append_ppg_batch(rows)

    def _handle_raw_mems_batch(self, acc_rows: List[Tuple], gyro_rows: List[Tuple]) -> None:
        if self.hdf5_writer is not None:
            self.hdf5_writer.append_mems_batch(acc_rows, gyro_rows)

    def _handle_snapshot(self, snap: MetricSnapshot) -> None:
        vec = snapshot_to_emotion_vector(snap, self.session_baseline)
        q = compute_snapshot_quality(snap)

        if self._ema_vec is None:
            self._ema_vec = dict(vec)
        else:
            a = EMA_SMOOTHING_ALPHA
            for k in CORE_EMOTIONS:
                self._ema_vec[k] = a * vec[k] + (1.0 - a) * self._ema_vec[k]

        rel_t = snap.timestamp - self.session_start_ts
        self._plot_t.append(rel_t)
        for k in CORE_EMOTIONS:
            self._plot_series[k].append(self._ema_vec[k])
        self._plot_hr.append(snap.heart_rate if snap.heart_rate is not None else np.nan)

        dom = max(self._ema_vec, key=self._ema_vec.get)
        hr_str = f" | ЧСС: {snap.heart_rate:.0f}" if snap.heart_rate else ""
        self._latest_verdict_text = (
            f"Текущий фон: {EMOTION_LABELS_RU[dom]} ({int(self._ema_vec[dom] * 100)}%){hr_str} | Качество: {int(q * 100)}%"
        )
        self._latest_verdict_color = EMOTION_COLORS[dom]
        self._charts_dirty = True

        if self.hdf5_writer is not None:
            self.hdf5_writer.append_emotion_point(
                timestamp_sec=snap.timestamp,
                vec=vec,
                heart_rate=snap.heart_rate,
                stress_index=snap.stress_index,
                quality=q,
            )

        if self._active_track is not None and self._active_track.is_playing:
            self._active_track_snaps.append(snap)

    # ------------------------------------------------------------------ Media / Spotify SMTC tracking
    def _on_manual_track_set(self) -> None:
        raw = self.inp_manual_track.text().strip()
        if not raw:
            return
        if " - " in raw:
            artist, title = raw.split(" - ", 1)
        else:
            artist, title = "Manual", raw
        self.media_monitor.set_manual_track(
            TrackInfo(title=title.strip(), artist=artist.strip(), source_app="Manual", is_playing=True)
        )

    def _on_media_tick(self) -> None:
        track = self.media_monitor.poll_current_track()
        now = time.time()

        if track is None or not track.is_playing:
            if self._active_track is not None:
                self._close_active_track_segment()
            if track is not None:
                self.lbl_track_name.setText(f"⏸ {track.display_name}")
                self.lbl_track_meta.setText(f"Плеер: {track.source_app} (пауза)")
            else:
                self.lbl_track_name.setText("Ожидание воспроизведения...")
                self.lbl_track_meta.setText("Включите трек в Spotify или другом плеере")
            return

        if self._active_track is None or self._active_track.track_key != track.track_key:
            if self._active_track is not None:
                self._close_active_track_segment()
            self._active_track = track
            self._active_track_start = now
            self._active_track_snaps = []
            self._log(f"🎵 Начат трек: {track.display_name} ({track.source_app})")

        elapsed = now - self._active_track_start
        self.lbl_track_name.setText(f"▶ {track.display_name}")
        self.lbl_track_meta.setText(
            f"Плеер: {track.source_app} | Записано по треку: {elapsed:.0f} с (срезов: {len(self._active_track_snaps)})"
        )

    def _close_active_track_segment(self) -> None:
        if self._active_track is None:
            return
        end_ts = time.time()
        duration = end_ts - self._active_track_start
        track = self._active_track
        snaps = list(self._active_track_snaps)

        self._active_track = None
        self._active_track_snaps = []

        if self.session_id is None or not snaps:
            return

        if duration < MIN_TRACK_LISTEN_SEC:
            self._log(
                f"Трек «{track.display_name}» пропущен ({duration:.0f} с < порога {MIN_TRACK_LISTEN_SEC:.0f} с)."
            )
            return

        rec = summarize_track_segment(
            session_id=self.session_id,
            track_key=track.track_key,
            artist=track.artist,
            title=track.title,
            source_app=track.source_app,
            start_time=self._active_track_start,
            end_time=end_ts,
            snapshots=snaps,
            session_baseline=self.session_baseline,
        )
        self.db.add_listen(rec)
        self.session_listens.append(rec)

        if self.hdf5_writer is not None:
            self.hdf5_writer.add_track_marker(
                {
                    "track_key": rec.track_key,
                    "artist": rec.artist,
                    "title": rec.title,
                    "start_time": rec.start_time,
                    "end_time": rec.end_time,
                    "dominant_emotion": rec.dominant_emotion,
                    "quality_score": rec.quality_score,
                    "scores": rec.scores,
                }
            )

        self._log(
            f"✓ Сохранено прослушивание «{track.display_name}»: "
            f"{EMOTION_LABELS_RU.get(rec.dominant_emotion, rec.dominant_emotion)} "
            f"({duration:.0f} с, качество {int(rec.quality_score * 100)}%)"
        )
        self._refresh_session_report_table()
        self._refresh_library_table()

    # ------------------------------------------------------------------ Report / HDF5 Tab
    def _refresh_session_report_table(self) -> None:
        self.tbl_session.setRowCount(len(self.session_listens))
        for row, rec in enumerate(self.session_listens):
            self.tbl_session.setItem(row, 0, QTableWidgetItem(f"{rec.artist} — {rec.title}"))
            self.tbl_session.setItem(row, 1, QTableWidgetItem(f"{rec.duration_sec:.0f} с"))
            emo_item = QTableWidgetItem(EMOTION_LABELS_RU.get(rec.dominant_emotion, rec.dominant_emotion))
            emo_item.setForeground(QColor(EMOTION_COLORS.get(rec.dominant_emotion, "#e2e8f0")))
            self.tbl_session.setItem(row, 2, emo_item)
            self.tbl_session.setItem(
                row, 3, QTableWidgetItem(f"{EMOTION_LABELS_RU.get(rec.peak_emotion, rec.peak_emotion)} @ {rec.peak_offset_sec:.0f}с")
            )
            self.tbl_session.setItem(row, 4, QTableWidgetItem(f"{rec.avg_heart_rate:.0f}" if rec.avg_heart_rate else "—"))
            self.tbl_session.setItem(row, 5, QTableWidgetItem(f"{int(rec.quality_score * 100)}%"))
            self.tbl_session.setItem(row, 6, QTableWidgetItem(rec.source_app))

    def _on_open_hdf5_clicked(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Выберите HDF5-файл сессии", str(SESSIONS_DIR), "HDF5 Files (*.h5 *.hdf5)"
        )
        if not path_str:
            return
        try:
            data = load_session_hdf5(Path(path_str))
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка чтения HDF5", str(exc))
            return

        self.lbl_h5_info.setText(
            f"Файл: {Path(path_str).name} | Каналы: {', '.join(data['channel_names']) or '—'} | "
            f"Треков в разметке: {len(data['track_markers'])}"
        )

        self.plot_report.clear()
        self.plot_report.addLegend(offset=(10, 10))
        tl = data.get("emotion_timeline")
        if tl is not None and len(tl) > 0:
            t0 = float(tl["timestamp_sec"][0])
            xs = tl["timestamp_sec"] - t0
            for emo in CORE_EMOTIONS:
                self.plot_report.plot(
                    xs, tl[emo], name=EMOTION_LABELS_RU[emo], pen=pg.mkPen(EMOTION_COLORS[emo], width=2)
                )
            for mk in data["track_markers"]:
                r0 = mk["start_time"] - t0
                r1 = mk["end_time"] - t0
                col = QColor(EMOTION_COLORS.get(mk.get("dominant_emotion", "neutral"), "#64748b"))
                col.setAlpha(45)
                region = pg.LinearRegionItem(values=(r0, r1), movable=False, brush=col)
                self.plot_report.addItem(region)

        markers = data["track_markers"]
        self.tbl_session.setRowCount(len(markers))
        for row, mk in enumerate(markers):
            dur = mk["end_time"] - mk["start_time"]
            dom = mk.get("dominant_emotion", "neutral")
            self.tbl_session.setItem(row, 0, QTableWidgetItem(f"{mk.get('artist', '')} — {mk.get('title', '')}"))
            self.tbl_session.setItem(row, 1, QTableWidgetItem(f"{dur:.0f} с"))
            emo_item = QTableWidgetItem(EMOTION_LABELS_RU.get(dom, dom))
            emo_item.setForeground(QColor(EMOTION_COLORS.get(dom, "#e2e8f0")))
            self.tbl_session.setItem(row, 2, emo_item)
            self.tbl_session.setItem(row, 3, QTableWidgetItem("—"))
            self.tbl_session.setItem(row, 4, QTableWidgetItem("—"))
            self.tbl_session.setItem(row, 5, QTableWidgetItem(f"{int(mk.get('quality_score', 1.0) * 100)}%"))
            self.tbl_session.setItem(row, 6, QTableWidgetItem("HDF5"))

    # ------------------------------------------------------------------ Track Library Tab
    def _refresh_library_table(self) -> None:
        flt = self.combo_filter.currentData()
        self._cached_profiles = self.db.get_all_track_profiles(emotion_filter=flt)
        self.tbl_library.setRowCount(len(self._cached_profiles))
        for row, prof in enumerate(self._cached_profiles):
            self.tbl_library.setItem(row, 0, QTableWidgetItem(f"{prof.artist} — {prof.title}"))
            self.tbl_library.setItem(row, 1, QTableWidgetItem(str(prof.listen_count)))
            rec_item = QTableWidgetItem(prof.recommendation_text)
            rec_item.setForeground(QColor(EMOTION_COLORS.get(prof.dominant_emotion, "#e2e8f0")))
            self.tbl_library.setItem(row, 2, rec_item)
            self.tbl_library.setItem(row, 3, QTableWidgetItem(f"{int(prof.confidence * 100)}%"))
            self.tbl_library.setItem(row, 4, QTableWidgetItem(prof.trend_note))
            self.tbl_library.setItem(row, 5, QTableWidgetItem(f"{prof.avg_heart_rate:.0f}" if prof.avg_heart_rate else "—"))
            self.tbl_library.setItem(row, 6, QTableWidgetItem(f"{int(prof.avg_quality * 100)}%"))

    def _on_library_track_selected(self) -> None:
        rows = self.tbl_library.selectionModel().selectedRows()
        if not rows:
            return
        idx = rows[0].row()
        if idx >= len(self._cached_profiles):
            return
        prof = self._cached_profiles[idx]
        history = self.db.get_track_history(prof.track_key)
        self.lbl_hist_title.setText(f"История прослушиваний: {prof.artist} — {prof.title} (всего {len(history)})")
        self.tbl_history.setRowCount(len(history))
        for r, rec in enumerate(history):
            dt_str = datetime.fromtimestamp(rec.start_time).strftime("%Y-%m-%d %H:%M")
            self.tbl_history.setItem(r, 0, QTableWidgetItem(dt_str))
            self.tbl_history.setItem(r, 1, QTableWidgetItem(rec.session_id))
            emo_item = QTableWidgetItem(EMOTION_LABELS_RU.get(rec.dominant_emotion, rec.dominant_emotion))
            emo_item.setForeground(QColor(EMOTION_COLORS.get(rec.dominant_emotion, "#e2e8f0")))
            self.tbl_history.setItem(r, 2, emo_item)
            self.tbl_history.setItem(r, 3, QTableWidgetItem(f"{rec.duration_sec:.0f} с"))
            self.tbl_history.setItem(r, 4, QTableWidgetItem(f"{int(rec.quality_score * 100)}%"))
            vec_str = " / ".join(f"{int(rec.scores.get(k, 0) * 100)}%" for k in CORE_EMOTIONS)
            self.tbl_history.setItem(r, 5, QTableWidgetItem(vec_str))

    def closeEvent(self, event) -> None:
        self._on_disconnect_clicked()
        self.media_monitor.close()
        self.db.close()
        super().closeEvent(event)
