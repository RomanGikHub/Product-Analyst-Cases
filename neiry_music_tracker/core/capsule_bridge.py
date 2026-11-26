"""Мост к реальному CapsuleClientPython API (точные сигнатуры) + режим DemoSimulator с оптимизацией CPU."""
from __future__ import annotations

import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from core.models import CalibrationProfile, MetricSnapshot


DEVICE_TYPE_NAMES = {
    0: "Headband",
    1: "Buds",
    2: "Headphones",
    3: "Impulse",
    4: "Any",
    6: "BrainBit",
    100: "SinWave",
    101: "Noise",
}


class CapsuleEventCallbacks:
    """Набор колбэков, которые UI регистрирует для получения событий от контроллера."""

    def __init__(self) -> None:
        self.on_log: Callable[[str], None] = lambda msg: None
        self.on_status: Callable[[str], None] = lambda status: None
        self.on_devices_found: Callable[[List[Dict[str, str]]], None] = lambda devs: None
        self.on_connected: Callable[[bool], None] = lambda ok: None
        self.on_scan_finished: Callable[[bool], None] = lambda found: None
        self.on_battery: Callable[[int], None] = lambda charge: None
        self.on_resistances: Callable[[Dict[str, float]], None] = lambda res: None
        self.on_snapshot: Callable[[MetricSnapshot], None] = lambda snap: None
        self.on_raw_eeg_batch: Callable[[List[Tuple]], None] = lambda rows: None
        self.on_raw_ppg_batch: Callable[[List[Tuple]], None] = lambda rows: None
        self.on_raw_mems_batch: Callable[[List[Tuple], List[Tuple]], None] = lambda acc, gyro: None
        self.on_calibration_progress: Callable[[str, float], None] = lambda stage, pct: None
        self.on_calibration_ready: Callable[[CalibrationProfile], None] = lambda prof: None


class RealCapsuleController:
    """
    Обёртка над оригинальными файлами из папки CapsuleClientPython.
    Оптимизирована по CPU:
      - Пакетная передача сырых данных EEG/PPG/MEMS (1 вызов на пакет вместо 250/сек)
      - Ограничение частоты публикации MetricSnapshot (~3–4 Гц вместо каждого суб-колбэка)
      - Возможность отключить запись сырых EEG/MEMS (record_raw_signals=False)
    """

    def __init__(
        self,
        capsule_py_dir: Path,
        dll_path: Path,
        logs_dir: Path,
        callbacks: CapsuleEventCallbacks,
        record_raw_signals: bool = True,
    ):
        self.capsule_py_dir = Path(capsule_py_dir)
        self.dll_path = Path(dll_path)
        self.logs_dir = Path(logs_dir)
        self.cb = callbacks
        self.record_raw_signals = record_raw_signals

        self._loaded = False
        self.capsule_lib = None
        self.device_locator = None
        self.device = None
        self.calibrator = None
        self.emotions = None
        self.cardio = None
        self.productivity = None
        self.phy_states = None
        self.mems = None

        self.channel_names: List[str] = ["O1-T3", "O2-T4"]
        self.device_name: str = "Neiry Headband"
        self.bipolar: bool = True
        self._is_started = False
        self._scanning = False
        self._scan_attempt = 0
        self._max_scan_attempts = 3
        self._scan_seconds = 12
        self._auto_connect_first = True
        self._target_device_type = None
        self._target_device_type_name = "Any"
        self.found_devices: List[Dict[str, str]] = []

        self._latest = MetricSnapshot(timestamp=time.time())
        self._last_snapshot_emit_t = 0.0
        self._snapshot_min_interval = 0.25  # Не чаще 4 раз в секунду

        self._calib_profile = CalibrationProfile(
            profile_name="default",
            created_at=datetime.now().isoformat(timespec="seconds"),
        )
        self._alpha_done = False
        self._prod_done = False
        self._phy_done = False

    def _import_sdk(self) -> None:
        if self._loaded:
            return
        py_dir_str = str(self.capsule_py_dir.resolve())
        if py_dir_str not in sys.path:
            sys.path.insert(0, py_dir_str)

        if not self.dll_path.exists():
            raise FileNotFoundError(
                f"Не найдена нативная библиотека CapsuleClient: {self.dll_path}\n"
                "Поместите CapsuleClient.dll рядом с app.py."
            )

        from Capsule import Capsule
        from DeviceLocator import DeviceLocator
        from DeviceType import DeviceType
        from Device import Device, Device_Connection_Status
        from Calibrator import Calibrator, IndividualNFBData, IndividualNFBCalibrationFailReason
        from Emotions import Emotions
        from Cardio import Cardio
        from Productivity import Productivity, Productivity_Baselines
        from PhysiologicalStates import PhysiologicalStates, PhysiologicalStates_Baselines
        from MEMS import MEMS
        from Error import CapsuleException

        self._sdk = {
            "Capsule": Capsule,
            "DeviceLocator": DeviceLocator,
            "DeviceType": DeviceType,
            "Device": Device,
            "Device_Connection_Status": Device_Connection_Status,
            "Calibrator": Calibrator,
            "IndividualNFBData": IndividualNFBData,
            "IndividualNFBCalibrationFailReason": IndividualNFBCalibrationFailReason,
            "Emotions": Emotions,
            "Cardio": Cardio,
            "Productivity": Productivity,
            "Productivity_Baselines": Productivity_Baselines,
            "PhysiologicalStates": PhysiologicalStates,
            "PhysiologicalStates_Baselines": PhysiologicalStates_Baselines,
            "MEMS": MEMS,
            "CapsuleException": CapsuleException,
        }

        self.capsule_lib = Capsule(str(self.dll_path))
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.device_locator = DeviceLocator.with_logs_path(str(self.logs_dir), self.capsule_lib.get_lib())
        self.cb.on_log(f"Capsule API загружен, версия: {self.capsule_lib.get_version()}")
        self._loaded = True

    def start_scan(
        self,
        device_type_name: str = "Any",
        bipolar: bool = True,
        auto_connect: bool = True,
        max_attempts: int = 3,
        scan_seconds: int = 12,
    ) -> None:
        self._import_sdk()
        self.bipolar = bipolar
        self._auto_connect_first = auto_connect
        self._max_scan_attempts = max_attempts
        self._scan_seconds = scan_seconds
        self._scan_attempt = 1
        self._scanning = True
        self.found_devices = []

        dt_cls = self._sdk["DeviceType"]
        type_map = {
            "Any": dt_cls.Any,
            "Headband": dt_cls.Headband,
            "BrainBit": dt_cls.BrainBit,
            "Headphones": dt_cls.Headphones,
            "Buds": dt_cls.Buds,
            "SinWave": dt_cls.SinWave,
            "Noise": dt_cls.Noise,
        }
        self._target_device_type_name = device_type_name
        self._target_device_type = type_map.get(device_type_name, dt_cls.Any)

        self.device_locator.set_on_devices_list(self._on_devices_list)
        self.cb.on_status(f"Сканирование BLE ({device_type_name}, попытка 1/{max_attempts})...")
        self.cb.on_log(
            f"Запуск BLE-сканирования ({device_type_name}, {scan_seconds} сек, попытка 1/{max_attempts})..."
        )
        self.device_locator.request_devices(self._target_device_type, self._scan_seconds)

    def stop_scan(self) -> None:
        self._scanning = False
        if self.device_locator is not None:
            try:
                self.device_locator.stop_scanning()
            except Exception:
                pass
        self.cb.on_status("Сканирование остановлено")

    def _on_devices_list(self, locator, info, fail_reason) -> None:
        if not self._scanning:
            return

        fr_val = fail_reason.value if hasattr(fail_reason, "value") else int(fail_reason)
        fr_map = {0: "OK", 1: "BluetoothDisabled", 2: "Unknown"}
        fr_name = fr_map.get(fr_val, str(fr_val))

        if fr_val == 1:
            self._scanning = False
            self.cb.on_log("[BLE Ошибка] Bluetooth выключен или недоступен (BluetoothDisabled).")
            self.cb.on_status("Bluetooth выключен")
            self.cb.on_scan_finished(False)
            return

        count = len(info)
        self.cb.on_log(f"Результат сканирования (попытка {self._scan_attempt}): найдено {count}, статус={fr_name}")

        if count == 0:
            if self._scan_attempt < self._max_scan_attempts:
                self._scan_attempt += 1
                self.cb.on_status(
                    f"Сканирование BLE ({self._target_device_type_name}, попытка {self._scan_attempt}/{self._max_scan_attempts})..."
                )
                self.device_locator.request_devices(self._target_device_type, self._scan_seconds)
                return

            self._scanning = False
            self.cb.on_status("Устройство не найдено")
            self.cb.on_log(
                "Устройства не найдены. Закройте оригинальную программу Capsule, переведите повязку в режим поиска и повторите."
            )
            self.cb.on_scan_finished(False)
            return

        self._scanning = False
        try:
            locator.stop_scanning()
        except Exception:
            pass

        devices_parsed: List[Dict[str, str]] = []
        for i in range(count):
            d_info = info[i]
            serial = str(d_info.get_serial())
            name = str(d_info.get_name())
            dtype_raw = d_info.get_type()
            dtype_val = dtype_raw.value if hasattr(dtype_raw, "value") else int(dtype_raw)
            dtype_str = DEVICE_TYPE_NAMES.get(dtype_val, str(dtype_val))
            devices_parsed.append(
                {
                    "serial": serial,
                    "name": name,
                    "type": dtype_str,
                    "label": f"{name} [{dtype_str}] — {serial}",
                }
            )
            self.cb.on_log(f"  [{i}] {name} | тип: {dtype_str} ({dtype_val}) | серийный №: {serial}")

        self.found_devices = devices_parsed
        self.cb.on_devices_found(devices_parsed)
        self.cb.on_scan_finished(True)

        if self._auto_connect_first and devices_parsed:
            self.connect_by_serial(devices_parsed[0]["serial"], devices_parsed[0]["name"], self.bipolar)

    def connect_by_serial(self, serial: str, name: str = "Neiry", bipolar: bool = True) -> None:
        self.bipolar = bipolar
        self.device_name = f"{name} ({serial})"
        Device = self._sdk["Device"]
        self.device = Device(self.device_locator, serial, self.device_locator.get_lib())
        self.device.set_on_connection_status_changed(self._on_connection_status)
        self.device.set_on_battery_charge_changed(self._on_battery)
        self.device.set_on_resistances(self._on_resistances)
        if self.record_raw_signals:
            self.device.set_on_eeg(self._on_eeg)
        self.device.set_on_eeg_artifacts(self._on_eeg_artifacts)
        self.device.set_on_error(lambda d, err: self.cb.on_log(f"[Device Error] {err}"))

        self.cb.on_status(f"Подключение к {name}...")
        self.cb.on_log(f"Подключаюсь к {self.device_name} (bipolar={bipolar})...")
        self.device.connect(bipolarChannels=bipolar)

    def _on_connection_status(self, device, status) -> None:
        Connected = self._sdk["Device_Connection_Status"].Device_ConnectionState_Connected
        status_val = status.value if hasattr(status, "value") else int(status)
        if status_val == Connected:
            self.cb.on_log("Устройство подключено! Инициализирую классификаторы...")
            try:
                ch_obj = self.device.get_channel_names()
                self.channel_names = [ch_obj.get_name_by_index(i) for i in range(len(ch_obj))]
                self.cb.on_log(f"Каналы ЭЭГ ({'биполярный' if self.bipolar else 'монополярный'}): {self.channel_names}")
            except Exception:
                self.channel_names = ["O1-T3", "O2-T4"] if self.bipolar else ["O1", "T3", "T4", "O2"]

            lib = self.capsule_lib.get_lib()
            self.calibrator = self._sdk["Calibrator"](self.device, lib)
            self.calibrator.set_on_calibration_finished(self._on_alpha_calibrated)

            self.emotions = self._sdk["Emotions"](self.device, lib)
            self.emotions.set_on_states_update(self._on_emotions)
            self.emotions.set_on_error(lambda e, err: self.cb.on_log(f"[Emotions Error] {err}"))

            self.cardio = self._sdk["Cardio"](self.device, lib)
            self.cardio.set_on_indexes_update(self._on_cardio)
            if self.record_raw_signals:
                self.cardio.set_on_ppg(self._on_ppg)

            self.productivity = self._sdk["Productivity"](self.device, lib)
            self.productivity.set_on_metrics_update(self._on_prod_metrics)
            self.productivity.set_on_indexes_update(self._on_prod_indexes)
            self.productivity.set_on_baseline_update(self._on_prod_baselines)
            self.productivity.set_on_calibration_progress(
                lambda p, prog: self.cb.on_calibration_progress("Productivity baselines", float(prog))
            )

            self.phy_states = self._sdk["PhysiologicalStates"](self.device, lib)
            self.phy_states.set_on_states(self._on_phy_states)
            self.phy_states.set_on_calibrated(self._on_phy_calibrated)
            self.phy_states.set_on_calibration_progress(
                lambda p, prog: self.cb.on_calibration_progress("Physiological States baselines", float(prog))
            )

            try:
                self.mems = self._sdk["MEMS"](self.device, lib)
                self.mems.set_on_update(self._on_mems)
            except Exception as exc:
                self.cb.on_log(f"MEMS недоступен для данного устройства: {exc}")

            self.device.start()
            self._is_started = True
            self.cb.on_status(f"Подключено: {self.device_name}")
            self.cb.on_connected(True)
            self.cb.on_log("Поток данных запущен (device.start()).")
        else:
            self._is_started = False
            self.cb.on_status("Отключено")
            self.cb.on_connected(False)
            self.cb.on_log(f"Статус соединения изменён: {status_val}")

    # --- Калибровки ---
    def start_quick_calibration(self) -> None:
        if not self.calibrator or not self._is_started:
            self.cb.on_log("Сначала подключите устройство.")
            return
        self._alpha_done = False
        self._prod_done = False
        self._phy_done = False
        self.cb.on_log("Этап 1/3: Быстрая калибровка (30 сек с закрытыми глазами). Сидите спокойно...")
        self.cb.on_calibration_progress("Закрытые глаза (30 с)", 0.05)
        self.calibrator.calibrate_quick()

    def import_calibration_profile(self, profile: CalibrationProfile) -> None:
        if not self.calibrator or not self._is_started:
            self.cb.on_log("Сначала подключите устройство.")
            return
        NFB = self._sdk["IndividualNFBData"]
        nfb_data = NFB(
            timestamp_milli=1,
            individual_frequency=profile.individual_frequency,
            individual_peak_frequency=profile.individual_peak_frequency,
            individual_peak_frequency_power=profile.individual_peak_frequency_power,
            individual_peak_frequency_suppression=profile.individual_peak_frequency_suppression,
            individual_bandwidth=profile.individual_bandwidth,
            individual_normalized_power=profile.individual_normalized_power,
            lower_frequency=profile.lower_frequency,
            upper_frequency=profile.upper_frequency,
        )
        self.calibrator.import_alpha(individual_nfb_data=nfb_data)

        if profile.has_productivity_baselines() and self.productivity:
            PB = self._sdk["Productivity_Baselines"]
            self.productivity.import_baselines(
                PB(
                    concentration=profile.prod_concentration,
                    fatigue=profile.prod_fatigue,
                    gravity=profile.prod_gravity,
                    productivity=profile.prod_productivity,
                    relaxation=profile.prod_relaxation,
                    reverse_fatigue=profile.prod_reverse_fatigue,
                    timestamp_milli=0,
                )
            )

        if profile.has_phy_baselines() and self.phy_states:
            PhyB = self._sdk["PhysiologicalStates_Baselines"]
            self.phy_states.import_baselines(
                baselines=PhyB(
                    timestamp_milli=0,
                    alpha=profile.phy_alpha,
                    beta=profile.phy_beta,
                    alpha_gravity=profile.phy_alpha_gravity,
                    beta_gravity=profile.phy_beta_gravity,
                    concentration=profile.phy_concentration,
                )
            )

        self._calib_profile = profile
        self.cb.on_log(f"Импортирован сохранённый профиль калибровки: {profile.profile_name}")
        self.cb.on_calibration_progress("Импортировано", 1.0)
        self.cb.on_calibration_ready(profile)

    def _on_alpha_calibrated(self, calibrator, data) -> None:
        FailReason = self._sdk["IndividualNFBCalibrationFailReason"]
        if data.failReason.value == FailReason.IndividualNFBCalibrationFailReason_TooManyArtifacts:
            self.cb.on_log("Калибровка с закрытыми глазами не удалась (много артефактов).")
            self.cb.on_calibration_progress("Ошибка (артефакты)", 0.0)
            return

        self._calib_profile.individual_frequency = float(data.individualFrequency)
        self._calib_profile.individual_peak_frequency = float(data.individualPeakFrequency)
        self._calib_profile.individual_peak_frequency_power = float(data.individualPeakFrequencyPower)
        self._calib_profile.individual_peak_frequency_suppression = float(data.individualPeakFrequencySuppression)
        self._calib_profile.individual_bandwidth = float(data.individualBandwidth)
        self._calib_profile.individual_normalized_power = float(data.individualNormalizedPower)
        self._calib_profile.lower_frequency = float(data.lowerFrequency)
        self._calib_profile.upper_frequency = float(data.upperFrequency)
        self._alpha_done = True

        self.cb.on_log(
            f"Этап 1/3 завершён! IAF = {data.individualPeakFrequency:.2f} Гц. Запускаю сбор бейслайнов..."
        )
        self.cb.on_calibration_progress("Сбор бейслайнов", 0.35)
        if self.productivity:
            self.productivity.calibrate_baselines()
        if self.phy_states:
            self.phy_states.calibrate_baselines()

    def _on_prod_baselines(self, prod, baselines) -> None:
        self._calib_profile.prod_concentration = float(baselines.concentration)
        self._calib_profile.prod_fatigue = float(baselines.fatigue)
        self._calib_profile.prod_gravity = float(baselines.gravity)
        self._calib_profile.prod_productivity = float(baselines.productivity)
        self._calib_profile.prod_relaxation = float(baselines.relaxation)
        self._calib_profile.prod_reverse_fatigue = float(baselines.reverseFatigue)
        self._prod_done = True
        self.cb.on_log("Этап 2/3 завершён: Productivity baselines получены.")
        self._check_all_calibrated()

    def _on_phy_calibrated(self, phy, baselines) -> None:
        self._calib_profile.phy_alpha = float(baselines.alpha)
        self._calib_profile.phy_beta = float(baselines.beta)
        self._calib_profile.phy_alpha_gravity = float(baselines.alphaGravity)
        self._calib_profile.phy_beta_gravity = float(baselines.betaGravity)
        self._calib_profile.phy_concentration = float(baselines.concentration)
        self._phy_done = True
        self.cb.on_log("Этап 3/3 завершён: Physiological States baselines получены.")
        self._check_all_calibrated()

    def _check_all_calibrated(self) -> None:
        if self._alpha_done and self._prod_done and self._phy_done:
            self._calib_profile.created_at = datetime.now().isoformat(timespec="seconds")
            self._calib_profile.profile_name = f"profile_{datetime.now().strftime('%Y%m%d_%H%M')}"
            self.cb.on_calibration_progress("Готово", 1.0)
            self.cb.on_calibration_ready(self._calib_profile)

    # --- Колбэки данных (пакетные и с троттлингом) ---
    def _on_battery(self, device, charge: int) -> None:
        self.cb.on_battery(int(charge))

    def _on_resistances(self, device, res) -> None:
        out: Dict[str, float] = {}
        for i in range(len(res)):
            out[res.get_channel_name(i)] = float(res.get_value(i))
        if out:
            self._latest.max_resistance_ohm = max(out.values())
        self.cb.on_resistances(out)

    def _on_eeg(self, device, eeg) -> None:
        if not self.record_raw_signals:
            return
        n_samples = eeg.get_samples_count()
        n_ch = eeg.get_channels_count()
        target_ch = len(self.channel_names)
        rows = []
        for idx in range(n_samples):
            ts_ms = int(eeg.get_timestamp(idx)) & 0xFFFFFFFF
            vals = [float(eeg.get_raw_value(ch, idx)) for ch in range(min(n_ch, target_ch))]
            if len(vals) < target_ch:
                vals.extend([0.0] * (target_ch - len(vals)))
            rows.append((ts_ms, *vals))
        if rows:
            self.cb.on_raw_eeg_batch(rows)

    def _on_eeg_artifacts(self, device, artifacts) -> None:
        n_ch = artifacts.get_channels_count()
        if n_ch > 0:
            bad = sum(1 for i in range(n_ch) if artifacts.get_artifacts_by_channel(i) > 0)
            self._latest.eeg_artifact_ratio = bad / float(n_ch)

    def _on_ppg(self, cardio, ppg) -> None:
        if not self.record_raw_signals:
            return
        n = len(ppg)
        if n:
            rows = [(int(ppg.get_timestamp(i)) & 0xFFFFFFFF, float(ppg.get_value(i))) for i in range(n)]
            self.cb.on_raw_ppg_batch(rows)

    def _on_mems(self, mems, mems_data) -> None:
        n = len(mems_data)
        if n == 0:
            return
        max_dev = 0.0
        if self.record_raw_signals:
            acc_rows = []
            gyro_rows = []
            for i in range(n):
                ts = int(mems_data.get_timestamp(i)) & 0xFFFFFFFF
                a = mems_data.get_accelerometer(i)
                g = mems_data.get_gyroscope(i)
                acc_rows.append((ts, float(a.x), float(a.y), float(a.z)))
                gyro_rows.append((ts, float(g.x), float(g.y), float(g.z)))
                mag = math.sqrt(a.x * a.x + a.y * a.y + a.z * a.z)
                dev = abs(mag - 1.0)
                if dev > max_dev:
                    max_dev = dev
            self.cb.on_raw_mems_batch(acc_rows, gyro_rows)
        else:
            # Берём только каждый 5-й сэмпл для детекции резких движений головы
            for i in range(0, n, 5):
                a = mems_data.get_accelerometer(i)
                mag = math.sqrt(a.x * a.x + a.y * a.y + a.z * a.z)
                dev = abs(mag - 1.0)
                if dev > max_dev:
                    max_dev = dev
        self._latest.mems_motion_g = max_dev

    def _maybe_emit_snapshot(self, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last_snapshot_emit_t) < self._snapshot_min_interval:
            return
        self._last_snapshot_emit_t = now
        self._latest.timestamp = now
        self.cb.on_snapshot(MetricSnapshot(**self._latest.__dict__))

    def _on_emotions(self, emotions, st) -> None:
        self._latest.attention = float(st.attention)
        self._latest.relaxation_emo = float(st.relaxation)
        self._latest.cognitive_load = float(st.cognitiveLoad)
        self._latest.cognitive_control = float(st.cognitiveControl)
        self._latest.self_control = float(st.selfControl)
        self._maybe_emit_snapshot()

    def _on_cardio(self, cardio, data) -> None:
        if data.metricsAvailable:
            self._latest.heart_rate = float(data.heartRate)
            self._latest.stress_index = float(data.stressIndex)
            self._latest.kaplan_index = float(data.kaplanIndex)
        self._latest.cardio_has_artifacts = bool(data.hasArtifacts or data.motionArtifacts)
        self._maybe_emit_snapshot()

    def _on_prod_metrics(self, prod, m) -> None:
        self._latest.prod_fatigue = float(m.fatigueScore)
        self._latest.prod_reverse_fatigue = float(m.reverseFatigueScore)
        self._latest.prod_gravity = float(m.gravityScore)
        self._latest.prod_relaxation = float(m.relaxationScore)
        self._latest.prod_concentration = float(m.concentrationScore)
        self._latest.prod_score = float(m.productivityScore)
        self._latest.prod_alpha = float(m.alpha)
        self._maybe_emit_snapshot()

    def _on_prod_indexes(self, prod, idx) -> None:
        self._latest.prod_stress_enum = int(idx.stress.value)
        self._maybe_emit_snapshot()

    def _on_phy_states(self, phy, st) -> None:
        self._latest.phy_relaxation = float(st.relaxation)
        self._latest.phy_fatigue = float(st.fatigue)
        self._latest.phy_concentration = float(st.concentration)
        self._latest.phy_involvement = float(st.involvement)
        self._latest.phy_stress = float(st.stress)
        self._maybe_emit_snapshot(force=True)

    def update(self) -> None:
        """Вызывается каждые 50 мс из главного потока Qt."""
        if self.device_locator is not None:
            self.device_locator.update()

    def stop_and_disconnect(self) -> None:
        self._scanning = False
        if self.device_locator is not None:
            try:
                self.device_locator.stop_scanning()
            except Exception:
                pass
        if self.device is not None:
            try:
                if self._is_started:
                    self.device.stop()
                self.device.disconnect()
            except Exception:
                pass
        self._is_started = False


class DemoCapsuleController:
    """Встроенный симулятор Neiry Headband для тестирования без повязки."""

    def __init__(self, callbacks: CapsuleEventCallbacks, record_raw_signals: bool = True):
        self.cb = callbacks
        self.record_raw_signals = record_raw_signals
        self.channel_names = ["O1-T3", "O2-T4"]
        self.device_name = "Neiry Headband (DEMO)"
        self.bipolar = True
        self._running = False
        self._t0 = time.time()
        self._last_metric_t = 0.0
        self._calib_start: Optional[float] = None
        self._phase = 0.0

    def start_scan(
        self,
        device_type_name: str = "Demo",
        bipolar: bool = True,
        auto_connect: bool = True,
        max_attempts: int = 1,
        scan_seconds: int = 1,
    ) -> None:
        self.bipolar = bipolar
        self.channel_names = ["O1-T3", "O2-T4"] if bipolar else ["O1", "T3", "T4", "O2"]
        self._running = True
        self._t0 = time.time()
        self.cb.on_devices_found([{"serial": "DEMO-0001", "name": "Neiry Headband (DEMO)", "type": "Demo", "label": "Neiry Headband (DEMO)"}])
        self.cb.on_scan_finished(True)
        self.cb.on_status("Подключено: DEMO-симулятор")
        self.cb.on_connected(True)
        self.cb.on_battery(87)
        self.cb.on_resistances({ch: 240_000.0 for ch in self.channel_names})
        self.cb.on_log("Запущен DEMO-режим (синтетические ЭЭГ/ФПГ/эмоции).")

    def connect_device(self, device_type_name: str = "Demo", bipolar: bool = True) -> None:
        self.start_scan(device_type_name=device_type_name, bipolar=bipolar)

    def start_quick_calibration(self) -> None:
        self._calib_start = time.time()
        self.cb.on_log("DEMO-калибровка запущена (ускорена до 3 секунд)...")

    def import_calibration_profile(self, profile: CalibrationProfile) -> None:
        self.cb.on_log(f"DEMO: импортирован профиль {profile.profile_name}")
        self.cb.on_calibration_progress("Импортировано", 1.0)
        self.cb.on_calibration_ready(profile)

    def update(self) -> None:
        if not self._running:
            return
        now = time.time()
        if self._calib_start is not None:
            elapsed = now - self._calib_start
            pct = min(1.0, elapsed / 3.0)
            self.cb.on_calibration_progress("DEMO-калибровка", pct)
            if pct >= 1.0:
                self._calib_start = None
                prof = CalibrationProfile(
                    profile_name=f"demo_{datetime.now().strftime('%Y%m%d_%H%M')}",
                    created_at=datetime.now().isoformat(timespec="seconds"),
                    individual_frequency=10.2,
                    individual_peak_frequency=10.2,
                    prod_concentration=1.1,
                    prod_fatigue=1.8,
                    prod_gravity=0.6,
                    prod_productivity=0.7,
                    prod_relaxation=2.2,
                    prod_reverse_fatigue=0.4,
                    phy_alpha=0.32,
                    phy_beta=0.48,
                    phy_alpha_gravity=4.5,
                    phy_beta_gravity=0.15,
                    phy_concentration=1.7,
                )
                self.cb.on_calibration_ready(prof)

        if now - self._last_metric_t >= 0.5:
            self._last_metric_t = now
            ts_ms = int((now - self._t0) * 1000) & 0xFFFFFFFF
            self._phase += 0.25
            if self.record_raw_signals:
                vals = [math.sin(self._phase + i) * 25e-6 + random.uniform(-3e-6, 3e-6) for i in range(len(self.channel_names))]
                self.cb.on_raw_eeg_batch([(ts_ms, *vals)])
                self.cb.on_raw_ppg_batch([(ts_ms, 0.5 + 0.3 * math.sin(self._phase * 0.5))])
                self.cb.on_raw_mems_batch([(ts_ms, 0.02, -0.01, 0.99)], [(ts_ms, 0.1, 0.0, -0.05)])

            t = now - self._t0
            relax = 0.5 + 0.35 * math.sin(t / 14.0)
            stress = 0.5 - 0.35 * math.sin(t / 14.0)
            focus = 0.5 + 0.25 * math.cos(t / 10.0)
            snap = MetricSnapshot(
                timestamp=now,
                attention=round(focus * 100, 1),
                relaxation_emo=round(relax * 100, 1),
                cognitive_load=round(stress * 100, 1),
                cognitive_control=round(focus * 90, 1),
                self_control=round((1.0 - stress * 0.6) * 100, 1),
                prod_relaxation=round(1.5 + relax * 2.0, 2),
                prod_concentration=round(0.6 + focus * 1.5, 2),
                prod_fatigue=round(1.2 + 0.3 * math.sin(t / 30.0), 2),
                prod_gravity=round(0.4 + focus * 0.6, 2),
                prod_stress_enum=2 if stress > 0.72 else (1 if stress > 0.55 else 0),
                heart_rate=round(66 + stress * 18 + random.uniform(-1, 1), 1),
                stress_index=round(90 + stress * 140, 1),
                max_resistance_ohm=240_000.0,
                eeg_artifact_ratio=0.0,
                mems_motion_g=0.05,
            )
            self.cb.on_snapshot(snap)

    def stop_and_disconnect(self) -> None:
        self._running = False
        self.cb.on_connected(False)
        self.cb.on_status("Отключено")
