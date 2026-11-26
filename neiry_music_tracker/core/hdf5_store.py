"""Запись и чтение сессий в формате HDF5 (совместимо со спецификацией Neiry 3.0 + маркеры треков)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None


class SessionHDF5Writer:
    """
    Поточно сохраняет сырые данные ЭЭГ/ФПГ/MEMS, рассчитанные метрики и
    автоматические маркеры треков в один HDF5-файл.
    Использует пакетную запись и крупный буфер, чтобы минимизировать нагрузку на CPU и диск.
    """

    def __init__(
        self,
        filepath: Path,
        session_id: str,
        channel_names: List[str],
        device_type: int = 0,
        bipolar: bool = True,
        flush_threshold: int = 5000,
    ):
        if h5py is None:
            raise RuntimeError("Для записи HDF5 установите пакет h5py: pip install h5py")

        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self.channel_names = list(channel_names)
        self.device_type = device_type
        self.bipolar = bipolar
        self.flush_threshold = flush_threshold
        self.start_time_us = int(time.time() * 1_000_000)

        self._h5 = h5py.File(str(self.filepath), "w")
        self._h5.attrs["session_id"] = session_id
        self._h5.attrs["start_time_us"] = self.start_time_us
        self._h5.attrs["bipolar"] = int(bipolar)
        self._h5.attrs["channel_names"] = json.dumps(self.channel_names, ensure_ascii=False)
        self._h5.attrs["device_type"] = device_type
        self._h5.attrs["track_markers_json"] = "[]"

        self._eeg_buf: List[Tuple] = []
        self._ppg_buf: List[Tuple] = []
        self._acc_buf: List[Tuple] = []
        self._gyro_buf: List[Tuple] = []
        self._metrics_buf: List[Tuple] = []
        self._track_markers: List[Dict[str, Any]] = []

        self._datasets: Dict[str, Any] = {}
        self._init_datasets()

    def _init_datasets(self) -> None:
        n_ch = max(1, len(self.channel_names))
        eeg_dtype = np.dtype(
            [("timestamp_name", np.uint32)]
            + [(f"eeg{i + 1}_name", np.float32) for i in range(n_ch)]
        )
        ds_eeg = self._h5.create_dataset(
            f"eeg{n_ch}_datasetName",
            shape=(0,),
            maxshape=(None,),
            dtype=eeg_dtype,
            chunks=(4096,),
        )
        ds_eeg.attrs[f"eeg{n_ch}_datasetAttributeStartTime"] = np.int64(self.start_time_us)
        ds_eeg.attrs[f"eeg{n_ch}_datasetAttribute"] = json.dumps(
            {
                "channelCount": n_ch,
                "channelNames": self.channel_names,
                "deviceType": self.device_type,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        self._datasets["eeg"] = ds_eeg

        ppg_dtype = np.dtype([("timestamp_name", np.uint32), ("ppg1_name", np.float32)])
        ds_ppg = self._h5.create_dataset(
            "ppg1_datasetName",
            shape=(0,),
            maxshape=(None,),
            dtype=ppg_dtype,
            chunks=(2048,),
        )
        ds_ppg.attrs["ppg1_datasetAttributeStartTime"] = np.int64(self.start_time_us)
        self._datasets["ppg"] = ds_ppg

        acc_dtype = np.dtype(
            [
                ("timestamp_name", np.uint32),
                ("accelerometerX_name", np.float32),
                ("accelerometerY_name", np.float32),
                ("accelerometerZ_name", np.float32),
            ]
        )
        ds_acc = self._h5.create_dataset(
            "accelerometer3_datasetName",
            shape=(0,),
            maxshape=(None,),
            dtype=acc_dtype,
            chunks=(4096,),
        )
        ds_acc.attrs["accelerometer3_datasetAttributeStartTime"] = np.int64(self.start_time_us)
        self._datasets["acc"] = ds_acc

        gyro_dtype = np.dtype(
            [
                ("timestamp_name", np.uint32),
                ("gyroscopeX_name", np.float32),
                ("gyroscopeY_name", np.float32),
                ("gyroscopeZ_name", np.float32),
            ]
        )
        ds_gyro = self._h5.create_dataset(
            "gyroscope3_datasetName",
            shape=(0,),
            maxshape=(None,),
            dtype=gyro_dtype,
            chunks=(4096,),
        )
        ds_gyro.attrs["gyroscope3_datasetAttributeStartTime"] = np.int64(self.start_time_us)
        self._datasets["gyro"] = ds_gyro

        metrics_dtype = np.dtype(
            [
                ("timestamp_sec", np.float64),
                ("relaxation", np.float32),
                ("focus", np.float32),
                ("involvement", np.float32),
                ("stress", np.float32),
                ("fatigue", np.float32),
                ("heart_rate", np.float32),
                ("stress_index", np.float32),
                ("quality", np.float32),
            ]
        )
        self._datasets["metrics"] = self._h5.create_dataset(
            "emotion_timeline",
            shape=(0,),
            maxshape=(None,),
            dtype=metrics_dtype,
            chunks=(256,),
        )

    def _append_rows(self, key: str, rows: List[Tuple]) -> None:
        if not rows:
            return
        ds = self._datasets[key]
        arr = np.array(rows, dtype=ds.dtype)
        old_len = ds.shape[0]
        ds.resize((old_len + len(arr),))
        ds[old_len:] = arr

    def append_eeg_batch(self, rows: List[Tuple]) -> None:
        if not rows:
            return
        self._eeg_buf.extend(rows)
        if len(self._eeg_buf) >= self.flush_threshold:
            self._append_rows("eeg", self._eeg_buf)
            self._eeg_buf.clear()

    def append_ppg_batch(self, rows: List[Tuple]) -> None:
        if not rows:
            return
        self._ppg_buf.extend(rows)
        if len(self._ppg_buf) >= self.flush_threshold // 2:
            self._append_rows("ppg", self._ppg_buf)
            self._ppg_buf.clear()

    def append_mems_batch(self, acc_rows: List[Tuple], gyro_rows: List[Tuple]) -> None:
        if not acc_rows:
            return
        self._acc_buf.extend(acc_rows)
        self._gyro_buf.extend(gyro_rows)
        if len(self._acc_buf) >= self.flush_threshold:
            self._append_rows("acc", self._acc_buf)
            self._append_rows("gyro", self._gyro_buf)
            self._acc_buf.clear()
            self._gyro_buf.clear()

    def append_emotion_point(
        self,
        timestamp_sec: float,
        vec: Dict[str, float],
        heart_rate: Optional[float],
        stress_index: Optional[float],
        quality: float,
    ) -> None:
        self._metrics_buf.append(
            (
                float(timestamp_sec),
                float(vec.get("relaxation", 0.0)),
                float(vec.get("focus", 0.0)),
                float(vec.get("involvement", 0.0)),
                float(vec.get("stress", 0.0)),
                float(vec.get("fatigue", 0.0)),
                float(heart_rate) if heart_rate is not None else np.nan,
                float(stress_index) if stress_index is not None else np.nan,
                float(quality),
            )
        )
        if len(self._metrics_buf) >= 128:
            self._append_rows("metrics", self._metrics_buf)
            self._metrics_buf.clear()

    def add_track_marker(self, marker: Dict[str, Any]) -> None:
        self._track_markers.append(marker)
        self._h5.attrs["track_markers_json"] = json.dumps(self._track_markers, ensure_ascii=False)

    def flush(self) -> None:
        self._append_rows("eeg", self._eeg_buf)
        self._eeg_buf.clear()
        self._append_rows("ppg", self._ppg_buf)
        self._ppg_buf.clear()
        self._append_rows("acc", self._acc_buf)
        self._acc_buf.clear()
        self._append_rows("gyro", self._gyro_buf)
        self._gyro_buf.clear()
        self._append_rows("metrics", self._metrics_buf)
        self._metrics_buf.clear()
        self._h5.attrs["track_markers_json"] = json.dumps(self._track_markers, ensure_ascii=False)
        self._h5.flush()

    def close(self) -> None:
        try:
            self.flush()
        finally:
            self._h5.close()


def load_session_hdf5(filepath: Path) -> Dict[str, Any]:
    """Читает HDF5-файл сессии (наш формат или штатный session_eeg.h5 Neiry)."""
    if h5py is None:
        raise RuntimeError("Установите h5py: pip install h5py")

    result: Dict[str, Any] = {
        "filepath": str(filepath),
        "session_id": filepath.stem,
        "start_time_us": 0,
        "channel_names": [],
        "track_markers": [],
        "emotion_timeline": None,
        "datasets": [],
    }

    with h5py.File(str(filepath), "r") as f:
        result["datasets"] = list(f.keys())
        if "session_id" in f.attrs:
            result["session_id"] = str(f.attrs["session_id"])
        if "start_time_us" in f.attrs:
            result["start_time_us"] = int(f.attrs["start_time_us"])
        if "channel_names" in f.attrs:
            raw = f.attrs["channel_names"]
            result["channel_names"] = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else str(raw))
        if "track_markers_json" in f.attrs:
            raw = f.attrs["track_markers_json"]
            result["track_markers"] = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else str(raw))

        for ds_name in f.keys():
            if ds_name.startswith("eeg") and ds_name.endswith("_datasetName"):
                ds = f[ds_name]
                for ak in ds.attrs.keys():
                    if ak.endswith("StartTime") and result["start_time_us"] == 0:
                        result["start_time_us"] = int(ds.attrs[ak])
                    elif ak.endswith("Attribute") and not result["channel_names"]:
                        raw_attr = ds.attrs[ak]
                        meta = json.loads(raw_attr.decode("utf-8") if isinstance(raw_attr, bytes) else str(raw_attr))
                        result["channel_names"] = meta.get("channelNames", [])

        if "emotion_timeline" in f:
            result["emotion_timeline"] = np.array(f["emotion_timeline"])

    return result
