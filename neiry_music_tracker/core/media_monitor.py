"""Монитор текущего трека через Windows SMTC (Spotify, браузер, Яндекс Музыка и любые плееры)."""
from __future__ import annotations

import asyncio
import sys
from typing import Optional

from core.models import TrackInfo

_WINRT_AVAILABLE = False
if sys.platform == "win32":
    try:
        from winrt.windows.media.control import (
            GlobalSystemMediaTransportControlsSessionManager as MediaManager,
            GlobalSystemMediaTransportControlsSessionPlaybackStatus as PlaybackStatus,
        )
        _WINRT_AVAILABLE = True
    except ImportError:
        _WINRT_AVAILABLE = False


class WindowsMediaMonitor:
    """
    Читает активный трек и статус воспроизведения из Windows SMTC.
    Кэширует MediaManager, чтобы не пересоздавать WinRT/COM-объект на каждом опросе.
    """

    def __init__(self, prefer_spotify: bool = True):
        self.prefer_spotify = prefer_spotify
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._manager = None
        self._manual_track: Optional[TrackInfo] = None

    @property
    def is_smtc_available(self) -> bool:
        return _WINRT_AVAILABLE

    def set_manual_track(self, track: Optional[TrackInfo]) -> None:
        """Позволяет задать трек вручную (для отладки или если SMTC недоступен)."""
        self._manual_track = track

    def poll_current_track(self) -> Optional[TrackInfo]:
        """Синхронный опрос текущего трека (вызывается по QTimer раз в 1.5 сек)."""
        if self._manual_track is not None:
            return self._manual_track
        if not _WINRT_AVAILABLE:
            return None

        try:
            if self._loop is None or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
            return self._loop.run_until_complete(self._fetch_smtc_async())
        except Exception:
            self._manager = None
            return None

    async def _fetch_smtc_async(self) -> Optional[TrackInfo]:
        if self._manager is None:
            self._manager = await MediaManager.request_async()
        manager = self._manager
        sessions = manager.get_sessions()
        if not sessions:
            session = manager.get_current_session()
            if session is None:
                return None
            sessions = [session]

        chosen = None
        if self.prefer_spotify:
            for s in sessions:
                app_id = (s.source_app_user_model_id or "").lower()
                if "spotify" in app_id:
                    chosen = s
                    break
        if chosen is None:
            chosen = manager.get_current_session() or sessions[0]
        if chosen is None:
            return None

        info = await chosen.try_get_media_properties_async()
        playback = chosen.get_playback_info()
        timeline = chosen.get_timeline_properties()

        title = (info.title or "").strip() if info else ""
        artist = (info.artist or "").strip() if info else ""
        album = (info.album_title or "").strip() if info else ""
        if not title and not artist:
            return None

        is_playing = False
        if playback is not None:
            is_playing = playback.playback_status == PlaybackStatus.PLAYING

        pos_sec = 0.0
        dur_sec = 0.0
        if timeline is not None:
            pos = timeline.position
            end = timeline.end_time
            pos_sec = pos.total_seconds() if hasattr(pos, "total_seconds") else getattr(pos, "duration", 0) / 1e7
            dur_sec = end.total_seconds() if hasattr(end, "total_seconds") else getattr(end, "duration", 0) / 1e7

        return TrackInfo(
            title=title,
            artist=artist,
            album=album,
            source_app=chosen.source_app_user_model_id or "Windows SMTC",
            is_playing=is_playing,
            position_sec=max(0.0, float(pos_sec)),
            duration_sec=max(0.0, float(dur_sec)),
        )

    def close(self) -> None:
        self._manager = None
        if self._loop is not None and not self._loop.is_closed():
            self._loop.close()
