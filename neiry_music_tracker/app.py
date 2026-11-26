"""Точка входа в приложение Neiry Music Emotion Tracker."""
import sys
from PySide6.QtWidgets import QApplication

from config import ensure_directories
from ui.main_window import MainWindow


def main() -> int:
    ensure_directories()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
