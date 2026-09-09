import json
import sys
import traceback
from pathlib import Path

from PySide6.QtWidgets import QApplication

from src.ui.main_window import MainWindow


def load_settings() -> dict:
    project_root = Path(__file__).resolve().parent.parent
    settings_path = project_root / "config" / "settings.json"

    with settings_path.open("r", encoding="utf-8") as handle:
        settings = json.load(handle)

    settings["project_root"] = str(project_root)
    return settings


def main() -> None:
    settings = load_settings()
    app = QApplication(sys.argv)
    window = MainWindow(settings)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    try:
        main()
    except Exception:
        crash_log_path = Path(__file__).resolve().parent.parent / "startup_crash.log"
        with crash_log_path.open("w", encoding="utf-8") as handle:
            traceback.print_exc(file=handle)
        traceback.print_exc()
        print(f"\n[FATAL] Startup failed. Full traceback written to {crash_log_path}", file=sys.stderr)
        sys.exit(1)
