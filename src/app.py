import json
from pathlib import Path

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
    app = MainWindow(settings)
    app.run()
