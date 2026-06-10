import json
from pathlib import Path

_GUI_CONFIG_PATH = Path(__file__).parent.parent / "backend" / "gui_config.json"


def load_gui_config() -> dict:
    if _GUI_CONFIG_PATH.exists():
        with open(_GUI_CONFIG_PATH) as f:
            return json.load(f)
    return {"language": "en", "theme": "light"}


def save_gui_config(config: dict) -> None:
    with open(_GUI_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)
