from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILES_DIR = ROOT / "config" / "game_profiles"

Frac = tuple[float, float, float, float]


@dataclass(frozen=True)
class GameProfile:
    id: str
    label: str
    require_hud_energy: bool
    hud_energy_stdev: float
    hud_energy_frac: tuple[Frac, ...]
    hud_mask_frac: tuple[Frac, ...]


def _as_frac(values: Any) -> Frac:
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError(f"expected 4-float rect, got {values!r}")
    return (float(values[0]), float(values[1]), float(values[2]), float(values[3]))


def load_game_profile(profile_id: str, profiles_dir: Path | None = None) -> GameProfile:
    directory = profiles_dir or DEFAULT_PROFILES_DIR
    path = directory / f"{profile_id}.json"
    if not path.is_file():
        available = sorted(p.stem for p in directory.glob("*.json")) if directory.is_dir() else []
        raise FileNotFoundError(f"unknown game profile {profile_id!r}; available={available}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return GameProfile(
        id=str(data.get("id") or profile_id),
        label=str(data.get("label") or profile_id),
        require_hud_energy=bool(data.get("require_hud_energy", True)),
        hud_energy_stdev=float(data.get("hud_energy_stdev", 12.0)),
        hud_energy_frac=tuple(_as_frac(item) for item in data.get("hud_energy_frac", [])),
        hud_mask_frac=tuple(_as_frac(item) for item in data.get("hud_mask_frac", [])),
    )


def list_game_profiles(profiles_dir: Path | None = None) -> list[GameProfile]:
    directory = profiles_dir or DEFAULT_PROFILES_DIR
    if not directory.is_dir():
        return []
    profiles = []
    for path in sorted(directory.glob("*.json")):
        profiles.append(load_game_profile(path.stem, directory))
    return profiles
