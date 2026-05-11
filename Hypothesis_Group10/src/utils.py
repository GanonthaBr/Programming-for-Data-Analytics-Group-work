"""Utility helpers for file paths, logging, and serialization."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any, Dict


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT_DIR / "SYB67_328_202411_Intentional homicides and other crimes.csv"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"


def ensure_directories() -> None:
    """Create required output directories if missing."""
    for path in (RESULTS_DIR, FIGURES_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)


@dataclass
class ExperimentPaths:
    """Collect common output locations for reproducibility."""

    root: Path = ROOT_DIR
    data_file: Path = DATA_FILE
    results: Path = RESULTS_DIR
    figures: Path = FIGURES_DIR
    models: Path = MODELS_DIR


def save_json(payload: Dict[str, Any], destination: Path) -> None:
    """Persist dictionaries with UTF-8 encoding for downstream reporting."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
