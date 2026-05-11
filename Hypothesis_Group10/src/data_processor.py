"""Data ingestion and cleaning helpers for the homicide dataset."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd


RAW_COLUMNS = ["RegionCode", "Region", "Year", "Series", "Value", "Footnotes", "Source"]


@dataclass
class DataProcessor:
    """Load the wide UNODC crime table and return modeling-ready frames."""

    data_path: Path

    def load_raw(self) -> pd.DataFrame:
        """Read the CSV after skipping the descriptive header rows."""
        df = pd.read_csv(self.data_path, header=None, skiprows=2)
        df.columns = RAW_COLUMNS
        return df

    def pivot_wide(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert the long-format table into a Region/Year wide matrix."""
        df = df.copy()
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
        wide = (
            df.pivot_table(index=["RegionCode", "Region", "Year"], columns="Series", values="Value")
            .reset_index()
        )
        wide.columns = [
            col.replace(" ", "_").replace(",", "").replace("/", "_").lower()
            for col in wide.columns
        ]
        return wide.sort_values(["region", "year"]).reset_index(drop=True)

    def summarize_missingness(self, df: pd.DataFrame, columns: List[str]) -> pd.Series:
        """Return missing-value share for quick reporting."""
        return df[columns].isna().mean().sort_values(ascending=False)
