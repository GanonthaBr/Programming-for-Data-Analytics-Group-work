"""Feature engineering routines extracted from EDA insights."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


TARGET_COLUMN = "intentional_homicide_rates_per_100000"

# create FeatureEngineer class to handle feature engineering tasks
@dataclass
class FeatureEngineer:
    """Create derived variables and filtered modeling samples."""

    base_features: List[str] = field(
        default_factory=lambda: [
            "assault_rate_per_100000_population",
            "kidnapping_at_the_national_level_rate_per_100000",
            "total_sexual_violence_at_the_national_level_rate_per_100000",
            "theft_at_the_national_level_rate_per_100000_population",
            "percentage_of_male_and_female_intentional_homicide_victims_male",
            "percentage_of_male_and_female_intentional_homicide_victims_female",
            "year",
        ]
    )
    drop_threshold: float = 0.95
    region_key: str = "region"
    winsor_limits: Tuple[float, float] = (0.01, 0.99)
    missing_actions: Dict[str, List[str]] = field(init=False, default_factory=dict)
    outlier_caps: Dict[str, Dict[str, float]] = field(init=False, default_factory=dict)

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Append engineered fields the hypotheses rely on."""
        # Create  gender and violent-crime indicators that the hypotheses reference.
        engineered = df.copy()
        engineered["male_share_gap"] = (
            engineered["percentage_of_male_and_female_intentional_homicide_victims_male"]
            - engineered["percentage_of_male_and_female_intentional_homicide_victims_female"]
        )
        engineered["violent_crime_combo"] = (
            engineered["assault_rate_per_100000_population"].fillna(0)
            + engineered["total_sexual_violence_at_the_national_level_rate_per_100000"].fillna(0)
            + engineered["kidnapping_at_the_national_level_rate_per_100000"].fillna(0)
        )
        engineered["homicide_rate_log1p"] = (
            engineered[TARGET_COLUMN] + 1
        ).apply(lambda x: pd.NA if pd.isna(x) else float(np.log1p(x)))
        return engineered

    def handle_missingness(self, df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        """Impute or drop columns depending on their missingness rate."""
        # Drop columns that exceed the missingness threshold; otherwise fill gaps via region/global medians.
        working = df.copy()
        dropped: List[str] = []
        imputed: List[str] = []
        for col in columns:
            if col not in working.columns:
                continue
            missing_rate = working[col].isna().mean()
            if missing_rate > self.drop_threshold:
                working = working.drop(columns=col)
                dropped.append(col)
                continue
            imputed.append(col)
            working[col] = working.groupby(self.region_key)[col].transform(
                lambda series: series.fillna(series.median())
            )
            working[col] = working[col].fillna(working[col].median())
        actions = {"dropped_columns": dropped, "imputed_columns": imputed}
        return working, actions

    def apply_winsorization(self, df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
        """Clip extreme values using quantile-based caps to address outliers."""
        # Cap each numeric column at the requested quantiles so rare spikes do not dominate the models.
        caps: Dict[str, Dict[str, float]] = {}
        adjusted = df.copy()
        for col in columns:
            if col not in adjusted.columns:
                continue
            lower = adjusted[col].quantile(self.winsor_limits[0])
            upper = adjusted[col].quantile(self.winsor_limits[1])
            if pd.isna(lower) or pd.isna(upper):
                continue
            adjusted[col] = adjusted[col].clip(lower=lower, upper=upper)
            caps[col] = {"lower": float(lower), "upper": float(upper)}
        return adjusted, caps

    def build_model_frame(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Return X/y with rows that have the target available."""
        # Orchestrate the full feature pipeline: engineer fields, clean missing values, winsorize, and split X/y.
        extended = self.add_features(df)
        feature_cols = self.base_features + ["male_share_gap", "violent_crime_combo"]
        available = extended.dropna(subset=[TARGET_COLUMN])
        prepared, actions = self.handle_missingness(available, feature_cols)
        final_features = [col for col in feature_cols if col in prepared.columns]
        prepared, caps = self.apply_winsorization(prepared, final_features)
        X = prepared[final_features]
        y = prepared[TARGET_COLUMN]
        actions["winsorized_columns"] = list(caps.keys())
        self.missing_actions = actions
        self.outlier_caps = caps
        return X, y
