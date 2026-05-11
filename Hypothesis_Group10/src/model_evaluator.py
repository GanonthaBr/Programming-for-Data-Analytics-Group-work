"""Evaluation helpers for comparing supervised models."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score

from .model_trainer import ModelResult, ModelTrainer

# create ModelEvaluator class to handle model evaluation and visualization
@dataclass
class ModelEvaluator:
    """Create comparison tables and figures aligned with the rubric."""

    trainer: ModelTrainer
    results: List[ModelResult]

    def to_dataframe(self) -> pd.DataFrame:
        # Turn the collected ModelResult objects into a tabular DataFrame for downstream use.
        return pd.DataFrame([r.__dict__ for r in self.results])

    def save_results(self, destination: Path) -> None:
        # Persist evaluation metrics to disk, creating the parent folders when missing.
        df = self.to_dataframe()
        destination.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(destination, index=False)

    def plot_feature_importance(self, model_name: str, destination: Path) -> None:
        # Render a horizontal bar chart of feature importances for the requested fitted model.
        importances = self.trainer.extract_feature_importance().get(model_name)
        if not importances:
            return
        fig, ax = plt.subplots(figsize=(8, 5))
        (pd.Series(importances).sort_values().plot.barh(ax=ax, color="#2a9d8f"))
        ax.set_title(f"Feature importance - {model_name}")
        ax.set_xlabel("Relative importance")
        fig.tight_layout()
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=300)
        plt.close(fig)

    def plot_predictions(self, model_name: str, X: pd.DataFrame, y: pd.Series, destination: Path) -> None:
        # Compare actual vs predicted values with a parity line to check regression calibration.
        model = self.trainer.fitted_models.get(model_name)
        if model is None:
            return
        preds = model.predict(X)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y, preds, alpha=0.6, edgecolor="none")
        line_min, line_max = y.min(), y.max()
        ax.plot([line_min, line_max], [line_min, line_max], color="black", linestyle="--")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"Actual vs predicted ({model_name})\nR^2={r2_score(y, preds):.2f}")
        fig.tight_layout()
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=300)
        plt.close(fig)

    def plot_all_predictions(self, X: pd.DataFrame, y: pd.Series, figures_dir: Path) -> None:
        """Generate and save actual-vs-predicted charts for every fitted model.

        The method iterates over `trainer.fitted_models` and calls
        `plot_predictions` for each model, saving PNG files into `figures_dir`.
        """
        figures_dir.mkdir(parents=True, exist_ok=True)
        for model_name in self.trainer.fitted_models.keys():
            dest = figures_dir / f"actual_vs_pred_{model_name}.png"
            # Reuse existing plotting routine to keep styling consistent.
            self.plot_predictions(model_name, X, y, dest)

    def plot_comparison_grid(
        self,
        model_names: List[str],
        X: pd.DataFrame,
        y: pd.Series,
        destination: Path,
    ) -> None:
        """Render a single figure with side-by-side actual-vs-predicted plots.

        The provided `model_names` sequence controls subplot order; models that
        have not been fitted are skipped.
        """
        fitted = [name for name in model_names if name in self.trainer.fitted_models]
        if not fitted:
            return
        n_cols = len(fitted)
        fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))
        if n_cols == 1:
            axes = [axes]
        for ax, name in zip(axes, fitted):
            model = self.trainer.fitted_models[name]
            preds = model.predict(X)
            ax.scatter(y, preds, alpha=0.5, edgecolor="none")
            line_min, line_max = y.min(), y.max()
            ax.plot([line_min, line_max], [line_min, line_max], color="black", linestyle="--")
            ax.set_title(f"{name} (R^2={r2_score(y, preds):.2f})")
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
        fig.tight_layout()
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=300)
        plt.close(fig)
