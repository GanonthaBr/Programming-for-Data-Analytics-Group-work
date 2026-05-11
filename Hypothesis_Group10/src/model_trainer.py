"""Model training module with shared preprocessing pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# create ModelResult dataclass to store results of model training
@dataclass
class ModelResult:
    name: str
    train_r2: float
    test_r2: float
    test_rmse: float
    test_mae: float
    cv_r2_mean: float
    cv_r2_std: float
    best_params: Dict[str, object]

# create ModelTrainer class to handle model training and evaluation
@dataclass
class ModelTrainer:
    """Train multiple supervised learners with shared preprocessing."""

    feature_names: List[str]
    test_size: float = 0.2
    random_state: int = 42

    def __post_init__(self) -> None:
        # Build a shared numeric preprocessing pipeline (median fill + scaling) reused by every model.
        self.preprocess = ColumnTransformer(
            [
                (
                    "num",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]),
                    self.feature_names,
                )
            ]
        )
        # Define each estimator along with the hyperparameter grid we plan to search.
        self.estimators = {
            "LinearRegression": (LinearRegression(), {}),
            "RandomForest": (
                RandomForestRegressor(random_state=self.random_state),
                {
                    "model__n_estimators": [200, 400],
                    "model__max_depth": [5, 10, None],
                    "model__min_samples_leaf": [1, 3, 5],
                },
            ),
            "GradientBoosting": (
                GradientBoostingRegressor(random_state=self.random_state),
                {
                    "model__n_estimators": [200, 400],
                    "model__learning_rate": [0.05, 0.1],
                    "model__max_depth": [2, 3],
                },
            ),
        }
        self.fitted_models: Dict[str, Pipeline] = {}

    def train(self, X: pd.DataFrame, y: pd.Series) -> List[ModelResult]:
        # Split data once so all models see identical train/test partitions.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        summaries: List[ModelResult] = []
        # Iterate through each configured estimator and fit with its respective grid (if any).
        for name, (estimator, grid) in self.estimators.items():
            pipeline = Pipeline([("preprocess", self.preprocess), ("model", estimator)])
            if grid:
                # Run cross-validated grid search when a hyperparameter sweep is defined.
                search = GridSearchCV(pipeline, grid, cv=5, n_jobs=-1, scoring="r2")
                search.fit(X_train, y_train)
                best_model = search.best_estimator_
                best_params = search.best_params_
            else:
                # Fall back to fitting the baseline pipeline directly when no grid is provided.
                best_model = pipeline.fit(X_train, y_train)
                best_params = {}
            # Collect standard regression metrics for both train and held-out data.
            y_pred_train = best_model.predict(X_train)
            y_pred_test = best_model.predict(X_test)
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
            mae = mean_absolute_error(y_test, y_pred_test)
            # Compute additional CV scores so we can report stability across folds.
            cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring="r2")
            summaries.append(
                ModelResult(
                    name=name,
                    train_r2=float(train_r2),
                    test_r2=float(test_r2),
                    test_rmse=rmse,
                    test_mae=float(mae),
                    cv_r2_mean=float(cv_scores.mean()),
                    cv_r2_std=float(cv_scores.std()),
                    best_params=best_params,
                )
            )
            self.fitted_models[name] = best_model
        return summaries

    def extract_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """Return feature importances or coefficients by model."""
        # Walk through the fitted models and expose whatever importance vector they provide.
        results: Dict[str, Dict[str, float]] = {}
        for name, model in self.fitted_models.items():
            estimator = model.named_steps["model"]
            if hasattr(estimator, "feature_importances_"):
                values = estimator.feature_importances_
            elif hasattr(estimator, "coef_"):
                values = estimator.coef_
            else:
                continue
            results[name] = {
                feature: float(weight)
                for feature, weight in zip(self.feature_names, values)
            }
        return results
