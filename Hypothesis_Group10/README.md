# Analytics Assignment II – Predictive Modeling & Hypothesis Testing

This repo follows the rubric requirements for 04-638 Programming for Data Analytics Assignment II. It contains modular Python code, a notebook-driven workflow, and reproducible outputs for the UNODC homicide dataset (233 regions, 2000‑2022). Three supervised-learning hypotheses are evaluated: (H1) assault intensity drives homicide, (H2) higher male victim share signals higher homicide burden, and (H3) theft rates are inversely related to homicide.

## Project layout
```
.
├── src/
│   ├── data_processor.py
│   ├── feature_engineer.py
│   ├── model_trainer.py
│   ├── model_evaluator.py
│   ├── utils.py
│   └── __init__.py
├── notebooks/
│   └── main_analysis.ipynb
├── results/
│   ├── figures/
│   └── models/
├── SYB67_328_202411_Intentional homicides and other crimes.csv
├── requirements.txt
└── README.md
```

## Quick start
1. `python -m venv .venv && .venv\Scripts\activate`
2. `pip install -r requirements.txt`
3. Launch JupyterLab or VS Code and open `notebooks/main_analysis.ipynb`.
4. Run all cells to regenerate cleaned data, tabulated hypothesis evidence, models, evaluation tables, and plots saved under `results/`.

Key outputs:
- `results/model_metrics.csv` – consolidated train/test/CV metrics for Linear Regression, Random Forest, and Gradient Boosting.
- `results/figures/feature_importance_GradientBoosting.png` & `actual_vs_pred_GradientBoosting.png` – champion-model diagnostics (test R² ≈ 0.88, RMSE ≈ 4.66).
- `results/summary.json` – overall dataset and modeling summary mirrored in notebook Table 7.

## Deliverables checklist
- **Hypotheses & rationale**: Section 0 of the notebook tabulates H1–H3 with null/alternative statements, evidence criteria, and regression framing. Section 6.2 records accept/reject decisions with quantitative evidence.
- **Modular scripts**: `src/` implements OOP-based components (data processor, feature engineer, trainer, evaluator, utilities) with docstrings and single-responsibility methods.
- **Notebook**: `notebooks/main_analysis.ipynb` orchestrates the workflow, surfaces visuals (EDA trend, model comparison charts, hypothesis scatterplots, feature importance), and documents insights beyond EDA.
- **Results**: Metrics/figures persist under `results/` for reuse in the PDF report (`report_outline.md` lists the recommended narrative order).

## Testing & linting
Optional tooling (install manually if desired):
- `pytest` for unit tests.
- `ruff` or `flake8` for style enforcement.

## Notes on data preparation & modeling
- Missing values: numeric features are imputed with region medians then global medians; features exceeding 95% missingness are automatically dropped (none in this iteration). A 1st/99th percentile winsorization step caps extreme values to retain scarce country-year records.
- Feature engineering: `male_share_gap`, `violent_crime_combo`, and log-scale diagnostic columns capture the EDA-informed relationships needed for hypotheses.
- Modeling: all learners share a preprocessing pipeline (median imputer + scaler). Random Forest and Gradient Boosting run GridSearchCV over estimator depth/size; Gradient Boosting currently performs best on the test split.
