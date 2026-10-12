# Notebook guide

The notebooks are kept exactly as they were run during the boot camp
(April 2024), outputs included, and are numbered in reading order. Only the
file names changed when the repository was reorganised.

| # | Notebook | Author | What it does |
|---|---|---|---|
| 01 | `01_data_cleaning_eda_feature_engineering` | Sze Hong Kwong | Builds a 30-column table from the base, static, credit-bureau, person and previous-application tables; analyses correlations overall and by income type; engineers the credit-bureau risk midpoint and weighted days-past-due features. Finding: individual features correlate weakly with default; the bureau risk band reaches r ≈ 0.23 on the 3.5% of applicants who have one. |
| 02 | `02_xgboost` | Sze Hong Kwong | XGBoost with native missing-value and categorical handling; search over class weight (`scale_pos_weight`) and tree depth; weekly Gini plots and feature importances. |
| 03 | `03_xgboost_feature_ablation_and_baseline` | Sze Hong Kwong | XGBoost with and without the engineered features, and an `XGBRFClassifier` (random forest) baseline. |
| 04 | `04_mlp` | Brady Ali Medina | scikit-learn pipeline (median imputation with missing indicators, scaling, one-hot encoding) feeding a 100-unit MLP; ROC and precision-recall curves, permutation importance, weekly Gini. AUC 0.796 / 0.595 / 0.591 and stability 0.577 / 0.109 / 0.111 on train / validation / test. |
| 05 | `05_lightgbm` | Brady Ali Medina | Polars-based loading and aggregation, 70 features, LightGBM with early stopping; ROC curve, gain-based feature importance, tree plot, weekly Gini with trend lines for all three splits, submission file. AUC 0.792 / 0.782 / 0.778 and stability 0.562 / 0.534 / 0.521. |
| - | `archive/mlp_preliminary` | - | Earlier draft of notebook 04. |

## Running them

The notebooks read the Kaggle CSV files from a hard-coded location that you
need to change in the first cells:

* 01-03: `file_path = 'home-credit-credit-risk-model-stability/csv_files/train'`
* 04-05: Google Drive paths (`/content/drive/My Drive/credit/kaggle_credit/`),
  because they were run on Google Colab; the `drive.mount` cell can be deleted
  when running locally.

After `bash scripts/download_data.sh`, the path from this folder is
`../data/csv_files/`. Loading everything needs roughly 16 GB of RAM.

## Relation to `src/creditrisk`

The notebooks are self-contained and do not import the package. The package
collects the pieces worth reusing (table loading and joins, the engineered
features, the stability metric, random and time-based splits) as tested
functions, for the next iteration of the project.
