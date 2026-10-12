# Predicting loan default with models that stay stable over time

[![tests](https://github.com/Bra-AM/erdos-homecreditmodelstability/actions/workflows/tests.yml/badge.svg)](https://github.com/Bra-AM/erdos-homecreditmodelstability/actions/workflows/tests.yml)

**Which loan applicants are likely to default, and will the model that says so
still be right six months from now?**

Lenders retrain credit models rarely, so a model that scores well today but
degrades as the applicant population shifts is costly: it approves loans it
should decline, and declines people who would have repaid, often those with
little credit history. Home Credit, an international consumer-finance
provider, posed this as a
[Kaggle competition](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability)
scored on a metric that rewards accuracy *and* penalises decay over time. This
repository is our entry point into that problem: 1.5 million loan applications,
three model families, and the competition's stability metric.

Project for the [Erdos Institute](https://www.erdosinstitute.org/) Data Science
Boot Camp, Spring 2024, by **Brady Ali Medina** and **Sze Hong Kwong**.
[Project page](https://www.erdosinstitute.org/project-database/spring-2024/data-science-boot-camp/home-credit---credit-risk-model-stability-ii)
· [Executive summary (PDF)](reports/executive_summary_spring_2024.pdf)

## At a glance

| | |
|---|---|
| **Task** | Binary classification: will this application default? |
| **Data** | 1,526,659 applications over 92 weeks (Jan 2019 - Oct 2020), ~3.1% defaults, spread over relational tables keyed by `case_id` |
| **Metric** | Gini stability: mean weekly Gini, minus penalties for a falling trend and for week-to-week volatility |
| **Models** | XGBoost, a multilayer perceptron (scikit-learn), LightGBM; random-forest baseline |
| **Best result** | LightGBM: test AUC 0.778, stability 0.521, with a train-test gap of only 0.014 AUC |
| **Stakeholders** | Consumer lenders; applicants, particularly those with thin credit files |

## The stability metric

For each week *w* of loan decisions, compute the Gini coefficient
`gini_w = 2·AUC_w − 1`. Fit a line `gini_w ≈ a·w + b` through the weekly
values. Then

```
stability = mean(gini_w) + 88 · min(0, a) − 0.5 · std(residuals)
```

A model is rewarded for ranking defaulters above non-defaulters, penalised if
that ability trends downward (`a < 0`), and penalised if it is erratic from
week to week. The implementation, with unit tests against the competition's
reference code, is in [`src/creditrisk/metrics.py`](src/creditrisk/metrics.py).

## Approach

1. **Data preparation** ([notebook 01](notebooks/01_data_cleaning_eda_feature_engineering.ipynb)).
   Join the base table with internal application data, credit-bureau data,
   previous applications and applicant details. Tables with several rows per
   application are aggregated to one row first.
2. **Exploratory analysis.** Individual features are only weakly correlated
   with default (|r| < 0.01 for income, annuity and debt). The strongest single
   signal is the credit bureau's own risk band (r ≈ 0.23), but only 3.5% of
   applicants have one.
3. **Feature engineering.** The bureau's text risk band (`'6% - 8%'`) becomes a
   numeric midpoint plus a has-assessment indicator; days-past-due columns over
   different horizons are combined into weighted summaries.
4. **Models.**
   * [XGBoost](notebooks/02_xgboost.ipynb) on ~30 selected features, with class
     weighting for the 31:1 imbalance, and a
     [feature ablation with a random-forest baseline](notebooks/03_xgboost_feature_ablation_and_baseline.ipynb).
   * [MLP](notebooks/04_mlp.ipynb) with median imputation, missing-value
     indicators, scaling and one-hot encoding in a scikit-learn pipeline.
   * [LightGBM](notebooks/05_lightgbm.ipynb) on 70 features (all amount,
     days-past-due and masked-category columns), with early stopping.
5. **Evaluation.** 60/20/20 train/validation/test split; AUC and the stability
   metric on each part; weekly Gini plotted over time.

## Results

| | MLP | LightGBM |
|---|---|---|
| AUC, train | 0.796 | 0.792 |
| AUC, validation | 0.595 | 0.782 |
| AUC, test | 0.591 | **0.778** |
| Stability, train | 0.577 | 0.562 |
| Stability, validation | 0.109 | 0.534 |
| Stability, test | 0.111 | **0.521** |

<sub>Outputs saved in notebooks 04 and 05. The XGBoost notebooks (02, 03) report their own experiments, including a class-weight and depth search and a feature ablation; they use a different feature set and scoring setup, so they are not tabulated side by side here.</sub>

LightGBM generalises well: its stability score moves from 0.562 on training
data to 0.521 on held-out data, with a train-test AUC gap of 0.014.

![Weekly Gini of the LightGBM model](reports/figures/lightgbm_weekly_gini.png)

*LightGBM, Gini per week with fitted trend lines.* Train, validation and test
track each other, and the trend is upward, so the falling-rate penalty is
zero. Volatility rises sharply after about week 60 (March 2020, the onset of
COVID-19), which is where the residual-volatility penalty comes from.

![Weekly Gini of the MLP](reports/figures/mlp_weekly_gini.png)

*MLP, Gini per week.* The neural network fits the training weeks far better
than held-out applications. It was given the decision date, month and week as
inputs, which lets it learn period-specific default rates that do not
transfer; boosted trees on the credit features proved the more robust choice.

The most important LightGBM features by gain are days-past-due history
(`avgdpdtolclosure24`), total settled amount, the reason the client's last
application was rejected, and the price of the credit
([chart](reports/figures/lightgbm_feature_importance.png)).

## Next steps

- [ ] Compare all model families on one shared feature table and one shared
      evaluation function (`creditrisk.metrics`).
- [ ] Evaluate with a time-based split: train on the first 60% of weeks,
      validate on the next 20%, test on the last 20%
      (`creditrisk.splits.time_based_split`), since the splits so far are
      random and every part contains every week.
- [ ] Remove date and week columns from model inputs and retrain the MLP.
- [ ] Use more of the depth-1 and depth-2 tables through richer aggregations.
- [ ] Exclude protected attributes and audit for proxies before any
      production use.
- [ ] Monitor drift with a population stability index per feature and per week.
- [ ] Explain the final model with SHAP values.

## Repository structure

```
.
├── notebooks/                # Analysis, in reading order (see notebooks/README.md)
│   ├── 01_data_cleaning_eda_feature_engineering.ipynb
│   ├── 02_xgboost.ipynb
│   ├── 03_xgboost_feature_ablation_and_baseline.ipynb
│   ├── 04_mlp.ipynb
│   ├── 05_lightgbm.ipynb
│   └── archive/              # Superseded drafts
├── src/creditrisk/           # Reusable, tested code
│   ├── data.py               #   loading tables, aggregating, joining on case_id
│   ├── features.py           #   risk-band parsing, weighted days-past-due features
│   ├── metrics.py            #   Gini and the stability metric
│   └── splits.py             #   random and time-based splits
├── tests/                    # pytest suite on synthetic data (runs in CI)
├── reports/                  # Executive summary
├── data/                     # Not in git; see data/README.md
├── scripts/download_data.sh  # Fetches the data with the Kaggle API
├── requirements.txt
└── pyproject.toml
```

## Getting started

```bash
git clone https://github.com/Bra-AM/erdos-homecreditmodelstability.git
cd erdos-homecreditmodelstability
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

pytest                           # tests need no data
bash scripts/download_data.sh    # ~4 GB; needs a Kaggle API token, see data/README.md
```

Using the package:

```python
from creditrisk import metrics, splits

train, valid, test = splits.time_based_split(applications)
test = test.assign(score=model.predict_proba(test[features])[:, 1])
metrics.stability_components(metrics.weekly_gini(test))
# {'mean_gini': ..., 'slope': ..., 'residual_std': ..., 'stability': ...}
```

## Team and credits

| | Contribution |
|---|---|
| [Brady Ali Medina](https://www.linkedin.com/in/brady-ali-487864133/) | LightGBM and MLP modelling and evaluation (notebooks 04, 05), stability-over-time analysis; later the `creditrisk` package, tests and documentation |
| Sze Hong Kwong | Data cleaning, exploratory analysis and feature engineering (notebook 01), XGBoost modelling and baseline (notebooks 02, 03) |

The LightGBM notebook builds on the competition's public starter notebook
(data-type handling, column selection, baseline parameters and the reference
implementation of the stability metric). Data © Home Credit, provided through
Kaggle under the competition rules; it is not redistributed here.
