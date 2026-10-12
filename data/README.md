# Data

The data is **not** stored in this repository. It belongs to the Kaggle
competition [Home Credit - Credit Risk Model Stability](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability)
and may only be downloaded by accepting the competition rules.

## Download

1. Create a Kaggle account and accept the rules on the competition page.
2. Create an API token (Kaggle > Settings > API > *Create New Token*) and
   save `kaggle.json` to `~/.kaggle/` (Windows: `C:\Users\<you>\.kaggle\`).
3. From the repository root:

   ```bash
   pip install kaggle
   bash scripts/download_data.sh
   ```

The files unpack to `data/csv_files/train/` and `data/csv_files/test/`
(about 4 GB of CSV, 1,526,659 training applications).

## Layout

| Table | Depth | Content |
|---|---|---|
| `base` | 0 | `case_id`, decision date, `WEEK_NUM`, `target` (1 = default) |
| `static_0_*` | 0 | Internal application data: amounts, instalments, days past due |
| `static_cb_0` | 0 | External credit-bureau summary, including a risk band |
| `person_1` | 1 | People attached to the application; `num_group1 == 0` is the applicant |
| `applprev_1_*` | 1 | Previous applications of the same client |
| `credit_bureau_*` | 1-2 | Credit-bureau contract and payment history |

Depth 1 and 2 tables have several rows per application and are aggregated
to one row per `case_id` before joining (see `src/creditrisk/data.py`).
