# WP1 Probabilistic Forecasting Conference Article Repository

This repository contains the code, article files, figures, and CSV outputs for the WP1 conference article on probabilistic forecasting of Western Cape school electricity demand.

## Repository structure

```text
data/
  schools_panel.csv.gz
  wp1_rb_fold_overall.csv
  wp1_rb_fold_per_school.csv
  wp1_rb_overall_summary.csv
  wp1_rb_historical_quantile_fold_overall.csv
  wp1_rb_historical_quantile_overall_summary.csv

src/
  prepare_schools_panel.py
  wp1_probabilistic_rolling_backtest.py
  run_rolling_historical_quantile_baseline.py

notebooks/
  wp1_baselines_and_significance.ipynb

paper/
  wp1_conference_article_adjusted_final.tex
  wp1_conference_article_adjusted_final.pdf
  references_revised_final.bib
  figures/

README.md
requirements.txt
```

The panel dataset is compressed as `data/schools_panel.csv.gz` to keep the repository package smaller. Pandas can read this file directly with `pd.read_csv("data/schools_panel.csv.gz")`. If a script expects `data/schools_panel.csv`, decompress it first:

```bash
gunzip -k data/schools_panel.csv.gz
```

## Main reproducibility steps

### 1. Prepare the panel dataset

If starting from individual school CSV files in `data/schools/`, run:

```bash
python src/prepare_schools_panel.py
```

This creates `data/schools_panel.csv`.

### 2. Run the main rolling-origin WP1 benchmark

```bash
python src/wp1_probabilistic_rolling_backtest.py
```

This produces:

```text
data/wp1_rb_fold_overall.csv
data/wp1_rb_fold_per_school.csv
data/wp1_rb_overall_summary.csv
```

It evaluates seasonal naive baselines, per-school quantile GB, pooled quantile GB, and the two-stage asymmetric calibration variant under 12 rolling-origin folds.

### 3. Run the rolling-origin historical-quantile baseline

To reproduce the historical-quantile row used in the article, use the same fold origins as the main rolling-origin benchmark:

```bash
python src/run_rolling_historical_quantile_baseline.py \
  --input data/schools_panel.csv \
  --timestamp timestamp \
  --school school_id \
  --target y \
  --origins 2023-05-30,2023-06-13,2023-06-27,2023-07-11,2023-07-25,2023-08-08,2023-08-22,2023-09-05,2023-09-19,2023-10-03,2023-10-17,2023-10-31 \
  --horizon-days 14 \
  --out-dir data
```

This produces:

```text
data/wp1_rb_historical_quantile_fold_overall.csv
data/wp1_rb_historical_quantile_overall_summary.csv
```

### 4. Significance and supporting analysis

The notebook `notebooks/wp1_baselines_and_significance.ipynb` contains supporting analysis for baselines and significance comparisons.

## Notes

- The conference article is based on rolling-origin evaluation, not the older fixed 80/20 chronological split.
- Weather and load-shedding variables are intentionally excluded from this WP1 benchmark and are discussed as future work.
- The two-stage calibration method is included as a diagnostic post-processing variant rather than as a separate forecasting architecture.
