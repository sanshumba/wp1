#!/usr/bin/env python3
"""
Rolling-origin historical-quantile baseline for half-hourly school electricity demand.

This script computes a simple probabilistic time-of-week baseline under the same rolling-origin
protocol used by the WP1 models. It uses only observations prior to each forecast origin.

Expected input columns (configurable by CLI):
  - school identifier: school_id
  - timestamp: timestamp
  - target: y  (kWh per 30-minute interval)

Baseline:
  For each fold, estimate empirical quantiles by (school_id, day_of_week, slot_of_day)
  from the training window, then predict those quantiles for the test window.
  Missing school/time-of-week cells fall back to pooled time-of-week quantiles, then global quantiles.

Outputs:
  - wp1_rb_historical_quantile_fold_overall.csv
  - wp1_rb_historical_quantile_overall_summary.csv
  - wp1_rb_historical_quantile_predictions.csv (optional, can be large)

Example:
  python run_rolling_historical_quantile_baseline.py \
      --input data/model_ready_panel.csv \
      --timestamp timestamp --school school_id --target y \
      --origins 2023-05-30,2023-06-13,2023-06-27,2023-07-11,2023-07-25,2023-08-08,2023-08-22,2023-09-05,2023-09-19,2023-10-03,2023-10-17,2023-10-31 \
      --horizon-days 14 \
      --out-dir data
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
QCOLS = {q: f"q{int(q*100):02d}" for q in QUANTILES}


def pinball_loss(y: np.ndarray, qhat: np.ndarray, q: float) -> np.ndarray:
    err = y - qhat
    return np.maximum(q * err, (q - 1.0) * err)


def make_time_keys(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out[ts_col])
    out["_dow"] = ts.dt.dayofweek.astype(int)
    # slot 0..47 for half-hour data. If timestamps are hourly, this still works (even slots only).
    out["_slot"] = (ts.dt.hour * 2 + (ts.dt.minute // 30)).astype(int)
    return out


def group_quantiles(train: pd.DataFrame, keys: list[str], target: str) -> pd.DataFrame:
    grouped = train.groupby(keys, observed=True)[target].quantile(QUANTILES).unstack()
    grouped.columns = [QCOLS[q] for q in QUANTILES]
    grouped = grouped.reset_index()
    return grouped


def add_predictions(
    test: pd.DataFrame,
    train: pd.DataFrame,
    school_col: str,
    target_col: str,
) -> pd.DataFrame:
    # Most specific: school + time-of-week
    by_school_tow = group_quantiles(train, [school_col, "_dow", "_slot"], target_col)
    pred = test.merge(by_school_tow, on=[school_col, "_dow", "_slot"], how="left")

    # Fallback 1: pooled time-of-week
    missing = pred[QCOLS[0.50]].isna()
    if missing.any():
        pooled_tow = group_quantiles(train, ["_dow", "_slot"], target_col)
        fallback = test.loc[missing, ["_dow", "_slot"]].merge(pooled_tow, on=["_dow", "_slot"], how="left")
        for q, col in QCOLS.items():
            pred.loc[missing, col] = fallback[col].to_numpy()

    # Fallback 2: global quantiles
    missing = pred[QCOLS[0.50]].isna()
    if missing.any():
        global_q = train[target_col].quantile(QUANTILES)
        for q, col in QCOLS.items():
            pred.loc[missing, col] = float(global_q.loc[q])

    # Enforce monotonicity after fallbacks.
    qmatrix = pred[[QCOLS[q] for q in QUANTILES]].to_numpy(dtype=float)
    qmatrix = np.maximum.accumulate(qmatrix, axis=1)
    pred[[QCOLS[q] for q in QUANTILES]] = qmatrix
    return pred


def evaluate(pred: pd.DataFrame, target_col: str) -> dict[str, float]:
    y = pred[target_col].to_numpy(dtype=float)
    out = {}
    out["mae_p50"] = float(np.mean(np.abs(y - pred[QCOLS[0.50]].to_numpy(dtype=float))))
    losses = []
    for q in QUANTILES:
        losses.append(pinball_loss(y, pred[QCOLS[q]].to_numpy(dtype=float), q))
    out["pinball_mean"] = float(np.mean(np.vstack(losses)))
    out["cov50_p25_p75"] = float(np.mean((y >= pred[QCOLS[0.25]]) & (y <= pred[QCOLS[0.75]])))
    out["cov80_p10_p90"] = float(np.mean((y >= pred[QCOLS[0.10]]) & (y <= pred[QCOLS[0.90]])))
    out["cov90_p05_p95"] = float(np.mean((y >= pred[QCOLS[0.05]]) & (y <= pred[QCOLS[0.95]])))
    out["width80_p10_p90"] = float(np.mean(pred[QCOLS[0.90]] - pred[QCOLS[0.10]]))
    return out


def parse_origins(origins: str | None) -> list[pd.Timestamp] | None:
    if not origins:
        return None
    return [pd.Timestamp(x.strip()) for x in origins.split(",") if x.strip()]


def infer_origins(df: pd.DataFrame, ts_col: str, n_folds: int, horizon_days: int) -> list[pd.Timestamp]:
    ts = pd.to_datetime(df[ts_col])
    start = ts.min() + pd.Timedelta(days=60)  # warm-up for weekly lags and stable historical cells
    latest_origin = ts.max() - pd.Timedelta(days=horizon_days)
    if latest_origin <= start:
        raise ValueError("Not enough data to infer rolling origins. Pass --origins explicitly or reduce horizon.")
    return list(pd.date_range(start=start, end=latest_origin, periods=n_folds))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Model-ready panel CSV/Parquet file.")
    p.add_argument("--timestamp", default="timestamp")
    p.add_argument("--school", default="school_id")
    p.add_argument("--target", default="y")
    p.add_argument("--origins", default=None, help="Comma-separated fold origins, e.g. 2023-05-30,2023-06-13")
    p.add_argument("--n-folds", type=int, default=12, help="Used only if --origins is omitted.")
    p.add_argument("--horizon-days", type=int, default=14)
    p.add_argument("--out-dir", default="data")
    p.add_argument("--save-predictions", action="store_true")
    args = p.parse_args()

    in_path = Path(args.input)
    if in_path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(in_path)
    else:
        df = pd.read_csv(in_path)

    required = [args.timestamp, args.school, args.target]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available columns: {list(df.columns)}")

    df = df.dropna(subset=required).copy()
    df[args.timestamp] = pd.to_datetime(df[args.timestamp])
    df = df.sort_values([args.school, args.timestamp])
    df = make_time_keys(df, args.timestamp)

    origins = parse_origins(args.origins)
    if origins is None:
        origins = infer_origins(df, args.timestamp, args.n_folds, args.horizon_days)

    fold_rows = []
    pred_parts = []
    for i, origin in enumerate(origins, start=1):
        end = origin + pd.Timedelta(days=args.horizon_days)
        train = df[df[args.timestamp] < origin]
        test = df[(df[args.timestamp] >= origin) & (df[args.timestamp] < end)]
        if train.empty or test.empty:
            print(f"Skipping fold {i}: train={len(train)}, test={len(test)}")
            continue

        pred = add_predictions(test, train, args.school, args.target)
        metrics = evaluate(pred, args.target)
        fold_rows.append({
            "fold": i,
            "origin_timestamp": origin,
            "end_timestamp": end,
            "method": "hist_quantile_tow",
            **metrics,
            "fit_time_sec": 0.0,
        })
        if args.save_predictions:
            keep = [args.school, args.timestamp, args.target, "_dow", "_slot"] + [QCOLS[q] for q in QUANTILES]
            part = pred[keep].copy()
            part.insert(0, "fold", i)
            pred_parts.append(part)

    fold_df = pd.DataFrame(fold_rows)
    if fold_df.empty:
        raise RuntimeError("No folds were evaluated.")

    summary = pd.DataFrame([{
        "method": "hist_quantile_tow",
        "folds": int(len(fold_df)),
        "mae_p50_mean": fold_df["mae_p50"].mean(),
        "mae_p50_std": fold_df["mae_p50"].std(ddof=1),
        "pinball_mean": fold_df["pinball_mean"].mean(),
        "pinball_std": fold_df["pinball_mean"].std(ddof=1),
        "cov50_mean": fold_df["cov50_p25_p75"].mean(),
        "cov80_mean": fold_df["cov80_p10_p90"].mean(),
        "cov90_mean": fold_df["cov90_p05_p95"].mean(),
        "width80_mean": fold_df["width80_p10_p90"].mean(),
    }])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_df.to_csv(out_dir / "wp1_rb_historical_quantile_fold_overall.csv", index=False)
    summary.to_csv(out_dir / "wp1_rb_historical_quantile_overall_summary.csv", index=False)
    if args.save_predictions and pred_parts:
        pd.concat(pred_parts, ignore_index=True).to_csv(out_dir / "wp1_rb_historical_quantile_predictions.csv", index=False)

    print("Wrote:")
    print(f"  {out_dir / 'wp1_rb_historical_quantile_fold_overall.csv'}")
    print(f"  {out_dir / 'wp1_rb_historical_quantile_overall_summary.csv'}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
