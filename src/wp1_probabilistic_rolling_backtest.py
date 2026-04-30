# src/wp1_probabilistic_rolling_backtest.py
# WP1 (paper-ready): Rolling-origin evaluation for (i) per-school (local) quantile GB,
# (ii) pooled quantile GB, and (iii) a two-stage median + per-school asymmetric calibration layer.
#
# Outputs:
# - data/wp1_rb_fold_overall.csv        (one row per fold per method)
# - data/wp1_rb_fold_per_school.csv     (one row per fold per school per method)
# - data/wp1_rb_overall_summary.csv     (summary mean ± std across folds)

import time
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

# ---------------------- configuration ----------------------

PANEL_CSV = "data/schools_panel.csv"
TARGET = "y"

# 30-minute data
STEPS_PER_DAY = 48
LAG_DAY = 48
LAG_WEEK = 336

# rolling evaluation
WARMUP_DAYS = 180
HORIZON_DAYS = 14
STEP_DAYS = 14
MAX_FOLDS = 12

# within-fold per-school calibration split (on TRAIN only)
SCHOOL_CALIB_FRAC = 0.20
MIN_SAMPLES_RESID = 500
MIN_SAMPLES_CALIB = 300

# Quantiles to fit
FIT_QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

PRIMARY_LO, PRIMARY_HI = 0.10, 0.90
TARGET_COVERAGE_PRIMARY = 0.80

# Drop known high-cardinality columns
DROP_ALWAYS = {"date"}  # prevents exploding one-hot

# Asymmetric alpha grids
ALPHA_GRID_LO = np.concatenate([np.linspace(0.25, 1.00, 16), np.linspace(1.05, 2.50, 30)])
ALPHA_GRID_HI = np.concatenate([np.linspace(0.25, 1.00, 16), np.linspace(1.05, 2.50, 30)])


# ---------------------- helpers ----------------------

def _fmt_secs(s: float) -> str:
    s = max(0.0, float(s))
    if s < 60:
        return f"{s:.1f}s"
    m = int(s // 60)
    r = s - 60 * m
    return f"{m}m{r:.0f}s"


def pinball_loss(y_true, y_pred, q: float) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    e = y_true - y_pred
    return float(np.mean(np.maximum(q * e, (q - 1) * e)))


def pinball_mean(y_true, preds: dict, quantiles) -> float:
    return float(np.mean([pinball_loss(y_true, preds[q], q) for q in quantiles]))


def coverage(y, lo, hi) -> float:
    y = np.asarray(y, float)
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    return float(np.mean((y >= lo) & (y <= hi)))


def mean_width(lo, hi) -> float:
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    return float(np.mean(np.maximum(hi - lo, 0.0)))


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    drop = {"timestamp", TARGET}
    X = df[[c for c in df.columns if c not in drop]].copy()

    for c in list(DROP_ALWAYS):
        if c in X.columns:
            X = X.drop(columns=[c])

    obj_cols = [c for c in X.columns if X[c].dtype == "object"]
    if obj_cols:
        X = pd.get_dummies(X, columns=obj_cols, drop_first=False)

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X


def align_columns(X_train: pd.DataFrame, X_test: pd.DataFrame):
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    return X_train, X_test


def fit_quantile_models(X_train, y_train, quantiles):
    models = {}
    for q in quantiles:
        m = GradientBoostingRegressor(loss="quantile", alpha=q, random_state=42)
        m.fit(X_train, y_train)
        models[q] = m
    return models


def predict_quantiles(models, X, quantiles):
    return {q: models[q].predict(X) for q in quantiles}


def calibrate_alpha_asymmetric(y_cal, p50_cal, p10_cal, p90_cal, target_cov, grid_lo, grid_hi):
    """
    Asymmetric calibration around p50:
      lo = p50 - a_lo*(p50-p10)
      hi = p50 + a_hi*(p90-p50)
    Choose (a_lo, a_hi) to minimise |coverage-target|, tie-break by narrower width.
    """
    y_cal = np.asarray(y_cal, float)
    p50_cal = np.asarray(p50_cal, float)
    p10_cal = np.asarray(p10_cal, float)
    p90_cal = np.asarray(p90_cal, float)

    w_lo = np.maximum(p50_cal - p10_cal, 0.0)
    w_hi = np.maximum(p90_cal - p50_cal, 0.0)

    best_err = 1e9
    best_width = 1e18
    best = (1.0, 1.0)

    for a_lo in grid_lo:
        lo = p50_cal - a_lo * w_lo
        for a_hi in grid_hi:
            hi = p50_cal + a_hi * w_hi
            cov = coverage(y_cal, lo, hi)
            err = abs(cov - target_cov)
            wid = mean_width(lo, hi)
            if (err < best_err) or (np.isclose(err, best_err) and wid < best_width):
                best_err = err
                best_width = wid
                best = (float(a_lo), float(a_hi))

    return best[0], best[1]


def eval_multi_level_coverages(y, preds):
    # preds must have 0.05,0.10,0.25,0.50,0.75,0.90,0.95
    cov50 = coverage(y, preds[0.25], preds[0.75])
    cov80 = coverage(y, preds[0.10], preds[0.90])
    cov90 = coverage(y, preds[0.05], preds[0.95])
    return cov50, cov80, cov90


def fold_indices(n_steps: int):
    warmup = WARMUP_DAYS * STEPS_PER_DAY
    horizon = HORIZON_DAYS * STEPS_PER_DAY
    step = STEP_DAYS * STEPS_PER_DAY

    origins = list(range(warmup, n_steps - horizon, step))
    if MAX_FOLDS is not None:
        origins = origins[:MAX_FOLDS]
    return origins, horizon


# ---------------------- rolling backtest ----------------------

def rolling_backtest(panel_csv=PANEL_CSV):
    t0 = time.time()
    print(f"[WP1-RB] Loading: {panel_csv}", flush=True)

    df = pd.read_csv(panel_csv, parse_dates=["timestamp"])

    if TARGET not in df.columns and "kWh" in df.columns:
        df = df.rename(columns={"kWh": TARGET})

    if "school_id" not in df.columns:
        df["school_id"] = "S1"

    df = df.sort_values(["timestamp", "school_id"]).reset_index(drop=True)

    schools = list(df["school_id"].unique())
    print(f"[WP1-RB] Rows: {len(df):,} | Schools: {len(schools)}", flush=True)
    print(f"[WP1-RB] Columns: {list(df.columns)}", flush=True)

    # Global time axis
    ts_unique = np.array(sorted(df["timestamp"].unique()))
    n_steps = len(ts_unique)
    origins, horizon = fold_indices(n_steps)

    print(f"[WP1-RB] Unique timestamps: {n_steps:,}", flush=True)
    print(f"[WP1-RB] Folds: {len(origins)} | horizon_steps={horizon} | step_days={STEP_DAYS}", flush=True)

    fold_overall_rows = []
    fold_school_rows = []

    for f, origin in enumerate(origins, start=1):
        f_start = time.time()
        t_origin = ts_unique[origin]
        t_end = ts_unique[min(origin + horizon, n_steps - 1)]

        train = df[df["timestamp"] < t_origin].copy()
        test = df[(df["timestamp"] >= t_origin) & (df["timestamp"] <= t_end)].copy()

        print(
            f"\n[WP1-RB] Fold {f}/{len(origins)} | origin={t_origin} | end={t_end} | "
            f"train={len(train):,} test={len(test):,}",
            flush=True
        )

        # Build positional maps
        train_pos_map = {idx: pos for pos, idx in enumerate(train.index)}
        test_pos_map = {idx: pos for pos, idx in enumerate(test.index)}

        # ---------------- seasonal naïve baselines (MAE only) ----------------
        # For each school, predict by lagging within TRAIN+TEST concatenated history (no future leakage).
        # We'll compute MAE on test timestamps where the lag exists.

        def seasonal_naive_mae(lag_steps: int):
            preds = np.full(len(test), np.nan, dtype=float)

            for sid in schools:
                # Build a local history using train+test for that school, but only using past values for prediction
                hist = pd.concat([train[train["school_id"] == sid], test[test["school_id"] == sid]]).sort_values("timestamp")
                y_hist = hist[TARGET].values
                idx_hist = hist.index.values

                # Create mapping from hist index -> position in hist
                pos_map = {idx_hist[i]: i for i in range(len(idx_hist))}

                # For each test row index, use y at (pos - lag_steps) if available
                for idx in test.index[test["school_id"] == sid]:
                    p = pos_map.get(idx, None)
                    if p is None or p - lag_steps < 0:
                        continue
                    preds[test_pos_map[idx]] = y_hist[p - lag_steps]

            mask = np.isfinite(preds)
            if mask.sum() == 0:
                return np.nan
            return float(np.mean(np.abs(test[TARGET].values[mask] - preds[mask])))

        mae_naive_day = seasonal_naive_mae(LAG_DAY)
        mae_naive_week = seasonal_naive_mae(LAG_WEEK)

        fold_overall_rows.append({
            "fold": f, "origin_timestamp": t_origin, "end_timestamp": t_end,
            "method": "naive_1day",
            "mae_p50": mae_naive_day,
            "pinball_mean": np.nan,
            "cov50_p25_p75": np.nan, "cov80_p10_p90": np.nan, "cov90_p05_p95": np.nan,
            "width80_p10_p90": np.nan,
            "fit_time_sec": 0.0
        })
        fold_overall_rows.append({
            "fold": f, "origin_timestamp": t_origin, "end_timestamp": t_end,
            "method": "naive_1week",
            "mae_p50": mae_naive_week,
            "pinball_mean": np.nan,
            "cov50_p25_p75": np.nan, "cov80_p10_p90": np.nan, "cov90_p05_p95": np.nan,
            "width80_p10_p90": np.nan,
            "fit_time_sec": 0.0
        })

        # ---------------- pooled probabilistic model ----------------
        p0 = time.time()
        X_tr = make_features(train)
        y_tr = train[TARGET].values
        X_te = make_features(test)
        y_te = test[TARGET].values
        X_tr, X_te = align_columns(X_tr, X_te)

        pooled_models = fit_quantile_models(X_tr, y_tr, FIT_QUANTILES)
        pooled_te = predict_quantiles(pooled_models, X_te, FIT_QUANTILES)

        pooled_mae = float(np.mean(np.abs(y_te - pooled_te[0.50])))
        pooled_cov50, pooled_cov80, pooled_cov90 = eval_multi_level_coverages(y_te, pooled_te)
        pooled_pin = pinball_mean(y_te, pooled_te, FIT_QUANTILES)

        fold_overall_rows.append({
            "fold": f,
            "origin_timestamp": t_origin,
            "end_timestamp": t_end,
            "method": "pooled",
            "mae_p50": pooled_mae,
            "pinball_mean": pooled_pin,
            "cov50_p25_p75": pooled_cov50,
            "cov80_p10_p90": pooled_cov80,
            "cov90_p05_p95": pooled_cov90,
            "width80_p10_p90": mean_width(pooled_te[0.10], pooled_te[0.90]),
            "fit_time_sec": float(time.time() - p0),
        })

        print(f"[WP1-RB] Pooled: mae={pooled_mae:.4f} pin={pooled_pin:.4f} cov80={pooled_cov80:.3f} time={_fmt_secs(time.time()-p0)}", flush=True)

        # ---------------- per-school (local) probabilistic models ----------------
        ps0 = time.time()
        ps_te = {q: np.empty(len(test), dtype=float) for q in FIT_QUANTILES}

        for sid in schools:
            tr_idx = train.index[train["school_id"] == sid]
            te_idx = test.index[test["school_id"] == sid]
            if len(te_idx) == 0 or len(tr_idx) == 0:
                continue

            tr_s = train.loc[tr_idx].sort_values("timestamp")
            te_s = test.loc[te_idx].sort_values("timestamp")

            X_tr_s = make_features(tr_s.drop(columns=["school_id"], errors="ignore"))
            y_tr_s = tr_s[TARGET].values
            X_te_s = make_features(te_s.drop(columns=["school_id"], errors="ignore"))
            y_te_s = te_s[TARGET].values

            X_tr_s, X_te_s = align_columns(X_tr_s, X_te_s)

            m_s = fit_quantile_models(X_tr_s, y_tr_s, FIT_QUANTILES)
            p_s = predict_quantiles(m_s, X_te_s, FIT_QUANTILES)

            te_pos = np.array([test_pos_map[i] for i in te_s.index], dtype=int)
            for q in FIT_QUANTILES:
                ps_te[q][te_pos] = p_s[q]

            mae_s = float(np.mean(np.abs(y_te_s - p_s[0.50])))
            cov50_s, cov80_s, cov90_s = eval_multi_level_coverages(y_te_s, p_s)

            fold_school_rows.append({
                "fold": f,
                "origin_timestamp": t_origin,
                "school_id": sid,
                "method": "per_school",
                "n_train_school": int(len(tr_s)),
                "n_test_school": int(len(te_s)),
                "mae_p50": mae_s,
                "pinball_mean": pinball_mean(y_te_s, p_s, FIT_QUANTILES),
                "cov50_p25_p75": cov50_s,
                "cov80_p10_p90": cov80_s,
                "cov90_p05_p95": cov90_s,
                "width80_p10_p90": mean_width(p_s[0.10], p_s[0.90]),
            })

        ps_mae = float(np.mean(np.abs(y_te - ps_te[0.50])))
        ps_cov50, ps_cov80, ps_cov90 = eval_multi_level_coverages(y_te, ps_te)
        ps_pin = pinball_mean(y_te, ps_te, FIT_QUANTILES)

        fold_overall_rows.append({
            "fold": f,
            "origin_timestamp": t_origin,
            "end_timestamp": t_end,
            "method": "per_school",
            "mae_p50": ps_mae,
            "pinball_mean": ps_pin,
            "cov50_p25_p75": ps_cov50,
            "cov80_p10_p90": ps_cov80,
            "cov90_p05_p95": ps_cov90,
            "width80_p10_p90": mean_width(ps_te[0.10], ps_te[0.90]),
            "fit_time_sec": float(time.time() - ps0),
        })

        print(f"[WP1-RB] Per-school: mae={ps_mae:.4f} pin={ps_pin:.4f} cov80={ps_cov80:.3f} time={_fmt_secs(time.time()-ps0)}", flush=True)

        # ---------------- two-stage median correction + per-school ASYMM calibration ----------------

        pooled_tr = predict_quantiles(pooled_models, X_tr, FIT_QUANTILES)

        p50_two = pooled_te[0.50].copy()
        p_two = {q: pooled_te[q].copy() for q in FIT_QUANTILES}

        out_q = {
            "p50": p50_two,
            "p10": np.empty_like(p50_two),
            "p90": np.empty_like(p50_two),
            "p25": np.empty_like(p50_two),
            "p75": np.empty_like(p50_two),
            "p05": np.empty_like(p50_two),
            "p95": np.empty_like(p50_two),
        }

        two_stage_start = time.time()

        for sid in schools:
            idx_tr_s_dfidx = train.index[train["school_id"] == sid]
            idx_te_s_dfidx = test.index[test["school_id"] == sid]
            if len(idx_te_s_dfidx) == 0:
                continue

            tr_s = train.loc[idx_tr_s_dfidx].sort_values("timestamp")
            cut = int(len(tr_s) * (1.0 - SCHOOL_CALIB_FRAC))
            fit_s = tr_s.iloc[:cut]
            cal_s = tr_s.iloc[cut:]

            fit_pos = np.array([train_pos_map[i] for i in fit_s.index], dtype=int) if len(fit_s) else np.array([], dtype=int)
            cal_pos = np.array([train_pos_map[i] for i in cal_s.index], dtype=int) if len(cal_s) else np.array([], dtype=int)
            te_pos  = np.array([test_pos_map[i] for i in idx_te_s_dfidx], dtype=int)

            # residual model (median correction)
            use_resid = len(fit_pos) >= MIN_SAMPLES_RESID
            if use_resid:
                resid_fit = y_tr[fit_pos] - pooled_tr[0.50][fit_pos]
                rm = GradientBoostingRegressor(loss="squared_error", random_state=42)
                rm.fit(X_tr.iloc[fit_pos], resid_fit)

                shift_te = rm.predict(X_te.iloc[te_pos])
                shift_cal = rm.predict(X_tr.iloc[cal_pos]) if len(cal_pos) else np.zeros(0, dtype=float)
            else:
                shift_te = np.zeros(len(te_pos), dtype=float)
                shift_cal = np.zeros(len(cal_pos), dtype=float)

            p50_s = pooled_te[0.50][te_pos] + shift_te
            p50_two[te_pos] = p50_s

            for q in FIT_QUANTILES:
                p_two[q][te_pos] = pooled_te[q][te_pos] + shift_te

            # asymmetric calibration on TRAIN-calib for PRIMARY interval
            alpha_lo, alpha_hi = 1.0, 1.0
            cov_unc_cal = np.nan
            if len(cal_pos) >= MIN_SAMPLES_CALIB:
                y_cal = y_tr[cal_pos]
                p50_cal = pooled_tr[0.50][cal_pos] + shift_cal
                p10_cal = pooled_tr[0.10][cal_pos] + shift_cal
                p90_cal = pooled_tr[0.90][cal_pos] + shift_cal

                cov_unc_cal = coverage(y_cal, p10_cal, p90_cal)

                alpha_lo, alpha_hi = calibrate_alpha_asymmetric(
                    y_cal=y_cal,
                    p50_cal=p50_cal,
                    p10_cal=p10_cal,
                    p90_cal=p90_cal,
                    target_cov=TARGET_COVERAGE_PRIMARY,
                    grid_lo=ALPHA_GRID_LO,
                    grid_hi=ALPHA_GRID_HI,
                )

            # build calibrated endpoints on TEST using same alphas around p50
            p10_s = p_two[0.10][te_pos]
            p90_s = p_two[0.90][te_pos]
            wlo80 = np.maximum(p50_s - p10_s, 0.0)
            whi80 = np.maximum(p90_s - p50_s, 0.0)
            lo80 = p50_s - alpha_lo * wlo80
            hi80 = p50_s + alpha_hi * whi80

            p25_s = p_two[0.25][te_pos]
            p75_s = p_two[0.75][te_pos]
            wlo50 = np.maximum(p50_s - p25_s, 0.0)
            whi50 = np.maximum(p75_s - p50_s, 0.0)
            lo50 = p50_s - alpha_lo * wlo50
            hi50 = p50_s + alpha_hi * whi50

            p05_s = p_two[0.05][te_pos]
            p95_s = p_two[0.95][te_pos]
            wlo90 = np.maximum(p50_s - p05_s, 0.0)
            whi90 = np.maximum(p95_s - p50_s, 0.0)
            lo90 = p50_s - alpha_lo * wlo90
            hi90 = p50_s + alpha_hi * whi90

            out_q["p10"][te_pos] = lo80
            out_q["p90"][te_pos] = hi80
            out_q["p25"][te_pos] = lo50
            out_q["p75"][te_pos] = hi50
            out_q["p05"][te_pos] = lo90
            out_q["p95"][te_pos] = hi90

            # per-school metrics (fold TEST)
            y_s = y_te[te_pos]
            mae_s = float(np.mean(np.abs(y_s - p50_s)))
            cov50_s = coverage(y_s, lo50, hi50)
            cov80_s = coverage(y_s, lo80, hi80)
            cov90_s = coverage(y_s, lo90, hi90)

            fold_school_rows.append({
                "fold": f,
                "origin_timestamp": t_origin,
                "school_id": sid,
                "method": "two_stage_cal_asymm",
                "n_train_school": int(len(idx_tr_s_dfidx)),
                "n_test_school": int(len(idx_te_s_dfidx)),
                "alpha_lo": float(alpha_lo),
                "alpha_hi": float(alpha_hi),
                "cov80_unc_on_calib": float(cov_unc_cal) if np.isfinite(cov_unc_cal) else np.nan,
                "mae_p50": mae_s,
                "pinball_mean": np.nan,  # optional per-school pinball for two-stage (can be added if needed)
                "cov50_p25_p75": cov50_s,
                "cov80_p10_p90": cov80_s,
                "cov90_p05_p95": cov90_s,
                "abs_err_cov80": float(abs(cov80_s - TARGET_COVERAGE_PRIMARY)),
                "width80_p10_p90": mean_width(lo80, hi80),
            })

        two_mae = float(np.mean(np.abs(y_te - p50_two)))
        two_cov50 = coverage(y_te, out_q["p25"], out_q["p75"])
        two_cov80 = coverage(y_te, out_q["p10"], out_q["p90"])
        two_cov90 = coverage(y_te, out_q["p05"], out_q["p95"])
        two_width80 = mean_width(out_q["p10"], out_q["p90"])

        # Build a preds dict for pinball on two-stage endpoints
        two_preds = {
            0.05: out_q["p05"], 0.10: out_q["p10"], 0.25: out_q["p25"],
            0.50: p50_two,      0.75: out_q["p75"], 0.90: out_q["p90"], 0.95: out_q["p95"]
        }
        two_pin = pinball_mean(y_te, two_preds, FIT_QUANTILES)

        fold_overall_rows.append({
            "fold": f,
            "origin_timestamp": t_origin,
            "end_timestamp": t_end,
            "method": "two_stage_cal_asymm",
            "mae_p50": two_mae,
            "pinball_mean": two_pin,
            "cov50_p25_p75": two_cov50,
            "cov80_p10_p90": two_cov80,
            "cov90_p05_p95": two_cov90,
            "width80_p10_p90": two_width80,
            "fit_time_sec": float(time.time() - two_stage_start),
        })

        print(
            f"[WP1-RB] Two-stage CAL(ASYMM): mae={two_mae:.4f} pin={two_pin:.4f} cov80={two_cov80:.3f} "
            f"time={_fmt_secs(time.time()-two_stage_start)}",
            flush=True
        )

        print(f"[WP1-RB] Fold time: {_fmt_secs(time.time()-f_start)}", flush=True)

    # ---------------------- outputs ----------------------
    overall_df = pd.DataFrame(fold_overall_rows)
    per_school_df = pd.DataFrame(fold_school_rows)

    overall_df.to_csv("data/wp1_rb_fold_overall.csv", index=False)
    per_school_df.to_csv("data/wp1_rb_fold_per_school.csv", index=False)

    def summarise(method_name: str):
        sub = overall_df[overall_df["method"] == method_name].copy()
        out = {
            "method": method_name,
            "folds": int(sub["fold"].nunique()),
            "mae_p50_mean": float(sub["mae_p50"].mean()),
            "mae_p50_std": float(sub["mae_p50"].std(ddof=0)),
            "pinball_mean": float(sub["pinball_mean"].mean()) if sub["pinball_mean"].notna().any() else np.nan,
            "pinball_std": float(sub["pinball_mean"].std(ddof=0)) if sub["pinball_mean"].notna().any() else np.nan,
            "cov50_mean": float(sub["cov50_p25_p75"].mean()) if sub["cov50_p25_p75"].notna().any() else np.nan,
            "cov80_mean": float(sub["cov80_p10_p90"].mean()) if sub["cov80_p10_p90"].notna().any() else np.nan,
            "cov90_mean": float(sub["cov90_p05_p95"].mean()) if sub["cov90_p05_p95"].notna().any() else np.nan,
            "width80_mean": float(sub["width80_p10_p90"].mean()) if sub["width80_p10_p90"].notna().any() else np.nan,
        }
        if method_name == "two_stage_cal_asymm":
            ps = per_school_df[per_school_df["method"] == method_name]
            if len(ps) > 0 and "abs_err_cov80" in ps.columns:
                err = ps["abs_err_cov80"].dropna().values
                out["abs_err_cov80_mean_per_school"] = float(np.mean(err)) if len(err) else np.nan
                out["abs_err_cov80_median_per_school"] = float(np.median(err)) if len(err) else np.nan
                out["abs_err_cov80_p90_per_school"] = float(np.percentile(err, 90)) if len(err) else np.nan
                if "alpha_lo" in ps.columns:
                    out["alpha_lo_mean"] = float(ps["alpha_lo"].mean())
                    out["alpha_hi_mean"] = float(ps["alpha_hi"].mean())
        return out

    summary = pd.DataFrame([
        summarise("naive_1day"),
        summarise("naive_1week"),
        summarise("per_school"),
        summarise("pooled"),
        summarise("two_stage_cal_asymm"),
    ])

    summary.to_csv("data/wp1_rb_overall_summary.csv", index=False)

    print("\n[WP1-RB] Wrote:", flush=True)
    print(" - data/wp1_rb_fold_overall.csv", flush=True)
    print(" - data/wp1_rb_fold_per_school.csv", flush=True)
    print(" - data/wp1_rb_overall_summary.csv", flush=True)
    print(f"[WP1-RB] Total time: {_fmt_secs(time.time() - t0)}", flush=True)

    print("\n[WP1-RB] Overall summary across folds:", flush=True)
    print(summary, flush=True)


if __name__ == "__main__":
    rolling_backtest()