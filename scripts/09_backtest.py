#!/usr/bin/env python3
"""
09_backtest.py

Walk-forward backtest: pick N random past cutoff dates, train on everything
knowable up to each cutoff, forecast the following FORECAST_HORIZON_DAYS,
and score those forecasts against what actually happened.

--------------------------------------------------------------------------
WHY THIS EXISTS (and what it adds over stage 06)
--------------------------------------------------------------------------
Stage 06 evaluates on ONE held-out chronological test block. That answers
"does the model generalize to the last ~15% of the catalog", which is a
single draw - if that window happened to contain (or miss) a big aftershock
sequence, the headline number moves a lot. This stage instead re-fits the
model from scratch at N independent points in time and scores each one's
next-7-day forecast against the real catalog, which is much closer to how
the model would actually be used operationally: "retrain on everything I
know today, forecast the coming week, see how it did".

--------------------------------------------------------------------------
LEAKAGE CONTROL (the whole point - read before changing anything)
--------------------------------------------------------------------------
For a cutoff date T, a row (cell, day, horizon_day) is only usable for
TRAINING if its label was already observable at T. The label covers the
window (day, day + horizon_day], so the condition is:

    day + horizon_day <= T

Not `day <= T` - that would train on labels peeking past the cutoff (e.g. a
row at day=T-2 with horizon_day=7 is labeled by events up to T+5, five days
the model is not allowed to know about). This distinction is the single
easiest way to get a flatteringly wrong backtest, so it is asserted below
rather than just commented.

Features are safe by construction: every feature in stage 04 is a trailing
window / expanding count / decayed sum over events at or before its own row's
day, so a row at day <= T never encodes anything after T.

The median imputer is re-fit on each cutoff's own training slice (fitting it
once on the full matrix would leak future distributions backward).

FORECAST rows are `day == T` at horizon_day 1..FORECAST_HORIZON_DAYS. Their
`target` column is the ground truth for scoring - available to us only
because T is in the past.

--------------------------------------------------------------------------
HYPERPARAMETERS
--------------------------------------------------------------------------
Re-running stage 05's full GridSearchCV at every cutoff would cost ~10x the
main training run (many hours each). Standard practice for a walk-forward
study is to hold the already-tuned hyperparameters fixed and refit the model
at each origin, which is what this does: max_depth / learning_rate /
n_estimators are read from stage 05's own 05_xgboost_best_params.json, and
only `scale_pos_weight` is recomputed per cutoff (it depends on that slice's
own class balance). XGBoost only - it is stage 06's best model, and fitting
all three at 10 cutoffs adds cost without changing what this stage answers.

Documented consequence: those hyperparameters were selected using the full
train split, which extends past some cutoffs. That is a mild optimism in
favour of the model, it is NOT the same as label leakage (no label is ever
seen), and it is the standard trade-off in walk-forward evaluation. Flagged
here rather than buried.

Input:  ../outputs/04_features/04_feature_matrix.csv
        ../outputs/05_models/05_xgboost_best_params.json
Output: ../outputs/09_backtest/09_backtest_per_horizon.csv
        ../outputs/09_backtest/09_backtest_per_date.csv
        ../outputs/09_backtest/09_backtest_top_cells.csv
        ../outputs/09_backtest/09_backtest_summary.json
        ../outputs/09_backtest/09_backtest_reliability.png
        ../outputs/09_backtest/09_backtest_skill_by_horizon.png
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier

RANDOM_SEED = 42
FORECAST_HORIZON_DAYS = 7
MIN_TRAIN_DAYS = 180      # a cutoff needs at least this much history behind it
TOP_N_CELLS = 10          # "of the N riskiest cells, how many actually had an event"


def build_matrices(train_df, fc_df, feature_cols):
    """Missing-indicator + median impute, imputer fit on the TRAIN slice only
    (see module docstring). Mirrors stage 05's build_features."""
    Xtr = train_df[feature_cols].copy()
    Xfc = fc_df[feature_cols].copy()

    nan_cols = [c for c in feature_cols if Xtr[c].isna().any() or Xfc[c].isna().any()]
    for c in nan_cols:
        Xtr[f"{c}_missing"] = Xtr[c].isna().astype(int)
        Xfc[f"{c}_missing"] = Xfc[c].isna().astype(int)

    imputer = SimpleImputer(strategy="median")
    Xtr[feature_cols] = imputer.fit_transform(Xtr[feature_cols])
    Xfc[feature_cols] = imputer.transform(Xfc[feature_cols])
    return Xtr, Xfc[Xtr.columns]


def safe_auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None


def safe_ap(y, p):
    return float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else None


def main():
    p = argparse.ArgumentParser(description="Walk-forward backtest at N random past cutoffs (stage 09).")
    p.add_argument("--features", type=str, default="../outputs/04_features/04_feature_matrix.csv")
    p.add_argument("--xgb-params", type=str, default="../outputs/05_models/05_xgboost_best_params.json")
    p.add_argument("--outdir", type=str, default="../outputs/09_backtest")
    p.add_argument("--n-dates", type=int, default=10)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--n-jobs", type=int, default=3,
                   help="XGBoost threads. Kept low by default so this can run alongside stage 05.")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"Loading {args.features} ...")
    df = pd.read_csv(args.features)
    df["day"] = pd.to_datetime(df["day"], utc=True)
    print(f"Loaded {len(df):,} rows, {df['day'].min().date()} -> {df['day'].max().date()}")

    with open(args.xgb_params, encoding="utf-8") as f:
        xgb_info = json.load(f)
    bp = xgb_info["best_params"]
    n_rounds = xgb_info.get("best_iteration") or 200
    print(f"Reusing stage 05's tuned hyperparameters: max_depth={bp['max_depth']}, "
          f"learning_rate={bp['learning_rate']}, n_estimators={n_rounds} (best_iteration)")

    feature_cols = [c for c in df.columns if c not in ("day", "target")]

    # --- choose cutoffs -------------------------------------------------------
    days = np.sort(df["day"].unique())
    first_ok = days[0] + pd.Timedelta(days=MIN_TRAIN_DAYS)
    # A cutoff also needs its own forecast rows to be fully labeled, i.e.
    # day == T at horizon 7 must exist in the matrix.
    last_ok = days[-1]
    eligible = [d for d in days if first_ok <= d <= last_ok]
    if len(eligible) < args.n_dates:
        raise SystemExit(f"only {len(eligible)} eligible cutoffs, need {args.n_dates}")
    cutoffs = sorted(rng.choice(eligible, size=args.n_dates, replace=False))
    print(f"\nEligible cutoff window: {pd.Timestamp(eligible[0]).date()} -> "
          f"{pd.Timestamp(eligible[-1]).date()} ({len(eligible)} days)")
    print(f"Chose {args.n_dates} random cutoffs (seed={args.seed}):")
    for c in cutoffs:
        print(f"  {pd.Timestamp(c).date()}")

    per_horizon, per_date, top_cells = [], [], []

    for i, T in enumerate(cutoffs, 1):
        T = pd.Timestamp(T)
        # label fully observable by T  <=>  day + horizon_day <= T
        label_end = df["day"] + pd.to_timedelta(df["horizon_day"], unit="D")
        train_mask = label_end <= T
        fc_mask = df["day"] == T

        train_df = df[train_mask]
        fc_df = df[fc_mask].sort_values(["horizon_day", "cell_lat", "cell_lon"])

        # Hard guarantee, not a comment: nothing in training may be labeled by
        # an event after the cutoff.
        assert (train_df["day"] + pd.to_timedelta(train_df["horizon_day"], unit="D") <= T).all(), \
            "LEAK: a training row's label window extends past the cutoff"
        assert len(fc_df) > 0, f"no forecast rows at {T.date()}"

        ytr = train_df["target"].to_numpy()
        pos, neg = int(ytr.sum()), int(len(ytr) - ytr.sum())
        spw = neg / max(pos, 1)

        Xtr, Xfc = build_matrices(train_df, fc_df, feature_cols)

        model = XGBClassifier(
            n_estimators=n_rounds, max_depth=bp["max_depth"], learning_rate=bp["learning_rate"],
            objective="binary:logistic", booster="gbtree", eval_metric="auc",
            tree_method="hist", random_state=RANDOM_SEED, scale_pos_weight=spw,
            n_jobs=args.n_jobs,
        )
        model.fit(Xtr, ytr, verbose=False)

        proba = model.predict_proba(Xfc)[:, 1]
        actual = fc_df["target"].to_numpy()
        out = fc_df[["horizon_day", "cell_lat", "cell_lon"]].copy()
        out["proba"] = proba
        out["actual"] = actual

        print(f"\n[{i}/{len(cutoffs)}] cutoff {T.date()}  train_rows={len(train_df):,} "
              f"(pos {pos:,}, {100*pos/len(train_df):.2f}%)  forecast_rows={len(fc_df):,}")

        for h in range(1, FORECAST_HORIZON_DAYS + 1):
            s = out[out["horizon_day"] == h]
            y, pr = s["actual"].to_numpy(), s["proba"].to_numpy()
            # Operational read: of the TOP_N riskiest cells, how many really had an event?
            top = s.nlargest(TOP_N_CELLS, "proba")
            hits = int(top["actual"].sum())
            per_horizon.append({
                "cutoff": T.date().isoformat(), "horizon_day": h, "n_cells": len(s),
                "actual_events": int(y.sum()), "actual_rate": float(y.mean()),
                "mean_predicted": float(pr.mean()),
                "roc_auc": safe_auc(y, pr), "pr_auc": safe_ap(y, pr),
                "brier": float(brier_score_loss(y, pr)) if len(np.unique(y)) > 1 else None,
                f"hits_in_top{TOP_N_CELLS}": hits,
                f"precision_at_top{TOP_N_CELLS}": hits / TOP_N_CELLS,
            })
            for _, r in top.iterrows():
                top_cells.append({
                    "cutoff": T.date().isoformat(), "horizon_day": h,
                    "cell_lat": r["cell_lat"], "cell_lon": r["cell_lon"],
                    "predicted_proba": round(float(r["proba"]), 4),
                    "actually_happened": int(r["actual"]),
                })

        y_all, p_all = out["actual"].to_numpy(), out["proba"].to_numpy()
        row = {
            "cutoff": T.date().isoformat(), "train_rows": int(len(train_df)),
            "train_positive_rate": float(ytr.mean()), "forecast_rows": int(len(fc_df)),
            "actual_events": int(y_all.sum()), "actual_rate": float(y_all.mean()),
            "mean_predicted": float(p_all.mean()),
            "roc_auc": safe_auc(y_all, p_all), "pr_auc": safe_ap(y_all, p_all),
            "brier": float(brier_score_loss(y_all, p_all)) if len(np.unique(y_all)) > 1 else None,
        }
        per_date.append(row)
        print(f"     ROC-AUC={row['roc_auc']:.4f}  PR-AUC={row['pr_auc']:.4f}  "
              f"Brier={row['brier']:.4f}  actual_rate={row['actual_rate']:.4f}  "
              f"mean_pred={row['mean_predicted']:.4f}")

    ph = pd.DataFrame(per_horizon)
    pdt = pd.DataFrame(per_date)
    tc = pd.DataFrame(top_cells)
    ph.to_csv(outdir / "09_backtest_per_horizon.csv", index=False)
    pdt.to_csv(outdir / "09_backtest_per_date.csv", index=False)
    tc.to_csv(outdir / "09_backtest_top_cells.csv", index=False)

    print("\n" + "=" * 78)
    print("PER-DATE SUMMARY (each row = an independent model, retrained from scratch)")
    print("=" * 78)
    print(pdt.round(4).to_string(index=False))

    print("\n" + "=" * 78)
    print("SKILL BY FORECAST HORIZON (averaged over all cutoffs)")
    print("=" * 78)
    agg = ph.groupby("horizon_day").agg(
        actual_rate=("actual_rate", "mean"), mean_predicted=("mean_predicted", "mean"),
        roc_auc=("roc_auc", "mean"), pr_auc=("pr_auc", "mean"),
        **{f"precision_at_top{TOP_N_CELLS}": (f"precision_at_top{TOP_N_CELLS}", "mean")},
    ).reset_index()
    print(agg.round(4).to_string(index=False))

    baseline_auc = 0.5
    summary = {
        "n_cutoffs": len(cutoffs), "seed": args.seed,
        "cutoffs": [pd.Timestamp(c).date().isoformat() for c in cutoffs],
        "horizon_days": FORECAST_HORIZON_DAYS,
        "hyperparameters_source": "stage 05 05_xgboost_best_params.json (held fixed, refit per cutoff)",
        "mean_roc_auc": float(pdt["roc_auc"].mean()), "std_roc_auc": float(pdt["roc_auc"].std()),
        "min_roc_auc": float(pdt["roc_auc"].min()), "max_roc_auc": float(pdt["roc_auc"].max()),
        "mean_pr_auc": float(pdt["pr_auc"].mean()),
        "mean_actual_rate_is_pr_auc_baseline": float(pdt["actual_rate"].mean()),
        "mean_brier": float(pdt["brier"].mean()),
        "roc_auc_baseline": baseline_auc,
        f"mean_precision_at_top{TOP_N_CELLS}": float(ph[f"precision_at_top{TOP_N_CELLS}"].mean()),
    }
    with open(outdir / "09_backtest_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 78)
    print("HEADLINE")
    print("=" * 78)
    print(f"  ROC-AUC across {len(cutoffs)} independent retrains: "
          f"{summary['mean_roc_auc']:.4f} +/- {summary['std_roc_auc']:.4f} "
          f"(min {summary['min_roc_auc']:.4f}, max {summary['max_roc_auc']:.4f})  [0.5 = no skill]")
    print(f"  PR-AUC: {summary['mean_pr_auc']:.4f}  vs base rate "
          f"{summary['mean_actual_rate_is_pr_auc_baseline']:.4f} = what random guessing scores")
    print(f"  Of the {TOP_N_CELLS} riskiest cells named each time, "
          f"{100*summary[f'mean_precision_at_top{TOP_N_CELLS}']:.1f}% really did have an M4.5+ event")

    # --- plots -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(agg["horizon_day"], agg["roc_auc"], marker="o", label="ROC-AUC")
    ax.plot(agg["horizon_day"], agg[f"precision_at_top{TOP_N_CELLS}"], marker="s",
            label=f"Precision @ top {TOP_N_CELLS} cells")
    ax.axhline(0.5, ls="--", c="grey", label="ROC-AUC no-skill (0.5)")
    ax.set_xlabel("Forecast horizon (days ahead)")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title(f"Walk-forward skill by horizon ({len(cutoffs)} random cutoffs, retrained each time)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "09_backtest_skill_by_horizon.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pdt["mean_predicted"], pdt["actual_rate"], s=70)
    for _, r in pdt.iterrows():
        ax.annotate(r["cutoff"][5:], (r["mean_predicted"], r["actual_rate"]),
                    fontsize=7, xytext=(4, 3), textcoords="offset points")
    lim = max(pdt["mean_predicted"].max(), pdt["actual_rate"].max()) * 1.15
    ax.plot([0, lim], [0, lim], ls="--", c="grey", label="perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Actual event rate")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_title("Predicted vs actual, per cutoff date")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "09_backtest_reliability.png", dpi=150)
    plt.close(fig)

    print(f"\nAll backtest artifacts written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
