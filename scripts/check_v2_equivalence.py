"""Vérifie que mlrp (v2) reproduit les prédictions et les rendements archivés du mémoire (v1, octobre 2024).

Cas de référence : Ridge, États-Unis, top 10, 2008-01 -> 2024-01 (les données brutes de 2024 sont identiques).
Compare : hyperparamètres optimaux, y_pred (193 x 50), poids long/short, rendements quotidiens « as_published ».

Usage : uv run python scripts/check_v2_equivalence.py [--n-jobs 1]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mlrp.config import RESULTS_DIR, RunSpec, TuningSpec  # noqa: E402
from mlrp.portfolio import build_weights, strategy_returns  # noqa: E402
from mlrp.runner import get_dataset, get_predictions  # noqa: E402

ARCHIVE = Path(__file__).resolve().parents[1] / "results" / "archive_2024" / "usa" / "2008-01 à 2024-01" / "Top 10"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args()

    spec = RunSpec(country="usa", period="2008-2024", model="ridge_regressor", signal="top10",
                   long_short_mode="as_published", tuning=TuningSpec(n_trials=50))
    ds = get_dataset(spec)
    pred = get_predictions(spec, ds, n_jobs=a.n_jobs, use_cache=not a.no_cache)

    y_old = pd.read_csv(ARCHIVE / "ridge_regressor_y_pred.csv.gz", index_col=0, parse_dates=True)
    y_new = pred.y_pred.loc[y_old.index, y_old.columns]
    diff_pred = float((y_old - y_new).abs().max().max())
    tune_old = pd.read_csv(ARCHIVE / "ridge_regressor_tuning_results.csv.gz").iloc[0]
    print("hyperparamètres 2024 :", {k: tune_old[k] for k in ["alpha", "max_iter", "tol", "solver"]})
    print("hyperparamètres v2   :", {k: pred.best_params.get(k) for k in ["alpha", "max_iter", "tol", "solver"]})
    print(f"écart maximal y_pred : {diff_pred:.3e}")

    lw, sw = build_weights(pred.y_pred, "top10", "regressor")
    lw_old = pd.read_csv(ARCHIVE / "ridge_regressor_long_weights.csv.gz", index_col=0, parse_dates=True)
    sw_old = pd.read_csv(ARCHIVE / "ridge_regressor_short_weights.csv.gz", index_col=0, parse_dates=True)
    diff_w = float(max((lw.loc[lw_old.index, lw_old.columns] - lw_old).abs().max().max(),
                       (sw.loc[sw_old.index, sw_old.columns] - sw_old).abs().max().max()))
    print(f"écart maximal poids  : {diff_w:.3e}")

    strat = strategy_returns(lw, sw, ds.prices_daily, mode="as_published", fee=0.0)
    r_old = pd.read_csv(ARCHIVE / "ridge_regressor_strategy_returns.csv.gz", index_col=0, parse_dates=True).iloc[:, 0]
    j = pd.concat([r_old, strat.returns], axis=1, join="inner")
    diff_r = float(np.nanmax(np.abs(j.iloc[:, 0] - j.iloc[:, 1])))
    print(f"écart maximal rendements quotidiens (as_published) : {diff_r:.3e} sur {len(j)} jours")

    ok = diff_pred < 1e-5 and diff_w < 1e-9 and diff_r < 1e-9
    out = RESULTS_DIR / "equivalence_v1_v2.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"y_pred_max_abs_diff": diff_pred, "weights_max_abs_diff": diff_w,
                   "returns_max_abs_diff": diff_r, "passed": ok}]).to_csv(out, index=False)
    print("ÉQUIVALENT" if ok else "ÉCART : voir ci-dessus", "->", out)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
