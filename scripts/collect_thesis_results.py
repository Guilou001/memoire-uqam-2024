"""Rassemble les sorties archivées du mémoire (exécution d'octobre 2024) dans results/.

Source : le dossier de résultats du mémoire, organisé en <pays>/<période>/<signal>/data/intermediate_data/...
(24 feuilles : 2 pays x 4 périodes x 3 signaux). Le chemin se passe en argument ou par la variable
d'environnement THESIS_RESULTS_DIR.

Sorties :
- results/archive_2024/<pays>/<période>/<signal>/ : métriques clés de chaque stratégie et du benchmark,
  métriques de prédiction par titre, et, pour la période principale 2008-01 à 2024-01, les prédictions
  (y_pred), les poids long/short et les rendements quotidiens de stratégie (csv.gz) ;
- results/figures/<pays>/<période>/<signal>/ : figures PNG (rendements cumulés, SHAP) ;
- results/tables/metriques_portefeuilles.csv : toutes les métriques clés à plat ;
- results/tables/tableau_4_1_regression.csv et tableau_4_2_classification.csv : reconstitution des deux
  tableaux du mémoire (période 2008-01 à 2024-01) avec, en plus, le TCAC recalculé sur années civiles.

Usage :
    python scripts/collect_thesis_results.py "/chemin/vers/04_resultats_2024"
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ml_returns_pred.paths import RESULTS_DIR  # noqa: E402

COUNTRIES = {"Canada Final": "canada", "US Final": "usa"}
MAIN_PERIOD = "2008-01 à 2024-01"
REGRESSORS = ["ridge_regressor", "xgboost_regressor", "ada_boost_regressor", "extra_trees_regressor"]
CLASSIFIERS = ["logistic_regression_classifier", "xgboost_classifier", "hist_gradient_boosting_classifier",
               "extra_trees_classifier"]
BENCHMARK_FILES = {"TSX60", "SP500", "NASDAQ", "equally_weighted"}


def cagr_calendar(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return np.nan
    years = (r.index[-1] - r.index[0]).days / 365.25
    return float((1 + r).prod() ** (1 / years) - 1)


def main(src: Path) -> int:
    import unicodedata

    def nfc(s: str) -> str:  # les noms de dossiers de l'archive sont en Unicode NFD (macOS)
        return unicodedata.normalize("NFC", s)

    rows = []
    for country_dir, country in COUNTRIES.items():
        for period_dir in sorted((src / country_dir).iterdir()):
            if not period_dir.is_dir():
                continue
            for signal_dir in sorted(period_dir.iterdir()):
                if not signal_dir.is_dir():
                    continue
                period_name, signal_name = nfc(period_dir.name), nfc(signal_dir.name)
                inter = signal_dir / "data" / "intermediate_data"
                dest = RESULTS_DIR / "archive_2024" / country / period_name / signal_name
                dest.mkdir(parents=True, exist_ok=True)
                # métriques clés (portefeuilles et benchmarks)
                for f in sorted((inter / "analyze_strategy_returns").glob("*key_metrics.csv")):
                    shutil.copy2(f, dest / f.name)
                    km = pd.read_csv(f, index_col=0).iloc[:, 0]
                    name = f.name.replace("_portfolio_key_metrics.csv", "").replace("_key_metrics.csv", "")
                    sr_file = inter / "compute_strategy_returns" / f"{name}_strategy_returns.csv"
                    cagr_cal = np.nan
                    if sr_file.exists():
                        sr = pd.read_csv(sr_file, index_col=0, parse_dates=True).iloc[:, 0]
                        cagr_cal = cagr_calendar(sr)
                    rows.append({"pays": country, "periode": period_name, "signal": signal_name,
                                 "strategie": name, **km.to_dict(), "CAGR_annees_civiles": cagr_cal})
                # métriques de prédiction par titre
                for f in sorted((inter / "evaluate_model_performance").glob("*metrics_levels.csv")):
                    shutil.copy2(f, dest / f.name)
                # période principale : prédictions, poids, rendements (compressés)
                if period_name == MAIN_PERIOD:
                    for sub, pattern in [("prediction_pipeline", "*_y_pred.csv"), ("weighting", "*_weights.csv"),
                                         ("compute_strategy_returns", "*_strategy_returns.csv"),
                                         ("benchmark_returns", "*.csv"), ("hyperparameters_tuned", "*.csv")]:
                        for f in sorted((inter / sub).glob(pattern)):
                            pd.read_csv(f, index_col=0).to_csv(dest / (f.name + ".gz"), compression="gzip")
                # figures
                figdest = RESULTS_DIR / "figures" / country / period_name / signal_name
                plots = signal_dir / "plots"
                if plots.exists():
                    figdest.mkdir(parents=True, exist_ok=True)
                    for f in plots.rglob("*.png"):
                        shutil.copy2(f, figdest / f.name)
    metrics = pd.DataFrame(rows)
    tables = RESULTS_DIR / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(tables / "metriques_portefeuilles.csv", index=False)

    cols = ["pays", "signal", "strategie", "CAGR", "CAGR_annees_civiles", "Sharpe", "Sortino", "Max_Drawdown",
            "Omega_Ratio", "Cumulative_Returns"]
    main = metrics[metrics["periode"] == MAIN_PERIOD]
    reg = main[main["strategie"].isin(REGRESSORS + list(BENCHMARK_FILES))][cols].sort_values(["pays", "signal", "strategie"])
    cla = main[main["strategie"].isin(CLASSIFIERS + list(BENCHMARK_FILES))][cols].sort_values(["pays", "signal", "strategie"])
    reg.to_csv(tables / "tableau_4_1_regression.csv", index=False)
    cla.to_csv(tables / "tableau_4_2_classification.csv", index=False)
    print(f"{len(metrics)} lignes de métriques ; tableaux écrits dans {tables}")
    print(reg[reg["signal"] == "Top 10"].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("THESIS_RESULTS_DIR")
    if not arg:
        raise SystemExit("Donnez le chemin du dossier de résultats (ou THESIS_RESULTS_DIR).")
    raise SystemExit(main(Path(arg)))
