"""Interface en ligne de commande du pipeline du mémoire.

Exemples::

    ml-returns-pred list
    ml-returns-pred run --strategy ridge_regressor --country canada --signal top --top 10 --period 2008-2024
    ml-returns-pred run --strategy xgboost_classifier --country usa --signal positive --period 2020-2024 --n-trials 5
    ml-returns-pred run --strategy ridge_regressor --country usa --long-short-mode corrected

Toutes les options surchargent les fichiers YAML de ``config/`` sans les modifier. Les
chemins relatifs du code historique sont résolus depuis ``workdir/`` (voir ``paths.py``).
"""

from __future__ import annotations

import argparse
import sys
import time

from ml_returns_pred.paths import ROOT, chdir_workdir

CANADIAN_TICKERS = [
    "ABX.TO", "AEM.TO", "ATD.TO", "BB.TO", "BBD-B.TO", "BCE.TO", "BMO.TO", "BN.TO", "BLDP.TO", "BNS.TO",
    "CAE.TO", "CCA.TO", "CCL-B.TO", "CCO.TO", "CM.TO", "CNR.TO", "CTC.TO", "CTC-A.TO", "EMA.TO", "EMP-A.TO",
    "ENGH.TO", "ENB.TO", "FTS.TO", "FTT.TO", "GIL.TO", "HR-UN.TO", "IMO.TO", "L.TO", "MFC.TO", "MFI.TO",
    "MRU.TO", "MTL.TO", "NA.TO", "ONEX.TO", "POW.TO", "RCI-B.TO", "RY.TO", "SAP.TO", "SJ.TO", "STN.TO",
    "SU.TO", "T.TO", "TCL-A.TO", "TECK-B.TO", "TRP.TO", "TD.TO", "WN.TO", "WDO.TO", "XIU.TO",
]  # 49 titres uniques : la liste du mémoire contenait ENB.TO deux fois et le FNB XIU.TO

US_TICKERS = [
    "AAPL", "ABT", "ACN", "AMGN", "AMZN", "AOS", "BA", "CAT", "CL", "CMCSA", "COST", "CRM", "CSCO", "CVX",
    "DHR", "DIS", "GE", "GIS", "GLW", "GOOGL", "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "LIN",
    "LLY", "LMT", "LOW", "MCD", "MDT", "MMM", "MRK", "MSFT", "NKE", "PEP", "PFE", "PG", "QCOM", "SBUX", "T",
    "TXN", "UNH", "UPS", "VZ", "WMT", "XOM",
]

COUNTRIES = {
    "canada": {
        "tickers": CANADIAN_TICKERS,
        "stocks_file_name": "canadian_stocks",
        "benchmark_ticker": "^GSPTSE",
        "benchmark_file_name": "TSX60",
        "relative_prices_data_path": "../data/raw_data/canadian_stocks_2000-01-01_to_2024-06-01.csv",
        "relative_macro_data_path": "../data/raw_data/macro_data.csv",
        "benchmark_prices_relative_path": "../data/raw_data/TSX60_2000-01-01_to_2024-06-01.csv",
    },
    "usa": {
        "tickers": US_TICKERS,
        "stocks_file_name": "us_stocks",
        "benchmark_ticker": "^IXIC",
        "benchmark_file_name": "NASDAQ",
        "relative_prices_data_path": "../data/raw_data/us_stocks_2000-01-01_to_2024-06-01.csv",
        "relative_macro_data_path": "../data/raw_data/Fred-MD.csv",
        "benchmark_prices_relative_path": "../data/raw_data/NASDAQ_2000-01-01_to_2024-06-01.csv",
    },
}

PERIODS = {
    "2008-2024": {"cutoff_date": "2007-12-31", "max_date": "2024-01-01"},
    "2008-2012": {"cutoff_date": "2007-12-31", "max_date": "2012-01-01"},
    "2012-2020": {"cutoff_date": "2011-12-31", "max_date": "2020-01-01"},
    "2020-2024": {"cutoff_date": "2019-12-31", "max_date": "2024-01-01"},
}

# Les huit modèles du mémoire (Tableaux 4.1 et 4.2) plus le portefeuille équipondéré de référence.
THESIS_STRATEGIES = [
    "equally_weighted",
    "ridge_regressor", "xgboost_regressor", "ada_boost_regressor", "extra_trees_regressor",
    "logistic_regression_classifier", "xgboost_classifier", "hist_gradient_boosting_classifier",
    "extra_trees_classifier",
]


def _pipeline_class(strategy: str):
    """Retourne la classe de pipeline correspondant au nom de stratégie des fichiers YAML."""
    from ml_returns_pred import main as m

    mapping = {
        "equally_weighted": m.EquallyWeightedPipeline,
        "ridge_regressor": m.RidgeRegressorPipeline,
        "xgboost_regressor": m.XGBoostRegressorPipeline,
        "ada_boost_regressor": m.AdaBoostRegressorPipeline,
        "extra_trees_regressor": m.ExtraTreesRegressorPipeline,
        "logistic_regression_classifier": m.LogisticRegressionClassifierPipeline,
        "xgboost_classifier": m.XGBoostClassifierPipeline,
        "hist_gradient_boosting_classifier": m.HistGradientBoostingClassifierPipeline,
        "extra_trees_classifier": m.ExtraTreesClassifierPipeline,
        "lasso_regressor": m.LassoRegressorPipeline,
        "gradient_boosting_regressor": m.GradientBoostingRegressorPipeline,
        "ada_boost_classifier": m.AdaBoostClassifierPipeline,
        "knn_classifier": m.KNNClassifierPipeline,
        "elastic_net_regressor": m.ElasticNetRegressorPipeline,
        "random_forest_regressor": m.RandomForestRegressorPipeline,
        "random_forest_classifier": m.RandomForestClassifierPipeline,
        "mlp_regressor": m.MLPRegressorPipeline,
        "mlp_classifier": m.MLPClassifierPipeline,
        "light_gbm_regressor": m.LightGBMRegressorPipeline,
        "catboost_regressor": m.CatBoostRegressorPipeline,
        "catboost_classifier": m.CatBoostClassifierPipeline,
        "linear_regression_regressor": m.LinearRegressionRegressorPipeline,
    }
    try:
        return mapping[strategy]
    except KeyError as exc:
        raise SystemExit(f"Stratégie inconnue : {strategy}. Voir `ml-returns-pred list`.") from exc


def apply_overrides(config: dict, *, country: str, period: str, signal: str, top: int,
                    n_trials: int | None, fee: float, long_short_mode: str, download: bool,
                    benchmark: str | None, tuning: bool) -> dict:
    """Surcharge la configuration fusionnée avec les options de la ligne de commande."""
    c = COUNTRIES[country]
    p = PERIODS[period]

    config["folder_cleaner"]["clean_data"] = False  # jamais de nettoyage implicite
    config["download_data"]["download_data"] = download
    config["download_data"]["tickers"] = c["tickers"]
    config["download_data"]["stocks_file_name"] = c["stocks_file_name"]
    config["download_data"]["benchmark_ticker"] = c["benchmark_ticker"]
    config["download_data"]["benchmark_file_name"] = c["benchmark_file_name"]

    config["read_data"]["relative_prices_data_path"] = c["relative_prices_data_path"]
    config["read_data"]["relative_macro_data_path"] = c["relative_macro_data_path"]
    config["read_data"]["benchmark_prices_relative_path"] = (
        f"../data/raw_data/{benchmark}_2000-01-01_to_2024-06-01.csv" if benchmark
        else c["benchmark_prices_relative_path"]
    )

    config["preprocess_data"]["max_date"] = p["max_date"]
    config["prediction_pipeline"]["cutoff_date"] = p["cutoff_date"]
    config["prediction_pipeline"]["optimize_hyperparameters"] = tuning
    if n_trials is not None:
        config["prediction_pipeline"]["bayes_search_params"]["n_trials"] = n_trials

    lsp = config["create_long_short_portfolio"]
    if signal == "top":
        lsp.update({"use_ranking": True, "use_ranking_as_signals": True, "fix_threshold": top,
                    "transform_continuous_to_binary": False, "transform_binary_classification_to_rank": False,
                    "selection_method": "fix", "method": "first"})
    elif signal == "positive":
        lsp.update({"use_ranking": False, "use_ranking_as_signals": False, "fix_threshold": 1,
                    "transform_continuous_to_binary": True, "transform_binary_classification_to_rank": True})
    # signal == "config" : on garde les valeurs des YAML

    config["compute_strategy_returns"]["transaction_fee"] = fee
    config["compute_strategy_returns"]["long_short_mode"] = long_short_mode
    return config


def cmd_run(args: argparse.Namespace) -> int:
    chdir_workdir()
    cls = _pipeline_class(args.strategy)
    pipeline = cls()
    pipeline.config = apply_overrides(
        pipeline.config, country=args.country, period=args.period, signal=args.signal, top=args.top,
        n_trials=args.n_trials, fee=args.fee, long_short_mode=args.long_short_mode, download=args.download,
        benchmark=args.benchmark, tuning=not args.no_tuning,
    )
    if args.strategy == "equally_weighted" and args.signal == "top":
        # le portefeuille de référence est long seul sur tous les titres : on garde sa configuration YAML
        pipeline.config["create_long_short_portfolio"] = cls().config["create_long_short_portfolio"]
    t0 = time.time()
    pipeline.main()
    print(f"\nTerminé : {args.strategy} / {args.country} / {args.signal}{args.top if args.signal=='top' else ''} / "
          f"{args.period} / mode {args.long_short_mode} en {time.time() - t0:.0f} s. Sorties : "
          f"{ROOT / 'data' / 'intermediate_data'} et {ROOT / 'reports'}")
    return 0


def cmd_list(_: argparse.Namespace) -> int:
    print("Stratégies du mémoire :", ", ".join(THESIS_STRATEGIES))
    print("Pays :", ", ".join(COUNTRIES))
    print("Périodes :", ", ".join(PERIODS))
    print("Signaux : top (--top N), positive, config")
    print("Modes long-short : as_published (code de 2024), corrected (long moins short)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ml-returns-pred", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="exécute un pipeline complet (données -> prédictions -> portefeuille -> métriques)")
    run.add_argument("--strategy", required=True)
    run.add_argument("--country", choices=list(COUNTRIES), default="canada")
    run.add_argument("--period", choices=list(PERIODS), default="2008-2024")
    run.add_argument("--signal", choices=["top", "positive", "config"], default="top")
    run.add_argument("--top", type=int, default=10, help="nombre de titres par jambe quand --signal top")
    run.add_argument("--n-trials", type=int, default=None, help="essais Optuna (défaut YAML : 50)")
    run.add_argument("--no-tuning", action="store_true", help="saute la recherche bayésienne")
    run.add_argument("--fee", type=float, default=0.0, help="coût de transaction proportionnel (0.001 = 10 pb)")
    run.add_argument("--long-short-mode", choices=["as_published", "corrected"], default="as_published")
    run.add_argument("--download", action="store_true", help="télécharge les prix via yfinance avant d'exécuter")
    run.add_argument("--benchmark", default=None, help="TSX60, SP500 ou NASDAQ (fichier dans data/raw_data)")
    run.set_defaults(func=cmd_run)

    lst = sub.add_parser("list", help="liste les stratégies, pays, périodes et signaux")
    lst.set_defaults(func=cmd_list)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
