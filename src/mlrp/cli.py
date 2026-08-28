"""Ligne de commande de la version 2.

Exemples::

    mlrp run --country usa --period 2008-2024 --models ridge_regressor --signals top10 --modes corrected,as_published
    mlrp thesis --country canada --jobs 6            # 8 modèles x 4 périodes x 3 signaux x 2 modes
    mlrp figures --country usa --period 2008-2024
    mlrp table --country usa --period 2008-2024 --signal top10
"""

from __future__ import annotations

import argparse
import sys

from mlrp.config import PERIODS, RESULTS_DIR, SIGNALS, THESIS_MODELS, RunSpec, TuningSpec


def _specs(args) -> list[RunSpec]:
    tuning = TuningSpec(n_trials=args.n_trials, tune=not args.no_tuning, select_best=args.select_best)
    countries = ["canada", "usa"] if args.country == "both" else [args.country]
    periods = list(PERIODS) if args.period == "all" else [args.period]
    models = THESIS_MODELS if args.models == "thesis" else args.models.split(",")
    signals = list(SIGNALS) if args.signals == "all" else args.signals.split(",")
    modes = args.modes.split(",")
    return [RunSpec(country=c, period=p, model=m, signal=s, long_short_mode=mode, fee=args.fee, tuning=tuning)
            for c in countries for p in periods for m in models for s in signals for mode in modes]


def cmd_run(args) -> int:
    from mlrp.runner import run_many

    specs = _specs(args)
    print(f"{len(specs)} exécutions, {len({s.prediction_key() for s in specs})} jeux de prédictions, {args.jobs} processus")
    table = run_many(specs, n_jobs=args.jobs)
    cols = ["country", "period", "model", "signal", "mode", "CAGR", "Sharpe", "Volatility", "Max_Drawdown", "seconds"]
    keys = {(s.country, s.period, s.model, s.signal, s.long_short_mode) for s in specs}
    mask = table.apply(lambda r: (r["country"], r["period"], r["model"], r["signal"], r["mode"]) in keys, axis=1)
    print(table.loc[mask, [c for c in cols if c in table.columns]].round(3).to_string(index=False))
    print(f"\nmétriques : {RESULTS_DIR / 'metrics.csv'}")
    return 0


def cmd_thesis(args) -> int:
    args.models, args.signals, args.period = "thesis", "all", "all"
    return cmd_run(args)


def cmd_figures(args) -> int:
    from mlrp.report import figures_for_period

    made = figures_for_period(RESULTS_DIR / "returns", args.country, args.period)
    for p in made:
        print(p)
    return 0


def cmd_shap(args) -> int:
    from mlrp.explain import EXPLAINABLE, shap_figures
    from mlrp.runner import get_dataset

    countries = ["canada", "usa"] if args.country == "both" else [args.country]
    models = list(EXPLAINABLE) if args.models == "all" else args.models.split(",")
    for country in countries:
        spec0 = RunSpec(country=country, period=args.period, model=models[0], signal="top10")
        ds = get_dataset(spec0)
        for model in models:
            if model not in EXPLAINABLE:
                print(f"{model} : pas d'explicateur SHAP adapté (ignoré)")
                continue
            spec = RunSpec(country=country, period=args.period, model=model, signal="top10")
            for p in shap_figures(spec, dataset=ds):
                print(p)
    return 0


def cmd_table(args) -> int:
    import pandas as pd

    from mlrp.report import comparison_table, write_markdown_table

    metrics = pd.read_csv(RESULTS_DIR / "metrics.csv")
    table = comparison_table(metrics, args.country, args.period, args.signal)
    out = write_markdown_table(table, RESULTS_DIR / "tables" / f"{args.country}_{args.period}_{args.signal}.md")
    print(table.round(3).to_string())
    print(f"\nécrit : {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mlrp", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--country", default="canada", choices=["canada", "usa", "both"])
        sp.add_argument("--period", default="2008-2024", choices=[*PERIODS, "all"])
        sp.add_argument("--modes", default="corrected,as_published")
        sp.add_argument("--fee", type=float, default=0.0)
        sp.add_argument("--n-trials", type=int, default=50)
        sp.add_argument("--no-tuning", action="store_true")
        sp.add_argument("--select-best", action="store_true",
                        help="retient le meilleur essai de la recherche (par défaut : pire essai pour les "
                             "régresseurs, artefact du pipeline de 2024 conservé pour la réplication)")
        sp.add_argument("--jobs", type=int, default=1, help="processus en parallèle pour les prédictions")

    r = sub.add_parser("run", help="exécute des modèles, signaux et modes choisis")
    common(r)
    r.add_argument("--models", default="ridge_regressor", help="liste séparée par des virgules, ou 'thesis'")
    r.add_argument("--signals", default="top10", help="top10,top20,positive ou 'all'")
    r.set_defaults(func=cmd_run)

    t = sub.add_parser("thesis", help="toutes les combinaisons du mémoire pour un pays (ou both)")
    common(t)
    t.set_defaults(func=cmd_thesis)

    f = sub.add_parser("figures", help="figures de croissance cumulée pour un pays et une période")
    f.add_argument("--country", default="canada")
    f.add_argument("--period", default="2008-2024")
    f.set_defaults(func=cmd_figures)

    sh = sub.add_parser("shap", help="figures SHAP (essaim et classement) par pays et modèle")
    sh.add_argument("--country", default="canada", choices=["canada", "usa", "both"])
    sh.add_argument("--period", default="2008-2024", choices=list(PERIODS))
    sh.add_argument("--models", default="all", help="liste séparée par des virgules, ou 'all'")
    sh.set_defaults(func=cmd_shap)

    tb = sub.add_parser("table", help="table modèles x modes en markdown")
    tb.add_argument("--country", default="canada")
    tb.add_argument("--period", default="2008-2024")
    tb.add_argument("--signal", default="top10")
    tb.set_defaults(func=cmd_table)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
