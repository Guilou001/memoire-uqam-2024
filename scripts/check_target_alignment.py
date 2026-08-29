"""Test-oracle : la cible du pipeline est-elle le mois à venir ou le mois écoulé ?

``arithmetic_returns`` est un ``pct_change`` : la ligne datée du 1er mars porte le rendement de
février. Le modèle prédit cette ligne, ``build_weights`` en tire les poids de la même date, et
``strategy_returns`` encaisse mars. Le signal a donc un mois de retard sur ce qu'il prétend prévoir.

Ce script le prouve sans toucher aux modèles : on donne au constructeur de portefeuille la
connaissance PARFAITE de la cible, une fois telle que le pipeline la définit, une fois décalée d'un
mois. Si l'alignement était bon, la première colonne écraserait la seconde. Elle perd de l'argent.

Sortie : ``results/v2/tables/alignement_cible.csv`` et un tableau à l'écran.
"""

from __future__ import annotations

import pandas as pd

from mlrp.config import PERIODS, RESULTS_DIR, RunSpec
from mlrp.metrics import cagr, sharpe
from mlrp.portfolio import build_weights, strategy_returns
from mlrp.runner import get_dataset

CIBLES = {"cible du pipeline (mois écoulé)": lambda r: r,
          "cible décalée (mois à venir)": lambda r: r.shift(-1).iloc[:-1]}


def main() -> int:
    lignes = []
    for pays in ("usa", "canada"):
        ds = get_dataset(RunSpec(country=pays, period="2008-2024"))
        fin = pd.Timestamp(PERIODS["2008-2024"]["max_date"])
        r = ds.returns_monthly.loc["2008-01-01":fin]
        for nom, transforme in CIBLES.items():
            long_w, short_w = build_weights(transforme(r), "top10", "regressor")
            strat = strategy_returns(long_w, short_w, ds.prices_daily, mode="corrected", fee=0.0)
            rendements = strat.returns.loc[:fin]
            lignes.append({"pays": pays, "cible": nom,
                           "tcac_pct": cagr(rendements) * 100.0, "sharpe": sharpe(rendements)})
    table = pd.DataFrame(lignes)
    out = RESULTS_DIR / "tables"
    out.mkdir(parents=True, exist_ok=True)
    table.round(3).to_csv(out / "alignement_cible.csv", index=False)
    print(table.round(2).to_string(index=False))
    print(f"\nécrit : {out / 'alignement_cible.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
