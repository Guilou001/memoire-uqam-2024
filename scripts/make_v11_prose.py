"""Rédige ``reports/memoire_v1.1/prose_resultats.tex`` : le texte des résultats, avec les chiffres du run v2.

Chaque nombre est lu dans ``results/v2/metrics.csv`` (mode long short, TCAC civil, univers canadien de 50
titres) ; rien n'est retapé à la main. À lancer après ``mlrp thesis --country both`` et avant la compilation
de ``memoire_v1_1.tex``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mlrp.config import ROOT  # noqa: E402

OUT = ROOT / "reports" / "memoire_v1.1" / "prose_resultats.tex"
NAMES = {"ridge_regressor": "le Ridge", "xgboost_regressor": "le XGBoost", "ada_boost_regressor": "l'Ada Boost",
         "extra_trees_regressor": "l'Extra Trees", "logistic_regression_classifier": "la régression logistique",
         "xgboost_classifier": "le XGBoost de classification",
         "hist_gradient_boosting_classifier": "le Hist Gradient Boosting",
         "extra_trees_classifier": "l'Extra Trees de classification"}
PERIOD_LABELS = {"2008-2024": "2008-01 à 2024-01", "2008-2012": "2008-01 à 2012-01",
                 "2012-2020": "2012-01 à 2020-01", "2020-2024": "2020-01 à 2024-01"}


def pc(x: float) -> str:
    return f"{x * 100:.1f}".replace(".", ",").replace("-", "$-$") + "~\\%"


def num(x: float) -> str:
    return f"{x:.2f}".replace(".", ",").replace("-", "$-$")


def paragraph(m: pd.DataFrame, period: str) -> str:
    sub = m[(m.period == period) & (m["mode"] == "corrected") & (m.signal == "top10")]
    if sub.empty:
        return ""
    lines = [f"\\subsection*{{Période {PERIOD_LABELS[period]}}}\n"]
    for country, label, bench_name in (("usa", "Aux États-Unis", "le S\\&P 500"),
                                       ("canada", "Au Canada", "le composite TSX")):
        c = sub[sub.country == country].set_index("model")
        if c.empty:
            continue
        best = c["CAGR"].idxmax()
        pos = int((c["CAGR"] > 0).sum())
        bench_cagr, bench_sharpe = float(c["bench_CAGR"].iloc[0]), float(c["bench_Sharpe"].iloc[0])
        ew_cagr, ew_sharpe = float(c["ew_CAGR"].iloc[0]), float(c["ew_Sharpe"].iloc[0])
        beats_ew = int((c["Sharpe"] > ew_sharpe).sum())
        lines.append(
            f"{label}, sur les huit portefeuilles long short top~10, {pos} affichent un TCAC positif. "
            f"Le meilleur est {NAMES[best]}, avec {pc(c.loc[best, 'CAGR'])} par an (Sharpe {num(c.loc[best, 'Sharpe'])}, "
            f"perte maximale {pc(abs(c.loc[best, 'Max_Drawdown']))}). "
            f"Sur la même période, {bench_name} rend {pc(bench_cagr)} (Sharpe {num(bench_sharpe)}) et le portefeuille "
            f"équipondéré des titres de l'univers {pc(ew_cagr)} (Sharpe {num(ew_sharpe)}) ; "
            f"{beats_ew} portefeuille(s) long short sur huit font mieux que l'équipondéré en Sharpe.\n"
        )
    reg = sub[sub.country == "usa"]
    if not reg.empty and reg["r2_oos_average"].notna().any():
        r2 = reg.set_index("model")["r2_oos_average"].dropna()
        lines.append(
            f"Les $R^2$ hors échantillon moyens des régressions restent négatifs ou nuls "
            f"(de {num(float(r2.min()))} à {num(float(r2.max()))} côté américain) : les prédictions expliquent moins "
            f"bien les rendements qu'une moyenne simple, et la valeur des portefeuilles vient du classement, non du "
            f"niveau prédit. Les modèles dont les prédictions sont identiques pour tous les titres à la plupart des "
            f"dates (voir \\texttt{{results/v2/tables/prediction\\_ties.csv}}) doivent se lire comme des portefeuilles "
            f"quasi fixes.\n"
        )
    return "\n".join(lines)


def main() -> int:
    m = pd.read_csv(ROOT / "results" / "v2" / "metrics.csv")
    parts = ["% Généré par scripts/make_v11_prose.py, ne pas éditer à la main.\n"]
    parts += [paragraph(m, p) for p in PERIOD_LABELS]
    OUT.write_text("\n".join(x for x in parts if x), encoding="utf-8")
    print(f"écrit : {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
