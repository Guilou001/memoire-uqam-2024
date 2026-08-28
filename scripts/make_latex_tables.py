"""Régénère les 8 tableaux LaTeX du mémoire (v1.1) depuis les résultats de la version 2.

Reproduit la mise en page du fichier de tableaux de 2024 (siunitx, lignes de groupe grises, meilleur chiffre
du groupe en gras vert) avec les résultats à jour : mode long short (short soustrait), TCAC en années civiles,
univers canadien de 50 titres. Un tableau « régression » et un tableau « classification » par période.

Colonnes régression : R² hors échantillon (moyenne), TCAC, Sharpe, Sortino, perte maximale, Oméga.
Colonnes classification : score de précision (SP), p-value de Pesaran-Timmermann (PT), puis les mêmes.
Sortie : ``reports/memoire_v1.1/tables/tableau_<periode>_<famille>.tex`` (inclus par ``memoire_v1_1.tex``).

Usage : ``uv run python scripts/make_latex_tables.py`` (après ``mlrp thesis --country both``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mlrp.config import CACHE_DIR, PERIODS, ROOT, RunSpec  # noqa: E402

OUT = ROOT / "reports" / "memoire_v1.1" / "tables"
MODE = "corrected"

REGRESSORS = [("ridge_regressor", "Ridge"), ("xgboost_regressor", "XGBoost"),
              ("ada_boost_regressor", "Ada Boost"), ("extra_trees_regressor", "Extra Trees")]
CLASSIFIERS = [("logistic_regression_classifier", "Régression logistique"),
               ("xgboost_classifier", "XGBoost Classification"),
               ("hist_gradient_boosting_classifier", "Hist Gradient Boosting Classification"),
               ("extra_trees_classifier", "Extra Trees Classification")]
SIGNALS = [("top10", r"\textbf{Long short 10}"), ("top20", r"\textbf{Long short 20}"),
           ("positive", r"\textbf{Long short $\gtrless$ 0}")]
PERIOD_LABELS = {"2008-2024": "2008-01/2024-01", "2008-2012": "2008-01/2012-01",
                 "2012-2020": "2012-01/2020-01", "2020-2024": "2020-01/2024-01"}


def fr(x: float, decimals: int = 2, pct: bool = False) -> str:
    """Nombre au format français du mémoire (virgule décimale), en pourcentage si demandé."""
    if pd.isna(x):
        return ""
    v = x * 100 if pct else x
    return f"{v:.{decimals}f}".replace(".", ",").replace("-", "-")


def classifier_stats(country: str, period: str, model: str) -> tuple[float, float]:
    """Score de précision et p-value de Pesaran-Timmermann (moyenne des titres), depuis le cache."""
    spec = RunSpec(country=country, period=period, model=model, signal="top10")
    ml = pd.read_parquet(CACHE_DIR / spec.prediction_key() / "metrics_levels.parquet")
    return float(ml.loc["average", "accuracy_score"]), float(ml.loc["average", "pesaran_timmermann_p_value"])


def stars(p: float) -> str:
    if pd.isna(p):
        return ""
    return "$^{***}$" if p < 0.01 else "$^{**}$" if p < 0.05 else "$^{*}$" if p < 0.10 else ""


def bold_green_mask(block: pd.DataFrame, col: str, maximize: bool = True) -> pd.Series:
    """Vrai pour le meilleur chiffre du groupe (gras vert du mémoire)."""
    s = block[col]
    if s.isna().all():
        return pd.Series(False, index=block.index)
    best = s.max() if maximize else s.min()
    return s == best


def cell(value: str, best: bool) -> str:
    return rf"\bfseries\color{{Green}}{value}" if best and value != "" else value


def rows_for_family(metrics: pd.DataFrame, period: str, family: str) -> list[str]:
    models = REGRESSORS if family == "regressor" else CLASSIFIERS
    lines: list[str] = []
    for signal, label in SIGNALS:
        ncols = 12 if family == "regressor" else 14
        lines.append(r"\rowcolor{Gray}")
        lines.append(label + " & " * ncols + r"\\")
        lines.append(r"\addlinespace[0.1cm]")
        sub = {c: metrics[(metrics.country == c) & (metrics.period == period) & (metrics.signal == signal)
                          & (metrics["mode"] == MODE)].set_index("model") for c in ("canada", "usa")}
        block = {}
        for c in ("canada", "usa"):
            b = sub[c].reindex([m for m, _ in models])
            if family == "classifier":
                sp_pt = [classifier_stats(c, period, m) for m in b.index]
                b = b.assign(SP=[v[0] for v in sp_pt], PT=[v[1] for v in sp_pt])
            block[c] = b
        # Max_Drawdown est négatif : le meilleur est le plus proche de zéro, donc le maximum (corrigé le
        # 2026-08-28 ; l'ancien masque en minimisation mettait la PIRE perte en gras vert). Seule la
        # p-value de Pesaran-Timmermann (PT) se minimise.
        best = {c: {col: bold_green_mask(block[c], col, maximize=(col != "PT"))
                    for col in block[c].columns} for c in ("canada", "usa")}
        for model, name in models:
            parts = [name]
            for c in ("canada", "usa"):
                b, row = block[c], block[c].loc[model]
                if family == "classifier":
                    parts += [cell(fr(row["SP"]), best[c]["SP"][model]),
                              cell(fr(row["PT"]) + stars(row["PT"]), best[c]["PT"][model])]
                else:
                    parts.append(cell(fr(row["r2_oos_average"]), best[c]["r2_oos_average"][model]))
                parts += [cell(fr(row["CAGR"], pct=True), best[c]["CAGR"][model]),
                          cell(fr(row["Sharpe"]), best[c]["Sharpe"][model]),
                          cell(fr(row["Sortino"]), best[c]["Sortino"][model]),
                          cell(fr(abs(row["Max_Drawdown"]), pct=True), best[c]["Max_Drawdown"][model]),
                          cell(fr(row["Omega_Ratio"]), best[c]["Omega_Ratio"][model])]
            lines.append(" & ".join(parts) + r" \\ \addlinespace[0.1cm]")
    # critères de référence (équipondéré du pays + indices), communs aux deux familles
    lines.append(r"\rowcolor{Gray}")
    ncols = 12 if family == "regressor" else 14
    lines.append(r"\textbf{Critères de références}" + " & " * ncols + r"\\")
    lines.append(r"\addlinespace[0.1cm]")
    pad = 1 if family == "regressor" else 2
    ref = metrics[(metrics.period == period) & (metrics.signal == "top10") & (metrics["mode"] == MODE)
                  & (metrics.model == "ridge_regressor")].set_index("country")
    ew = " & ".join(["Même pondération"]
                    + sum(([""] * pad + [fr(ref.loc[c, "ew_CAGR"], pct=True), fr(ref.loc[c, "ew_Sharpe"]), "", "", ""]
                           for c in ("canada", "usa")), []))
    bench = " & ".join(["Indice (TSX / S\\&P 500)"]
                       + sum(([""] * pad + [fr(ref.loc[c, "bench_CAGR"], pct=True), fr(ref.loc[c, "bench_Sharpe"]), "", "", ""]
                              for c in ("canada", "usa")), []))
    lines.append(ew + r" \\ \addlinespace[0.1cm]")
    lines.append(bench + r" \\ \addlinespace[0.1cm]")
    return lines


def table_for(metrics: pd.DataFrame, period: str, family: str) -> str:
    label = PERIOD_LABELS[period]
    fam_label = "régression" if family == "regressor" else "classification"
    if family == "regressor":
        colspec = r"l *{6}{S[table-format=3.2]} | *{6}{S[table-format=3.2]}"
        span, cmid = 6, r"\cmidrule(lr){2-7} \cmidrule(lr){8-13}"
        heads = (r"& \textit{\(R^2_{\text{HÉ}}\)} & \textit{\(r^A\)} & \textit{SR} & \textit{ST} & "
                 r"\textit{P\,$^{\text{max}}$} & \textit{$\Omega$} ") * 2 + r"\\"
    else:
        colspec = r"l *{7}{S[table-format=3.2]} | *{7}{S[table-format=3.2]}"
        span, cmid = 7, r"\cmidrule(lr){2-8} \cmidrule(lr){9-15}"
        heads = (r"& \textit{SP} & \textit{PT} & \textit{\(r^A\)} & \textit{SR} & \textit{ST} & "
                 r"\textit{P\,$^{\text{max}}$} & \textit{$\Omega$} ") * 2 + r"\\"
    body = "\n".join(rows_for_family(metrics, period, family))
    return "\n".join([
        r"\begin{table}[h]", r"\centering", r"\small",
        rf"\caption{{Résumé statistique des portefeuilles long short, {label} ({fam_label})}}",
        r"\vspace{.2cm}", r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", r"\midrule",
        rf"& \multicolumn{{{span}}}{{c}}{{Canada}} & \multicolumn{{{span}}}{{c}}{{États-Unis}} \\", cmid,
        heads, r"\midrule", body, r"\bottomrule", r"\end{tabular}}", r"\end{table}",
    ])


def main() -> int:
    metrics = pd.read_csv(ROOT / "results" / "v2" / "metrics.csv")
    OUT.mkdir(parents=True, exist_ok=True)
    for period in PERIODS:
        for family in ("regressor", "classifier"):
            have = metrics[(metrics.period == period) & (metrics["mode"] == MODE)]
            if have.empty:
                print(f"{period} : pas encore de résultats, tableau non produit")
                continue
            out = OUT / f"tableau_{period}_{family}.tex"
            out.write_text(table_for(metrics, period, family) + "\n", encoding="utf-8")
            print(f"écrit : {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
