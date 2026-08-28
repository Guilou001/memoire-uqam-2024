"""Tables et figures de la version 2 (matplotlib, style sobre, export PNG et PDF vectoriel)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

from mlrp.config import MODEL_LABELS, RESULTS_DIR

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000", "#999999"]


def _style() -> None:
    plt.rcParams.update({
        "figure.figsize": (7.5, 4.2), "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight", "axes.prop_cycle":
        matplotlib.cycler(color=OKABE_ITO), "figure.constrained_layout.use": True,
    })


def cumulative_plot(returns: dict[str, pd.Series], title: str, out: Path, log_scale: bool = False,
                    reference: str | None = None) -> Path:
    """Courbes de croissance d'un dollar (échelle linéaire ou logarithmique).

    ``reference`` : nom d'une série à tracer en pointillés gris épais (repère, hors cycle de couleurs).
    """
    _style()
    fig, ax = plt.subplots()
    for name, r in returns.items():
        wealth = (1 + r.dropna()).cumprod()
        if name == reference:
            ax.plot(wealth, label=name, lw=1.8, ls="--", color="#666666")
        else:
            ax.plot(wealth, label=name, lw=1.2)
    if log_scale:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Valeur d'un dollar investi")
    ax.legend(ncol=2, fontsize=8)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=160)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out.with_suffix(".png")


def comparison_table(metrics: pd.DataFrame, country: str, period: str, signal: str,
                     cols: tuple[str, ...] = ("CAGR", "Sharpe", "Volatility", "Max_Drawdown", "Sortino", "Omega_Ratio",
                                              "Gross_Exposure", "r2_oos_average")) -> pd.DataFrame:
    """Table modèles x modes (as_published contre corrected) pour un pays, une période, un signal."""
    sub = metrics[(metrics["country"] == country) & (metrics["period"] == period) & (metrics["signal"] == signal)]
    present = [c for c in cols if c in sub.columns]
    table = sub.pivot_table(index="model", columns="mode", values=present)
    # colonnes plates et dans l'ordre demandé : « CAGR (as_published) », « CAGR (corrected) », …
    modes = [m for m in ("as_published", "corrected") if m in sub["mode"].unique()]
    table = table.reindex(columns=pd.MultiIndex.from_product([present, modes]))
    table.columns = [f"{metric} ({mode})" for metric, mode in table.columns]
    return table


def write_markdown_table(df: pd.DataFrame, out: Path, floatfmt: str = ".3f") -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        text = df.to_markdown(floatfmt=floatfmt)
    except ImportError:  # tabulate absent
        text = df.round(3).to_string()
    out.write_text(text + "\n")
    return out


def _equal_weight_reference(country: str, period: str, start: pd.Timestamp) -> pd.Series | None:
    """Rendements quotidiens de l'équipondéré du pays (repère des figures) ; None si les données brutes manquent."""
    from mlrp.config import PERIODS, RunSpec
    from mlrp.portfolio import equally_weighted_long_only
    from mlrp.runner import get_dataset

    try:
        ds = get_dataset(RunSpec(country=country, period=period))
    except (FileNotFoundError, KeyError):
        return None
    end = pd.Timestamp(PERIODS[period]["max_date"])
    return equally_weighted_long_only(ds.returns_monthly, ds.prices_daily, start=str(start.date())).returns.loc[:end]


def figures_for_period(returns_dir: Path, country: str, period: str, out_dir: Path = RESULTS_DIR / "figures") -> list[Path]:
    """Une figure par mode : tous les modèles d'un pays et d'une période (signal top10 si présent),
    plus la courbe équipondérée du pays en repère (pointillés gris)."""
    made = []
    labels = {"usa": "États-Unis", "canada": "Canada"}
    mode_labels = {"corrected": "long short", "as_published": "réplication du code de 2024"}
    reference = "Équipondéré (référence)"
    for mode in ("corrected", "as_published"):
        series = {}
        for f in sorted((returns_dir / country / period).glob(f"*_top10_{mode}.csv.gz")):
            name = f.name.replace(f"_top10_{mode}.csv.gz", "")
            series[MODEL_LABELS.get(name, name)] = pd.read_csv(f, index_col=0, parse_dates=True).iloc[:, 0]
        if series:
            ew = _equal_weight_reference(country, period, min(s.index[0] for s in series.values()))
            if ew is not None:
                series[reference] = ew
            title = f"{labels.get(country, country)}, {period}, top 10, {mode_labels[mode]}, brut de coûts"
            made.append(cumulative_plot(series, title, out_dir / country / f"{period}_top10_{mode}",
                                        reference=reference))
    return made
