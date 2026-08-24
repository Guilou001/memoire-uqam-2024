"""Tables et figures de la version 2 (matplotlib, style sobre, export PNG et PDF vectoriel)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

from mlrp.config import RESULTS_DIR

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000", "#999999"]


def _style() -> None:
    plt.rcParams.update({
        "figure.figsize": (7.5, 4.2), "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight", "axes.prop_cycle":
        matplotlib.cycler(color=OKABE_ITO), "figure.constrained_layout.use": True,
    })


def cumulative_plot(returns: dict[str, pd.Series], title: str, out: Path, log_scale: bool = False) -> Path:
    """Courbes de croissance d'un dollar (échelle linéaire ou logarithmique)."""
    _style()
    fig, ax = plt.subplots()
    for name, r in returns.items():
        ax.plot((1 + r.dropna()).cumprod(), label=name, lw=1.2)
    if log_scale:
        ax.set_yscale("log")
    ax.set_title(title)
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


def figures_for_period(returns_dir: Path, country: str, period: str, out_dir: Path = RESULTS_DIR / "figures") -> list[Path]:
    """Une figure par mode : tous les modèles d'un pays et d'une période (signal top10 si présent)."""
    made = []
    labels = {"usa": "États-Unis", "canada": "Canada"}
    mode_labels = {"corrected": "long short", "as_published": "réplication du code de 2024"}
    for mode in ("corrected", "as_published"):
        series = {}
        for f in sorted((returns_dir / country / period).glob(f"*_top10_{mode}.csv.gz")):
            name = f.name.replace(f"_top10_{mode}.csv.gz", "")
            series[name] = pd.read_csv(f, index_col=0, parse_dates=True).iloc[:, 0]
        if series:
            title = f"{labels.get(country, country)}, {period}, top 10, {mode_labels[mode]}, brut de coûts"
            made.append(cumulative_plot(series, title, out_dir / country / f"{period}_top10_{mode}"))
    return made
