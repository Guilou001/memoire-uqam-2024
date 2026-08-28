"""Figure de synthèse du README : TCAC par modèle, États-Unis et Canada côte à côte, repère équipondéré.

Lit ``results/v2/metrics.csv`` (portefeuilles top 10, mode long short, 2008-2024, sans coûts) et trace des
barres alignées par modèle pour les deux pays, avec une ligne pointillée par pays au niveau du portefeuille
équipondéré. Palette Okabe-Ito, police 11, DPI 200.

Usage : ``uv run python scripts/make_summary_figure.py`` (après ``mlrp thesis --country both``).
Sortie : ``results/v2/figures/summary_cagr_par_modele.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mlrp.config import MODEL_LABELS, ROOT, THESIS_MODELS  # noqa: E402

OUT = ROOT / "results" / "v2" / "figures" / "summary_cagr_par_modele.png"
PERIOD, SIGNAL, MODE = "2008-2024", "top10", "corrected"
BLUE, ORANGE = "#0072B2", "#D55E00"  # Okabe-Ito


def main() -> int:
    m = pd.read_csv(ROOT / "results" / "v2" / "metrics.csv")
    sub = m[(m.period == PERIOD) & (m.signal == SIGNAL) & (m["mode"] == MODE)]
    usa = sub[sub.country == "usa"].set_index("model").reindex(THESIS_MODELS)
    can = sub[sub.country == "canada"].set_index("model").reindex(THESIS_MODELS)
    ew_usa, ew_can = float(usa["ew_CAGR"].iloc[0]), float(can["ew_CAGR"].iloc[0])

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "figure.constrained_layout.use": True})
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(THESIS_MODELS))
    ax.bar(x - 0.2, usa["CAGR"] * 100, width=0.38, color=BLUE, label="États-Unis")
    ax.bar(x + 0.2, can["CAGR"] * 100, width=0.38, color=ORANGE, label="Canada")
    def pc(v: float) -> str:
        return f"{v * 100:.1f}".replace(".", ",") + " %"

    ax.axhline(ew_usa * 100, color=BLUE, ls="--", lw=1.4, label=f"Équipondéré É.-U. ({pc(ew_usa)})")
    ax.axhline(ew_can * 100, color=ORANGE, ls="--", lw=1.4, label=f"Équipondéré Canada ({pc(ew_can)})")
    ax.axhline(0, color="#000000", lw=0.8)
    ax.set_xticks(x, [MODEL_LABELS[mdl] for mdl in THESIS_MODELS], rotation=30, ha="right")
    ax.set_ylabel("TCAC 2008-2024 (% par an)")
    ax.set_xlabel("Modèle (portefeuille long short top 10, sans coûts)")
    ax.set_title("Aucun portefeuille long short ne rejoint l'équipondéré de son pays")
    ax.legend(ncol=2, fontsize=10)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"écrit : {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
