"""Spécifications d'exécution et constantes partagées (pays, périodes, modèles, signaux)."""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
RAW_DIR = ROOT / "data" / "raw_data"
CACHE_DIR = ROOT / "data" / "cache_v2"
RESULTS_DIR = ROOT / "results" / "v2"

COUNTRIES: dict[str, dict] = {
    "canada": {
        "prices": "canadian_stocks_2000-01-01_to_2024-06-01.csv",
        "macro": "macro_data.csv",
        "macro_kind": "lcdma",
        "benchmark": "TSX60_2000-01-01_to_2024-06-01.csv",
        "benchmark_name": "TSX60",
    },
    "usa": {
        "prices": "us_stocks_2000-01-01_to_2024-06-01.csv",
        "macro": "Fred-MD.csv",
        "macro_kind": "fredmd",
        "benchmark": "NASDAQ_2000-01-01_to_2024-06-01.csv",
        "benchmark_name": "NASDAQ",
    },
}

PERIODS: dict[str, dict[str, str]] = {
    "2008-2024": {"cutoff": "2007-12-31", "max_date": "2024-01-01"},
    "2008-2012": {"cutoff": "2007-12-31", "max_date": "2012-01-01"},
    "2012-2020": {"cutoff": "2011-12-31", "max_date": "2020-01-01"},
    "2020-2024": {"cutoff": "2019-12-31", "max_date": "2024-01-01"},
}

REGRESSORS = ["ridge_regressor", "xgboost_regressor", "ada_boost_regressor", "extra_trees_regressor"]
CLASSIFIERS = ["logistic_regression_classifier", "xgboost_classifier", "hist_gradient_boosting_classifier",
               "extra_trees_classifier"]
THESIS_MODELS = REGRESSORS + CLASSIFIERS
SIGNALS = ("top10", "top20", "positive")
LONG_SHORT_MODES = ("corrected", "as_published")


def model_family(model: str) -> str:
    """``"classifier"`` si le nom se termine par classifier, sinon ``"regressor"``."""
    return "classifier" if model.endswith("classifier") else "regressor"


@dataclass(frozen=True)
class TuningSpec:
    """Paramètres de la recherche bayésienne (identiques au mémoire par défaut)."""

    n_trials: int = 50
    lags_grid: tuple[int, ...] = (12, 24)
    lags_default: int = 6
    seed: int = 123
    tune: bool = True


@dataclass(frozen=True)
class RunSpec:
    """Une exécution = un pays, une période, un modèle, un signal, un mode de calcul du long-short."""

    country: str = "canada"
    period: str = "2008-2024"
    model: str = "ridge_regressor"
    signal: str = "top10"
    long_short_mode: str = "corrected"
    fee: float = 0.0
    tuning: TuningSpec = field(default_factory=TuningSpec)

    def __post_init__(self) -> None:
        if self.country not in COUNTRIES:
            raise ValueError(f"pays inconnu : {self.country}")
        if self.period not in PERIODS:
            raise ValueError(f"période inconnue : {self.period}")
        if self.signal not in SIGNALS:
            raise ValueError(f"signal inconnu : {self.signal} (attendu : {SIGNALS})")
        if self.long_short_mode not in LONG_SHORT_MODES:
            raise ValueError(f"mode inconnu : {self.long_short_mode}")

    @property
    def cutoff(self) -> str:
        return PERIODS[self.period]["cutoff"]

    @property
    def max_date(self) -> str:
        return PERIODS[self.period]["max_date"]

    @property
    def family(self) -> str:
        return model_family(self.model)

    @property
    def top_k(self) -> int | None:
        return {"top10": 10, "top20": 20}.get(self.signal)

    def prediction_key(self) -> str:
        """Clé de cache des prédictions : ne dépend ni du signal, ni du mode long-short, ni des coûts."""
        payload = {"country": self.country, "period": self.period, "model": self.model, "tuning": asdict(self.tuning)}
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]

    def label(self) -> str:
        return f"{self.country}/{self.period}/{self.model}/{self.signal}/{self.long_short_mode}"


def _parse_tuples(value):
    """Les YAML du mémoire écrivent les intervalles comme des chaînes ``"(0.01, 1.0)"``."""
    if isinstance(value, str) and value.startswith("(") and value.endswith(")"):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value
    if isinstance(value, dict):
        return {k: _parse_tuples(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_parse_tuples(v) for v in value]
    return value


def load_model_space(model: str, config_dir: Path = CONFIG_DIR) -> tuple[dict, dict]:
    """Retourne (paramètres de construction de l'estimateur, espace de recherche bayésienne) du YAML de stratégie."""
    path = config_dir / "strategy_config" / f"{model}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"pas de configuration pour {model} : {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)["prediction_pipeline"]
    params = _parse_tuples(cfg.get("regressor_dict") or {})
    space = _parse_tuples(cfg.get("bayes_search_params_grid") or {})
    return dict(params), dict(space)
