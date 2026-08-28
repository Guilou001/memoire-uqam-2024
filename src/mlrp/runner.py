"""Exécution d'une ou plusieurs spécifications : cache des prédictions, portefeuilles, métriques, parallélisme.

Une prédiction (pays, période, modèle, réglages de recherche) est calculée une fois, stockée en parquet/JSON
dans ``data/cache_v2/<clé>/`` et réutilisée par tous les signaux et modes long-short. ``run_many`` calcule
d'abord les prédictions manquantes en parallèle (processus joblib, un cœur chacun pour skforecast), puis
les portefeuilles, peu coûteux.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from mlrp.config import CACHE_DIR, RAW_DIR, RESULTS_DIR, RunSpec
from mlrp.data import Dataset, binarize, build_dataset
from mlrp.metrics import performance_table, r2_oos_by_level
from mlrp.models import PredictionResult, predict
from mlrp.portfolio import StrategyResult, build_weights, equally_weighted_long_only, strategy_returns


# ---------------------------------------------------------------------------------------------- cache
def _cache_paths(key: str, cache_dir: Path) -> dict[str, Path]:
    d = cache_dir / key
    return {"dir": d, "y_pred": d / "y_pred.parquet", "levels": d / "metrics_levels.parquet",
            "tuning": d / "tuning_results.parquet", "meta": d / "meta.json"}


def load_cached(key: str, cache_dir: Path = CACHE_DIR,
                expected_fingerprint: str | None = None) -> PredictionResult | None:
    p = _cache_paths(key, cache_dir)
    if not p["y_pred"].exists() or not p["meta"].exists():
        return None
    meta = json.loads(p["meta"].read_text())
    if expected_fingerprint is not None and meta.get("data_fingerprint") != expected_fingerprint:
        return None  # les données brutes ont changé depuis la mise en cache : prédictions périmées, on recalcule
    tuning = pd.read_parquet(p["tuning"]) if p["tuning"].exists() else None
    return PredictionResult(y_pred=pd.read_parquet(p["y_pred"]), metrics_levels=pd.read_parquet(p["levels"]),
                            tuning_results=tuning, best_params=meta["best_params"], train_size=meta["train_size"])


def save_cached(key: str, pred: PredictionResult, spec: RunSpec, dataset: Dataset, cache_dir: Path = CACHE_DIR,
                seconds: float | None = None) -> None:
    p = _cache_paths(key, cache_dir)
    p["dir"].mkdir(parents=True, exist_ok=True)
    pred.y_pred.to_parquet(p["y_pred"])
    pred.metrics_levels.to_parquet(p["levels"])
    if pred.tuning_results is not None:
        t = pred.tuning_results.copy()
        for c in t.columns:  # les colonnes objets (listes de lags, dicts) deviennent des chaînes
            if t[c].dtype == object:
                t[c] = t[c].astype(str)
        t.to_parquet(p["tuning"])
    p["meta"].write_text(json.dumps({"spec": {**asdict(spec), "tuning": asdict(spec.tuning)},
                                     "best_params": pred.best_params, "train_size": pred.train_size,
                                     "data_fingerprint": dataset.fingerprint(),
                                     "seconds": None if seconds is None else round(seconds, 1),
                                     "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=1, default=str))


# ------------------------------------------------------------------------------------------ exécution
def get_dataset(spec: RunSpec, raw_dir: Path = RAW_DIR, _cache: dict = {}) -> Dataset:  # noqa: B006
    key = (spec.country, spec.max_date, str(raw_dir))
    if key not in _cache:
        _cache[key] = build_dataset(spec.country, spec.max_date, raw_dir)
    return _cache[key]


def get_predictions(spec: RunSpec, dataset: Dataset, cache_dir: Path = CACHE_DIR, n_jobs: int | str = 1,
                    use_cache: bool = True) -> PredictionResult:
    key = spec.prediction_key()
    if use_cache:
        cached = load_cached(key, cache_dir, expected_fingerprint=dataset.fingerprint())
        if cached is not None:
            return cached
    series = dataset.returns_monthly
    if spec.family == "classifier":
        series = binarize(series)
    t0 = time.perf_counter()
    pred = predict(spec.model, series, dataset.exog_monthly, spec.cutoff, spec.tuning, n_jobs=n_jobs)
    if use_cache:
        save_cached(key, pred, spec, dataset, cache_dir, seconds=time.perf_counter() - t0)
    return pred


@dataclass
class RunResult:
    spec: RunSpec
    performance: pd.Series
    benchmark: pd.Series
    equal_weight: pd.Series
    r2_levels: pd.Series | None
    strategy: StrategyResult
    best_params: dict
    seconds: float


def run(spec: RunSpec, raw_dir: Path = RAW_DIR, cache_dir: Path = CACHE_DIR, n_jobs: int | str = 1,
        dataset: Dataset | None = None) -> RunResult:
    t0 = time.time()
    ds = dataset or get_dataset(spec, raw_dir)
    pred = get_predictions(spec, ds, cache_dir, n_jobs=n_jobs)
    long_w, short_w = build_weights(pred.y_pred, spec.signal, spec.family)
    strat = strategy_returns(long_w, short_w, ds.prices_daily, mode=spec.long_short_mode, fee=spec.fee)
    perf = performance_table(strat.returns, spec.model)
    perf["Gross_Exposure"] = float(strat.gross_exposure.median())
    # le premier rééquilibrage (mise en place, rotation forcée à 0) est exclu de la moyenne
    perf["Turnover_Monthly"] = float(strat.turnover.iloc[1:].mean()) if len(strat.turnover) > 1 else 0.0

    start, end = strat.returns.index[0], pd.Timestamp(spec.max_date)
    bench_ret = ds.benchmark_daily.iloc[:, 0].pct_change().loc[start:end]
    bench = performance_table(bench_ret, ds.benchmark_daily.columns[0])
    ew = performance_table(equally_weighted_long_only(ds.returns_monthly, ds.prices_daily, start=str(start.date()),
                                                      fee=spec.fee).returns.loc[:end], "equally_weighted")

    r2 = None
    if spec.family == "regressor":
        y_true = ds.returns_monthly
        r2 = r2_oos_by_level(y_true, pred.y_pred, y_true.iloc[: pred.train_size])
    return RunResult(spec=spec, performance=perf, benchmark=bench, equal_weight=ew, r2_levels=r2, strategy=strat,
                     best_params=pred.best_params, seconds=time.time() - t0)


def _predict_job(spec: RunSpec, raw_dir: Path, cache_dir: Path) -> str:
    ds = build_dataset(spec.country, spec.max_date, raw_dir)
    get_predictions(spec, ds, cache_dir, n_jobs=1)
    return spec.prediction_key()


def run_many(specs: list[RunSpec], n_jobs: int = 1, raw_dir: Path = RAW_DIR, cache_dir: Path = CACHE_DIR,
             results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """Calcule les prédictions manquantes en parallèle puis tous les portefeuilles ; écrit results/v2/*.csv."""
    from joblib import Parallel, delayed

    pending = {}
    for s in specs:
        k = s.prediction_key()
        if k not in pending and load_cached(k, cache_dir) is None:
            pending[k] = s
    if pending:
        Parallel(n_jobs=n_jobs)(delayed(_predict_job)(s, raw_dir, cache_dir) for s in pending.values())

    rows = []
    for s in specs:
        res = run(s, raw_dir, cache_dir, n_jobs=1)
        tuning_label = "none" if not s.tuning.tune else f"bayes{s.tuning.n_trials}" + (
            "+best" if s.tuning.select_best else "")
        row = {"country": s.country, "period": s.period, "model": s.model, "signal": s.signal,
               "mode": s.long_short_mode, "fee": s.fee, "tuning": tuning_label, **res.performance.to_dict(),
               "bench_CAGR": res.benchmark["CAGR"], "bench_Sharpe": res.benchmark["Sharpe"],
               "ew_CAGR": res.equal_weight["CAGR"], "ew_Sharpe": res.equal_weight["Sharpe"],
               "r2_oos_average": None if res.r2_levels is None else res.r2_levels.get("average"),
               "r2_oos_pooling": None if res.r2_levels is None else res.r2_levels.get("pooling"),
               "best_params": json.dumps(res.best_params, default=str), "seconds": round(res.seconds, 1)}
        rows.append(row)
        out = results_dir / "returns" / s.country / s.period
        out.mkdir(parents=True, exist_ok=True)
        res.strategy.returns.to_frame().to_csv(out / f"{s.model}_{s.signal}_{s.long_short_mode}.csv.gz",
                                               compression="gzip")
    table = pd.DataFrame(rows)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "metrics.csv"
    if path.exists():
        old = pd.read_csv(path)
        if "tuning" not in old.columns:  # CSV antérieurs au 2026-08-28 : produits avec le réglage par défaut
            old["tuning"] = "bayes50"
        # le réglage de tuning fait partie de la clé : un run --no-tuning n'écrase plus la ligne tunée
        keys = ["country", "period", "model", "signal", "mode", "fee", "tuning"]
        table = pd.concat([old, table]).drop_duplicates(subset=keys, keep="last")
    table.to_csv(path, index=False)
    return table
