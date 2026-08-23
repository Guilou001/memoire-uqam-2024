"""Impact du décalage des exogènes : macro du mois M+1 (mémoire) contre macro du mois M-1 (temps réel), Ridge US top10."""
import pandas as pd

from mlrp.config import PERIODS, RAW_DIR, TuningSpec
from mlrp.data import build_dataset
from mlrp.metrics import cagr, r2_oos_by_level, sharpe
from mlrp.models import predict
from mlrp.portfolio import build_weights, strategy_returns

per = PERIODS["2008-2024"]
ds = build_dataset("usa", per["max_date"], RAW_DIR)
tun = TuningSpec()  # 50 essais, graine 123, comme le mémoire
rows = {}
for name, (series, exog) in {
    "memoire (macro M+1)": (ds.returns_monthly, ds.exog_monthly),
    "temps reel (macro M-1)": (ds.returns_monthly.iloc[1:], ds.exog_monthly.shift(1).iloc[1:]),
}.items():
    res = predict("ridge_regressor", series, exog, per["cutoff"], tun, n_jobs=8)
    lw, sw = build_weights(res.y_pred, "top10", "regressor")
    r2 = r2_oos_by_level(series.loc[res.y_pred.index], res.y_pred, series.iloc[:res.train_size])
    line = {"r2_pooling": float(r2.loc["pooling"]), "r2_average": float(r2.loc["average"])}
    for mode in ("as_published", "corrected"):
        out = strategy_returns(lw, sw, ds.prices_daily, mode=mode)
        line[f"CAGR_{mode}"] = cagr(out.returns)
        line[f"Sharpe_{mode}"] = sharpe(out.returns)
    rows[name] = line
    print("fini :", name, flush=True)
t = pd.DataFrame(rows).T
pd.set_option("display.width", 200)
print(t.round(4).to_string())
