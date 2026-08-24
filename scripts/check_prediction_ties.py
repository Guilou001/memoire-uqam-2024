"""Part des dates où un modèle prédit la même valeur pour tous les titres (égalités de rangs).

Quand toutes les prédictions d'une date sont égales, le classement « top 10 » ne dépend plus du modèle mais
de l'ordre des colonnes (alphabétique). Ce script mesure cette part pour chaque modèle et chaque pays sur la
période principale, depuis le cache de prédictions de ``mlrp run``, et écrit le résultat en CSV.
"""

import json

import pandas as pd

from mlrp.config import CACHE_DIR, RESULTS_DIR, THESIS_MODELS, RunSpec, load_model_space

rows = []
for country in ("usa", "canada"):
    for model in THESIS_MODELS:
        spec = RunSpec(country=country, period="2008-2024", model=model, signal="top10")
        d = CACHE_DIR / spec.prediction_key()
        if not (d / "y_pred.parquet").exists():
            print(f"{country} {model} : pas de cache (lancer mlrp run), ignoré")
            continue
        y = pd.read_parquet(d / "y_pred.parquet")
        meta = json.loads((d / "meta.json").read_text())
        base, _ = load_model_space(model)
        std_cross = y.std(axis=1)
        rows.append({
            "country": country, "model": model,
            "ccp_alpha_base": base.get("ccp_alpha"), "ccp_alpha_retenu": meta["best_params"].get("ccp_alpha"),
            "std_transversale_mediane": float(std_cross.median()),
            "part_dates_predictions_identiques": round(float((std_cross < 1e-12).mean()), 4),
            "n_dates": int(len(y)),
        })

table = pd.DataFrame(rows)
out = RESULTS_DIR / "tables" / "prediction_ties.csv"
out.parent.mkdir(parents=True, exist_ok=True)
table.to_csv(out, index=False)
pd.set_option("display.width", 220)
print(table.to_string(index=False))
print(f"\nécrit : {out}")
