# Commandes courantes. Prérequis : uv (https://docs.astral.sh/uv/) ; macOS : libomp pour xgboost/lightgbm
# (voir README, section Reproduire).
UV ?= uv
PY  := $(UV) run python

.PHONY: setup data test lint run-ridge-ca run-ridge-us run-thesis-ca run-thesis-us results clean

setup:            ## crée l'environnement épinglé (.venv) depuis uv.lock
	$(UV) sync --locked --all-extras

data:             ## télécharge les prix (yfinance) et place les fichiers macro dans data/raw_data (voir scripts/fetch_data.py)
	$(PY) scripts/fetch_data.py

test:             ## tests unitaires (aucune donnée externe requise)
	$(UV) run pytest

lint:
	$(UV) run ruff check src tests scripts

run-ridge-ca:     ## réplication rapide : Ridge, Canada, top 10, 2008-2024 (environ 3 min)
	$(UV) run ml-returns-pred run --strategy ridge_regressor --country canada --signal top --top 10 --period 2008-2024

run-ridge-us:
	$(UV) run ml-returns-pred run --strategy ridge_regressor --country usa --signal top --top 10 --period 2008-2024

run-thesis-ca:    ## les 8 modèles + équipondéré, Canada, top 10, 2008-2024 (long : plusieurs heures)
	for s in equally_weighted ridge_regressor xgboost_regressor ada_boost_regressor extra_trees_regressor \
	         logistic_regression_classifier xgboost_classifier hist_gradient_boosting_classifier extra_trees_classifier; do \
	  $(UV) run ml-returns-pred run --strategy $$s --country canada --signal top --top 10 --period 2008-2024 || exit 1; done

run-thesis-us:
	for s in equally_weighted ridge_regressor xgboost_regressor ada_boost_regressor extra_trees_regressor \
	         logistic_regression_classifier xgboost_classifier hist_gradient_boosting_classifier extra_trees_classifier; do \
	  $(UV) run ml-returns-pred run --strategy $$s --country usa --signal top --top 10 --period 2008-2024 || exit 1; done

results:          ## reconstruit results/tables/*.csv à partir des sorties archivées du mémoire
	$(PY) scripts/collect_thesis_results.py

# ---- version 2 (paquet mlrp) ----
v2-equivalence:   ## vérifie que mlrp reproduit les prédictions et rendements archivés (Ridge US, environ 1 min)
	$(PY) scripts/check_v2_equivalence.py

v2-main:          ## 8 modèles x 2 pays x 3 signaux x 2 modes, période 2008-2024, 8 processus
	$(UV) run mlrp run --country both --period 2008-2024 --models thesis --signals all --jobs 8

v2-thesis:        ## toutes les périodes (long)
	$(UV) run mlrp thesis --country both --jobs 8

v2-figures:
	$(UV) run mlrp figures --country usa --period 2008-2024 && $(UV) run mlrp figures --country canada --period 2008-2024

clean:            ## vide les sorties intermédiaires (jamais data/raw_data ni results/)
	find data/intermediate_data data/concatenated_data plots reports -type f \( -name '*.csv' -o -name '*.pkl' -o -name '*.html' -o -name '*.png' \) -delete
