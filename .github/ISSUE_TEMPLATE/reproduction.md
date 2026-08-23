---
name: Problème de reproduction
about: Une commande ne redonne pas le résultat attendu
title: "[repro] "
labels: reproduction
---

**Commande lancée** (par exemple `uv run ml-returns-pred run --strategy ridge_regressor --country usa ...`) :

**Résultat attendu** (tableau ou fichier de `results/` auquel vous comparez) :

**Résultat obtenu** (sortie, métriques, message d'erreur) :

**Environnement** : système, `uv --version`, `uv run python -c "import skforecast, sklearn; print(skforecast.__version__, sklearn.__version__)"`, source et date des données brutes.
