# Contribuer

Ce dépôt archive le code d'un mémoire et sa vérification. Les contributions bienvenues :

1. Signaler une erreur de reproduction (ouvrir une issue avec la commande lancée, la sortie et les versions).
2. Proposer une correction méthodologique : ouvrir une issue d'abord ; le code historique (`src/ml_returns_pred`,
   2024) reste figé pour la reproductibilité, les corrections passent par des options (comme `long_short_mode`)
   ou par le projet successeur `memoire-2.0`.
3. Avant une pull request : `uv run ruff check src tests scripts` et `uv run pytest` doivent passer ; un changement
   de résultat s'accompagne de la table mise à jour dans `results/tables/` et d'une ligne dans `docs/`.

Langue : français ou anglais. Code de conduite : `CODE_OF_CONDUCT.md`.
