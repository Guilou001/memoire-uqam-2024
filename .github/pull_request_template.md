## Objet

(une phrase : ce que change cette PR et pourquoi)

## Vérifications

- [ ] `uv run ruff check src tests scripts` passe
- [ ] `uv run pytest` passe
- [ ] si un résultat change : table mise à jour dans `results/tables/` et note dans `docs/`
- [ ] le code historique de 2024 (`src/ml_returns_pred`, hors `cli.py`, `paths.py`) n'est pas modifié, ou la modification est une option documentée
