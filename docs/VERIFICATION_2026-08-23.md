# Journal de vérification (23 août 2026)

Objet : faire tourner le code du mémoire dans un environnement épinglé et vérifier qu'il régénère les résultats
publiés. Machine : macOS 26, puce Apple M2, 15 cœurs ; Python 3.12.14 (uv), skforecast 0.13.0, scikit-learn 1.4.2,
pandas 2.2.2, numpy 1.26.4, optuna 3.6.1, quantstats-lumi 1.1.5, xgboost 2.1.0, lightgbm 4.4.0, catboost 1.2.5.
Statuts : M = mesuré, R = rapporté.

## 1. Provenance du code (M)

Le code final n'était pas dans l'archive universitaire (notebooks de 2022-2023 seulement). Il a été retrouvé sur
un disque externe (`ml_returns_pred`, 40 commits du 2024-06-14 au 2024-10-04, dépôt distant privé), copié avec son
historique git, puis emballé ici en paquet (`src/ml_returns_pred`, imports `src.` réécrits en `ml_returns_pred.`,
chemins relatifs résolus depuis `workdir/`). Les 21 fichiers `__init__.py` ont été ajoutés ; ruff a trié les imports
et retiré 169 imports inutilisés ; la logique n'a pas été modifiée, sauf l'ajout du paramètre `long_short_mode`.

## 2. Réexécution États-Unis, Ridge, top 10, 2008-01 → 2024-01 (M)

Données brutes : fichiers de 2024 présents dans l'archive (`us_stocks`, `NASDAQ`, `Fred-MD`), identiques (md5) dans
les 24 feuilles de résultats. Configuration : `cutoff_date` 2007-12-31, `max_date` 2024-01-01, 50 essais Optuna,
`fix_threshold` 10, `transaction_fee` 0.

| Grandeur | Archive (oct. 2024) | Réexécution (août 2026) |
|---|---|---|
| Meilleurs hyperparamètres (alpha, max_iter, tol, solver) | 0,4829 ; 9 400 ; 4,07e-5 ; auto | identiques |
| Prédictions `y_pred` (193 × 50) | | écart absolu maximal 2,8 × 10⁻⁷ |
| Rendement cumulé | 15,879 | 15,879 |
| Sharpe / Volatilité / Perte max. / Sortino / Oméga | 0,644 / 0,397 / −0,737 / 0,926 / 1,135 | identiques |
| TCAC (quantstats) | 12,97 % | 19,33 % |

Le seul écart, le TCAC, vient de la bibliothèque : la version 2024 de quantstats calculait les années comme
`jours civils / 252` ; la version 1.1.5 utilise `jours / 365,25`. Sur 16 ans : (1 + 15,879)^(252/5 840) − 1 = 12,97 % ;
(1 + 15,879)^(1/16) − 1 = 19,33 %.

## 3. Réexécution Canada, Ridge, top 10 (M)

Le fichier de prix canadiens de 2024 n'existe plus nulle part (ni archive, ni disque externe) ; les prix ont été
retéléchargés (yfinance 1.6.0, clôtures ajustées, 49 titres, 6 179 jours, 2000-01-03 → 2024-05-31). Hyperparamètres
optimaux identiques à 2024 ; prévisions corrélées à 0,962 ; écart absolu moyen 0,0097. Métriques : Sharpe 0,90
(archive 0,81), volatilité 0,350 (0,352), perte maximale −0,667 (−0,644), rendement cumulé 55,6 (33,5).
Les différences viennent de la révision des prix ajustés par Yahoo (dividendes postérieurs, corrections).

## 4. Audit du calcul des rendements de stratégie (M)

`StrategyReturnsCalculator` (code 2024) : les poids de la jambe « short » sont positifs et normalisés à 1
(`RankBasedWeightingStrategy`, méthode `equal`) ; `compute_strategy_returns` fait
`strategy_returns = long + short` ; la dérive journalière applique `(1 − r)` aux deux jambes dès que
`is_long_only` est faux. Recalcul depuis les poids archivés et les prix de 2024 (États-Unis) : écart maximal
1 × 10⁻¹⁶ avec les rendements archivés, donc c'est bien ce calcul qui a produit les tableaux.
Décomposition (États-Unis, Ridge) : jambe longue 11,42 % de TCAC ; jambe « short » tenue longue 10,95 % ;
somme telle que publiée 19,35 % ; vrai long-short (long − short) −0,60 %. Tableaux complets :
`results/tables/long_short_modes_*.csv` (Ridge, XGBoost, Extra Trees États-Unis ; Ridge Canada).

## 5. Autres constats (M)

- La liste des 50 tickers canadiens contenait `ENB.TO` deux fois : 49 titres uniques, dont le FNB `XIU.TO`.
- Le fichier `TSX60_2000-01-01_to_2024-06-01.csv` contient les niveaux du composite `^GSPTSE` (8 202 au 2000-01-04).
- La recherche bayésienne utilise `initial_train_size = len(train)` et `refit = False` : la métrique d'un candidat est
  son R² hors échantillon sur 2008-2024, la même période que le backtest final. Les hyperparamètres sont donc choisis
  sur la période de test.
- Les résultats Ada Boost régression du mémoire n'existent dans l'archive qu'en HTML et en pickle (pas de `y_pred`,
  pas de `key_metrics`).
- `transaction_fee` vaut 0 dans la configuration finale : tous les résultats sont bruts de coûts.

## Addendum du 2026-08-23 (soir) : version 2

La refonte `mlrp` (voir `V2_ARCHITECTURE.md`) reproduit les prédictions et rendements de 2024 (Ridge US : 2,8 × 10⁻⁷ et
1,2 × 10⁻¹⁶) et a permis de recalculer les huit modèles, trois signaux et deux modes de long-short sur 2008-2024 en une
exécution de 2 h 40 (`results/v2/metrics.csv`). Elle a révélé un troisième comportement du code de 2024 : les
rééquilibrages dont le 1er du mois n'est pas un jour de bourse sont sautés. Tableau des huit modèles : README, section 3.

## Addendum du 2026-08-23 (nuit) : revue ligne à ligne de la v2

Revue indépendante avec recalculs (détail dans `V2_ARCHITECTURE.md`, section 5 bis) : implémentation confirmée
exacte ; quatrième biais méthodologique identifié et mesuré, hérité de 2024 : exogènes macro un mois en avance
sur le rendement prédit (Ridge US top 10 en alignement temps réel : TCAC corrigé −0,9 % → −3,3 %, R² moyen
−0,40 → −0,46, `scripts/check_exog_alignment.py`). Ajouté à la table des limites du README.
