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

## Addendum du 2026-08-23 (soir) : prédictions constantes et figures SHAP

Cinquième constat, mesuré (`scripts/check_prediction_ties.py`) : plusieurs modèles retenus prédisent la même
valeur pour tous les titres à la plupart des dates (Extra Trees régression : 100 % des dates, ses arbres étant
élagués à une feuille par un ccp_alpha de 0,42 à 0,5 ; classifieurs : classes 0/1 souvent unanimes). Le « top
10 » de ces modèles se réduit alors à l'ordre des colonnes. Les figures SHAP de 2024 moyennaient par ailleurs
les contributions entre titres et expliquaient des modèles aux hyperparamètres par défaut ; figures refaites
dans `results/v2/figures/shap/` (détail : `V2_ARCHITECTURE.md`, section 5 ter).

## Addendum du 2026-08-28 : trois artefacts supplémentaires, deux corrigés, un documenté

### 1. L'indice « S&P 500 » du pipeline était le NASDAQ (M, corrigé)

`config.py` pointait le benchmark américain vers `NASDAQ_2000-01-01_to_2024-06-01.csv` (niveaux du composite
NASDAQ) alors que toute la prose, du mémoire au README, appelait cet indice « S&P 500 ». Le fichier
`SP500_2000-01-01_to_2024-06-01.csv` (le vrai S&P 500) existait dans `data/raw_data/` sans être branché.
Chiffres des deux indices sur 2008-01 → 2024-01, mesurés par le pipeline (colonnes `bench_CAGR` et
`bench_Sharpe` de `results/v2/metrics.csv`, avant et après bascule) : NASDAQ, TCAC 11,45 %, Sharpe 0,59 ;
S&P 500, TCAC 7,65 %, Sharpe 0,46. La comparaison publiée flattait donc l'indice de référence d'environ
4 points de TCAC. `config.py` charge désormais le fichier S&P 500 ; tableaux, prose et PDF v1.1 régénérés.

### 2. L'équipondéré de référence sautait son premier mois (M, corrigé)

`runner.run` passait à `equally_weighted_long_only` une date de départ en jour de bourse (2008-01-02) ; le
masque mensuel `mask.loc[start:]` supprimait alors la ligne du 1er du mois et le portefeuille ne détenait
rien pendant son premier mois. Corrigé en ramenant la date au début de son mois
(`to_period("M").to_timestamp()`), comportement figé par le test
`test_equal_weight_holds_from_month_of_start`. Chiffres 2008-01 → 2024-01 (`ew_CAGR`, `ew_Sharpe` de
`results/v2/metrics.csv`) : États-Unis, TCAC 11,08 % → 10,79 %, Sharpe 0,659 → 0,645 ; Canada,
11,86 % → 11,45 %, 0,803 → 0,777. Les métriques des stratégies sont inchangées (vérifié colonne par colonne
contre l'ancien CSV) et `scripts/check_v2_equivalence.py` passe toujours (2,8 × 10⁻⁷ ; 0 ; 1,2 × 10⁻¹⁶) :
l'équivalence porte sur les stratégies, pas sur cette référence, qui diffère désormais des tableaux de 2024
et c'est assumé. Relance des 384 combinaisons depuis le cache de prédictions : 12 secondes.

### 3. La sélection d'hyperparamètres retient le pire essai pour les régresseurs (M, documenté, défaut conservé)

`metrics.py` déclare la direction « maximize » pour les régresseurs (R² en première métrique), mais
skforecast 0.13.0 trie toujours les essais par la première métrique en ordre croissant, et `return_best`
comme `models.py` retiennent la ligne 0 : pour un R² à maximiser, c'est le PIRE des 50 essais. Mesuré dans le
cache (`data/cache_v2/<clé>/tuning_results.parquet`, colonne `r_squared_modified__average`) : Ridge
États-Unis, essai retenu R² −741,0 contre −81,3 au meilleur ; Ridge Canada, −20,9 contre −0,90. Les
classifieurs ne sont pas touchés (première métrique : p-value de Pesaran-Timmermann, à minimiser, donc le tri
croissant est le bon). L'artefact vient des YAML de 2024 : il est fidèle à la réplication mais n'était pas
déclaré, et le README affirmait même que la sélection « avantage les modèles ». Le comportement par défaut
est conservé (l'équivalence avec 2024 doit tenir) ; l'option `--select-best` (`TuningSpec.select_best`)
resélectionne le meilleur essai et reconstruit le forecaster avant le backtest, sans toucher aux clés de
cache des exécutions par défaut.

### 4. Deux correctifs mineurs (M)

- `scripts/make_latex_tables.py` mettait en gras vert la PIRE perte maximale (`s.min()` sur des valeurs
  négatives) ; le meilleur drawdown est le plus proche de zéro. Corrigé, les huit tableaux `.tex` régénérés.
- `runner.py` : la clé de déduplication de `results/v2/metrics.csv` ignorait le réglage de tuning ; un run
  `--no-tuning` écrasait la ligne tunée. Colonne `tuning` ajoutée à la table et à la clé.

Régénérations du 2026-08-28 : figures de croissance des deux pays (légendes lisibles, courbe équipondérée en
repère, axe des x étiqueté), 28 figures SHAP (titres sans tirets cadratins), nouvelle figure de synthèse
`results/v2/figures/summary_cagr_par_modele.png` (`scripts/make_summary_figure.py`), tableaux LaTeX, prose et
PDF `reports/memoire_v1.1/memoire_v1_1.pdf` (tectonic).
