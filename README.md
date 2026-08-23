# Prédire les rendements d'actions canadiennes et américaines avec des données macroéconomiques et l'apprentissage automatique

Pipeline complet du mémoire de maîtrise *Évaluation empirique d'actifs canadiens par l'apprentissage
automatique* (Guillaume Vaudescal, UQAM, décembre 2024, dir. Philippe Goulet Coulombe et Dalibor Stevanovic),
rendu reproductible en août 2026 : environnement épinglé, ligne de commande, tests, résultats archivés, et un
audit du code qui a révélé deux corrections à apporter à la lecture des résultats publiés.

[![ci](https://github.com/Guilou001/memoire-uqam-2024/actions/workflows/ci.yml/badge.svg)](https://github.com/Guilou001/memoire-uqam-2024/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.12-blue)
![licence](https://img.shields.io/badge/code-MIT-green)
![rapport](https://img.shields.io/badge/m%C3%A9moire-PDF-lightgrey)

**Résultat en une phrase.** Le pipeline de 2024 se réexécute à l'identique (prédictions Ridge États-Unis
égales à 2,8 × 10⁻⁷ près, mêmes hyperparamètres), mais les portefeuilles « longs-courts » du mémoire sont en
réalité longs sur les deux jambes : une fois le short réellement soustrait, le Ridge top 10 passe d'un TCAC de
19,3 % à −0,7 % aux États-Unis et de 30,4 % à −8,6 % au Canada (2008-2024, brut de coûts).

*English summary.* Reproducible pipeline of my MSc thesis (UQAM, 2024): monthly returns of 49 TSX and 50 S&P 500
stocks predicted with Canadian (LCDMA) and US (FRED-MD) macro data, eight ML regressors and classifiers,
long-short portfolios, 2008-2024. The 2024 code re-runs exactly (Ridge US predictions match to 2.8e-7). An audit
shows the published "long-short" portfolios added the short leg instead of subtracting it (both legs long, 200 %
gross exposure) and that the CAGR came from a library bug that divided calendar days by 252. Corrected long-short
CAGRs: Ridge top 10 −0.7 % (US), −8.6 % (Canada); Extra Trees US +6.9 % (Sharpe 0.76). See sections 3 and 6.

## 1. Ce que c'est, ce que ce n'est pas

Le dépôt contient le code exact qui a produit les tableaux et figures du mémoire (pipeline `ml_returns_pred`,
2024), emballé en paquet Python avec une interface en ligne de commande, plus les sorties archivées de l'exécution
d'octobre 2024 (`results/`). Les paramètres sont ceux du mémoire, pas des valeurs ajustées après coup.
Ce n'est pas un conseil d'investissement et ce n'est pas encore la version corrigée de l'étude : la correction du
long-short est offerte comme option (`--long-short-mode corrected`) et documentée, l'étude complète refaite
(univers point-in-time, validation purgée, coûts) est le projet `memoire-2.0`.

Le dépôt contient deux implémentations de la même méthode. `ml_returns_pred` (v1) est le code de 2024, figé pour
la reproductibilité. `mlrp` (v2, août 2026) est une refonte : mêmes estimateurs et mêmes réglages, mais prédictions
calculées une fois par modèle puis partagées entre les trois signaux, portefeuille vectorisé, long-short corrigé par
défaut, exécution parallèle, cache, tests d'équivalence avec 2024 (voir `docs/V2_ARCHITECTURE.md`).

## 2. Question, données et méthode (résumé du mémoire)

Question : des algorithmes d'apprentissage machine nourris de données macroéconomiques prévoient-ils les rendements
mensuels d'actions au Canada et aux États-Unis, et des portefeuilles longs-courts bâtis sur ces prévisions
battent-ils les indices ?

| Élément | Canada | États-Unis |
|---|---|---|
| Titres | 49 titres du TSX 60 (dont le FNB XIU.TO), prix Yahoo ajustés | 50 titres du S&P 500 |
| Macro | LCDMA, panel équilibré mensuel, 410 variables (1981 →) | FRED-MD, 126 variables (1959 →) |
| Indices | S&P/TSX (TSX60) | S&P 500, NASDAQ |
| Fréquence | mensuelle (prix rééchantillonnés en début de mois) | idem |
| Entraînement / test | 2000 → 2007-12 / 2008-01 → 2024-01 (193 mois) ; sous-périodes 2008-2012, 2012-2020, 2020-2024 | idem |

Méthode : `ForecasterAutoregMultiSeries` (skforecast 0.13) avec 12 retards des rendements et les variables macro
en exogènes standardisées ; hyperparamètres par recherche bayésienne (Optuna, 50 essais) ; backtest à un pas avec
réajustement mensuel ; régressions (Ridge, XGBoost, AdaBoost, Extra Trees) et classifications (logistique,
XGBoost, Hist Gradient Boosting, Extra Trees) ; signaux top 10, top 20 et signe de la prévision ; poids égaux par
jambe ; rendements quotidiens des portefeuilles avec dérive des poids et coûts de transaction à 0 ; métriques
quantstats (TCAC, Sharpe, Sortino, perte maximale, Oméga) et R² hors échantillon ; importance des variables par SHAP.

## 3. Ce que la réexécution de 2026 a établi (mesuré)

| Vérification | Résultat |
|---|---|
| Réexécution Ridge top 10 États-Unis, 2008-2024, données brutes de 2024 | prédictions identiques à 2,8 × 10⁻⁷ près ; mêmes hyperparamètres Optuna ; Sharpe, volatilité, perte maximale, Sortino, Oméga identiques à l'archive |
| Réexécution Ridge top 10 Canada avec prix retéléchargés en 2026 | mêmes hyperparamètres ; prévisions corrélées à 0,96 avec 2024 ; Sharpe 0,90 contre 0,81 (les prix ajustés Yahoo ont été révisés) |
| Recalcul des rendements de stratégie depuis les poids archivés (États-Unis) | écart maximal 1 × 10⁻¹⁶ : la classe de calcul est bien celle qui a produit les tableaux |
| Jambe « short » | poids positifs, sommés à 1, et **additionnés** à la jambe longue (`compute_strategy_returns`) ; exposition brute médiane 2,0 |
| TCAC publié | calculé par quantstats (version 2024) en divisant les jours civils par 252 ; 12,97 % publié contre 19,33 % en années civiles pour Ridge États-Unis ; 16,51 % contre 24,78 % pour Ridge Canada |
| Univers canadien | 49 titres et non 50 (ENB.TO figurait deux fois dans la liste) ; XIU.TO est un FNB ; fichier « TSX60 » = niveaux du composite ^GSPTSE |
| Sélection des hyperparamètres | la recherche bayésienne évalue chaque candidat sur la période de test elle-même (`initial_train_size` = fin de l'entraînement, sans réajustement) : les hyperparamètres sont choisis hors échantillon sur les mêmes mois que le backtest |

### Tel que publié contre long-short corrigé (top 10, 2008-01 → 2024-01, brut de coûts)

| Pays, modèle | Mode | TCAC (années civiles) | Sharpe | Volatilité | Perte max. | Jambe longue seule | Jambe « short » tenue longue |
|---|---|---|---|---|---|---|---|
| États-Unis, Ridge | tel que publié | 19,3 % | 0,64 | 39,7 % | −73,7 % | 11,4 % | 11,0 % |
| États-Unis, Ridge | corrigé (long − short) | −0,7 % | 0,02 | 13,1 % | −37,2 % | | |
| États-Unis, XGBoost | tel que publié | 19,3 % | 0,65 | 39,2 % | −70,5 % | 11,3 % | 11,0 % |
| États-Unis, XGBoost | corrigé | −0,9 % | 0,00 | 13,7 % | −62,4 % | | |
| États-Unis, Extra Trees | tel que publié | 21,8 % | 0,71 | 37,8 % | −66,6 % | 16,2 % | 8,4 % |
| États-Unis, Extra Trees | corrigé | 6,9 % | 0,76 | 9,5 % | −19,4 % | | |
| Canada, Ridge (prix 2026) | tel que publié | 30,4 % | 0,93 | 35,2 % | −64,1 % | 12,3 % | 18,7 % |
| Canada, Ridge (prix 2026) | corrigé | −8,6 % | −0,37 | 19,4 % | −78,6 % | | |

Lecture : la surperformance publiée tient à une exposition longue de 200 % sur des marchés haussiers, pas au
classement des prévisions ; la jambe « short » tenue longue fait aussi bien que la jambe longue (et mieux au
Canada). Un seul modèle garde un vrai alpha long-short dans ce tableau, Extra Trees aux États-Unis.
Scripts : `scripts/compare_long_short_modes.py` ; tables : `results/tables/long_short_modes_*.csv`.

### Les huit modèles, top 10, version 2 (mesuré le 2026-08-23, `results/v2/metrics.csv`)

TCAC en années civiles, brut de coûts, 2008-01 → 2024-01. « Publié » : mode `as_published` de la v2, qui reproduit le code de
2024 (short additionné, rééquilibrages sautés les 1ers non ouvrés) ; « corrigé » : short soustrait et rééquilibrage au premier
jour de bourse suivant. Volet États-Unis sur les prix bruts de 2024 ; volet Canada sur les prix retéléchargés en 2026. Les écarts avec le tableau
précédent (Ridge corrigé −0,7 % → −0,9 % aux États-Unis, −8,6 % → −7,5 % au Canada) viennent du rééquilibrage au jour
ouvré suivant et, pour le Canada, des prix retéléchargés.

| Modèle | É.-U. TCAC publié | É.-U. TCAC corrigé | É.-U. Sharpe publié | É.-U. Sharpe corrigé | Canada TCAC publié | Canada TCAC corrigé | Canada Sharpe publié | Canada Sharpe corrigé |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ridge (rég.) | 19,3 % | −0,9 % | 0,64 | 0,00 | 28,7 % | −7,5 % | 0,90 | −0,35 |
| XGBoost (rég.) | 19,3 % | −0,6 % | 0,65 | 0,02 | 23,4 % | −1,3 % | 0,77 | 0,02 |
| AdaBoost (rég.) | 17,5 % | 0,4 % | 0,60 | 0,10 | 24,8 % | −0,5 % | 0,80 | 0,07 |
| Extra Trees (rég.) | 21,8 % | 6,9 % | 0,71 | 0,76 | 21,6 % | −2,3 % | 0,71 | −0,07 |
| Logistique (class.) | 21,7 % | 6,7 % | 0,71 | 0,72 | 18,8 % | −2,0 % | 0,66 | −0,05 |
| XGBoost (class.) | 21,2 % | 4,9 % | 0,70 | 0,52 | 22,6 % | 1,1 % | 0,75 | 0,15 |
| Hist. gradient boosting (class.) | 20,3 % | 7,4 % | 0,68 | 0,80 | 20,9 % | −3,2 % | 0,69 | −0,12 |
| Extra Trees (class.) | 21,5 % | 5,6 % | 0,71 | 0,63 | 21,7 % | −2,2 % | 0,72 | −0,06 |

Repères sur la même période : S&P 500 TCAC 11,5 %, Sharpe 0,59 ; équipondéré des 50 titres
américains 11,1 %, Sharpe 0,66 ; composite TSX 2,6 %, Sharpe 0,24 ; équipondéré des 49 titres
canadiens 11,5 %, Sharpe 0,79. Lecture : une fois le short réellement soustrait, aucun modèle canadien ne
dépasse 1,1 % de TCAC ; aux États-Unis, six modèles restent positifs (AdaBoost à 0,4 % ; Extra Trees régresseur et
classifieur, Logistique, XGBoost classifieur et Hist. gradient boosting entre 4,9 % et 7,4 %), tous sous l'équipondéré et
sous l'indice. Les 96 exécutions (3 signaux × 2 modes) sont dans `results/v2/`.

## 4. Arborescence

```
memoire-uqam-2024/
├── src/ml_returns_pred/      v1 : code du pipeline (2024) + cli.py, paths.py (2026)
├── src/mlrp/                 v2 : config, data, models, portfolio, metrics, runner, report, cli (2026)
├── docs/                     VERIFICATION_2026-08-23.md (audit), V2_ARCHITECTURE.md
├── config/                   YAML : méta (pipeline, régression, classification) et par stratégie
├── scripts/                  fetch_data.py, collect_thesis_results.py, compare_long_short_modes.py, check_v2_equivalence.py
├── tests/                    pytest (31) : long-short, rangs, CLI, équivalence v2/v1 (dérive, poids, rendements, métriques)
├── results/                  archive_2024/<pays>/<période>/<signal>/ (métriques, prédictions, poids, rendements),
│                             figures/ (PNG), tables/ (tableaux 4.1-4.2 reconstitués, comparaisons long-short)
│                             v2/ : metrics.csv et rendements quotidiens par exécution (mlrp)
├── reports/                  mémoire (PDF, 8 Mo) et résumé officiel
├── data/raw_data/            non versionné : prix Yahoo, LCDMA, FRED-MD (voir data/raw_data/README.md)
└── workdir/                  répertoire de travail : les chemins relatifs du code historique s'y résolvent
```

## 5. Reproduire

```bash
uv sync --locked --all-extras            # Python 3.12, versions épinglées (uv.lock)
uv run pytest                            # 31 tests, sans données externes
uv run python scripts/fetch_data.py      # prix Yahoo ; déposer macro_data.csv (LCDMA) et Fred-MD.csv à la main
uv run ml-returns-pred list
uv run ml-returns-pred run --strategy ridge_regressor --country usa --signal top --top 10 --period 2008-2024
uv run ml-returns-pred run --strategy ridge_regressor --country canada --long-short-mode corrected
```

Durées mesurées (M2, 15 cœurs) : Ridge top 10 sur 2008-2024 environ 3 minutes (50 essais Optuna + 193 réajustements) ;
les modèles d'arbres sont plus longs. macOS : xgboost et lightgbm exigent `libomp` (`brew install libomp`, ou la
bibliothèque extraite d'une bouteille Homebrew puis `install_name_tool -add_rpath`). La graine Optuna est 123,
celle des estimateurs 42 ; les sorties vont dans `data/intermediate_data/`, `plots/` et `reports/`.

### Version 2 (`mlrp`)

```bash
uv run python scripts/check_v2_equivalence.py   # Ridge US : mlrp contre les sorties archivées de 2024 (environ 1 min)
uv run mlrp run --country usa --period 2008-2024 --models ridge_regressor --signals top10 --modes corrected,as_published
uv run mlrp run --country both --period 2008-2024 --models thesis --signals all --jobs 8   # 16 jeux de prédictions, 96 portefeuilles
uv run mlrp table --country canada --period 2008-2024 && uv run mlrp figures --country canada --period 2008-2024
```

Les prédictions sont mises en cache dans `data/cache_v2/` (clé : pays, période, modèle, réglage) ; les trois signaux
et les deux modes de long-short en dérivent sans réentraîner. Équivalence mesurée avec le code de 2024 (Ridge US,
top 10, 2008-2024) : mêmes hyperparamètres, prédictions à 2,8 × 10⁻⁷, poids identiques, rendements quotidiens à
1,2 × 10⁻¹⁶ ; la dérive vectorisée égale la boucle de 2024 à 10⁻¹² sur données aléatoires avec trous. Le mode
`as_published` reproduit aussi un comportement du code de 2024 : un rééquilibrage dont le 1er du mois tombe un
samedi, un dimanche ou un férié est sauté ; le mode `corrected` rééquilibre au premier jour de bourse suivant.

Ce qui se reproduit exactement : tout le volet États-Unis (les prix bruts de 2024 sont identiques à ceux du mémoire) et
tout ce qui part des prédictions archivées (`results/archive_2024`). Ce qui se reproduit approximativement : le volet
Canada, faute du fichier de prix de 2024 dans l'archive ; les prix ajustés retéléchargés diffèrent un peu.

## 6. Limites et biais (statut)

| Limite | Statut |
|---|---|
| Jambe « short » additionnée et non soustraite (portefeuille long des deux côtés) | quantifié (section 3) ; correction disponible par `--long-short-mode corrected` ; texte du mémoire non modifié |
| TCAC sous-estimé par la version 2024 de quantstats (jours / 252) | quantifié ; `results/tables/*` donnent les deux définitions |
| Hyperparamètres choisis sur la période de test (fuite de sélection) | reconnu ; à corriger par validation purgée dans `memoire-2.0` |
| Univers figé de titres survivants (biais de survie), FNB XIU.TO dans l'univers, 49 titres au lieu de 50 | reconnu ; univers point-in-time prévu dans `memoire-2.0` |
| Coûts de transaction à zéro, poids égaux, rééquilibrage mensuel | reconnu ; paramètre `--fee` disponible, non utilisé dans le mémoire |
| Données macro en millésime final (révisions non simulées) | reconnu |
| Prix Yahoo ajustés révisés dans le temps (volet Canada non identique) | mesuré (corrélation 0,96 des prévisions) |

## 7. Crédits, licence, citation

Code et résultats : Guillaume Vaudescal. Infrastructure initiale du pipeline en 2024 (lecture de configuration,
gestion de fichiers, structure des modules) : Thomas Vaudescal. Données : LCDMA (Fortin-Gagnon, Leroux, Stevanovic et
Surprenant, 2022, *Canadian Journal of Economics*), FRED-MD (McCracken et Ng, 2016, *JBES*), Yahoo Finance.
Bibliothèques : skforecast, scikit-learn, Optuna, quantstats-lumi, shap.

Code sous licence MIT ; texte du mémoire et figures sous CC BY 4.0. Citation : voir `CITATION.cff`.
