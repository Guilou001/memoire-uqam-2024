# Version 2 (paquet `mlrp`) : architecture, choix et équivalence avec 2024

Date : 2026-08-23. Statuts : M = mesuré, P = précepte (choix d'ingénierie motivé).

## 1. Pourquoi une version 2

Le code de 2024 (`ml_returns_pred`, conservé tel quel et étiqueté `v1.0.0`) fonctionne et se reproduit, mais il
recalcule les prédictions pour chacun des trois signaux (top 10, top 20, positif) alors qu'elles sont identiques,
déroule la dérive des poids en boucles Python jour par jour, dépend de chemins relatifs au répertoire de travail,
mélange configuration et exécution dans une hiérarchie de 25 classes, et additionne la jambe « short » au lieu de
la soustraire. La v2 garde la méthode du mémoire (mêmes estimateurs, mêmes réglages skforecast, même sélection
d'hyperparamètres) et change l'architecture, la vitesse et les défauts documentés.

## 2. Architecture (P)

```
src/mlrp/
├── config.py     RunSpec / TuningSpec (dataclasses immuables), pays, périodes, lecture des espaces YAML de 2024
├── data.py       lecture, remplissage avant borné, alignement, rééchantillonnage, rendements, jeu synthétique
├── models.py     registre d'estimateurs (imports différés), forecaster, recherche bayésienne, backtest
├── portfolio.py  rangs, masques top-k / signe, poids égaux, dérive vectorisée par bloc, long-short, coûts
├── metrics.py    R² hors échantillon et Pesaran-Timmermann (2024), TCAC civil, Sharpe, Sortino, MDD, Oméga
├── runner.py     cache des prédictions (parquet + JSON), exécution d'une spécification, parallélisme joblib
├── report.py     tables markdown et figures matplotlib (palette Okabe-Ito, PDF vectoriel)
└── cli.py        mlrp run | thesis | figures | table
```

Principes : fonctions pures sur DataFrames (testables sans réseau), une seule source de vérité pour les
réglages (`RunSpec`), prédictions calculées une fois par (pays, période, modèle) puis partagées, chemins absolus
depuis la racine du dépôt, aucune écriture implicite (pas de nettoyage de dossiers), graines explicites.

## 3. Vitesse (M)

| Étape | 2024 | v2 |
|---|---|---|
| Prédictions (Ridge, 50 essais + 193 réajustements) | environ 3 min par signal, refaites pour chaque signal | 63 s, une fois, puis cache |
| Dérive des poids et rendements de stratégie | boucle Python sur 4 027 jours, environ 2 s par jambe | cumul vectorisé numpy par bloc mensuel, moins de 50 ms |
| Jeu complet 8 modèles x 2 pays x 3 signaux x 2 modes (période principale) | 48 exécutions complètes en série | 16 jeux de prédictions en parallèle (8 processus), 96 portefeuilles dérivés |

Le temps de calcul est dominé par les réajustements mensuels des estimateurs scikit-learn et xgboost (déjà en C/C++
sous le capot). Réécrire en Rust ou C++ n'apporterait rien là où le temps passe ; le gain vient du partage des
prédictions, du parallélisme par processus et de la vectorisation de la partie portefeuille. C'est un choix
délibéré (P).

## 4. Équivalence avec le code de 2024 (M)

`scripts/check_v2_equivalence.py` (Ridge, États-Unis, top 10, 2008-2024, données brutes de 2024) : mêmes
hyperparamètres Optuna, écart maximal des prédictions 2,8 × 10⁻⁷, poids long et short identiques (0), rendements
quotidiens « as_published » identiques (1,2 × 10⁻¹⁶ sur 4 027 jours). Tests unitaires : la dérive vectorisée égale la
boucle de 2024 à 10⁻¹² près sur données aléatoires avec trous (`tests/test_v2_portfolio.py`), le prétraitement et
le rééchantillonnage égalent les classes de 2024 (`tests/test_v2_data.py`), les métriques de prédiction égalent
celles de 2024 (`tests/test_v2_metrics.py`) ; 31 tests, sans réseau.

## 5. Deux comportements de 2024 découverts en écrivant la v2 (M)

1. **Rééquilibrages sautés.** Les dates de poids cibles sont les premiers du mois ; le code de 2024 n'applique un
   rééquilibrage que si cette date est un jour de bourse. Un 1er tombant un samedi, un dimanche ou un férié est
   ignoré et les poids continuent de dériver jusqu'au mois suivant. Mode `as_published` : reproduit ; mode
   `corrected` : rééquilibrage au premier jour de bourse suivant.
2. **Coûts de transaction.** La fonction de 2024 indexait les poids dérivés par position entière des dates de
   rééquilibrage (les 193 premiers jours au lieu des 193 dates mensuelles) ; sans effet dans le mémoire puisque le
   coût était nul. La v2 calcule la rotation sur les jours de rééquilibrage effectifs dans les deux modes.

S'ajoutent aux constats du journal de vérification : jambe « short » additionnée, TCAC sous-estimé par quantstats,
hyperparamètres choisis sur la période de test, 49 titres canadiens.

## 6. Ce que la v2 ne change pas (encore)

La sélection des hyperparamètres sur la période de test, l'univers figé de titres survivants, l'absence de
caractéristiques financières par titre, la pondération égale : ce sont les chantiers du projet `memoire-2.0`
(validation purgée, univers point-in-time, coûts, Sharpe déflaté).
