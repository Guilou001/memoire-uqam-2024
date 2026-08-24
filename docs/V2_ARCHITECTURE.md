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

Exécution complète mesurée le 2026-08-23 (`mlrp run --country both --period 2008-2024 --models thesis --signals all
--modes corrected,as_published --jobs 8`, Apple M2, 16 jeux dont 1 déjà en cache) : 9 578 s de mur, soit 2 h 40, pour
15 jeux de prédictions et 96 portefeuilles. Ordre d'achèvement depuis le lancement : Ridge Canada 4 min, Logistique
Canada 7 min, XGBoost régresseur 12 à 19 min, Logistique États-Unis 24 min, XGBoost classifieur et Hist. gradient
boosting Canada 37 min, Extra Trees classifieur Canada 43 min, Hist. gradient boosting États-Unis 57 min, Extra Trees
États-Unis 64 à 66 min, AdaBoost États-Unis 105 min, Extra Trees régresseur Canada 148 min, AdaBoost Canada 158 min
(horodatages `created` du cache ; 8 processus en concurrence, donc des durées par modèle surestimées par rapport à
une exécution isolée ; les prochains jeux portent un champ `seconds`). Le code de 2024 aurait fait 48 exécutions
complètes en série, chacune avec sa propre recherche d'hyperparamètres. Le journal contient des
`ChildProcessError: [Errno 10] No child processes` émis par `multiprocessing.resource_tracker` à l'arrêt des
processus loky sous Python 3.12 : message de fin de vie, sans effet sur les résultats (96 lignes écrites, code de
sortie 0).

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

## 5 bis. Revue du 2026-08-23 : un troisième constat et des correctifs

Une relecture ligne à ligne (avec recalculs indépendants : Pesaran-Timmermann identique au papier de 1992 à
4,9 × 10⁻¹⁵, dérive identique à une boucle naïve à 6 × 10⁻¹⁸, métriques à 10⁻¹²) a établi un constat
méthodologique supplémentaire, hérité de 2024 : les exogènes macro sont alignées sur le tampon du rendement
prédit, or FRED-MD tamponne chaque ligne au mois couvert. Le rendement du mois M (tamponné en fin de mois M)
est donc prédit avec la macro du mois M+1, publiée en M+2. Impact mesuré (Ridge États-Unis, top 10, 2008-2024,
recherche d'hyperparamètres refaite) : en alignement temps réel (macro du mois M-1), le TCAC corrigé passe de
−0,9 % à −3,3 %, le Sharpe de 0,00 à −0,18, le R² moyen de −0,40 à −0,46 ; le mode « tel que publié » passe de
19,3 % à 21,8 %. L'ordre de grandeur des conclusions ne change pas, le biais est déclaré dans le README
(section 6) et l'alignement d'origine est conservé pour la fidélité au mémoire ; `memoire-2.0` le corrigera.

Correctifs appliqués à la suite de la revue : invalidation du cache de prédictions quand l'empreinte des
données change (l'empreinte était écrite mais jamais relue), rotation moyenne calculée hors rééquilibrage de
mise en place, « pooling » du R² robuste aux NaN, filtres d'avertissements ciblés au lieu d'un ignore global,
`ffill` explicite avant `pct_change` (comportement de 2024 figé face à pandas 3, et documenté : un titre radié
reste à plat au lieu de sortir). Constats conservés tels quels et documentés : jambe short au seuil global de
rangs (sans effet avec un univers constant), départage alphabétique du top 10 des classifieurs (hérité de 2024).

## 5 ter. SHAP et prédictions constantes (2026-08-23, soir)

Trois constats mesurés en régénérant les figures SHAP (`mlrp shap`, module `mlrp/explain.py`) :

1. **Les figures SHAP archivées de 2024 moyennaient les contributions entre les titres**, position par position ;
   deux effets de signes opposés s'annulaient et plusieurs figures (volet canadien surtout) s'écrasaient vers
   zéro. La v2 empile les observations (titre, mois) au lieu de les moyenner, et garde la colonne du niveau
   skforecast pendant le calcul (l'écarter décale l'attribution des variables).
2. **Le module d'explicabilité de 2024 réajustait le modèle avec les hyperparamètres par défaut**, pas ceux
   retenus par la recherche bayésienne ; ses figures ne décrivent donc pas les modèles qui ont produit les
   portefeuilles. La v2 relit les hyperparamètres retenus dans le cache de prédictions.
3. **Plusieurs modèles retenus prédisent une constante.** Le ccp_alpha des Extra Trees (0,42 à 0,48 retenu, 0,5
   dans le YAML de base, contre 0,0 par défaut dans scikit-learn) élague chaque arbre jusqu'à une feuille unique :
   la forêt prédit la même valeur pour tous les titres à toutes les dates (vérifié : 1 nœud, profondeur 0,
   variance nulle des prédictions). Les classifieurs livrent des classes 0/1, identiques pour tous les titres à
   la plupart des dates. Part des dates à prédictions toutes égales (`results/v2/tables/prediction_ties.csv`,
   script `check_prediction_ties.py`) : Extra Trees régression 100 % (deux pays) ; Extra Trees classifieur 94 %
   (É.-U.) et 99 % (Canada) ; Hist Gradient Boosting Canada 95 % ; logistique 69 à 73 % ; XGBoost classifieur
   2 % (É.-U.) et 58 % (Canada). À ces dates, le « top 10 » se réduit à l'ordre des colonnes (alphabétique) ;
   les SHAP nuls des Extra Trees sont donc exacts. Ridge, XGBoost et AdaBoost en régression classent réellement.

## 6. Ce que la v2 ne change pas (encore)

La sélection des hyperparamètres sur la période de test, l'univers figé de titres survivants, l'absence de
caractéristiques financières par titre, la pondération égale : ce sont les chantiers du projet `memoire-2.0`
(validation purgée, univers point-in-time, coûts, Sharpe déflaté).
