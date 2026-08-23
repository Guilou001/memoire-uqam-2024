"""mlrp : version 2 (2026) du pipeline du mémoire, même méthode, architecture refaite.

Modules : ``config`` (spécifications d'exécution), ``data`` (chargement, prétraitement, rééchantillonnage),
``models`` (estimateurs et encapsulation skforecast : recherche bayésienne et backtest), ``portfolio``
(rangs, poids, rendements de stratégie vectorisés), ``metrics`` (mesures de prédiction et de performance),
``runner`` (exécution avec cache des prédictions et parallélisme), ``report`` (tables et figures), ``cli``.

Différences voulues avec ``ml_returns_pred`` (code de 2024) : les prédictions sont calculées une fois par
(pays, période, modèle) et partagées entre les signaux ; le long-short est correct par défaut
(``long_short_mode="corrected"``) ; le TCAC est en années civiles ; tout est testé et sans chemins relatifs.
"""

__version__ = "2.0.0"
