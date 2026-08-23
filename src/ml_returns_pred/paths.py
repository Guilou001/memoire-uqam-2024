"""Résolution des chemins du dépôt.

Le code original (2024) utilise partout des chemins relatifs du type ``../config/...`` ou
``../data/...`` qui supposent un répertoire de travail situé un niveau sous la racine du
dépôt (à l'époque : ``src/``). Pour ne pas réécrire ces chemins dans une vingtaine de
modules, le point d'entrée se place dans ``<racine>/workdir/`` avant toute exécution.
La racine peut être forcée avec la variable d'environnement ``ML_RETURNS_PRED_ROOT``.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("ML_RETURNS_PRED_ROOT", Path(__file__).resolve().parents[2]))
WORKDIR = ROOT / "workdir"
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


def chdir_workdir() -> Path:
    """Se place dans ``workdir/`` (créé au besoin) et retourne son chemin."""
    WORKDIR.mkdir(parents=True, exist_ok=True)
    os.chdir(WORKDIR)
    return WORKDIR
