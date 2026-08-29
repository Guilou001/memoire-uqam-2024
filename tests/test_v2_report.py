"""Le libellé de coûts d'une figure dit ce que les rendements portent, ou déclare son ignorance."""

import pandas as pd

from mlrp.report import cost_label

LIGNES = [
    {"country": "usa", "period": "2008-2024", "mode": "corrected", "fee": 0.0},
    {"country": "usa", "period": "2008-2024", "mode": "as_published", "fee": 0.001},
    {"country": "canada", "period": "2008-2024", "mode": "corrected", "fee": 0.0},
    {"country": "canada", "period": "2008-2024", "mode": "corrected", "fee": 0.002},
]


def _metrics(tmp_path):
    p = tmp_path / "metrics.csv"
    pd.DataFrame(LIGNES).to_csv(p, index=False)
    return p


def test_frais_nuls_annonce_avant_couts(tmp_path):
    assert cost_label("usa", "2008-2024", "corrected", _metrics(tmp_path)) == "avant coûts de transaction"


def test_frais_uniformes_donnent_le_montant_en_points_de_base(tmp_path):
    assert cost_label("usa", "2008-2024", "as_published", _metrics(tmp_path)) == \
        "net de 10 points de base par unité de rotation"


def test_frais_melanges_declarent_l_ignorance(tmp_path):
    # deux exécutions du même couple pays / période / mode avec des frais différents : le titre ne
    # doit affirmer ni l'un ni l'autre
    assert cost_label("canada", "2008-2024", "corrected", _metrics(tmp_path)) == \
        "coûts de transaction non renseignés"


def test_fichier_absent_declare_l_ignorance(tmp_path):
    assert cost_label("usa", "2008-2024", "corrected", tmp_path / "absent.csv") == \
        "coûts de transaction non renseignés"
