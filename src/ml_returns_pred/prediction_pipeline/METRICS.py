import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
)


def r_squared_modified(y_true: pd.Series, y_pred: pd.Series, y_train: pd.Series) -> float:
    """
    Calcule le R² ajusté pour les valeurs en échantillon et hors échantillon.

    :param y_true: Véritables étiquettes.
    :param y_pred: Étiquettes prédites.
    :param y_train: Valeurs en échantillon.
    :return: Valeur de R² ajusté.
    """
    SSR = np.sum((y_true - y_pred) ** 2)
    SST = np.sum((y_true - np.mean(y_train)) ** 2)
    return 1 - SSR / SST


def pesaran_timmermann_stat(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Calcule la statistique de Pesaran-Timmermann.
    Cette fonction gère les prédictions binaires et les probabilités.

    Paramètres :
    - y_true : Array des vraies étiquettes binaires (0 ou 1).
    - y_pred : Array des prédictions (binaires ou probabilités entre 0 et 1).

    Retourne :
    - pt_stat : La statistique de Pesaran-Timmermann.
    """
    # Conversion en arrays NumPy si nécessaire
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Vérifier si y_pred contient des probabilités
    if y_pred.dtype.kind in 'fc' and np.any((y_pred > 0) & (y_pred < 1)):
        # Convertir les probabilités en prédictions binaires avec un seuil de 0,5
        y_pred = np.where(y_pred >= 0.5, 1, 0)
    else:
        # S'assurer que y_pred est de type entier
        y_pred = y_pred.astype(int)

    # S'assurer que y_true est binaire
    y_true = np.where(y_true >= 0.5, 1, 0)

    # Nombre d'observations
    size = y_true.shape[0]

    # Score de précision directionnelle (DAC)
    directional_accuracy = np.sum(y_true == y_pred) / size

    # Proportions de valeurs positives vraies et prédites
    proportion_true_positive = np.sum(y_true == 1) / size
    variance_true = proportion_true_positive * (1 - proportion_true_positive) / size

    proportion_pred_positive = np.sum(y_pred == 1) / size
    variance_pred = proportion_pred_positive * (1 - proportion_pred_positive) / size

    # Proportion attendue de prédictions directionnelles correctes sous indépendance
    expected_proportion = (proportion_true_positive * proportion_pred_positive) + \
                          ((1 - proportion_true_positive) * (1 - proportion_pred_positive))

    # Variance de la proportion attendue
    variance_expected = expected_proportion * (1 - expected_proportion) / size

    # Terme de correction de variance
    correction_variance = ((2 * proportion_true_positive - 1) ** 2) * variance_pred + \
                          ((2 * proportion_pred_positive - 1) ** 2) * variance_true + \
                          4 * variance_true * variance_pred

    # Éviter la racine carrée d'un nombre négatif
    variance_diff = variance_expected - correction_variance
    if variance_diff <= 0:
        # Gérer le cas où variance_diff est zéro ou négatif
        pt_stat = 0.0
    else:
        # Statistique de Pesaran-Timmermann
        pt_stat = (directional_accuracy - expected_proportion) / np.sqrt(variance_diff)

    return pt_stat

def pesaran_timmermann_p_value(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Calcule la p-valeur de la statistique de Pesaran-Timmermann.
    Cette fonction gère les prédictions binaires et les probabilités.

    Paramètres :
    - y_true : Array des vraies étiquettes binaires (0 ou 1).
    - y_pred : Array des prédictions (binaires ou probabilités entre 0 et 1).

    Retourne :
    - p_value : La p-valeur associée à la statistique de Pesaran-Timmermann.
    """
    # Conversion en arrays NumPy si nécessaire
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Vérifier si y_pred contient des probabilités
    if y_pred.dtype.kind in 'fc' and np.any((y_pred > 0) & (y_pred < 1)):
        # Convertir les probabilités en prédictions binaires avec un seuil de 0,5
        y_pred = np.where(y_pred >= 0.5, 1, 0)
    else:
        # S'assurer que y_pred est de type entier
        y_pred = y_pred.astype(int)

    # S'assurer que y_true est binaire
    y_true = np.where(y_true >= 0.5, 1, 0)

    # Nombre d'observations
    size = y_true.shape[0]

    # Score de précision directionnelle (DAC)
    directional_accuracy = np.sum(y_true == y_pred) / size

    # Proportions de valeurs positives vraies et prédites
    proportion_true_positive = np.sum(y_true == 1) / size
    variance_true = proportion_true_positive * (1 - proportion_true_positive) / size

    proportion_pred_positive = np.sum(y_pred == 1) / size
    variance_pred = proportion_pred_positive * (1 - proportion_pred_positive) / size

    # Proportion attendue de prédictions directionnelles correctes sous indépendance
    expected_proportion = (proportion_true_positive * proportion_pred_positive) + \
                          ((1 - proportion_true_positive) * (1 - proportion_pred_positive))

    # Variance de la proportion attendue
    variance_expected = expected_proportion * (1 - expected_proportion) / size

    # Terme de correction de variance
    correction_variance = ((2 * proportion_true_positive - 1) ** 2) * variance_pred + \
                          ((2 * proportion_pred_positive - 1) ** 2) * variance_true + \
                          4 * variance_true * variance_pred

    # Éviter la racine carrée d'un nombre négatif
    variance_diff = variance_expected - correction_variance
    if variance_diff <= 0:
        # Gérer le cas où variance_diff est zéro ou négatif
        pt_stat = 0.0
    else:
        # Statistique de Pesaran-Timmermann
        pt_stat = (directional_accuracy - expected_proportion) / np.sqrt(variance_diff)

    # P-valeur à partir de la distribution normale (test unilatéral)
    p_value = 1 - stats.norm.cdf(pt_stat)

    return p_value


METRICS = {
    'mean_squared_error': mean_squared_error,
    'r2_score': r2_score,
    'accuracy_score': accuracy_score,
    'f1_score': f1_score,
    'confusion_matrix': confusion_matrix,
    'r_squared_modified': r_squared_modified,
    'pesaran_timmermann_stat': pesaran_timmermann_stat,
    'pesaran_timmermann_p_value': pesaran_timmermann_p_value,
    'precision_score': precision_score,
    'recall_score': recall_score,
    'mean_absolute_error': mean_absolute_error,
    'median_absolute_error': median_absolute_error
}

# Example usage
if __name__ == "__main__":
    # Generate random binary classification data
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)  # True labels (binary: 0 or 1)
    y_pred = np.random.randint(0, 2, 100)  # Predicted labels (binary: 0 or 1)

    # Calculate Accuracy Score
    accuracy = accuracy_score(y_true, y_pred)

    # Calculate Pesaran-Timmermann statistic


