"""Patient distance matrices under a given ground metric and their agreement."""

import numpy as np
from scipy.stats import spearmanr


def patient_distances(dataset, ground_metric, diag_bures_approx: bool) -> np.ndarray:
    """All-patient OT distance matrix with the exact or the diagonal Bures term."""
    distances = dataset.compute_OT(
        ground_metric=ground_metric, plot_type=False, diag_bures_approx=diag_bures_approx
    )
    return distances.cpu().numpy() if hasattr(distances, "cpu") else np.asarray(distances)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation of two distance matrices over all patient pairs."""
    upper = np.triu_indices_from(a, k=1)
    return float(spearmanr(a[upper], b[upper]).statistic)


def nearest_neighbour_spearman(reference: np.ndarray, other: np.ndarray, k: int = 10) -> float:
    """Mean over patients of the Spearman correlation over the patient's k nearest neighbours
    (by ``reference``)."""
    values = []
    for i in range(len(reference)):
        neighbours = [j for j in np.argsort(reference[i]) if j != i][:k]
        values.append(spearmanr(reference[i, neighbours], other[i, neighbours]).statistic)
    return float(np.nanmean(values))


def within_class_spearman(a: np.ndarray, b: np.ndarray, labels) -> float:
    """Spearman correlation over pairs of patients of the same class only."""
    labels = np.asarray(labels)
    rows, cols = np.triu_indices(len(labels), k=1)
    same = labels[rows] == labels[cols]
    return float(spearmanr(a[rows[same], cols[same]], b[rows[same], cols[same]]).statistic)
