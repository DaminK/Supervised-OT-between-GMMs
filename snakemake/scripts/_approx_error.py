"""Approximation error of the diagonal Bures term for a learned map on a fitted GMM (Figure 5)."""

import numpy as np
from ggml_ot.distances import pairwise_gaussian_distance
from ggml_ot.optimization.regularization import mutual_information_loss

try:
    import torch
except Exception:  # pragma: no cover - optional runtime dependency detail
    torch = None


def to_numpy(array_like) -> np.ndarray:
    if torch is not None and isinstance(array_like, torch.Tensor):
        return array_like.detach().cpu().numpy()
    return np.asarray(array_like)


def flatten_gmm_model_arrays(model: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.asarray(model["mu"], dtype=float)
    var = np.asarray(model["var"], dtype=float)

    pi = model.get("pi")
    if pi is not None:
        pi_arr = np.asarray(pi, dtype=float)
        if pi_arr.shape != mu.shape[:-1]:
            raise ValueError(
                "GMM mixture weights shape mismatch: "
                f"expected {mu.shape[:-1]}, got {pi_arr.shape}"
            )
        active_mask = pi_arr > 0
        flat_mu = mu[active_mask]
        flat_var = var[active_mask]
        flat_pi = pi_arr[active_mask]
    else:
        flat_mu = mu.reshape(-1, mu.shape[-1])
        flat_var = var.reshape(-1, var.shape[-2], var.shape[-1])
        flat_pi = np.ones(flat_mu.shape[0]) / flat_mu.shape[0]

    return flat_mu, flat_var, flat_pi


def map_metrics(
    sigma: np.ndarray,
    pi: np.ndarray,
    linear_map: np.ndarray,
    mu: np.ndarray | None = None,
) -> dict[str, float]:
    """Per-epoch map-level metrics: Frobenius norm, projected MI, and measures of dimensional
    collapse and redundancy of the latent space.

    Of the map alone (rows = latent axes, s = singular values):
      axis_redundancy   mean |cos| between distinct rows (0 = orthogonal axes)
      gram_dependence   -1/2 log det of the row-normalized Gram matrix (0 = orthogonal axes;
                        the latent MI for an isotropic input; NaN if the rows are dependent)
      effective_rank    (sum s^2)^2 / sum s^4 (participation ratio of the spectrum, 1 to d)
      stable_rank       sum s^2 / max s^2 (1 to d)
    Of the projected GMM components (weighted by the mixture weights pi):
      latent_correlation_within   mean |off-diagonal Pearson correlation| between the latent
                                  dimensions within each component (0 = uncorrelated; what the
                                  diagonal Bures term drops)
      latent_participation_ratio  (sum l)^2 / sum l^2 of the eigenvalues of the projected
                                  covariance, how many latent dimensions carry variance (1 to d)
    Of the whole mixture (only with the component means mu):
      latent_correlation          mean |off-diagonal Pearson correlation| between the latent
                                  dimensions over all data, i.e. under the mixture covariance
                                  sum_k w_k (Sigma_k + mu_k mu_k^T) - mu_bar mu_bar^T (the standard
                                  measure, as penalized by VICReg / Barlow Twins over a batch)
      effective_dof               (sum m)^2 / sum m^2 of the eigenvalues m of that correlation
                                  matrix = d / (1 + (d-1) mean r^2), the effective number of
                                  uncorrelated latent dimensions (1 to d; Bretherton et al. 1999),
                                  in the paper "effective dimensionality" (Del Giudice 2021)"""
    W = np.asarray(linear_map)
    fro_norm = float(np.linalg.norm(W, "fro"))

    unit_rows = W / np.linalg.norm(W, axis=1, keepdims=True)
    gram = unit_rows @ unit_rows.T
    sign, logdet = np.linalg.slogdet(gram)
    squared_singular_values = np.linalg.svd(W, compute_uv=False) ** 2

    # Weighted MI using the package's mutual_information_loss
    weighted_mi = float(to_numpy(mutual_information_loss(sigma, W, weights=pi)))
    total_mi = float(to_numpy(mutual_information_loss(sigma, W, weights=None)))

    weights = np.asarray(pi, dtype=float) / np.sum(pi)
    projected = np.einsum("ij,njk,lk->nil", W, np.asarray(sigma, dtype=float), W)
    sd = np.sqrt(np.einsum("nii->ni", projected))
    correlation = np.abs(projected / (sd[:, :, None] * sd[:, None, :]))
    off_diagonal = ~np.eye(W.shape[0], dtype=bool)
    eigenvalues = np.linalg.eigvalsh(projected)

    metrics = {
        "fro_norm": fro_norm,
        "projected_mi_weighted": weighted_mi,
        "projected_mi_total": total_mi,
        "axis_redundancy": float(np.abs(gram[off_diagonal]).mean()),
        "gram_dependence": float(-0.5 * logdet) if sign > 0 else float("nan"),
        "effective_rank": float(squared_singular_values.sum() ** 2 / (squared_singular_values ** 2).sum()),
        "stable_rank": float(squared_singular_values.sum() / squared_singular_values.max()),
        "latent_correlation_within": float(np.sum(weights * correlation[:, off_diagonal].mean(axis=1))),
        "latent_participation_ratio": float(np.sum(weights * eigenvalues.sum(axis=1) ** 2 / (eigenvalues ** 2).sum(axis=1))),
    }
    if mu is not None:
        # latent covariance of the whole mixture: within-component plus between-component part
        projected_means = np.asarray(mu, dtype=float) @ W.T
        mean = weights @ projected_means
        pooled = (np.einsum("n,nij->ij", weights, projected)
                  + np.einsum("n,ni,nj->ij", weights, projected_means, projected_means) - np.outer(mean, mean))
        pooled_sd = np.sqrt(np.diag(pooled))
        pooled_correlation = pooled / np.outer(pooled_sd, pooled_sd)
        metrics["latent_correlation"] = float(np.abs(pooled_correlation)[off_diagonal].mean())
        correlation_eigenvalues = np.linalg.eigvalsh(pooled_correlation)
        metrics["effective_dof"] = float(correlation_eigenvalues.sum() ** 2 / (correlation_eigenvalues ** 2).sum())
    return metrics


def initial_map(shape: tuple[int, int], init_strategy: str, seed: int) -> np.ndarray:
    """Initial map of one training run of approx_error_bound_epochs.py (stored maps are the
    trained ones only): ggml_ot's initialization for *init_strategy* after seeding with *seed*
    (random_seed + split index, as _set_seed there). Callers check the result against the
    epoch-0 metrics the training stores."""
    import ggml_ot
    from types import SimpleNamespace
    from ggml_ot.optimization._ggml import _ggml_init

    n_comps, dim = shape
    previous = ggml_ot.settings.init_strategy, ggml_ot.settings.random_seed
    ggml_ot.settings.init_strategy = init_strategy
    ggml_ot.settings.random_seed = seed  # seeds torch, as in the training
    try:
        linear_map = _ggml_init(SimpleNamespace(dataset=SimpleNamespace(dim=dim)), n_comps)
    finally:
        ggml_ot.settings.init_strategy, ggml_ot.settings.random_seed = previous
    return to_numpy(linear_map).astype(float)


def gaussian_emd_approx_errors(
    mu: np.ndarray,
    sigma: np.ndarray,
    linear_map: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    min_distance_threshold: float = 0.0,
    min_active_fraction: float = 0.5,
) -> dict[str, float]:
    # Use squared_ground_cost=True so the mean term is ||Wμ_i - Wμ_j||²,
    # matching the squared Bures term B²(Σ_i, Σ_j) for a proper W²_2 decomposition.
    exact_d = to_numpy(pairwise_gaussian_distance(mu, sigma, linear_map, diag_bures_approx=False, squared_ground_cost=True))
    approx_d = to_numpy(pairwise_gaussian_distance(mu, sigma, linear_map, diag_bures_approx=True, squared_ground_cost=True))

    # Compute Bures (covariance) distances directly to avoid catastrophic
    # cancellation from subtracting two large, nearly-equal numbers (total - mean).
    zero_mu = np.zeros_like(mu)
    bures_exact_mat = to_numpy(pairwise_gaussian_distance(zero_mu, sigma, linear_map, diag_bures_approx=False, squared_ground_cost=True))
    bures_approx_mat = to_numpy(pairwise_gaussian_distance(zero_mu, sigma, linear_map, diag_bures_approx=True, squared_ground_cost=True))

    tri = np.triu_indices(exact_d.shape[0], k=1)
    n_pairs_total = len(tri[0])

    exact_vals = np.maximum(np.abs(exact_d[tri]), 0.0)
    bures_exact_vals = np.maximum(bures_exact_mat[tri], 0.0)
    bures_approx_vals = np.maximum(bures_approx_mat[tri], 0.0)

    # Pair weights: w_i * w_j for each upper-triangular pair
    if weights is not None:
        pair_weights = weights[tri[0]] * weights[tri[1]]
    else:
        pair_weights = np.ones(n_pairs_total)

    # Distance threshold: exclude pairs where exact distance is below threshold.
    # This prevents numerically degenerate pairs from dominating relative errors.
    active = exact_vals >= min_distance_threshold
    n_pairs_active = int(active.sum())
    active_frac = n_pairs_active / max(n_pairs_total, 1)

    # If too many pairs are filtered, report NaN (metric is unreliable)
    if active_frac < min_active_fraction:
        nan = float("nan")
        return {
            "approx_error": nan,
            "relative_approx_error": nan,
            "bures_gap": nan,
            "bures_relative_error": nan,
            "bures_fraction": nan,
            "n_pairs_active": n_pairs_active,
            "n_pairs_total": n_pairs_total,
        }

    # Apply mask
    w = pair_weights[active]
    ev = exact_vals[active]
    bev = bures_exact_vals[active]
    bav = bures_approx_vals[active]
    abs_gap = np.abs(exact_d[tri][active] - approx_d[tri][active])
    bures_gap = np.abs(bev - bav)

    # Weighted sums
    total_sum = float(np.sum(w * ev))
    bures_exact_sum = float(np.sum(w * bev))

    # Bures fraction: share of total distance due to covariance
    bures_frac = bures_exact_sum / total_sum if total_sum > 0 else float("nan")

    # Weighted relative errors: sum(w * gap) / sum(w * exact)
    rel_approx = float(np.sum(w * abs_gap)) / total_sum if total_sum > 0 else float("nan")
    bures_rel = float(np.sum(w * bures_gap)) / bures_exact_sum if bures_exact_sum > 0 else float("nan")

    return {
        "approx_error": float(np.sum(w * abs_gap) / np.sum(w)) if np.sum(w) > 0 else float("nan"),
        "relative_approx_error": rel_approx,
        "bures_gap": float(np.sum(w * bures_gap) / np.sum(w)) if np.sum(w) > 0 else float("nan"),
        "bures_relative_error": bures_rel,
        "bures_fraction": bures_frac,
        "n_pairs_active": n_pairs_active,
        "n_pairs_total": n_pairs_total,
    }
