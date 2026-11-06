"""Helpers for the Figure 2 Bures approximation-bound notebook."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib.ticker as tck
import numpy as np
import seaborn as sns
from sklearn.base import clone
from sklearn.mixture import GaussianMixture

from ggml_ot.distances import pairwise_gaussian_distance


# Helper functions (GMM construction, MI/KL, etc.)

def compute_KL(covs: np.ndarray) -> np.ndarray:
    """Compute KL-like MI term used in the original notebook for a batch of covariance matrices."""
    eyes = np.identity(len(covs[0]))[np.newaxis, :, :]
    diag_covs = covs * eyes

    n_comps = len(covs)
    KL = np.zeros((n_comps))
    for i in range(n_comps):
        KL[i] = 0.5 * np.log(np.linalg.det(diag_covs[i]) / np.linalg.det(covs[i]))
    return KL


def create_gmm(covs, means, weights=None) -> GaussianMixture:
    if weights is None:
        weights = np.ones(len(means)) / len(means)
    estimator = GaussianMixture(n_components=len(means))
    estimator.covariances_ = np.asarray(covs)
    estimator.means_ = np.asarray(means)
    estimator.weights_ = np.asarray(weights)
    return estimator


def diag_gmm(gmm: GaussianMixture) -> GaussianMixture:
    gmm_d = clone(gmm)
    gmm_d.means_ = gmm.means_
    gmm_d.weights_ = gmm.weights_
    eyes = np.identity(len(gmm.means_[0]))[np.newaxis, :, :]
    gmm_d.covariances_ = gmm.covariances_ * eyes
    return gmm_d


def ratio_gmm(gmm: GaussianMixture) -> float:
    """Alignment proxy used during experimentation (kept for completeness)."""
    l_Us = []
    for n in range(len(gmm.weights_)):
        if gmm.covariance_type == 'full':
            covariances = gmm.covariances_[n]
        elif gmm.covariance_type == 'tied':
            covariances = gmm.covariances_
        elif gmm.covariance_type == 'diag':
            covariances = np.diag(gmm.covariances_[n])
        elif gmm.covariance_type == 'spherical':
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        v, w = np.linalg.eigh(covariances)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        l_u = w[:, -1]
        l_Us.append(l_u)
    l_Us = np.asarray(l_Us)
    return float(np.inner(l_Us[0], l_Us[1]))


def compute_gmm_marginals(
    data=None,
    cov_type: str = 'full',
    dims=(0, 1),
    gmm: GaussianMixture | None = None,
    title: str = '',
    height: int = 6,
    plot: bool = True,
    ax=None,
    show: bool = True,
    fill: bool = True,
    diag_bures_approx: bool = False,
    hide_axis: bool = False,
):
    """
    Fit or use a 2D GMM, plot ellipses (optional), then compute pairwise distances and MI terms.

    `diag_bures_approx=True` uses the diagonal Bures approximation in `pairwise_gaussian_distance`.
    """
    # Fit GMM if needed
    if gmm is None:
        n_classes = len(np.unique(data.target))
        estimator = GaussianMixture(n_components=n_classes, covariance_type=cov_type, max_iter=20, random_state=0)
        estimator.means_init = np.array([data.data[data.target == i].mean(axis=0) for i in range(n_classes)])
        estimator.fit(data.data)
    else:
        estimator = gmm

    dims = np.asarray(dims)

    # Plot GMM (optional)
    if plot:
        cmap = mpl.colormaps['tab10'].colors

        if data is not None:
            data.target_str = [data.target_names[t] for t in data.target]
            ax = sns.jointplot(
                x=data.data[:, dims[0]],
                y=data.data[:, dims[1]],
                hue=data.target_str,
                palette=cmap,
                height=height,
            )
            ax.set_axis_labels(data.feature_names[dims[0]], data.feature_names[dims[1]])
            ax = ax.ax_joint
        else:
            if ax is None:
                ax = plt.subplot(aspect='equal')
            ax.set_box_aspect(1)
            ax.set_xlim(0, 5)
            ax.set_ylim(0, 5)
            ax.spines['top'].set_color('none')
            ax.spines['bottom'].set_position('zero')
            ax.spines['left'].set_position('zero')
            ax.spines['right'].set_color('none')
            if hide_axis:
                ax.spines[['right', 'left', 'bottom', 'top']].set_visible(False)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlim(1, 4)
                ax.set_ylim(1, 4)
            else:
                ax.xaxis.set_major_locator(tck.MultipleLocator(base=1))
                ax.yaxis.set_major_locator(tck.MultipleLocator(base=1))

        def make_ellipses(gmm: GaussianMixture, ax, colors):
            for n in range(len(gmm.weights_)):
                if gmm.covariance_type == 'full':
                    covariances = gmm.covariances_[n][dims, :][:, dims]
                elif gmm.covariance_type == 'tied':
                    covariances = gmm.covariances_[dims, :][:, dims]
                elif gmm.covariance_type == 'diag':
                    covariances = np.diag(gmm.covariances_[n][dims])
                elif gmm.covariance_type == 'spherical':
                    covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]

                v, w = np.linalg.eigh(covariances)
                u = w[0] / np.linalg.norm(w[0])
                angle = np.arctan2(u[1], u[0])
                angle = 180 * angle / np.pi
                v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
                ell = mpl.patches.Ellipse(
                    gmm.means_[n, dims], v[0], v[1], angle=180 + angle, color=colors[n], fill=fill
                )
                ell.set_clip_box(ax.bbox)
                ell.set_alpha(0.5)
                ax.add_artist(ell)

        make_ellipses(estimator, ax, colors=cmap)
        ax.title.set_text(title)
        if show:
            plt.show()

    # Compute Wasserstein/Bures between Gaussians via ggml_ot
    if estimator.covariances_.ndim == 2:
        covs = np.asarray([np.diag(covs) for covs in estimator.covariances_])
    else:
        covs = estimator.covariances_

    emd_distances = pairwise_gaussian_distance(
        mu=estimator.means_,
        sigma=covs,
        w=np.identity(len(estimator.means_[0])),
        diag_bures_approx=diag_bures_approx,
    ).numpy()

    # Compute MI term (called KL in the original notebook)
    eyes = np.identity(len(estimator.means_[0]))[np.newaxis, :, :]
    diag_covs = estimator.covariances_ * eyes

    n_comps = len(estimator.means_)
    KL = np.zeros((n_comps))
    for i in range(n_comps):
        KL[i] = 0.5 * np.log(np.linalg.det(diag_covs[i]) / np.linalg.det(estimator.covariances_[i]))

    return estimator, emd_distances, KL


# Plotting utilities

def contour(
    X,
    Y,
    Z,
    title: str = '',
    axis_label: str = 'Rotation Angle',
    ax=None,
    fig=None,
    ticks_pi: bool = False,
    invert_cbar: bool = False,
    cbar_title: str | None = None,
    show_diag_line: bool = True,
    diag_line_kwargs: dict | None = None,
    save_path: Path | None = None,
    show: bool = True,
):
    if ax is None:
        if fig is None:
            fig = plt.figure()
        ax = plt.gca()
    if fig is None:
        fig = ax.figure

    ax.contour(X, Y, Z, levels=17, linewidths=0.5, colors='k', vmin=0)
    cntr = ax.contourf(X, Y, Z, levels=17, vmin=0, cmap='viridis_r' if invert_cbar else None)

    # Overlay the diagonal slice used by contour_diag (i == j)
    if show_diag_line:
        X_arr = np.asarray(X)
        Y_arr = np.asarray(Y)
        if X_arr.ndim == 1 and Y_arr.ndim == 1 and X_arr.shape == Y_arr.shape:
            kw = {'linestyle': '--', 'linewidth': 1.25, 'alpha': 0.9, 'color': 'white'}
            if diag_line_kwargs:
                kw.update(diag_line_kwargs)
            ax.plot(X_arr, Y_arr, **kw)

    x_min, x_max = float(np.min(X)), float(np.max(X))
    y_min, y_max = float(np.min(Y)), float(np.max(Y))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal', adjustable='box')

    cbar = fig.colorbar(cntr, ax=ax)
    if cbar_title:
        cbar.set_label(cbar_title)
    if invert_cbar:
        cbar.ax.invert_yaxis()

    if title:
        fig.suptitle(title)
    ax.invert_yaxis()

    if ticks_pi:
        step = 0.125
        x_start = step * np.floor(x_min / step)
        x_stop = step * np.ceil(x_max / step)
        y_start = step * np.floor(y_min / step)
        y_stop = step * np.ceil(y_max / step)
        xticks = np.arange(x_start, x_stop + 0.5 * step, step)
        yticks = np.arange(y_start, y_stop + 0.5 * step, step)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

        def _pi_formatter(value, pos):
            return rf'${value:g}\pi$'

        formatter = plticker.FuncFormatter(_pi_formatter)
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        # set_xticks/set_yticks expands the view to include rounded tick positions;
        # restore the actual contour domain afterwards.
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min)

    ax.set_xlabel(axis_label + r' $\mathcal{Y}$')
    ax.set_ylabel(axis_label + r' $\mathcal{X}$')

    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=mpl.rcParams.get('figure.dpi', 300), bbox_inches='tight')
    if show:
        plt.show()


def contour_diag(
    X,
    Y,
    Z,
    title: str = '',
    axis_label: str = 'Rotation Angle',
    ax=None,
    fig=None,
    ticks_pi: bool = False,
    invert_cbar: bool = False,
    cbar_title: str | None = None,
    line_fmt: dict | None = None,
):
    """Line plot of the diagonal of Z as a function of the diagonal coordinate from X/Y."""
    created_ax = ax is None
    if ax is None:
        if fig is None:
            fig = plt.figure()
        ax = plt.gca()
    if fig is None:
        fig = ax.figure

    Z_arr = np.asarray(Z)
    if Z_arr.ndim != 2:
        raise ValueError(f'Z must be 2D, got shape {Z_arr.shape}')

    n = min(Z_arr.shape[0], Z_arr.shape[1])
    z_diag = np.diag(Z_arr)[:n]

    X_arr = np.asarray(X)
    Y_arr = np.asarray(Y)

    if X_arr.ndim == 2:
        x_diag = np.diag(X_arr)[:n]
    elif X_arr.ndim == 1:
        x_diag = X_arr[:n]
    else:
        x_diag = np.arange(n)

    if Y_arr.ndim == 2:
        y_diag = np.diag(Y_arr)[:n]
    elif Y_arr.ndim == 1:
        y_diag = Y_arr[:n]
    else:
        y_diag = None

    if y_diag is None or x_diag.shape != y_diag.shape:
        t_diag = x_diag
    else:
        t_diag = x_diag if np.allclose(x_diag, y_diag) else 0.5 * (x_diag + y_diag)

    (line,) = ax.plot(t_diag, z_diag, label=cbar_title if cbar_title else None, **(line_fmt or {}))

    if created_ax:
        if title:
            ax.set_title(title)
        ax.set_xlabel(axis_label)
        ax.set_ylabel('Value')
        if invert_cbar:
            ax.invert_yaxis()
        if ticks_pi:
            def _pi_formatter(value, pos):
                return rf'${value:g}\pi$'

            formatter = plticker.FuncFormatter(_pi_formatter)
            ax.xaxis.set_major_formatter(formatter)

        fig.tight_layout()
        plt.show()

    return line


# Upper bounds for diagonal approximation error (see quantify_bonds_diagW_KL.ipynb)

def _sqrtm_spd(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    w, V = np.linalg.eigh(M)
    w = np.clip(w, eps, None)
    return (V * np.sqrt(w)) @ V.T


def _bures_cov_distance_sq(S1: np.ndarray, S2: np.ndarray) -> float:
    """Bures/W2^2 between zero-mean Gaussians with covariances S1, S2."""
    S1_sqrt = _sqrtm_spd(S1)
    middle = S1_sqrt @ S2 @ S1_sqrt
    middle_sqrt = _sqrtm_spd(middle)
    d = float(np.trace(S1) + np.trace(S2) - 2.0 * np.trace(middle_sqrt))
    return float(max(d, 0.0))


def _mi_of_cov(Sigma: np.ndarray) -> float:
    """Mutual-information term (Section 3.4): 0.5 log det(diag(Sigma))/det(Sigma)."""
    D = np.diag(np.diag(Sigma))
    return float(0.5 * np.log(np.linalg.det(D) / np.linalg.det(Sigma)))


def bound_mi_only(Xi: np.ndarray, Yj: np.ndarray, A: np.ndarray) -> float:
    """Baseline: MI_i + MI_j after push-forward under metric A."""
    Sigma_i_A = A @ Xi @ A.T
    Sigma_j_A = A @ Yj @ A.T
    return _mi_of_cov(Sigma_i_A) + _mi_of_cov(Sigma_j_A)

def bound_sqrt_mi_only(Xi: np.ndarray, Yj: np.ndarray, A: np.ndarray) -> float:
    """Baseline: MI_i + MI_j after push-forward under metric A."""
    Sigma_i_A = A @ Xi @ A.T
    Sigma_j_A = A @ Yj @ A.T
    return np.sqrt(_mi_of_cov(Sigma_i_A)) + np.sqrt(_mi_of_cov(Sigma_j_A))

def bound_bures_to_diag_sum(Xi: np.ndarray, Yj: np.ndarray, A: np.ndarray) -> float:
    """Sum of Bures distances to diagonalized covariances (triangle-inequality-style bound)."""
    Sigma_i_A = A @ Xi @ A.T
    Sigma_j_A = A @ Yj @ A.T
    D_i_A = np.diag(np.diag(Sigma_i_A))
    D_j_A = np.diag(np.diag(Sigma_j_A))
    return _bures_cov_distance_sq(Sigma_i_A, D_i_A) + _bures_cov_distance_sq(Sigma_j_A, D_j_A)


def bound_1_elegant(Xi: np.ndarray, Yj: np.ndarray, A: np.ndarray) -> float:
    """
    Bound 1 (Theorem 3.6): geometric-mean condition number.

    Delta <= sqrt(2sqrt(kappa_i*kappa_j)) * (sqrt(tr(Sigma_i^A)^2*MI_i) + sqrt(tr(Sigma_j^A)^2*MI_j))
    """
    Sigma_i_A = A @ Xi @ A.T
    Sigma_j_A = A @ Yj @ A.T

    MI_i = _mi_of_cov(Sigma_i_A)
    MI_j = _mi_of_cov(Sigma_j_A)

    eig_i = np.linalg.eigvalsh(Sigma_i_A)
    eig_j = np.linalg.eigvalsh(Sigma_j_A)
    kappa_i = float(np.max(eig_i) / np.min(eig_i))
    kappa_j = float(np.max(eig_j) / np.min(eig_j))
    kappa_geom = float(np.sqrt(kappa_i * kappa_j))

    term_i = float(np.sqrt((np.trace(Sigma_i_A) ** 2) * MI_i))
    term_j = float(np.sqrt((np.trace(Sigma_j_A) ** 2) * MI_j))

    return float(np.sqrt(2 * kappa_geom) * (term_i + term_j))


def bound_2_conservative(Xi: np.ndarray, Yj: np.ndarray, A: np.ndarray) -> float:
    """
    Bound 2 (Theorem 3.6): max condition number.

    Delta <= sqrt(2*kappa_bar) * (sqrt(tr(Sigma_i^A)^2*MI_i) + sqrt(tr(Sigma_j^A)^2*MI_j))
    """
    Sigma_i_A = A @ Xi @ A.T
    Sigma_j_A = A @ Yj @ A.T

    MI_i = _mi_of_cov(Sigma_i_A)
    MI_j = _mi_of_cov(Sigma_j_A)

    eig_i = np.linalg.eigvalsh(Sigma_i_A)
    eig_j = np.linalg.eigvalsh(Sigma_j_A)
    kappa_i = float(np.max(eig_i) / np.min(eig_i))
    kappa_j = float(np.max(eig_j) / np.min(eig_j))
    kappa_max = float(max(kappa_i, kappa_j))

    term_i = float(np.sqrt((np.trace(Sigma_i_A) ** 2) * MI_i))
    term_j = float(np.sqrt((np.trace(Sigma_j_A) ** 2) * MI_j))

    return float(np.sqrt(2 * kappa_max) * (term_i + term_j))


BOUND_FNS = {
    'mi_only': bound_mi_only,
    'sqrt_mi_only': bound_sqrt_mi_only,
    'bures_to_diag_sum': bound_bures_to_diag_sum,
    'thm36_bound1_geom': bound_1_elegant,
    'thm36_bound2_max': bound_2_conservative,
}


def fmt_radians_for_path(x: float) -> str:
    s = f'{x/np.pi:.2f}'.rstrip('0').rstrip('.')
    return '0' if s == '-0' else s


def pi_formatter(value, pos):
                return rf'${value:g}\pi$'


__all__ = [
    "BOUND_FNS",
    "compute_KL",
    "compute_gmm_marginals",
    "contour",
    "contour_diag",
    "create_gmm",
    "diag_gmm",
    "fmt_radians_for_path",
    "pi_formatter",
]
