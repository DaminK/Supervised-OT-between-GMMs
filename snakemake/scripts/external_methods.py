"""
External (non-GGML) method wrappers for the GGML-OT benchmark pipeline.

Each function takes a dataset and method-specific params, runs its own
training, and returns ``(ground_metric, epoch_time)`` where ground_metric
is a callable/matrix accepted by ``ggml_ot.test``. The benchmark calls it via
``ggml_ot.benchmark.train_test_external``, once per split on the training
distributions only; ``max_iter`` and ``n_comps`` come from the config's
``train_test`` section unless the method's ``params`` override them.

Adding a new external method:
  1. Write a function with the signature below.
  2. Register it in EXTERNAL_REGISTRY.
  3. Add a YAML block in the experiment config (``external: <name>``).
"""

import inspect
import logging
import time
import numpy as np

logger = logging.getLogger(__name__)


def _to_numpy(x):
    """Convert torch/numpy inputs to a CPU numpy array."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class POTProjection:
    """Pickle-safe linear projection callable returned by POT wrappers."""

    def __init__(self, proj_matrix, mean=None):
        self.proj_matrix = np.asarray(proj_matrix)
        self.mean = None if mean is None else np.asarray(mean)

    def __call__(self, x):
        x_np = _to_numpy(x)
        if self.mean is not None:
            x_np = x_np - self.mean.reshape((1, -1))
        return np.asarray(x_np.dot(self.proj_matrix))


def _as_picklable_projection(proj_fn, fallback_proj_matrix=None, fallback_mean=None):
    """Extract POT closure state and return a pickle-safe projection object."""
    proj_matrix = fallback_proj_matrix
    mean = fallback_mean
    try:
        closure_nonlocals = inspect.getclosurevars(proj_fn).nonlocals
        proj_matrix = closure_nonlocals.get("Popt", proj_matrix)
        mean = closure_nonlocals.get("mx", mean)
    except Exception:
        pass

    if proj_matrix is None:
        raise ValueError("Could not extract projection matrix from external method.")
    # POT >=0.9: closure may capture OptimizerResult instead of raw matrix
    if hasattr(proj_matrix, "point"):
        proj_matrix = proj_matrix.point
    return POTProjection(proj_matrix=proj_matrix, mean=mean)


def wda_method(dataset, n_comps: int, max_iter: int, **kwargs):
    """Wasserstein Discriminant Analysis via POT ``ot.dr.wda()``.

    The initial projection is seeded. Left to POT, ``P0=None`` makes pymanopt draw
    the starting point on the Stiefel manifold from the *global legacy* NumPy RNG
    (``pymanopt.manifolds.Stiefel.random_point`` -> ``np.random.normal``), which the
    pipeline never seeds: ``ggml_ot.settings.random_seed`` initialises its own
    ``default_rng`` instead. WDA results therefore varied between otherwise
    identical runs. We pass an explicit orthonormal ``P0`` drawn from
    ``ggml_ot.settings.numpy_generator``, which the config's ``random_seed`` seeds.
    As for GGML's initialization, each split thus starts from a different
    projection, and repeated runs start from the same ones.

    Training stops after ``max_iter`` iterations or on pymanopt's convergence
    criteria, as GGML stops after ``max_iter`` epochs or at ``stop_thr``. POT's
    default optimizer additionally stops after 1000 s of wall-clock time, which
    made the number of iterations depend on machine load, so we pass the same
    optimizer without that limit. The epoch time is the optimizer's measured time
    divided by the iterations it actually ran.

    Returns
    -------
    projwda : callable
        Projection function returned by wda — suitable as ground_metric.
    epoch_time : float
        Average seconds per iteration.
    """
    from ot.dr import wda
    from pymanopt.optimizers import SteepestDescent
    from ggml_ot import settings

    X = _to_numpy(dataset.supports)
    y = _to_numpy(dataset.distribution_labels)
    p, n, d = X.shape
    X_flat = X.reshape(p * n, d)
    y_flat = np.repeat(y, n)

    # Orthonormal d x n_comps starting point = a valid point on the Stiefel manifold.
    P0, _ = np.linalg.qr(settings.numpy_generator.standard_normal((d, n_comps)))

    optimizer = SteepestDescent(max_iterations=max_iter, max_time=np.inf)
    result, projwda = wda(X_flat, y_flat, p=n_comps, maxiter=max_iter, P0=P0, solver=optimizer)
    # POT returns only the optimized matrix; pymanopt's result, with the iteration
    # count and optimization time, is held in the projection's closure.
    optimizer_result = inspect.getclosurevars(projwda).nonlocals["Popt"]
    epoch_time = optimizer_result.time / max(optimizer_result.iterations, 1)
    logger.info(
        f"WDA: {optimizer_result.iterations} iterations, stopping criterion: "
        f"{optimizer_result.stopping_criterion}, {epoch_time:.2f} s per iteration"
    )
    # POT >=0.9 returns an OptimizerResult; extract the raw matrix via .point
    proj_matrix = result.point if hasattr(result, "point") else result
    projection = _as_picklable_projection(
        projwda,
        fallback_proj_matrix=proj_matrix,
        fallback_mean=X_flat.mean(axis=0),
    )
    return projection, epoch_time


def fda_method(dataset, n_comps: int, **kwargs):
    """Fisher Discriminant Analysis via POT ``ot.dr.fda()``.

    Returns
    -------
    projfda : callable
        Projection function returned by fda.
    epoch_time : float
        Total wall-clock seconds (fda is a one-shot solver).
    """
    from ot.dr import fda

    X = _to_numpy(dataset.supports)
    y = _to_numpy(dataset.distribution_labels)
    p, n, d = X.shape
    X_flat = X.reshape(p * n, d)
    y_flat = np.repeat(y, n)

    t0 = time.time()
    result, projfda = fda(X_flat, y_flat, p=n_comps)
    epoch_time = time.time() - t0
    # POT >=0.9 returns an OptimizerResult; extract the raw matrix via .point
    proj_matrix = result.point if hasattr(result, "point") else result
    projection = _as_picklable_projection(
        projfda,
        fallback_proj_matrix=proj_matrix,
        fallback_mean=X_flat.mean(axis=0),
    )
    return projection, epoch_time


EXTERNAL_REGISTRY = {
    "wda": wda_method,
    "fda": fda_method,
}
