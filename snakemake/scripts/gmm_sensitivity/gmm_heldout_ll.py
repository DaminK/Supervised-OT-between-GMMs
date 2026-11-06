#!/usr/bin/env python
"""Held-out GMM goodness-of-fit for the fixed-k component-count sweep.

For a single fixed number of components k, fit a per-patient full-covariance GMM
on a deterministic train split and score the mean log-likelihood on the held-out
split. This measures how well a k-component GMM fits each patient's cells, so it
can be cross-referenced against the downstream KNN/ARI at the same k: it separates
"bad downstream performance because the fit is poor" from "two comparably good
fits at different k that give similar downstream performance".

This is an independent fit-quality proxy (sklearn EM, matched settings), not the
exact ggml torch-EM used downstream. The per-patient train/val split is fixed
across k, so the held-out log-likelihood is comparable across the sweep.

Usage:
  python gmm_heldout_ll.py PREPARED_H5AD PATIENT_COL USE_REP N_FEATS K SEED OUT_CSV
"""
import sys
import numpy as np
import pandas as pd
import anndata as ad
from sklearn.mixture import GaussianMixture

MIN_CELLS = 250
TRAIN_SIZE = 0.5
# Full-covariance GMMs in 100-dim PCA need a covariance floor to avoid singular
# fits and catastrophic held-out log-likelihoods. 1e-3 matches the guarded eps
# regime of the downstream fit and keeps the per-patient held-out LL well-behaved.
REG_COVAR = 1e-3
MAX_ITER = 100


def main():
    h5ad, patient_col, use_rep, n_feats, k, seed, out_csv = sys.argv[1:8]
    n_feats, k, seed = int(n_feats), int(k), int(seed)

    a = ad.read_h5ad(h5ad)
    X = np.asarray(a.obsm[use_rep][:, :n_feats], dtype=np.float64)
    patients = a.obs[patient_col].values

    per_patient = []
    for pid in pd.Series(patients).unique():
        Xi = X[patients == pid]
        n = len(Xi)
        if n < MIN_CELLS:
            continue
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        n_tr = int(round(TRAIN_SIZE * n))
        tr, va = idx[:n_tr], idx[n_tr:]
        if len(va) < k or n_tr < k:
            continue
        try:
            gm = GaussianMixture(
                n_components=k, covariance_type="full",
                max_iter=MAX_ITER, n_init=1, reg_covar=REG_COVAR,
                random_state=seed,
            ).fit(Xi[tr])
            ll = float(gm.score(Xi[va]))  # mean per-cell held-out log-likelihood
        except Exception as e:  # singular covariance etc.
            print(f"  patient {pid}: skipped ({e})", file=sys.stderr)
            continue
        per_patient.append((pid, n, ll))

    lls = np.array([p[2] for p in per_patient], dtype=float)
    # Median is the primary aggregate: the per-patient held-out LL has a heavy
    # left tail (a collapsed component sends a few patients very negative), so the
    # mean is outlier-dominated while the median is stable across reg_covar.
    med = float(np.median(lls))
    q25, q75 = (float(np.percentile(lls, 25)), float(np.percentile(lls, 75)))
    header = not _exists_nonempty(out_csv)
    with open(out_csv, "a") as fh:
        if header:
            fh.write("k,heldout_ll_median,heldout_ll_q25,heldout_ll_q75,heldout_ll_mean,n_patients\n")
        fh.write(f"{k},{med:.4f},{q25:.4f},{q75:.4f},{float(np.mean(lls)):.4f},{len(per_patient)}\n")
    print(f"k={k}: held-out LL median {med:.2f} (IQR {q25:.2f}..{q75:.2f}) "
          f"over {len(per_patient)} patients")


def _exists_nonempty(path):
    try:
        with open(path) as fh:
            return bool(fh.read(1))
    except FileNotFoundError:
        return False


if __name__ == "__main__":
    main()
