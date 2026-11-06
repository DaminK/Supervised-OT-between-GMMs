#!/usr/bin/env bash
# Controlled fixed-k GMM component-count sensitivity sweep (appendix, tab:gmm_sensitivity).
#
# For each k in K_VALUES, force every patient to exactly k GMM components, run
# the two-method benchmark (Ours Diag., Ours Exact) on the myocardial cohort with
# fixed hyperparameters (tuning off), and collect the aggregated scores.
#
# Everything is method-hash-isolated via `benchmark_only`, so this never touches
# the shared plot/aggregate/paper-export tree.
#
# Usage, from snakemake/:  bash scripts/gmm_sensitivity/sweep_fixed_k.sh "5 8 10 12 15"
set -euo pipefail

# Cap BLAS threads: fit_gmm inherits this env. Without it, full-covariance GMM
# EM oversubscribes ~100 cores and runs ~18x slower (65s vs 3.5s per patient).
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMK="$(cd "$SCRIPT_DIR/../.." && pwd)"           # snakemake/
REPO="$(dirname "$SMK")"                           # pipeline repo root
BASE="config/ablation/gmm_sensitivity.yml"
OUT="$REPO/data/generated_tables/gmm_sensitivity"
CFG_DIR="$(mktemp -d)"                             # per-k configs, removed on exit
trap 'rm -rf "$CFG_DIR"' EXIT
DATASET="myocardial_infarction"
K_VALUES="${1:-3 5 8 10 12 15}"

mkdir -p "$OUT"
SUMMARY="$OUT/fixed_k_sensitivity.csv"           # downstream: KNN / ARI per (k, method)
GOF="$OUT/fixed_k_heldout_ll.csv"                # goodness of fit: held-out LL per k
PREPARED="$REPO/data/snakemake/$DATASET/prepared/X_pca_100.h5ad"
PATIENT_COL="sample"
SEED=42
: > "$SUMMARY"; echo "k,method,knn_acc,ari" > "$SUMMARY"
: > "$GOF"

cd "$SMK"
for k in $K_VALUES; do
  echo "===== k=$k ====="
  TMP="$CFG_DIR/k${k}.yml"
  sed "s/^  k_comps:.*/  k_comps: [$k]/" "$BASE" > "$TMP"

  # (1) downstream benchmark (method-hash-isolated)
  ../.venv/bin/snakemake --configfile "$TMP" -j8 benchmark_only
  for m in GMM_FullBures GMM_DiagBures; do
    f=$(ls -t "$REPO/data/snakemake/$DATASET/benchmark/$m"/*/scores.csv 2>/dev/null | head -1)
    if [ -f "$f" ]; then
      ../.venv/bin/python - "$k" "$m" "$f" "$SUMMARY" <<'PY'
import sys, csv
k, method, path, summary = sys.argv[1:5]
rows = list(csv.reader(open(path)))
hdr, data = rows[0], rows[2]         # row1 is a "mean/SD" label row, data is row2
knn = data[1]                        # first 'knn' column is the mean
ari = data[hdr.index("ari")]         # first 'ari' column is the mean
with open(summary, "a") as fh:
    fh.write(f"{k},{method},{knn},{ari}\n")
PY
    fi
  done

  # (2) held-out goodness of fit at this k (independent, fast)
  ../.venv/bin/python "$SCRIPT_DIR/gmm_heldout_ll.py" \
      "$PREPARED" "$PATIENT_COL" "X_pca" 100 "$k" "$SEED" "$GOF"

  rm -f "$TMP"
done
echo "===== done ====="
echo "-- downstream ($SUMMARY) --"; cat "$SUMMARY"
echo "-- goodness of fit ($GOF) --"; cat "$GOF"
