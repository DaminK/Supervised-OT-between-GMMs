#!/usr/bin/env bash
# GMM-fitting sensitivity sweep (appendix, tab:gmm_sensitivity): vary one GMM-fitting choice at a
# time on the myocardial cohort (k=10, tuning off, fixed hyperparameters) and
# record downstream KNN / ARI under average linkage. Covers covariance type
# (full / diagonal / low-rank), PCA dimension (100 / 50), and the covariance
# floor eps (1e-4 default / 1e-6 / 1e-2).
#
# Self-contained: every variant config is generated inline from the base config,
# so this needs no pre-generated files and no sibling working directory. Runs
# sequentially in the main snakemake dir (Snakemake's per-directory lock makes
# concurrent runs in the same dir unsafe; sequential is the portable form).
#
# The full / 100-dim / eps=1e-4 baseline is identical to the k=10 point of
# sweep_fixed_k.sh and is re-run here (cached, so instant) to make the table
# self-contained.
#
# Usage, from snakemake/:  bash scripts/gmm_sensitivity/sweep_gmm_sensitivity.sh
set -uo pipefail

export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMK="$(cd "$SCRIPT_DIR/../.." && pwd)"           # snakemake/
REPO="$(dirname "$SMK")"                           # pipeline repo root
BASE="config/ablation/gmm_sensitivity.yml"
D="$(mktemp -d)"                                   # per-variant configs, removed on exit
trap 'rm -rf "$D"' EXIT
OUT="$REPO/data/generated_tables/gmm_sensitivity"
mkdir -p "$OUT"
SUMMARY="$OUT/gmm_fitting_sensitivity.csv"
echo "variant,method,knn_acc,ari" > "$SUMMARY"

cd "$SMK"

# run_variant LABEL SED_EXPR   (empty SED_EXPR = baseline, config unchanged)
run_variant () {
  local label="$1" sed_expr="$2"
  echo "===== $label ====="
  local tmp="$D/_sens_${label}.yml"
  if [ -z "$sed_expr" ]; then cp "$BASE" "$tmp"; else sed -e "$sed_expr" "$BASE" > "$tmp"; fi
  ../.venv/bin/snakemake --configfile "$tmp" --unlock >/dev/null 2>&1 || true
  local log; log=$(mktemp)
  if ../.venv/bin/snakemake --configfile "$tmp" -j8 benchmark_only >"$log" 2>&1; then
    for m in GMM_FullBures GMM_DiagBures; do
      local f; f=$(grep -oE "/[^ ]*/benchmark/$m/[a-f0-9]+/scores.csv" "$log" | head -1)
      if [ -f "$f" ]; then
        ../.venv/bin/python - "$label" "$m" "$f" "$SUMMARY" <<'PY'
import sys, csv
variant, method, path, summary = sys.argv[1:5]
rows = list(csv.reader(open(path))); hdr, data = rows[0], rows[2]
open(summary, "a").write(f"{variant},{method},{data[1]},{data[hdr.index('ari')]}\n")
PY
      else
        echo "  WARN: no scores path for $m"
      fi
    done
  else
    echo "===== $label FAILED ====="; tail -6 "$log"
  fi
  rm -f "$log" "$tmp"
}

run_variant baseline_full_100 ""
run_variant covtype_diag      's/covariance_type: "full"/covariance_type: "diag"/'
run_variant covtype_lowrank   's/covariance_type: "full"/covariance_type: "lowrank"\n    factor_rank: 20/'
run_variant pca50             's/^n_feats: 100/n_feats: 50/'
run_variant eps_1e-6          's/covariance_type: "full"/covariance_type: "full"\n    eps: 0.000001/'
run_variant eps_1e-2          's/covariance_type: "full"/covariance_type: "full"\n    eps: 0.01/'

echo "===== done -> $SUMMARY ====="
cat "$SUMMARY"
