Scalable Supervised Optimal Transport of Gaussian Mixture Models
================================================================

This repository reproduces the results of the paper *Scalable Supervised Optimal Transport of
Gaussian Mixture Models*. The method is implemented in the Python package
[ggml-ot](https://github.com/DaminK/ggml-ot). This repository contains the Snakemake pipeline,
the configurations and the notebooks that produce every table and figure of the paper.

## Abstract

Optimal Transport (OT) is a principled framework for comparing probability distributions, but its effectiveness depends critically on the ground metric, the cost used to compare observations.
In high-dimensional settings, fixed ground metrics like the Euclidean distance can make OT distances reflect task-irrelevant variation rather than low-dimensional class structure, limiting OT in tasks such as classification and clustering.
Supervised OT addresses this by learning a parameterized ground metric such that the induced OT distance reflects class structure.
However, existing approaches on point clouds scale poorly with the number of observations.
We propose a scalable supervised OT framework for Gaussian Mixture Models (GMMs) that lifts ground metric learning from observations to mixture components, replacing large point-cloud OT problems with smaller component-level problems whose transport costs admit closed forms.
We define a learnable Wasserstein-type distance between GMMs using the Generalized Bures-Wasserstein (GBW) distance between Gaussian components as the ground metric. The ground metric is parameterized as a projection of Gaussian components into a low-dimensional latent space inducing an OT distance between GMMs.
For additional scalability, we introduce a diagonal approximation of the GBW metric, that reduces the cost of the covariance computation between components to be linear.
We bound the resulting approximation error by the mutual information between latent Gaussian variables and the Frobenius norm of the learned map, yielding a principled regularization strategy that also enables interpretation of latent axes.
Across synthetic and scRNA-seq benchmarks, our method significantly improves classification and clustering over fixed-metric GMM-based OT and achieves competitive performance with substantially lower OT distance computation time than point-cloud baselines.

## Repository layout

```
snakemake/            Snakemake pipeline (see snakemake/README.md)
  Snakefile
  config/             defaults, datasets, and the configs of the paper (paper/, ablation/)
  rules/              one file per pipeline stage
  scripts/            scripts called by the rules
notebooks/            notebooks for the figures not built by the pipeline
data/                 outputs (created by the runs, not tracked)
requirements.txt      pinned versions of the environment that produced the paper results
```

## Setup

Linux and Python 3.11. All methods run on CPU. The paper results were computed on one node with
128 cores and 1 TB RAM; the default memory limit of the runs is 512 GB.

```bash
make setup-env          # .venv with the pinned packages, opens a shell with it activated
make setup-gmvae-env    # .venv_gmvae, only for the PILOT-GM-VAE comparison (Table 2)
```

Datasets are prepared by the pipeline: the four scRNA-seq cohorts are downloaded from
[CELLxGENE](https://cellxgene.cziscience.com/) (dataset ids in
[`snakemake/config/datasets.yml`](snakemake/config/datasets.yml)) and the two synthetic datasets
are generated from the parameters in the same file.

## Reproduce the results

Run the main benchmark first. All other runs reuse its preprocessing, GMM fits and tuning,
since the pipeline stores every output under a hash of its settings.

```bash
make benchmark                    # main benchmark, about 11 h with 12 jobs on 128 cores
make ablation-approx-error        # regularization ablation (Figure 5)
make ablation-hyperparameters     # hyperparameter sweep on Myocardial Infarction

cd snakemake
# Table 2: per-patient GMMs from PILOT-GM-VAE, then the table
snakemake --configfile config/paper/main_benchmark.yml config/paper/pilot_gmvae.yml -j8 -- benchmark_only
snakemake --configfile config/paper/main_benchmark.yml -j8 -R scrna_baselines_table -- scrna_baselines_table_only
# Point-cloud baselines on raw cells instead of GMM samples
snakemake --configfile config/paper/main_benchmark.yml config/ablation/_raw_cells.yml -j8 -- generate_paper_results
# Number-of-components check
snakemake --configfile config/paper/main_benchmark.yml config/ablation/_component_stability.yml -j8 -- patient_component_stability_only
# GMM-fitting sensitivity: fixed k, covariance type, PCA dimension, covariance floor
bash scripts/gmm_sensitivity/sweep_fixed_k.sh "5 8 10 12 15" && bash scripts/gmm_sensitivity/sweep_gmm_sensitivity.sh
```

Then run the notebooks in [`notebooks/`](notebooks/) for the remaining figures. The tables and
figures of the paper are written to `data/generated_tables/` and `data/generated_plots/`, under
the paths the paper uses.

| Paper | Produced by | Output |
|---|---|---|
| Table 1, per-dataset tables (App. B), Figure 3, Figures B.1-B.2 | `make benchmark` | `generated_tables/benchmark_table.tex`, `generated_tables/appendix/benchmark_table_*.tex`, `generated_plots/pareto/`, `generated_plots/clustermaps/` |
| Table A.1, Tables B.8-B.9, Tables C.1-C.3, Figure A.1 (a), Figure C.2, Figure C.3 (b) | `make benchmark` | `datasets_table.tex`, `appendix/complete_linkage_table.tex`, `appendix/paired_tests_all.tex`, `appendix/runtime_table.tex`, `distance_agreement*.tex`; `generated_plots/appendix/{gmm_validation,tuning,scaling}/` |
| Table 2, Table B.10 | PILOT-GM-VAE overlay, then `scrna_baselines_table_only` | `scrna_baselines_table.tex`, `appendix/scrna_baselines_per_dataset.tex` |
| Table B.11 | raw-cell overlay | `appendix/raw_cells_table.tex` |
| Figure 5, Figure C.4, Table C.4 | `make ablation-approx-error` | `generated_plots/approx_error_bound/selected/`, `map_faithfulness.tex` |
| Figure C.1, Figure C.3 (a, c) | `make ablation-hyperparameters`, then `supp_hyperparameters.ipynb` | `generated_plots/hyperparameter_sensitivity/`, `generated_plots/appendix/latent_dim/` |
| Table A.2 | GMM-sensitivity sweeps | CSVs in `generated_tables/gmm_sensitivity/` |
| Number-of-components check (App. A.2) | component-stability overlay | `snakemake/component_stability_analysis/` |
| Figure 1 | `fig1_overview.ipynb` | `generated_plots/overview/` |
| Figure 2, Figures D.1-D.2 | `fig2_bures_approx_bound.ipynb`, run once per angle (`initial_pair_angle` = π/2, π/3, π/4) | `generated_plots/rotation_bound/` |
| Figure 4, Figure A.2 | `fig6_biological_analysis.ipynb` | `generated_plots/interpretability/` |
| Figure A.1 (b) | `supp_gmm_validation.ipynb` | `generated_plots/appendix/gmm_validation/` |

Paths in the last column are relative to `data/`; table paths without a folder are in
`data/generated_tables/`.

Common overrides of the `make` targets:

```bash
make benchmark JOBS=4                    # fewer parallel jobs (default 8)
make benchmark MEM_MB=256000             # lower memory limit in MB (default 512000)
make benchmark SMKFLAGS="--dry-run"      # extra Snakemake flags
```

## License

MIT, see [LICENSE](LICENSE).
