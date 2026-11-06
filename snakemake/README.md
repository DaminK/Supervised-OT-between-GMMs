Snakemake workflow
==================

The pipeline behind all results of the paper. For setup and the commands that reproduce each
table and figure, see the [main README](../README.md). Run Snakemake from this directory, for
example

```bash
snakemake --configfile config/paper/main_benchmark.yml -j8 -- generate_paper_results
```

## Stages

For every dataset: download (or generate) → preprocess (log1p, PCA) → fit per-patient GMMs →
tune hyperparameters (learned metrics) → train and evaluate each method on five train-test
splits. Aggregation rules then build the tables and figures across datasets, and the export copies
the paper's outputs to `../data/generated_plots/` and `../data/generated_tables/`.

## Configs

`config/defaults.yml` is loaded automatically. The paper configs override it: `config/paper/`
holds the main-text results, `config/ablation/` the ablations and sensitivity analyses.

| Config | Purpose |
|---|---|
| `config/datasets.yml` | datasets: CELLxGENE ids and columns, synthetic generator parameters |
| `config/paper/main_benchmark.yml` | main benchmark; states the full protocol in its header |
| `config/paper/pilot_gmvae.yml` | overlay: GMMs from PILOT-GM-VAE instead of EM (Table 2) |
| `config/ablation/approx_error_vs_bound.yml` | regularization ablation (Figure 5) |
| `config/ablation/full_hyperparameter_sweep.yml` | hyperparameter sweep on one dataset |
| `config/ablation/gmm_sensitivity.yml` | base config of the GMM-sensitivity sweeps in `scripts/gmm_sensitivity/` |
| `config/ablation/_raw_cells.yml` | overlay: point-cloud baselines on raw cells |
| `config/ablation/_component_stability.yml` | overlay: per-patient number-of-components check |
| `config/ablation/myocard_dataset.yml` | dataset file of the single-dataset runs |
| `config/matplotlibrc` | plot settings of all scripts and notebooks (TrueType fonts in PDFs) |

An overlay is passed after the main config to the same `--configfile` flag, e.g.
`--configfile config/paper/main_benchmark.yml config/paper/pilot_gmvae.yml`. Later files win per key.
In `config/ablation/`, a leading underscore marks an overlay; the other configs run on their own.

## Targets

| Target | Builds |
|---|---|
| `generate_paper_results` | all outputs of the config and their export to `../data/generated_*` |
| `all` | full pipeline including per-dataset tables and plots |
| `benchmark_only`, `tune_only`, `gmm_only`, `preprocess_only` | the pipeline up to that stage |
| `scrna_baselines_table_only` | Table 2 (after the PILOT-GM-VAE overlay) |
| `paired_tests_only` | paired Wilcoxon and mixed-model tests with Holm correction |
| `distance_agreement_only` | agreement of exact and diagonal patient distances |
| `runtime_table_only` | runtime table from the jobs' timing files |
| `patient_component_stability_only` | number-of-components check (overlay `ablation/_component_stability.yml`) |

## Rules

| File | Content |
|---|---|
| `download.smk`, `synthetic.smk`, `preprocess.smk` | dataset download, synthetic data, preprocessing |
| `gmm.smk`, `gmvae.smk` | per-patient GMMs by EM or by PILOT-GM-VAE |
| `tune.smk`, `benchmark.smk` | hyperparameter tuning, training and evaluation |
| `plot.smk` | per-dataset and aggregate tables and plots |
| `paired_tests.smk`, `distance_agreement.smk`, `runtime_table.smk` | appendix analyses of the main benchmark |
| `scrna_baselines_table.smk`, `raw_cells_table.smk`, `component_stability.smk` | tables of the overlays |
| `approx_error_bound.smk` | regularization ablation (Figure 5) |
| `export.smk` | export of the paper's outputs |
| `common.smk` | path helpers shared by the rules |

## Outputs and caching

Intermediate outputs are written to `../data/snakemake/`:

```
<dataset>/raw.h5ad, prepared/, gmm/              data, preprocessing, GMM fits
<dataset>/tune/<method>/<hash>/                  tuning results
<dataset>/benchmark/<method>/<hash>/             scores per split, trained metric
<dataset>/timings/                               wall time and memory of each job
plots/aggregate/<label>/                         tables and figures across datasets
```

Each folder name contains a hash of the settings that affect its content, so runs with
different settings get separate folders and reuse everything they share (for example the GMM
fits of the main benchmark). The order of keys in a config does not matter.
