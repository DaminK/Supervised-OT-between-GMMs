.PHONY: setup-env setup-gmvae-env benchmark ablation-approx-error ablation-hyperparameters ablations help


# --------- Environment Setup ---------
PYTHON ?= python3.11
VENV ?= .venv
GMVAE_VENV ?= .venv_gmvae


setup-env:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/python -m pip install -U pip -r requirements.txt
	@printf "\nEnvironment ready.\nOpening a subshell with %s activated.\nExit that shell when you are done.\n\n" "$(VENV)"
	@bash -lc '. $(VENV)/bin/activate && exec bash -i'

# Isolated environment for the PILOT-GM-VAE representation (config/paper/pilot_gmvae.yml).
setup-gmvae-env:
	$(PYTHON) -m venv $(GMVAE_VENV)
	$(GMVAE_VENV)/bin/python -m pip install -U pip -r requirements-gmvae.txt

# --------- Run Benchmark & Ablations ---------
JOBS ?= 8
MEM_MB ?= 512000
SMKFLAGS ?=

benchmark:
	cd snakemake && snakemake --configfile config/paper/main_benchmark.yml -j$(JOBS) --resources mem_mb=$(MEM_MB) $(SMKFLAGS) -- generate_paper_results

ablation-approx-error:
	cd snakemake && snakemake --configfile config/ablation/approx_error_vs_bound.yml -j$(JOBS) --resources mem_mb=$(MEM_MB) $(SMKFLAGS) -- generate_paper_results

ablation-hyperparameters:
	cd snakemake && snakemake --configfile config/ablation/full_hyperparameter_sweep.yml -j$(JOBS) --resources mem_mb=$(MEM_MB) $(SMKFLAGS) -- generate_paper_results

ablations: ablation-approx-error ablation-hyperparameters

# ---------- Help ----------

help:
	@printf "Available targets:\n\n"
	@printf "  make setup-env\n"
	@printf "    Create %s, install Python dependencies, and open an activated subshell.\n\n" "$(VENV)"
	@printf "  make setup-gmvae-env\n"
	@printf "    Create %s with PILOT-GM-VAE, used by config/paper/pilot_gmvae.yml.\n\n" "$(GMVAE_VENV)"
	@printf "  make benchmark\n"
	@printf "    Run the main benchmark workflow (Table 1, Figure 3, appendix benchmark outputs).\n\n"
	@printf "  make ablation-approx-error\n"
	@printf "    Run the regularization ablation workflow (Figure 5 and its appendix figures and tables).\n\n"
	@printf "  make ablation-hyperparameters\n"
	@printf "    Run the full hyperparameter sweep workflow (appendix hyperparameter sensitivity and training time vs. d).\n\n"
	@printf "  make ablations\n"
	@printf "    Run both ablation workflows.\n\n"
	@printf "Outputs are written to data/generated_plots and data/generated_tables.\n"
	@printf "Defaults: JOBS=%s, MEM_MB=%s (512 GB RAM limit).\n" "$(JOBS)" "$(MEM_MB)"
	@printf "A fresh full benchmark run is typically a many-hour / overnight CPU job; reruns with cached intermediates are much faster.\n"
