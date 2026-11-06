"""Paired Wilcoxon + LMM tests with Holm correction (appendix, tab:paired_tests_all).

The split pairing and gap computation come from scripts/significance.py.
"""

PAIRED_TESTS_CONFIG = {
    "references": ["GMM_FullBures", "GMM_DiagBures"],
    "baselines": None,  # None: every other method of Table 1 (main_table not false)
    "metrics": ["knn", "ari", "ari_complete", "silhouette"],
    "alpha": 0.05,
    **dict(config.get("paired_tests") or {}),
}
PAIRED_TESTS_REFERENCES = [m for m in PAIRED_TESTS_CONFIG["references"] if m in ALL_METHODS]
PAIRED_TESTS_BASELINES = PAIRED_TESTS_CONFIG["baselines"] or [
    m for m in ALL_METHODS
    if m not in PAIRED_TESTS_REFERENCES and config["methods"][m].get("main_table", True)
]
PAIRED_TESTS_ENABLED = bool(PAIRED_TESTS_REFERENCES) and bool(PAIRED_TESTS_BASELINES)


def _paired_tests_pairs(group: str) -> list[tuple[str, str]]:
    methods = PAIRED_TESTS_REFERENCES + [m for m in PAIRED_TESTS_BASELINES if m not in PAIRED_TESTS_REFERENCES]
    return [(method, dataset) for dataset in DATASET_GROUPS[group] for method in methods]


def paired_tests_path(suffix: str) -> str:
    return aggregate_output_path(f"paired_tests/{suffix}")


rule paired_tests:
    """Paired tests of our methods against each baseline for one dataset group."""
    wildcard_constraints:
        group="|".join(DATASET_GROUPS.keys())
    input:
        scores=lambda w: [bench_path(dataset, method, "scores_per_split.csv") for method, dataset in _paired_tests_pairs(w.group)],
    output:
        gaps=paired_tests_path("paired_gaps_{group}.csv"),
        tests=paired_tests_path("paired_tests_{group}.csv"),
        table=paired_tests_path("paired_tests_{group}.tex"),
    params:
        method_dataset_pairs=lambda w: _paired_tests_pairs(w.group),
        dataset_names=lambda w: DATASET_GROUPS[w.group],
        references=PAIRED_TESTS_REFERENCES,
        baselines=PAIRED_TESTS_BASELINES,
        metrics=PAIRED_TESTS_CONFIG["metrics"],
        alpha=PAIRED_TESTS_CONFIG["alpha"],
        display_names={m: config["methods"][m].get("display_name", m) for m in ALL_METHODS},
        group=lambda w: w.group,
    log:
        paired_tests_path("paired_tests_{group}.log")
    resources:
        mem_mb=lambda w: _get_resources("significance", "mem_mb", 8000),
    script:
        "../scripts/paired_tests.py"
