"""Compare average- and complete-linkage ARI from existing score summaries."""

import math
import sys
from pathlib import Path

sys.path.insert(0, snakemake.scriptdir)  # noqa: F821

from _benchmark_display import (  # noqa: E402
    _escape_latex,
    _format_value,
    get_display_name,
    get_sectioned_methods,
    load_score_summary,
)
from _utils import setup_logging  # noqa: E402


def run(snakemake):
    logger = setup_logging(snakemake.log[0])
    cfg = snakemake.params.config
    pairs = snakemake.params.method_dataset_pairs
    methods = list(dict.fromkeys(method for method, _ in pairs))
    datasets = list(dict.fromkeys(dataset for _, dataset in pairs))
    labels = snakemake.params.dataset_labels
    metrics = ("ari", "ari_complete")
    scores = {}
    for (method, dataset), path in zip(pairs, snakemake.input.scores):
        summary = load_score_summary(path, extra_metrics=["ari_complete"])
        for metric in metrics:
            mean, sd = (summary[(metric, stat)] for stat in ("mean", "SD"))
            if not (math.isfinite(mean) and math.isfinite(sd)):
                raise ValueError(f"Missing or nonfinite {metric} in {path}")
            scores[(method, dataset, metric)] = _format_value(mean, sd, rank=0, unit=None)

    n_splits = int(cfg["train_test"]["n_splits"])
    lines = [
        r"\begin{table}[ht!]",
        r"\centering",
        r"\caption{\textbf{Average- and complete-linkage clustering performance.} "
        f"ARI, mean $\\pm$ SD over {n_splits} train-test splits per dataset. "
        "Both linkage criteria are evaluated on the same trained metrics.}",
        r"\label{tab:complete_linkage}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.1}",
    ]
    for start in range(0, len(datasets), 3):
        panel = datasets[start:start + 3]
        if start:
            lines.append(r"\par\vspace{0.3cm}")
        headers = [
            r"\multicolumn{2}{c}{" + _escape_latex(labels.get(ds, ds.replace("_", " "))) + "}"
            for ds in panel
        ]
        lines += [
            r"\resizebox{\linewidth}{!}{%",
            r"\begin{tabular}{l" + "cc" * len(panel) + "}",
            r"\toprule",
            " & " + " & ".join(headers) + r" \\",
            " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(panel))),
            "Method & " + " & ".join(["Average & Complete"] * len(panel)) + r" \\",
            r"\midrule",
        ]
        for section, section_methods in get_sectioned_methods(cfg, methods).items():
            lines.append(f"\\multicolumn{{{1 + 2 * len(panel)}}}{{l}}{{\\textsc{{{section}}}}}" + r" \\")
            for method in section_methods:
                lines.append(get_display_name(cfg, method) + " & "
                             + " & ".join(scores[(method, ds, metric)] for ds in panel for metric in metrics)
                             + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}}"]
    lines.append(r"\end{table}")
    output = Path(snakemake.output.table)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    logger.info("Wrote %s (%d methods, %d datasets)", output, len(methods), len(datasets))


run(snakemake)  # noqa: F821
