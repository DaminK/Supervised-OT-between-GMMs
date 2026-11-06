"""Helpers for the Figure 6 latent-axis enrichment dot plot."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from gprofiler import GProfiler

import ggml_ot

SOURCES = ("GO:BP", "GO:CC", "GO:MF", "REAC", "WP")
SHORT_NAMES = {"extracellular matrix structural constituent": "ECM structural constituent"}


def _end_label(axis: int, sign: str) -> str:
    return f"Axis {axis} {'positive' if sign == '+' else 'negative'}"


def enrich_axis_ends(
    adata,
    ends: list[tuple[int, str]],
    n_genes: int = 100,
    alpha: float = 0.05,
    max_gene_set_size: int = 1000,
    sources=SOURCES,
    gene_symbols: str = "feature_name",
) -> dict[str, pd.DataFrame]:
    """Run g:Profiler on the top-loading genes of each latent-axis end.

    ``ends`` lists ``(axis, sign)`` pairs with one-based axes and sign ``"+"`` (largest positive
    loadings) or ``"-"`` (largest negative loadings). Terms annotated to more than
    ``max_gene_set_size`` genes are removed after testing, so the remaining p-values are unchanged.
    """
    gp = GProfiler(return_dataframe=True)
    enrich = {}
    for axis, sign in ends:
        genes = ggml_ot.gene.top_ranked(adata, components=[axis], n_genes=n_genes, gene_symbols=gene_symbols,
                                        up_regulated=sign == "+", down_regulated=sign == "-")[0]
        r = gp.profile(query=list(genes), ordered=True, user_threshold=alpha, organism="hsapiens",
                       sources=list(sources), no_evidences=False)
        enrich[_end_label(axis, sign)] = r[r["term_size"] <= max_gene_set_size].reset_index(drop=True)
    return enrich


def select_terms(enrich: dict[str, pd.DataFrame], n_terms: int) -> dict[str, list[str]]:
    """Top ``n_terms`` terms per axis end, skipping terms already shown for an earlier end."""
    blocks = {}
    for end, r in enrich.items():
        shown = {t for b in blocks.values() for t in b}
        blocks[end] = [t for t in r["name"] if t not in shown][:n_terms]
    return blocks


def enrichment_dotplot(
    adata,
    ends: list[tuple[int, str]],
    n_genes: int = 100,
    n_terms: int = 4,
    alpha: float = 0.05,
    max_gene_set_size: int = 1000,
    sources=SOURCES,
    gene_symbols: str = "feature_name",
    figsize: tuple[float, float] = (3.0, 1.6),
    font_size: float = 6,
    vertical: bool = False,
    save: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Scanpy dot plot of the enriched terms of several latent-axis ends.

    Rows are axis ends, columns the selected terms grouped by the end they were selected for
    (``vertical=True`` swaps them: terms as rows, axis ends as columns).
    Color is -log10 adjusted p, dot size the fraction of query genes annotated to the term.
    Empty cells mark terms not significant at that end. Prints the shown terms with their genes
    and returns the full enrichment table per end.
    """
    enrich = enrich_axis_ends(adata, ends, n_genes, alpha, max_gene_set_size, sources, gene_symbols)
    blocks = select_terms(enrich, n_terms)
    terms = [t for b in blocks.values() for t in b]
    for end, b in blocks.items():
        r = enrich[end].set_index("name")
        print(end)
        for t in b:
            print(f"  {t}: p={r.loc[t, 'p_value']:.1e}, genes={', '.join(r.loc[t, 'intersections'])}")

    short_end = {_end_label(a, s): f"Axis {a} {'+' if s == '+' else '−'}" for a, s in ends}
    # vertical: short group names as column labels, no brackets (each column already names its end)
    groups = [short_end[e] if vertical else e for e in enrich]
    neg_log_p = pd.DataFrame({e: -np.log10(r.set_index("name")["p_value"]) for e, r in enrich.items()})
    gene_ratio = pd.DataFrame({e: r.set_index("name")["intersection_size"] / n_genes for e, r in enrich.items()})
    color_df = neg_log_p.reindex(terms)[list(enrich)].T.fillna(0).rename(columns=SHORT_NAMES)
    size_df = gene_ratio.reindex(terms)[list(enrich)].T.fillna(0).rename(columns=SHORT_NAMES)
    color_df.index, size_df.index = groups, groups

    # scanpy's DotPlot takes the precomputed tables; the AnnData only defines groups and variables
    dummy = AnnData(np.zeros((len(groups), len(terms)), dtype=np.float32),
                    obs=pd.DataFrame({"end": pd.Categorical(groups, categories=groups)}))
    dummy.var_names = list(color_df.columns)
    if vertical:
        var_names = list(color_df.columns)
    else:
        var_names = {short_end[e]: [SHORT_NAMES.get(t, t) for t in b] for e, b in blocks.items()}
    with plt.rc_context({"font.size": font_size}):
        dp = sc.pl.DotPlot(dummy, var_names=var_names, groupby="end", dot_color_df=color_df, dot_size_df=size_df,
                           var_group_rotation=0, vmin=0, vmax=float(np.ceil(color_df.values.max())), figsize=figsize)
        dp.style(cmap="Reds", dot_min=0, dot_max=0.25, smallest_dot=0)
        dp.legend(size_title="Query genes\nin term (%)", colorbar_title=r"$-\log_{10}$ adj. p", width=1.0)
        if vertical:
            dp.swap_axes()
        dp.make_figure()
        if vertical:
            dp.get_axes()["mainplot_ax"].tick_params(axis="x", labelrotation=0)
        if save is not None:
            dp.fig.savefig(save, bbox_inches="tight")
        else:
            plt.show()
    return enrich
