"""Step 10: functional enrichment of the DR differentially expressed genes (g:Profiler).

Tests GO Biological Process, KEGG and Reactome terms for over-representation among
the DEGs from step 9, against a CUSTOM background: the genes actually tested by
DESeq2 (expressed in retina). A whole-genome background would make any
retina-expressed gene look "enriched".

Requires internet access (g:Profiler web service) and `pip install gprofiler-official`.

Examples
--------
python scripts/10_enrichment.py                          # DEGs at padj < 0.05, |log2FC| >= 0.58
python scripts/10_enrichment.py --gene-set padj          # all genes at padj < 0.05 (95)
python scripts/10_enrichment.py --direction up           # up-regulated DEGs only

Outputs (molecular/enrichment/):
  enrichment_<set>.tsv        all significant terms: source, term, p (BH-adjusted), sizes, genes
  enrichment_<set>.png        top terms, bar length = -log10 adjusted p
  enrichment_<set>_info.json  query/background sizes, sources, g:Profiler version info
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dendritic_dr.utils import REPO_ROOT

GEO_DIR = REPO_ROOT / "molecular" / "geo2r"
OUT_DIR = REPO_ROOT / "molecular" / "enrichment"
SOURCE_COLORS = {"GO:BP": "#0072B2", "KEGG": "#D55E00", "REAC": "#009E73"}


def load_sets(args):
    res = pd.read_csv(GEO_DIR / "de_results.tsv", sep="\t")
    background = res["gene_id_stripped"].dropna().astype(str).unique().tolist()
    sel = res[res["padj"] < args.alpha]
    if args.gene_set == "deg":
        sel = sel[sel["log2FoldChange"].abs() >= args.lfc]
    if args.direction == "up":
        sel = sel[sel["log2FoldChange"] > 0]
    elif args.direction == "down":
        sel = sel[sel["log2FoldChange"] < 0]
    query = sel["gene_id_stripped"].astype(str).tolist()
    symbols = {}
    sym_file = GEO_DIR / "gene_symbols.tsv"
    if sym_file.exists():
        symbols = pd.read_csv(sym_file, sep="\t", index_col=0)["symbol"].dropna().to_dict()
    return query, background, symbols


def run_gprofiler(query, background, sources, alpha):
    from gprofiler import GProfiler

    gp = GProfiler(return_dataframe=True, user_agent="dendritic-dr-course-project")
    df = gp.profile(organism="hsapiens", query=query, sources=sources, user_threshold=alpha,
                    significance_threshold_method="fdr", domain_scope="custom", background=background,
                    no_evidences=False)
    return df


def plot_top(df, title, out, n=15):
    top = df.nsmallest(n, "p_value").iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(top) + 1.4))
    y = np.arange(len(top))
    ax.barh(y, -np.log10(top["p_value"]), color=[SOURCE_COLORS.get(s, "#777777") for s in top["source"]])
    labels = [f"{n_[:55]}{'…' if len(n_) > 55 else ''} ({k}/{t})"
              for n_, k, t in zip(top["name"], top["intersection_size"], top["term_size"])]
    ax.set_yticks(y, labels, fontsize=7)
    ax.axvline(-np.log10(0.05), ls="--", lw=0.8, color="black")
    ax.set_xlabel("-log10 adjusted p-value (BH)")
    ax.set_title(title, fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for s, c in SOURCE_COLORS.items() if s in set(top["source"])]
    ax.legend(handles, [s for s in SOURCE_COLORS if s in set(top["source"])], fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gene-set", choices=["deg", "padj"], default="deg",
                   help="deg: padj < alpha and |log2FC| >= lfc; padj: padj < alpha only")
    p.add_argument("--direction", choices=["both", "up", "down"], default="both")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--lfc", type=float, default=0.58)
    p.add_argument("--sources", nargs="+", default=["GO:BP", "KEGG", "REAC"])
    args = p.parse_args()

    query, background, symbols = load_sets(args)
    tag = f"{args.gene_set}_{args.direction}"
    print(f"query: {len(query)} genes ({tag}); background: {len(background):,} tested genes")
    if len(query) < 5:
        raise SystemExit("fewer than 5 query genes; nothing meaningful to test")

    df = run_gprofiler(query, background, args.sources, args.alpha)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    info = {"date": str(date.today()), "tool": "g:Profiler (gprofiler-official)", "organism": "hsapiens",
            "sources": args.sources, "correction": "Benjamini-Hochberg FDR", "alpha": args.alpha,
            "domain_scope": "custom (genes tested by DESeq2)", "query_size": len(query),
            "background_size": len(background), "gene_set": args.gene_set, "direction": args.direction,
            "n_terms": int(len(df))}
    (OUT_DIR / f"enrichment_{tag}_info.json").write_text(json.dumps(info, indent=2))
    if df is None or df.empty:
        print("no terms significant at FDR < %.2f against the custom background" % args.alpha)
        print("(a valid result: the DEGs do not concentrate in any annotated pathway)")
        return

    if "intersections" in df:
        df["genes"] = df["intersections"].apply(
            lambda ids: ", ".join(sorted(symbols.get(i, i) for i in ids)) if isinstance(ids, (list, tuple)) else "")
    cols = [c for c in ["source", "native", "name", "p_value", "term_size", "query_size", "intersection_size",
                        "precision", "recall", "genes"] if c in df]
    df = df.sort_values("p_value")[cols]
    df.to_csv(OUT_DIR / f"enrichment_{tag}.tsv", sep="\t", index=False)
    with pd.option_context("display.width", 160, "display.max_colwidth", 60):
        print(df.head(15)[["source", "name", "p_value", "intersection_size", "term_size"]].to_string(index=False))
    plot_top(df, f"Enrichment of DR DEGs ({tag}, n={len(query)}) vs. tested-gene background",
             OUT_DIR / f"enrichment_{tag}.png")
    print(f"wrote {OUT_DIR / f'enrichment_{tag}.tsv'}")


if __name__ == "__main__":
    main()
