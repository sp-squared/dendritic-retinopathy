"""Step 9: differential expression in human DR retina (GSE160306) with PyDESeq2.

Input (download once; see molecular/README.md):
  molecular/geo2r/raw/GSE160306_series_matrix.txt.gz          sample metadata
  molecular/geo2r/raw/GSE160306_human_retina_DR_totalRNA_counts.txt.gz   raw counts (authors')

The series matrix stores per-sample characteristics as "key: value" cells whose
rows are NOT aligned across samples (some samples have fewer fields), so they are
parsed per sample by key, never by row position.

Design: macula samples only (each donor gave a macula and a periphery sample, so
one region keeps one sample per donor), cases vs. controls chosen by a metadata
column, adjusting for covariates (default: age + sex). Wald test, Benjamini-Hochberg
adjusted p-values.

Examples
--------
# 1) look at the metadata groups and the counts file first
python scripts/09_differential_expression.py --inspect

# 2) run the comparison (group labels as printed by --inspect)
python scripts/09_differential_expression.py --group-col disease_group_detailed --case PDR --control Control

Outputs (molecular/geo2r/):
  samples.csv              parsed metadata for all samples
  de_results.tsv           all tested genes: baseMean, log2FC, p, padj (sorted by padj)
  de_significant.tsv       padj < --alpha and |log2FC| >= --lfc
  string_input.txt         significant gene IDs (version stripped), for STRING
  volcano.png              volcano plot
  run_info.json            parameters, sample counts, package versions
"""
from __future__ import annotations

import os

# PyDESeq2 fits genes in parallel worker processes, which ignore warnings filters set
# in this process; the environment variable reaches them. The warnings are harmless
# overflow messages during dispersion fitting.
os.environ.setdefault("PYTHONWARNINGS", "ignore::RuntimeWarning")

import argparse
import csv
import warnings
import gzip
import json
import platform
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dendritic_dr.utils import REPO_ROOT

GEO_DIR = REPO_ROOT / "molecular" / "geo2r"
RAW_DIR = GEO_DIR / "raw"


# --- metadata ----------------------------------------------------------------

def parse_series_matrix(path: Path) -> pd.DataFrame:
    """One row per sample. Characteristics are parsed by key per sample column."""
    opener = gzip.open if str(path).endswith(".gz") else open
    simple, chars = {}, []
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        for row in csv.reader(f, delimiter="\t", quotechar='"'):
            if not row or not row[0].startswith("!Sample_"):
                continue
            key, vals = row[0][len("!Sample_"):], row[1:]
            if key.startswith("characteristics"):
                chars.append(vals)
            elif key in ("title", "geo_accession", "source_name_ch1"):
                simple[key] = vals
    n = len(simple["title"])
    records = []
    for j in range(n):
        rec = {"title": simple["title"][j], "gsm": simple.get("geo_accession", [""] * n)[j],
               "source": simple.get("source_name_ch1", [""] * n)[j]}
        for vals in chars:
            if j < len(vals) and ":" in vals[j]:
                k, v = vals[j].split(":", 1)
                k = k.strip().lower().replace(" ", "_")
                if k not in rec:  # first occurrence wins
                    rec[k] = v.strip()
        records.append(rec)
    df = pd.DataFrame(records)
    for col in ("age", "dr_severity_score", "rin", "post_mortem_interval"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def site_column(meta: pd.DataFrame) -> str:
    for c in ("sample_site", "tissue", "source"):
        if c in meta:
            return c
    raise SystemExit("no sample-site column found in metadata")


# --- counts --------------------------------------------------------------------

SYMBOL_COLS = ("gene_name", "symbol", "gene_symbol", "genename", "name")


def detect_sep(path: Path) -> str:
    """Tab or comma, from the header line (the GSE160306 counts file is comma-separated)."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        header = f.readline()
    return "\t" if header.count("\t") >= header.count(",") else ","


def load_counts(path: Path, meta: pd.DataFrame):
    """Genes x samples integer matrix (columns renamed to sample titles) and a
    gene-id -> symbol Series when the file carries a symbol column."""
    df = pd.read_csv(path, sep=detect_sep(path), index_col=0, compression="infer")
    numeric = df.select_dtypes(include=[np.number])
    dropped = [c for c in df.columns if c not in numeric.columns]
    symbols = None
    for c in dropped:
        if str(c).lower() in SYMBOL_COLS:
            symbols = df[c].astype(str)
    if dropped:
        print(f"note: non-numeric columns set aside: {dropped[:5]}" + (" (used as gene symbols)" if symbols is not None else ""))
    titles, gsms = set(meta["title"]), dict(zip(meta["gsm"], meta["title"]))

    def norm(c):
        c = str(c).strip()
        if c in titles:
            return c
        if c in gsms:
            return gsms[c]
        m = re.search(r"(\d+)$", c)  # e.g. 'S1' / 'sample1' -> 'sample_01'
        if m:
            cand = f"sample_{int(m.group(1)):02d}"
            if cand in titles:
                return cand
        return c

    numeric.columns = [norm(c) for c in numeric.columns]
    unmatched = [c for c in numeric.columns if c not in titles]
    if unmatched:
        print(f"warning: {len(unmatched)} count columns not matched to metadata, e.g. {unmatched[:5]}")
    return numeric.round().astype(int), symbols


# --- gene symbols ---------------------------------------------------------------

def annotate_symbols(gene_ids, cache: Path) -> pd.Series:
    """Ensembl gene id -> HGNC symbol via MyGene.info, cached to a TSV so the lookup
    runs once and later runs are offline and reproducible. Missing -> NaN."""
    ids = pd.Index(gene_ids).astype(str).str.replace(r"\.\d+$", "", regex=True)
    known = pd.Series(dtype=str)
    if cache.exists():
        known = pd.read_csv(cache, sep="\t", index_col=0)["symbol"]
    todo = [i for i in ids.unique() if i not in known.index]
    if todo:
        try:
            import mygene

            hits = mygene.MyGeneInfo().querymany(todo, scopes="ensembl.gene", fields="symbol",
                                                 species="human", verbose=False, as_dataframe=False)
            new = {h["query"]: h.get("symbol") for h in hits if not h.get("notfound")}
            known = pd.concat([known, pd.Series(new, dtype=object)])
            known = known[~known.index.duplicated()]
            known.rename("symbol").rename_axis("gene_id").to_csv(cache, sep="\t")
            print(f"annotated {len(new):,} of {len(todo):,} new gene ids (cached in {cache.name})")
        except Exception as e:  # no network or mygene not installed
            print(f"note: symbol lookup skipped ({type(e).__name__}: {e}); using Ensembl ids")
    return pd.Series(ids.map(known).values, index=pd.Index(gene_ids))


# --- DE ---------------------------------------------------------------------------

def run_deseq(counts: pd.DataFrame, meta: pd.DataFrame, covariates: list[str]):
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    design = "~" + " + ".join(covariates + ["condition"])
    dds = DeseqDataSet(counts=counts, metadata=meta, design=design, refit_cooks=True, quiet=True)
    dds.deseq2()
    ds = DeseqStats(dds, contrast=["condition", "case", "control"], quiet=True)
    ds.summary()
    return ds.results_df, design


def volcano(res: pd.DataFrame, alpha: float, lfc: float, title: str, out: Path):
    r = res.dropna(subset=["padj"])
    y = -np.log10(r["padj"].clip(lower=1e-300))
    sig = (r["padj"] < alpha) & (r["log2FoldChange"].abs() >= lfc)
    up = sig & (r["log2FoldChange"] > 0)
    down = sig & (r["log2FoldChange"] < 0)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(r.loc[~sig, "log2FoldChange"], y[~sig], s=4, color="#bbbbbb", alpha=0.6, label="not significant")
    ax.scatter(r.loc[up, "log2FoldChange"], y[up], s=6, color="#D55E00", label=f"up in cases (n={up.sum()})")
    ax.scatter(r.loc[down, "log2FoldChange"], y[down], s=6, color="#0072B2", label=f"down in cases (n={down.sum()})")
    ax.axhline(-np.log10(alpha), ls="--", lw=0.8, color="black")
    for x in (-lfc, lfc):
        ax.axvline(x, ls="--", lw=0.8, color="black")
    top = r[sig].nsmallest(8, "padj")
    for k, (gene, row) in enumerate(top.iterrows()):
        label = row["symbol"] if isinstance(row.get("symbol"), str) else str(gene)
        ax.annotate(label, (row["log2FoldChange"], -np.log10(max(row["padj"], 1e-300))), fontsize=6,
                    xytext=(4, 6 if k % 2 == 0 else -8), textcoords="offset points",
                    arrowprops=dict(arrowstyle="-", lw=0.4, color="#555555"))
    ax.set_xlabel("log2 fold change (cases vs. controls)")
    ax.set_ylabel("-log10 adjusted p-value")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix", default=str(RAW_DIR / "GSE160306_series_matrix.txt.gz"))
    p.add_argument("--counts", default=str(RAW_DIR / "GSE160306_human_retina_DR_totalRNA_counts.txt.gz"))
    p.add_argument("--inspect", action="store_true", help="print metadata groups and counts layout, then stop")
    p.add_argument("--site", default="Macula", help="sample site to analyse (Macula or Periphery)")
    p.add_argument("--group-col", default="disease_group_detailed")
    p.add_argument("--case", nargs="+", default=["PDR"], help="value(s) of --group-col counted as cases")
    p.add_argument("--control", nargs="+", default=["Control"], help="value(s) counted as controls")
    p.add_argument("--covariates", nargs="*", default=["age", "gender"])
    p.add_argument("--min-count", type=int, default=10, help="keep genes with >= this count in >= smallest-group samples")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--lfc", type=float, default=1.0, help="|log2FC| threshold for de_significant / STRING")
    p.add_argument("--out-dir", default=str(GEO_DIR))
    p.add_argument("--no-symbols", action="store_true", help="skip the MyGene.info symbol lookup")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = parse_series_matrix(Path(args.matrix))
    meta.to_csv(out / "samples.csv", index=False)
    site = site_column(meta)

    if args.inspect:
        print(f"{len(meta)} samples; metadata columns:\n  {', '.join(meta.columns)}\n")
        for c in [site, "disease_group", "disease_group_detailed", "disease_group_dme", "dme", "dr_severity_score"]:
            if c in meta:
                print(meta[c].value_counts(dropna=False).to_string(), "\n")
        if "disease_group_detailed" in meta:
            print(pd.crosstab(meta[site], meta["disease_group_detailed"]).to_string(), "\n")
        counts = pd.read_csv(args.counts, sep=detect_sep(Path(args.counts)), index_col=0, nrows=5, compression="infer")
        print(f"counts file: first columns {list(counts.columns[:6])} ...; {counts.shape[1]} columns")
        print(counts.iloc[:5, :4].to_string())
        titles = set(meta["title"])
        hits = sum(c in titles for c in counts.columns)
        print(f"{hits} of {counts.shape[1]} count columns match sample titles exactly "
              "(other names are matched by GSM id or trailing number when the analysis runs)")
        print(f"\nwrote {out / 'samples.csv'}")
        return

    sel = meta[meta[site].str.contains(args.site, case=False, na=False)].copy()
    sel["condition"] = np.where(sel[args.group_col].isin(args.case), "case",
                                np.where(sel[args.group_col].isin(args.control), "control", None))
    sel = sel.dropna(subset=["condition"])
    for cov in args.covariates:
        if sel[cov].isna().any():
            raise SystemExit(f"covariate {cov!r} has missing values in the selected samples")
    dropped_dupes = []
    if "donor" in sel and sel["donor"].duplicated().any():
        # Some donors gave two samples from the same site (e.g. both eyes). Keep one per
        # donor so samples stay independent: the higher-RIN sample, ties -> first listed.
        dup = sel[sel["donor"].duplicated(keep=False)]
        cols = [c for c in ["title", "donor", "eye", args.group_col, "rin"] if c in dup]
        print("donors with more than one sample at this site:")
        print(dup.sort_values("donor")[cols].to_string(index=False))
        rin = sel["rin"] if "rin" in sel else pd.Series(0, index=sel.index)
        keep_idx = sel.assign(_rin=rin.fillna(-1)).sort_values("_rin", ascending=False).drop_duplicates("donor").index
        dropped_dupes = sel.loc[~sel.index.isin(keep_idx), "title"].tolist()
        sel = sel.loc[sel.index.isin(keep_idx)]
        print(f"kept the higher-RIN sample per donor; dropped: {', '.join(dropped_dupes)}\n")
    n_case, n_ctrl = (sel["condition"] == "case").sum(), (sel["condition"] == "control").sum()
    print(f"{args.site}: {n_case} cases ({', '.join(args.case)}) vs. {n_ctrl} controls ({', '.join(args.control)})")
    if min(n_case, n_ctrl) < 3:
        raise SystemExit("need at least 3 samples per group; check --group-col/--case/--control with --inspect")

    counts, symbols = load_counts(Path(args.counts), meta)
    missing = [t for t in sel["title"] if t not in counts.columns]
    if missing:
        raise SystemExit(f"samples missing from counts file: {missing[:5]}")
    X = counts[sel["title"]].T  # samples x genes
    keep = (X >= args.min_count).sum(axis=0) >= min(n_case, n_ctrl)
    X = X.loc[:, keep]
    print(f"genes kept after filtering: {X.shape[1]:,} of {counts.shape[0]:,}")

    md = sel.set_index("title")[["condition"] + args.covariates].copy()
    for cov in args.covariates:
        if md[cov].dtype == object:
            md[cov] = md[cov].str.lower()
    with warnings.catch_warnings():  # PyDESeq2 emits harmless overflow warnings during fitting
        warnings.simplefilter("ignore", RuntimeWarning)
        res, design = run_deseq(X, md, args.covariates)
    res = res.sort_values("padj")
    res.insert(0, "gene_id", res.index)
    res["gene_id_stripped"] = res["gene_id"].astype(str).str.replace(r"\.\d+$", "", regex=True)
    if symbols is not None:
        res.insert(1, "symbol", res["gene_id"].map(symbols))
    elif not args.no_symbols:
        sig_ids = res.loc[res["padj"] < args.alpha, "gene_id"]
        sym = annotate_symbols(sig_ids, out / "gene_symbols.tsv")
        res.insert(1, "symbol", res["gene_id"].map(sym))
    res.to_csv(out / "de_results.tsv", sep="\t", index=False)
    sig = res[(res["padj"] < args.alpha) & (res["log2FoldChange"].abs() >= args.lfc)]
    sig.to_csv(out / "de_significant.tsv", sep="\t", index=False)
    # STRING maps Ensembl ids itself; ids are unambiguous, symbols are for people
    (out / "string_input.txt").write_text("\n".join(sig["gene_id_stripped"].astype(str)) + "\n")
    n_p_only = int((res["padj"] < args.alpha).sum())
    print(f"genes with padj < {args.alpha} (any fold change): {n_p_only}")
    n_up, n_down = (sig["log2FoldChange"] > 0).sum(), (sig["log2FoldChange"] < 0).sum()
    print(f"significant (padj < {args.alpha}, |log2FC| >= {args.lfc}): {len(sig)} ({n_up} up, {n_down} down)")
    cols = ["gene_id"] + (["symbol"] if "symbol" in res else []) + ["baseMean", "log2FoldChange", "pvalue", "padj"]
    print(res.head(15)[cols].round(4).to_string(index=False))

    volcano(res.set_index("gene_id"), args.alpha, args.lfc,
            f"GSE160306 {args.site}: cases (n={n_case}: {', '.join(args.case)}) vs. controls (n={n_ctrl})",
            out / "volcano.png")

    import pydeseq2
    info = {"dataset": "GSE160306", "site": args.site, "group_col": args.group_col, "case": args.case,
            "control": args.control, "n_case": int(n_case), "n_control": int(n_ctrl), "design": design,
            "min_count": args.min_count, "genes_tested": int(X.shape[1]), "alpha": args.alpha, "lfc": args.lfc,
            "dropped_duplicate_donor_samples": dropped_dupes,
            "n_padj_only": n_p_only,
            "n_significant": int(len(sig)), "n_up": int(n_up), "n_down": int(n_down),
            "samples": sel[["title", "gsm", "donor", "condition"] + args.covariates].to_dict(orient="records")
            if "donor" in sel else sel["title"].tolist(),
            "pydeseq2": pydeseq2.__version__, "python": platform.python_version()}
    (out / "run_info.json").write_text(json.dumps(info, indent=2, default=str))
    print(f"wrote {out}/de_results.tsv, de_significant.tsv, string_input.txt, run_info.json")


if __name__ == "__main__":
    main()
