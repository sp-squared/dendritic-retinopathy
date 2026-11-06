"""Step 23 (extension, study 2): a pathway-wired dANN for DR from retinal gene expression.

The image experiments suggested that fixed dendritic wiring is a prior that helps only when data
are scarce. GSE160306 is far scarcer than any image task (about 30 donors per comparison), so
this asks whether a biologically meaningful prior helps there.

Models (same splits, same preprocessing):
  pathway-dann   genes -> one dendrite per Reactome pathway (only its genes) -> one soma per
                 top-level Reactome category -> logit
  random-dann    the same network, each dendrite wired to as many random genes (new draw per
                 repeat): identical parameters, no biology. Pathway vs. random isolates the prior.
  dense-mlp      a dense MLP with about the same number of parameters (second layer as wide as
                 the soma layer)
  logreg         L2 logistic regression on all genes (C chosen by inner donor-grouped CV)
  pathway-score  L2 logistic regression on pathway scores (mean z-score of each pathway's
                 genes): the linear use of the same prior, without dendritic nonlinearity

Data: macula and periphery samples (or one site) of GSE160306; cases vs. controls by
--group-col (default: any DR vs. non-diabetic controls). Evaluation: stratified 5-fold
cross-validation grouped by donor (a donor's samples never sit in both training and test),
repeated 10 times with different splits. Inside each fold: z-scores from training samples only;
the gene universe (CPM >= 1 in >= 20% of samples) and the Reactome masks use no labels.

Prediction (stated before running): pathway-dann beats random-dann (AUC) and is at least as good
as the parameter-matched dense MLP; its most influential dendrites are immune, extracellular-
matrix and TGF-beta/BMP pathways, the processes enriched among the course project's DEGs.

python scripts/23_pathway_dann.py --download-reactome          # once: fetch the two Reactome files
python scripts/23_pathway_dann.py                              # any DR vs. control
python scripts/23_pathway_dann.py --control Diabetic --name dr_vs_diabetic
python scripts/23_pathway_dann.py --smoke

Outputs (results/pathway/<name>/): oof.csv (out-of-fold probability per sample, model and
repeat), contributions.csv (mean dendrite contribution per pathway), masks.json (pathways,
categories, sizes), run_info.json
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
import warnings
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from dendritic_dr.pathway import (ENRICHR_URL, REACTOME_GMT_URL, REACTOME_REL_URL, DenseMLP, PathwayDANN, build_masks,
                                  dendrite_contributions, expressed, log_cpm, matched_dense, overlap_groups, predict, random_like,
                                  read_gmt, read_relations, smoke_expression, train_torch)
from dendritic_dr.utils import REPO_ROOT, results_root

GEO = REPO_ROOT / "molecular" / "geo2r"
REACTOME = REPO_ROOT / "molecular" / "reactome"
MODELS = ["pathway-dann", "random-dann", "dense-mlp", "logreg", "pathway-score"]


def script09():
    spec = spec_from_file_location("de", REPO_ROOT / "scripts" / "09_differential_expression.py")
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def download_reactome():
    REACTOME.mkdir(parents=True, exist_ok=True)
    for url in (REACTOME_GMT_URL, REACTOME_REL_URL):
        dest = REACTOME / url.rsplit("/", 1)[1]
        print(f"downloading {url} -> {dest}")
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as e:
            dest.unlink(missing_ok=True)
            raise SystemExit(f"download failed ({e}). Download the two files in a browser from "
                             f"https://reactome.org/download-data (ReactomePathways.gmt.zip and "
                             f"ReactomePathwaysRelation.txt) and save them in {REACTOME}/, or use the "
                             f"Enrichr copy: --download-enrichr")


def download_enrichr(library: str):
    """Reactome gene sets from Enrichr's server (no hierarchy file there: somas then come from
    gene-overlap groups, see --soma-groups)."""
    REACTOME.mkdir(parents=True, exist_ok=True)
    url, dest = ENRICHR_URL.format(library), REACTOME / f"{library}.gmt"
    print(f"downloading {url} -> {dest}")
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise SystemExit(f"download failed ({e}). Open {url} in a browser and save the page as {dest}")
    n = sum(1 for line in dest.read_text().splitlines() if line.strip())
    print(f"{n:,} gene sets")


def find_gene_sets(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    for pattern in ("ReactomePathways.gmt*", "Reactome*.gmt", "Reactome*.txt"):
        hits = sorted(p for p in REACTOME.glob(pattern) if "Relation" not in p.name)
        if hits:
            return hits[-1]
    raise SystemExit(f"no Reactome gene sets in {REACTOME}/: run --download-reactome or --download-enrichr "
                     "(see molecular/README.md)")


def load_real(args):
    de = script09()
    meta = pd.read_csv(GEO / "samples.csv") if (GEO / "samples.csv").exists() else \
        de.parse_series_matrix(Path(args.matrix))
    counts, symbols = de.load_counts(Path(args.counts), meta)
    site = de.site_column(meta)
    sel = meta.copy()
    if args.site != "both":
        sel = sel[sel[site].str.contains(args.site, case=False, na=False)]
    grp = sel[args.group_col].astype(str)
    sel = sel[grp.isin(args.case + args.control)].copy()
    sel["y"] = sel[args.group_col].astype(str).isin(args.case).astype(int)
    sel = sel[sel["title"].isin(counts.columns)]
    counts = counts[sel["title"]]
    # Ensembl id -> symbol: the counts file's own column, else the course project's DE table
    ids = counts.index.astype(str).str.replace(r"\.\d+$", "", regex=True)
    if symbols is not None:
        sym = pd.Series(symbols.values, index=ids)
    else:
        table = pd.read_csv(GEO / "de_results.tsv", sep="\t", usecols=["gene_id", "symbol"])
        table["gene_id"] = table["gene_id"].astype(str).str.replace(r"\.\d+$", "", regex=True)
        sym = table.dropna().drop_duplicates("gene_id").set_index("gene_id")["symbol"]
    counts.index = ids
    counts = counts[counts.index.isin(sym.index)]
    counts.index = sym.reindex(counts.index).values
    counts = counts.groupby(level=0).sum()  # several ids for one symbol -> summed
    src = find_gene_sets(args.gmt)
    gmt = read_gmt(src)
    rel = Path(args.relations) if args.relations and args.relations != "none" else REACTOME / "ReactomePathwaysRelation.txt"
    parents = read_relations(rel) if rel.exists() and args.relations != "none" else None
    print(f"gene sets: {src.name} ({len(gmt):,}); somas: " +
          (f"Reactome top-level categories ({rel.name})" if parents is not None else
           f"{args.soma_groups} gene-overlap groups (no hierarchy file)"))
    return counts, sel[["title", "donor", "y"]].reset_index(drop=True), gmt, parents


def fold_logreg(Xtr, ytr, gtr, seed):
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    inner = list(GroupKFold(n_splits=min(3, len(np.unique(gtr)))).split(Xtr, ytr, gtr))
    m = LogisticRegressionCV(Cs=np.logspace(-4, 2, 13), cv=inner, scoring="roc_auc", max_iter=5000,
                             class_weight="balanced", random_state=seed)
    return m.fit(Xtr, ytr)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="dr_vs_control")
    p.add_argument("--group-col", default="disease_group_detailed")
    p.add_argument("--case", nargs="+", default=["NPDR", "NPDR + DME", "PDR + DME"])
    p.add_argument("--control", nargs="+", default=["Control"])
    p.add_argument("--site", default="both", help="both, Macula or Periphery")
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--min-size", type=int, default=10, help="smallest pathway (genes in the universe)")
    p.add_argument("--max-size", type=int, default=300, help="largest pathway")
    p.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    p.add_argument("--stable-top", type=int, default=25, help="top-k used for the attribution stability column")
    p.add_argument("--matrix", default=str(GEO / "raw" / "GSE160306_series_matrix.txt.gz"))
    p.add_argument("--counts", default=str(GEO / "raw" / "GSE160306_human_retina_DR_totalRNA_counts.txt.gz"))
    p.add_argument("--download-reactome", action="store_true", help="fetch the Reactome files, then stop")
    p.add_argument("--download-enrichr", action="store_true",
                   help="fetch Enrichr's copy of the Reactome gene sets (if reactome.org is unreachable), then stop")
    p.add_argument("--enrichr-library", default="Reactome_2022")
    p.add_argument("--gmt", default=None, help="gene-set file (default: the Reactome file in molecular/reactome/)")
    p.add_argument("--relations", default=None, help="Reactome hierarchy file (default: molecular/reactome/...); 'none' = use gene-overlap somas")
    p.add_argument("--soma-groups", type=int, default=25,
                   help="without the hierarchy file: number of gene-overlap groups used as somas")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.download_reactome:
        download_reactome()
        return
    if args.download_enrichr:
        download_enrichr(args.enrichr_library)
        return
    if args.smoke:
        counts, meta, gmt, parents = smoke_expression()
        if args.relations == "none":
            parents = None
        args.repeats, args.epochs, args.name = min(args.repeats, 3), min(args.epochs, 150), "smoke"
    else:
        counts, meta, gmt, parents = load_real(args)

    genes = expressed(counts)
    universe = sorted(set(genes) & set().union(*(m for _, m in gmt.values())))
    masks = build_masks(universe, gmt, parents or {}, args.min_size, args.max_size)
    universe = [g for g, used in zip(universe, masks["M1"].any(0)) if used]  # genes on some dendrite
    masks = build_masks(universe, gmt, parents or {}, args.min_size, args.max_size)
    soma_source = "reactome-hierarchy"
    if parents is None:  # no hierarchy: somas = gene-overlap groups of the kept pathways (no labels used)
        parents, gnames = overlap_groups(masks["pathways"], masks["pathway_names"], masks["M1"], args.soma_groups)
        masks = build_masks(universe, gmt, parents, args.min_size, args.max_size, extra_names=gnames)
        soma_source = f"gene-overlap-{args.soma_groups}"
    X_all = log_cpm(counts).loc[universe, meta["title"]].T.to_numpy(np.float64)
    y, groups = meta["y"].to_numpy(), meta["donor"].astype(str).to_numpy()
    M1, M2 = masks["M1"], masks["M2"]
    n_params = PathwayDANN(M1, M2).effective_params()
    h1, h2 = matched_dense(len(universe), n_params, M2.shape[0])
    print(f"{len(meta)} samples from {len(np.unique(groups))} donors ({y.sum()} case, {len(y) - y.sum()} control); "
          f"{len(universe):,} genes, {M1.shape[0]:,} pathway dendrites, {M2.shape[0]} category somas; "
          f"dANN {n_params:,} parameters, dense MLP {h1} x {h2} hidden "
          f"({DenseMLP(len(universe), h1, h2).effective_params():,} parameters)")

    out = results_root(args.smoke) / "pathway" / args.name
    out.mkdir(parents=True, exist_ok=True)
    oof_rows, contrib = [], np.zeros((2, M1.shape[0]))  # [sum |c|, sum c(case) - c(control)]
    n_contrib = 0
    contrib_rep = np.zeros((args.repeats, M1.shape[0]))  # per repeat, for rank stability
    t0 = time.time()
    for rep in range(args.repeats):
        rng = np.random.default_rng(1000 + rep)
        M1_rand = random_like(M1, rng)
        cv = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=rep)
        for fold, (tr, te) in enumerate(cv.split(X_all, y, groups)):
            mu, sd = X_all[tr].mean(0), X_all[tr].std(0) + 1e-6
            Xtr, Xte = (X_all[tr] - mu) / sd, (X_all[te] - mu) / sd
            seed = rep * 100 + fold
            probs = {}
            if "pathway-dann" in args.models:
                torch.manual_seed(seed)  # initial weights depend only on the seed
                m = train_torch(PathwayDANN(M1, M2, args.dropout), Xtr, y[tr], args.epochs, args.lr, args.weight_decay, seed)
                probs["pathway-dann"] = predict(m, Xte)
                c = dendrite_contributions(m, Xte)
                contrib[0] += np.abs(c).sum(0)
                contrib_rep[rep] += np.abs(c).sum(0)
                if y[te].any() and (1 - y[te]).any():
                    contrib[1] += c[y[te] == 1].mean(0) - c[y[te] == 0].mean(0)
                n_contrib += 1
            if "random-dann" in args.models:
                torch.manual_seed(seed)
                m = train_torch(PathwayDANN(M1_rand, M2, args.dropout), Xtr, y[tr], args.epochs, args.lr,
                                args.weight_decay, seed)
                probs["random-dann"] = predict(m, Xte)
            if "dense-mlp" in args.models:
                torch.manual_seed(seed)
                m = train_torch(DenseMLP(len(universe), h1, h2, args.dropout), Xtr, y[tr], args.epochs, args.lr,
                                args.weight_decay, seed)
                probs["dense-mlp"] = predict(m, Xte)
            if "logreg" in args.models:
                probs["logreg"] = fold_logreg(Xtr, y[tr], groups[tr], seed).predict_proba(Xte)[:, 1]
            if "pathway-score" in args.models:
                W = M1 / M1.sum(1, keepdims=True)
                probs["pathway-score"] = fold_logreg(Xtr @ W.T, y[tr], groups[tr], seed).predict_proba(Xte @ W.T)[:, 1]
            for name, pr in probs.items():
                for i, pv in zip(te, pr):
                    oof_rows.append({"repeat": rep, "fold": fold, "model": name, "title": meta["title"][i],
                                     "donor": groups[i], "y": int(y[i]), "prob": float(pv)})
        print(f"  repeat {rep + 1}/{args.repeats} done ({time.time() - t0:.0f} s)", flush=True)

    pd.DataFrame(oof_rows).to_csv(out / "oof.csv", index=False)
    cat_of = [[masks["category_names"][c] for c in np.flatnonzero(M2[:, j])] for j in range(M1.shape[0])]
    pd.DataFrame({"pathway": masks["pathways"], "name": masks["pathway_names"], "size": M1.sum(1),
                  "categories": ["; ".join(c) for c in cat_of],
                  "mean_abs_contribution": contrib[0] / max(len(meta) * args.repeats, 1),
                  "case_minus_control": contrib[1] / max(n_contrib, 1),
                  # share of repeats in which the pathway is among the 25 most influential dendrites:
                  # one model's attributions are unstable, so only consistent pathways deserve weight
                  f"top{args.stable_top}_share": (np.argsort(np.argsort(-contrib_rep, 1), 1) < args.stable_top).mean(0)}) \
        .sort_values("mean_abs_contribution", ascending=False).to_csv(out / "contributions.csv", index=False)
    json.dump({"pathways": masks["pathways"], "categories": masks["categories"],
               "category_names": masks["category_names"], "sizes": M1.sum(1).tolist()}, open(out / "masks.json", "w"))
    json.dump({k: v for k, v in vars(args).items()} | {
        "n_samples": int(len(meta)), "n_donors": int(len(np.unique(groups))), "n_case": int(y.sum()),
        "n_genes": len(universe), "n_pathways": int(M1.shape[0]), "n_categories": int(M2.shape[0]),
        "dann_params": n_params, "dense_hidden": [h1, h2], "soma_source": soma_source}, open(out / "run_info.json", "w"), indent=2)
    print(f"wrote {out / 'oof.csv'}, {out / 'contributions.csv'}, {out / 'masks.json'}, {out / 'run_info.json'}")


if __name__ == "__main__":
    main()
