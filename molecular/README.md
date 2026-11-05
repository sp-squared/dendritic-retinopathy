# Molecular context analysis

Goal: give the image results a biological explanation by identifying the molecular processes
behind the lesions that define each DR grade. This is a *supporting* analysis. It does not feed
into the model, and the report should not claim the image model learns molecular pathways.

Differential expression and enrichment are scripted (`scripts/09`, `scripts/10`); the PPI network uses the
STRING web tool, so reproducibility comes from **saving every export in this folder and recording every
setting below**, like a lab notebook.

## Folder layout

```
molecular/
├── README.md                      this protocol
├── geo2r/                         written by scripts/09_differential_expression.py
│   ├── raw/                       GEO downloads (git-ignored, see step 1)
│   ├── samples.csv                sample metadata parsed from the series matrix (all samples)
│   ├── run_info.json              design, groups, thresholds, samples used, DEG counts, versions
│   ├── de_results.tsv             PyDESeq2 results for every tested gene
│   ├── de_significant.tsv         DEGs (padj < 0.05 and |log2FC| >= 0.58)
│   ├── gene_symbols.tsv           cached Ensembl -> symbol lookup (MyGene.info)
│   ├── string_input.txt           DEG list pasted into STRING
│   ├── background.txt             all tested genes (enrichment background)
│   └── volcano.png
├── string/                        STRING web-tool exports
│   ├── string_mapping.tsv         input ID -> STRING protein mapping
│   ├── string_interactions_54_nodes.tsv   network edges with evidence scores
│   └── ppi_enrichment.tsv         nodes, edges, expected edges, PPI-enrichment p-value
└── enrichment/                    written by scripts/10_enrichment.py
    └── enrichment_<set>.tsv / .png / _info.json   for deg_both, deg_up, padj_both
```

## Protocol (record what you actually did)

### 1. Dataset (NCBI GEO)
- Accession: GSE160306 (SubSeries of GSE160310), human post-mortem retina, total RNA-seq
- Citation: Becker et al. (2021), *Scientific Reports* 11:10494
- Groups compared: macula, diabetic retinopathy (NPDR, NPDR + DME, PDR + DME; n = 16 donors) vs.
  non-diabetic controls (n = 10 donors); details in `geo2r/run_info.json`
- Download (from the repo root):
  ```bash
  mkdir -p molecular/geo2r/raw && cd molecular/geo2r/raw
  wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE160nnn/GSE160306/matrix/GSE160306_series_matrix.txt.gz
  wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE160nnn/GSE160306/suppl/GSE160306_human_retina_DR_totalRNA_counts.txt.gz
  cd ../../..
  ```

### 2. Differential expression (scripted: `scripts/09_differential_expression.py`)
- PyDESeq2, design `~ age + gender + condition`, Wald test, Benjamini-Hochberg
- Macula only (each donor gave a macula and a periphery sample); where a donor contributed both eyes,
  the sample with the higher RIN is kept (3 duplicate samples dropped, listed in `run_info.json`)
- Genes kept: >= 10 counts in at least as many samples as the smaller group
- DEG threshold used: padj < 0.05 and |log2FC| >= 0.58 (1.5-fold), chosen because effects in bulk
  post-mortem retina are modest; the count at padj < 0.05 alone is also reported (`run_info.json`)
- Gene symbols: MyGene.info lookup, cached in `geo2r/gene_symbols.tsv`
- Outputs: `geo2r/de_results.tsv`, `geo2r/de_significant.tsv`, `geo2r/string_input.txt`,
  `geo2r/volcano.png`, `geo2r/run_info.json`
- Genes tested: 23,256. DEGs: 58 (49 up, 9 down); 95 genes at padj < 0.05 without the fold-change filter

### 3. PPI network (STRING, https://string-db.org)
- STRING version: 12.5   Species: Homo sapiens
- Input: the 58 DEGs in `geo2r/string_input.txt` (54 mapped)   Minimum interaction score: 0.7 (high confidence)
- Evidence channels used: STRING defaults (the export contains text-mining, co-expression and database evidence)
- Result: 54 nodes, 6 edges vs. 3 expected, PPI-enrichment p = 0.074 (not significant). The DEGs do not form
  a connected module, so no hub-gene ranking is reported
- Saved: `string/string_mapping.tsv`, `string/string_interactions_54_nodes.tsv`, `string/ppi_enrichment.tsv`

### 4. Functional enrichment (scripted: `scripts/10_enrichment.py`)
- Tool: g:Profiler via `gprofiler-official` (record the data version from the g:Profiler website)
- Sources: GO Biological Process, KEGG, Reactome
- Background: the genes tested by DESeq2, restricted to annotated genes (`domain_scope="custom_annotated"`);
  plain `custom` produced root-term "enrichment" (annotation bias), see `enrichment_*_info.json`
- Display filter (print/plot only): term size <= 1500, >= 3 DEGs per term
- Significance: Benjamini-Hochberg FDR < 0.05
- Gene sets: `deg_both` (all 58 DEGs; 157 significant terms), `deg_up` (up-regulated DEGs only),
  `padj_both` (all genes at padj < 0.05, no fold-change filter)
- Saved: `enrichment/enrichment_<set>.tsv`, `.png`, `_info.json` for each set

### 5. Lesion → process table
For each grade-defining lesion: the biological process, the enriched terms from steps 3–4 that support it,
and literature references. The completed table lives in the report; this is the template.

| Grade-defining lesion | Biological process | Supporting terms / hubs (this analysis) | References |
|---|---|---|---|
| Microaneurysms (grade 1) | Pericyte loss, capillary wall weakening | [TODO] | [TODO] |
| Hemorrhages, hard exudates (grade 2–3) | Vascular permeability, blood–retina barrier breakdown | [TODO] | [TODO] |
| Cotton-wool spots (grade 2–3) | Retinal ischemia, hypoxia signaling | [TODO] | [TODO] |
| Venous beading, IRMA (grade 3) | Progressive capillary non-perfusion | [TODO] | [TODO] |
| Neovascularization (grade 4) | VEGF-driven angiogenesis | [TODO] | [TODO] |

## How it connects to the image results
Use the table to interpret `results/figures/aptos_confusion_*.png` and the activation analysis:
for example, which grade boundaries the model confuses, and whether the change between those
grades shows up visually as small or as large lesions.

## Tools to cite
GEO (Barrett et al., 2013), DESeq2 (Love et al., 2014) / PyDESeq2 (Muzellec et al., 2023), STRING (Szklarczyk et al., 2023; version 12.5),
g:Profiler (Kolberg et al., 2023).
