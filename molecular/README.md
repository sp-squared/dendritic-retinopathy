# Molecular context analysis

Goal: give the image results a biological explanation by identifying the molecular processes
behind the lesions that define each DR grade. This is a *supporting* analysis. It does not feed
into the model, and the report should not claim the image model learns molecular pathways.

These steps use web tools (GEO2R, STRING, Enrichr/STRING enrichment), so reproducibility comes from
**saving every export in this folder and recording every setting below**, like a lab notebook.

## Folder layout

```
molecular/
├── README.md            this protocol: fill in every [TODO]
├── geo2r/               DE results from script 09 (raw/ downloads are git-ignored)
├── string/              STRING network (.tsv edges, .png image) + hub ranking
├── enrichment/          GO / KEGG / Reactome tables (.tsv)
└── lesion_process_table.md   final table used in the report
```

## Protocol (record what you actually did)

### 1. Dataset (NCBI GEO)
- Accession: GSE160306 (SubSeries of GSE160310), human post-mortem retina, total RNA-seq
- Citation: Becker et al. (2021), *Scientific Reports* 11:10494
- Groups compared: macula, PDR vs. healthy controls (exact n in `geo2r/run_info.json`)
- Download (from the repo root):
  ```bash
  mkdir -p molecular/geo2r/raw && cd molecular/geo2r/raw
  wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE160nnn/GSE160306/matrix/GSE160306_series_matrix.txt.gz
  wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE160nnn/GSE160306/suppl/GSE160306_human_retina_DR_totalRNA_counts.txt.gz
  cd ../../..
  ```

### 2. Differential expression (scripted: `scripts/09_differential_expression.py`)
- PyDESeq2, design `~ age + gender + condition`, Wald test, Benjamini-Hochberg
- Macula only (each donor gave a macula and a periphery sample)
- Genes kept: >= 10 counts in at least as many samples as the smaller group
- DEG thresholds: padj < 0.05, |log2FC| >= 1
- DEG threshold used: padj < 0.05 and |log2FC| >= 0.58 (1.5-fold), chosen because effects in bulk
  post-mortem retina are modest; the count at padj < 0.05 alone is also reported (`run_info.json`)
- Gene symbols: MyGene.info lookup, cached in `geo2r/gene_symbols.tsv`
- Outputs: `geo2r/de_results.tsv`, `geo2r/de_significant.tsv`, `geo2r/string_input.txt`,
  `geo2r/volcano.png`, `geo2r/run_info.json`
- Number of DEGs: up [TODO] / down [TODO] (from `run_info.json`)

### 3. PPI network (STRING, https://string-db.org)
- STRING version: 12.5   Species: Homo sapiens
- Input: DEGs from step 2   Minimum interaction score: [0.7, high confidence]
- Evidence channels used: [TODO]
- Hub genes: top 10 by degree (optionally confirm with Cytoscape cytoHubba MCC)
- Saved: `string/edges.tsv`, `string/network.png`, `string/hubs.tsv`

### 4. Functional enrichment (scripted: `scripts/10_enrichment.py`)
- Tool: g:Profiler via `gprofiler-official` (record the data version from the g:Profiler website)
- Sources: GO Biological Process, KEGG, Reactome
- Background: custom = the 23,256 genes tested by DESeq2 (`domain_scope="custom"`)
- Significance: Benjamini-Hochberg FDR < 0.05
- Saved: `enrichment/enrichment_deg_both.tsv`, `.png`, `_info.json`

### 5. Lesion → process table
Fill `lesion_process_table.md`: for each grade-defining lesion, the biological process, the
enriched terms / hub genes from steps 3–4 that support it, and literature references.

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
GEO (Barrett et al., 2013), DESeq2 (Love et al., 2014) / PyDESeq2 (Muzellec et al., 2023), STRING (Szklarczyk et al.;
cite the version you used), Enrichr (Kuleshov et al., 2016) or STRING enrichment,
Cytoscape (Shannon et al., 2003) if used.
