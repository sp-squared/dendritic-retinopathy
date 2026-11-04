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
├── geo2r/               GEO2R full results table (.tsv) + the R script GEO2R shows
├── string/              STRING network (.tsv edges, .png image) + hub ranking
├── enrichment/          GO / KEGG / Reactome tables (.tsv)
└── lesion_process_table.md   final table used in the report
```

## Protocol (record what you actually did)

### 1. Dataset (NCBI GEO)
- Accession: [TODO, e.g. GSExxxxx]   Platform: [TODO]
- Tissue / species: [TODO: retina; human preferred]
- Groups compared: [TODO: e.g. DR (n=?) vs. non-diabetic control (n=?)]
- Why this dataset: [TODO: tissue relevance, sample size, publication]
- Citation: [TODO: the dataset's original paper]

### 2. Differential expression (GEO2R)
- Settings: log transform [TODO], adjustment = Benjamini-Hochberg, other [TODO]
- DEG thresholds: adj. p < [0.05], |log2FC| > [1]
- Number of DEGs: up [TODO] / down [TODO]
- Saved: `geo2r/full_table.tsv`, `geo2r/geo2r_script.R`

### 3. PPI network (STRING, https://string-db.org)
- STRING version: [TODO]   Species: Homo sapiens
- Input: DEGs from step 2   Minimum interaction score: [0.7, high confidence]
- Evidence channels used: [TODO]
- Hub genes: top 10 by degree (optionally confirm with Cytoscape cytoHubba MCC)
- Saved: `string/edges.tsv`, `string/network.png`, `string/hubs.tsv`

### 4. Functional enrichment
- Tool + version: [TODO: STRING enrichment tab or Enrichr]
- Libraries: GO Biological Process, KEGG, [Reactome]
- Background: [TODO: whole genome or all genes on the array; state which]
- Significance: FDR < 0.05
- Saved: `enrichment/go_bp.tsv`, `enrichment/kegg.tsv`

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
GEO (Barrett et al., 2013), GEO2R/limma (Ritchie et al., 2015), STRING (Szklarczyk et al.;
cite the version you used), Enrichr (Kuleshov et al., 2016) or STRING enrichment,
Cytoscape (Shannon et al., 2003) if used.
