# Dendritic ANNs for Diabetic Retinopathy Grading

Course project for Bioinformatics (CSUSB). Author: Colin Morris-Moncada.

**Research question.** Dendritic artificial neural networks (dANNs) have been reported to match or beat
standard ANNs on image benchmarks with far fewer trainable parameters (Chavlis & Poirazi, 2025).
Those benchmarks use small, low-resolution images. Does that advantage carry over to grading diabetic
retinopathy (DR) from retinal fundus photographs, where the diagnostic signal is small, local lesions?

**Approach.**
1. *Replication*: dANN vs. a fully connected ANN on Fashion-MNIST.
2. *Main experiment*: fundus images (APTOS 2019) pass through a frozen ImageNet-pretrained CNN; a dendritic
   head and a dense head of matched size are trained on the extracted features and compared by
   quadratic weighted kappa (QWK) against effective parameter count, over multiple seeds.
3. *Representation analysis*: the trained heads' hidden-unit activations are treated like a
   samples × genes expression matrix: differential activation across DR grades with FDR control,
   clustering, PCA, and unit selectivity (testing the paper's "mixed selectivity" claim on clinical data).
4. *Statistics*: bootstrap confidence intervals and paired tests across seeds.
5. *Molecular context*: a small GEO → STRING PPI → enrichment analysis linking each grade's defining
   lesions to biological processes, used to interpret the image results (see `molecular/`).

> The dANN here is a simplified re-implementation (restricted input sampling + dendrite-to-single-soma
> connectivity), not the authors' official code.

---

## Requirements

- Python 3.10+ (tested with 3.13 on Windows and 3.14 on Ubuntu 26.04)
- PyTorch 2.2+ and torchvision (GPU optional; cached-feature heads train quickly on CPU)
- Roughly 10 GB free disk for APTOS raw + processed images, plus a few GB for cached features
  (spatial feature files are the largest; see `data/README.md`)
- A clone is about 75 MB: the trained heads in `results/models/` are stored compactly (see below)

## Installation

PyTorch first: pick the right command for your OS/GPU at <https://pytorch.org/get-started/locally/>.

**Linux / macOS**
```bash
git clone https://github.com/sp-squared/dendritic-retinopathy.git
cd dendritic-retinopathy
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision          # or the CUDA-specific command from pytorch.org
pip install -e ".[dev]"
pytest                                 # should report all tests passing
```

**Windows (PowerShell)**
```powershell
git clone https://github.com/sp-squared/dendritic-retinopathy.git
cd dendritic-retinopathy
py -m venv .venv
.venv\Scripts\Activate.ps1             # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install torch torchvision          # or the CUDA-specific command from pytorch.org
pip install -e ".[dev]"
pytest
```

All commands below are identical on both platforms (use `python` inside the activated venv).
Paths are resolved relative to the repository root, so scripts can be run from the repo root.

## Input data

See [`data/README.md`](data/README.md). In short, place the Kaggle APTOS 2019 files at
`data/raw/aptos/train.csv` and `data/raw/aptos/train_images/`. Fashion-MNIST downloads automatically.

## Directory structure

```
dendritic-retinopathy/
├── README.md
├── LICENSE                   MIT
├── pyproject.toml            package + dependencies (extras: [dev], [molecular])
├── requirements.txt
├── .gitattributes            LF line endings in the repo, binary figures
├── .gitignore                credentials, raw data, caches, smoke outputs, *.patch
├── data/                     git-ignored except README: APTOS raw/processed images, cached features
│   └── README.md             download instructions (Kaggle) and generated folders
├── scripts/                  numbered in the order they are run
│   ├── 01_replicate_fmnist.py   dANN vs. vanilla on Fashion-MNIST
│   ├── 02_prepare_aptos.py      crop/resize fundus images, stratified splits
│   ├── 03_extract_features.py   frozen ResNet-18 features -> .npz (pooled, --spatial, --layer, --grid)
│   ├── 04_train_heads.py        dendritic / dense heads on cached features
│   ├── 05_plot_results.py       report figures (QWK vs. params, confusion matrices, summary table)
│   ├── 06_activation_analysis.py  differential activation, clustering, PCA, selectivity
│   ├── 07_stats.py              bootstrap CIs, paired Wilcoxon, FDR across comparisons
│   ├── 08_inspect_subcluster.py exploratory: acquisition properties of a PCA subcluster
│   ├── 09_differential_expression.py  PyDESeq2 on GSE160306 retina RNA-seq
│   ├── 10_enrichment.py         g:Profiler GO/KEGG/Reactome enrichment, custom background
│   ├── 11_receptive_field_sweep.py  locality effect (LRF - random) across ResNet stages
│   ├── 12_extract_pixels.py     downsampled raw fundus pixels (no CNN) -> .npz, same format as features
│   ├── 13_pixel_comparison.py   dendritic vs. dense heads on raw pixels: LRF vs. random, dANN vs. dense
│   ├── 14_global_baseline.py    global image statistics vs. the one-unit raw-pixel model
│   └── compact_checkpoints.py   one-off utility: convert old dense checkpoints to the compact format
├── src/dendritic_dr/         importable package (pip install -e .)
│   ├── masks.py              dendritic connectivity masks (random, local receptive field, soma)
│   ├── models.py             MaskedLinear (+ memory-light IndexedMaskedLinear), DANN, VanillaMLP, compact checkpoints
│   ├── metrics.py            QWK, ordinal (cumulative-link) loss
│   ├── analysis.py           FDR, differential activation, selectivity, PCA, bootstrap
│   ├── features.py           loading/standardising cached features (pooled and spatial)
│   ├── data.py               fundus preprocessing, raw pixels, image properties, splits, Dataset
│   ├── plotting.py           display names, colour-blind-safe styles, axis formatting
│   └── utils.py              paths, seeding, device, CSV logging
├── tests/                    unit tests (pytest): models/masks, analysis, plotting
├── results/                  committed outputs of the real runs
│   ├── fmnist.csv            one row per Fashion-MNIST run
│   ├── aptos_heads.csv       one row per APTOS head run
│   ├── figures/              report figures + aptos_summary.csv
│   ├── analysis/             activation analysis, statistics, subcluster, sweep outputs
│   │   └── units/            per-unit tables, one CSV per analysed run
│   ├── models/               compact checkpoints, one .pt per run
│   └── predictions/          per-image test predictions, one CSV per run
└── molecular/                GEO -> DESeq2 -> STRING -> enrichment (protocol in molecular/README.md)
    ├── geo2r/                DE results, volcano plot, STRING input and background gene lists
    ├── string/               STRING ID mapping, network edges, PPI-enrichment statistics
    └── enrichment/           g:Profiler tables, plots and run settings
```

Outputs from `--smoke` runs go to `results/smoke/` (git-ignored), never into the folders above.

### Run names

Every run has an ID built from its settings, used for checkpoint, prediction and unit-table file names
and in the `config` column of the CSVs:

```
<features>_<model>_s<somas>_d<dendrites>_y<synapses>[_rf<k>]_<loss>_seed<n>    dendritic heads
<features>_vanilla_h<hidden>_<loss>_seed<n>                                   dense heads
```

For example `resnet18_224_spatial_dann-lrf_s128_d16_y16_rf3_ce_seed0` is a dANN-LRF head with 128 somas,
16 dendrites per soma, 16 synapses per dendrite and a 3 x 3 receptive field, trained with cross-entropy on
the layer4 spatial features. The ID without `_seed<n>` is the *configuration*, which is what the statistics
and plotting scripts group by.

## Reproducing the analyses

Quick offline check that the whole pipeline runs (synthetic data, a few minutes on CPU;
outputs go to `results/smoke/`, which is git-ignored):
```bash
python scripts/01_replicate_fmnist.py --smoke --model dann-lrf
python scripts/04_train_heads.py --smoke --model dann-r  --somas 64 --seeds 0 1 2 3 4 5
python scripts/04_train_heads.py --smoke --model vanilla --hidden 32 --seeds 0 1 2 3 4 5
python scripts/05_plot_results.py --smoke
python scripts/06_activation_analysis.py --smoke
python scripts/07_stats.py --smoke
python scripts/04_train_heads.py --smoke --features smoke_pixels --model dann-lrf --somas 32 --rf-size 4 --seeds 0 1 2
python scripts/04_train_heads.py --smoke --features smoke_pixels --model dann-r   --somas 32 --seeds 0 1 2
python scripts/04_train_heads.py --smoke --features smoke_pixels --model vanilla  --hidden 4 --seeds 0 1 2
python scripts/13_pixel_comparison.py --smoke
```

**1. Fashion-MNIST replication**
```bash
python scripts/01_replicate_fmnist.py --model dann-lrf --somas 256 --dendrites 16 --rf-size 4 --seeds 0 1 2
python scripts/01_replicate_fmnist.py --model dann-r   --somas 256 --dendrites 16 --synapses 16 --seeds 0 1 2
python scripts/01_replicate_fmnist.py --model vanilla  --hidden 256 --seeds 0 1 2
```

**2. Prepare APTOS** (one time; the split seed is fixed at 42 for the whole project)
```bash
python scripts/02_prepare_aptos.py --size 512
```

**3. Extract features**
```bash
python scripts/03_extract_features.py --backbone resnet18 --input-size 224
```

**4. Train heads**: sweep sizes so the comparison is at matched parameter budgets. Use 10 seeds:
with n paired seeds the smallest possible two-sided Wilcoxon p-value is 2/2ⁿ (0.0625 for n = 5).
```bash
python scripts/04_train_heads.py --features resnet18_224 --model dann-r  --somas 32  --seeds 0 1 2 3 4 5 6 7 8 9
python scripts/04_train_heads.py --features resnet18_224 --model dann-r  --somas 128 --seeds 0 1 2 3 4 5 6 7 8 9
python scripts/04_train_heads.py --features resnet18_224 --model vanilla --hidden 8  --seeds 0 1 2 3 4 5 6 7 8 9
python scripts/04_train_heads.py --features resnet18_224 --model vanilla --hidden 64 --seeds 0 1 2 3 4 5 6 7 8 9
```
Each run saves its checkpoint (`results/models/`) and per-image test predictions (`results/predictions/`).
Add `--loss ordinal` to treat grades as ordered, or `--class-weights` to counter class imbalance.

**4b. Spatial follow-up: does the dendritic advantage need spatial structure?**
Keep ResNet-18's last convolutional map (512 channels x 7 x 7 grid) instead of the pooled vector.
dANN-LRF dendrites sample 16 entries inside a local 3 x 3 window; dANN-R dendrites sample 16 entries
anywhere. Both have identical parameter counts, so the comparison isolates locality. (A size-matched
dense head is impossible on 25,088 inputs; the pooled dense head from step 4 is the practical benchmark.)
```bash
python scripts/03_extract_features.py --backbone resnet18 --input-size 224 --spatial
for s in 32 64 128 256; do
  python scripts/04_train_heads.py --features resnet18_224_spatial --model dann-lrf --somas $s --rf-size 3 --seeds 0 1 2 3 4 5 6 7 8 9
  python scripts/04_train_heads.py --features resnet18_224_spatial --model dann-r   --somas $s --seeds 0 1 2 3 4 5 6 7 8 9
done
python scripts/07_stats.py --pair resnet18_224_spatial_dann-lrf_s128_d16_y16_rf3_ce resnet18_224_spatial_dann-r_s128_d16_y16_ce \
                           --pair resnet18_224_spatial_dann-lrf_s128_d16_y16_rf3_ce resnet18_224_vanilla_h73_ce
```

**4c. Receptive-field sweep: does the locality advantage shrink with depth?**
Hypothesis (stated in advance): dANN-LRF beats dANN-R when map cells are local (early stages)
and the gap shrinks to zero at layer4. Each stage is pooled to ~25k inputs so only locality differs.
```bash
python scripts/03_extract_features.py --backbone resnet18 --input-size 224 --spatial --layer 1 --grid 20
python scripts/03_extract_features.py --backbone resnet18 --input-size 224 --spatial --layer 2 --grid 14
python scripts/03_extract_features.py --backbone resnet18 --input-size 224 --spatial --layer 3 --grid 10
for f in resnet18_224_spatial_l1_g20 resnet18_224_spatial_l2_g14 resnet18_224_spatial_l3_g10; do
  python scripts/04_train_heads.py --features $f --model dann-lrf --somas 128 --rf-size 3 --seeds 0 1 2 3 4 5 6 7 8 9
  python scripts/04_train_heads.py --features $f --model dann-r   --somas 128 --seeds 0 1 2 3 4 5 6 7 8 9
done
python scripts/11_receptive_field_sweep.py     # layer4 runs from step 4b are reused
```

*Window-size control.* A fixed 3 x 3 window covers 15% of the map width at layer1 but 43% at layer4, so
depth and window size change together. Rerun dANN-LRF with the window scaled to ~30% of the width at every
stage (6/20, 4/14, 3/10, 2/7). layer3 (rf 3) and all dANN-R runs are reused, so only 30 new runs are needed:
```bash
python scripts/04_train_heads.py --features resnet18_224_spatial_l1_g20 --model dann-lrf --somas 128 --rf-size 6 --seeds 0 1 2 3 4 5 6 7 8 9
python scripts/04_train_heads.py --features resnet18_224_spatial_l2_g14 --model dann-lrf --somas 128 --rf-size 4 --seeds 0 1 2 3 4 5 6 7 8 9
python scripts/04_train_heads.py --features resnet18_224_spatial       --model dann-lrf --somas 128 --rf-size 2 --seeds 0 1 2 3 4 5 6 7 8 9
python scripts/11_receptive_field_sweep.py --rf-sizes 6 4 3 2   # -> rf_sweep_scaled.csv, fig_rf_sweep_scaled.png
```

*Unpooled layer1.* The sweep average-pooled layer1 from 56 x 56 to 20 x 20, which may have blurred
away fine-scale locality. This reruns layer1 unpooled (64 x 56 x 56 = 200,704 inputs) with a 3 x 3 window
(the most local setting, 5% of the width) and an 8 x 8 window (15%, the same image area as the pooled
3 x 3 window). Layers this large switch automatically to a memory-light sparse layer
(`IndexedMaskedLinear`, mathematically identical). The feature file is ~1.2 GB and loading it needs
roughly 10 GB of RAM and 3 GB of GPU memory.
```bash
python scripts/03_extract_features.py --backbone resnet18 --input-size 224 --spatial --layer 1
python scripts/04_train_heads.py --features resnet18_224_spatial_l1 --model dann-lrf --somas 128 --rf-size 3 --seeds 0 1 2 3 4 5 6 7 8 9
python scripts/04_train_heads.py --features resnet18_224_spatial_l1 --model dann-lrf --somas 128 --rf-size 8 --seeds 0 1 2 3 4 5 6 7 8 9
python scripts/04_train_heads.py --features resnet18_224_spatial_l1 --model dann-r   --somas 128 --seeds 0 1 2 3 4 5 6 7 8 9
python scripts/11_receptive_field_sweep.py --features resnet18_224_spatial_l1 --rf-size 3 --tag l1_unpooled_rf3
python scripts/11_receptive_field_sweep.py --features resnet18_224_spatial_l1 --rf-size 8 --tag l1_unpooled_rf8
```

**4d. Raw pixels: do dendrites regain their advantage without a CNN?**
Steps 4b–4c found no locality advantage on CNN features at any depth. This step tests the
interpretation directly: the heads are trained on downsampled fundus pixels (green channel, 128 x 128 =
16,384 inputs). dANN-LRF dendrites see a 4 x 4 pixel patch (16 synapses, as on Fashion-MNIST); dANN-R
dendrites see 16 random pixels. Prediction, stated before running: if locality is what helped on
Fashion-MNIST, LRF beats random and matches or beats a dense ANN of similar size.
A dense ANN needs about 16k parameters per hidden unit here, so its sizes are 1–16 hidden units.
```bash
python scripts/12_extract_pixels.py            # data/features/pixels_green_128.npz (about 1 minute)
for s in 32 64 128 256; do
  python scripts/04_train_heads.py --features pixels_green_128 --model dann-lrf --somas $s --rf-size 4 --seeds 0 1 2 3 4 5 6 7 8 9
  python scripts/04_train_heads.py --features pixels_green_128 --model dann-r   --somas $s --seeds 0 1 2 3 4 5 6 7 8 9
done
for h in 1 2 4 8 16; do
  python scripts/04_train_heads.py --features pixels_green_128 --model vanilla --hidden $h --seeds 0 1 2 3 4 5 6 7 8 9
done
python scripts/13_pixel_comparison.py          # -> pixels_by_config.csv, pixels_paired.csv, fig_pixels.png
```
Raw-pixel runs are left out of the step 5 figures (`--exclude pixels`, the default) and plotted by script 13.

**4e. Global-statistics check.** On raw pixels the best model was a dense ANN with one hidden unit, i.e. a
single weighted sum over the image. This step checks whether that unit mostly reads image-wide properties
(brightness, contrast, sharpness; camera proxies from step 8) rather than lesions: logistic-regression
baselines on a few global statistics per image, compared with the one-unit model by a paired bootstrap,
plus the hidden unit's Spearman correlation with each property.
```bash
python scripts/14_global_baseline.py     # -> image_properties.csv, global_baseline.csv, global_unit_corr.csv
```

**5. Figures**
```bash
python scripts/05_plot_results.py
```

**6. Representation analysis** (best config per model type by validation QWK; or `--configs ...`)
```bash
python scripts/06_activation_analysis.py --features resnet18_224
```

**7. Statistics** (each dendritic config paired with the closest-sized dense config; or `--pair A B`)
```bash
python scripts/07_stats.py --features resnet18_224
```

**Exploratory: inspect a PCA subcluster** (e.g. the grade-0 cluster in `fig_pca_*.png`): compares it with the
rest of that grade on original resolution (camera proxy), brightness, contrast and sharpness, with an example sheet
```bash
python scripts/08_inspect_subcluster.py --config resnet18_224_dann-r_s128_d16_y16_ce --pc1-max 0 --pc2-min 2.5
```

**8. Molecular context**: differential expression in human DR retina (GSE160306), then STRING and enrichment
```bash
pip install -e ".[molecular]"   # adds PyDESeq2
python scripts/09_differential_expression.py --inspect
python scripts/09_differential_expression.py --group-col disease_group_detailed --case "NPDR" "NPDR + DME" "PDR + DME" --control "Control" --lfc 0.58
python scripts/10_enrichment.py                  # GO:BP, KEGG, Reactome vs. tested-gene background (needs internet)
```
Download instructions and the STRING web-tool step: [`molecular/README.md`](molecular/README.md)
(GEO2R, STRING and enrichment are web tools; all exports and settings are saved in that folder).

## Expected outputs

| File | Contents |
|---|---|
| `results/fmnist.csv` | one row per run: model, config, effective params, train/test accuracy and loss |
| `results/aptos_heads.csv` | one row per run: config, effective params, best epoch, val/test QWK, test accuracy, test confusion matrix (JSON) |
| `results/figures/fmnist_acc_vs_params.png` | test accuracy vs. parameters (mean ± SD over seeds) |
| `results/figures/aptos_qwk_vs_params.png` | test QWK vs. parameters (mean ± SD over seeds) |
| `results/figures/aptos_confusion_<model>.png` | row-normalised confusion matrix, best config per model (selected by validation QWK) |
| `results/figures/aptos_summary.csv` | per-configuration mean/SD table |
| `results/models/<run_id>.pt` | best checkpoint (compact: only unmasked weights plus a mask fingerprint; masks are regenerated from the seed when loading), model config, feature standardisation stats. Convert older dense checkpoints with `python scripts/compact_checkpoints.py` |
| `results/predictions/<run_id>.csv` | per-image test predictions (`id_code`, `true`, `pred`) |
| `results/analysis/activation_summary*.csv` | per run / per config: fraction of FDR-significant units, selectivity, progression score |
| `results/analysis/units/<run_id>.csv` | per-unit "DE table": Kruskal–Wallis and Spearman p/q values, grade means, selectivity |
| `results/analysis/fig_heatmap_<config>.png` | clustered heatmap of grade-associated units, images sorted by grade |
| `results/analysis/fig_pca_<config>.png` | PCA of hidden activations coloured by grade, with progression score |
| `results/analysis/fig_selectivity.png` | number of grades each unit responds to, dANN vs. dense |
| `results/analysis/stats_runs.csv`, `stats_by_config.csv`, `stats_paired.csv` | bootstrap CIs per run, per-config seed CIs, paired comparisons with BH q-values |
| `results/analysis/fig_subcluster_pca.png`, `fig_subcluster_contact.png` | selected PCA subcluster, and example images of it vs. the rest of the grade |
| `results/analysis/subcluster_images.csv`, `subcluster_stats.csv` | per-image acquisition properties; Mann–Whitney tests with BH q-values |
| `results/analysis/rf_sweep.csv`, `fig_rf_sweep.png` | step 4c: LRF − R difference per ResNet stage with Wilcoxon/bootstrap tests and the depth trend |
| `results/analysis/rf_sweep_l1_unpooled_rf3.csv`, `_rf8.csv` (+ `fig_rf_sweep_*.png`) | step 4c unpooled layer1: LRF − R with a 3 x 3 and an 8 x 8 window on the 56 x 56 map |
| `results/analysis/rf_sweep_scaled.csv`, `fig_rf_sweep_scaled.png` | step 4c window-size control: same analysis with the window at ~30% of the map width at every stage |
| `results/analysis/pixels_by_config.csv`, `pixels_paired.csv`, `fig_pixels.png` | step 4d: raw-pixel heads; QWK vs. parameters, LRF − random and dANN − nearest-dense differences with Wilcoxon/bootstrap tests |
| `results/analysis/image_properties.csv` | per-image brightness, contrast, sharpness, green mean/SD, original resolution and file size |
| `results/analysis/global_baseline.csv`, `global_unit_corr.csv` | step 4e: global-statistics baselines vs. the one-unit raw-pixel model; the unit's correlation with each property |
| `molecular/geo2r/de_results.tsv`, `de_significant.tsv` | PyDESeq2 results for all tested genes; DEGs passing padj and fold-change thresholds |
| `molecular/geo2r/run_info.json`, `samples.csv` | design, thresholds, sample list (one per donor), DEG counts, software versions |
| `molecular/geo2r/volcano.png` | volcano plot |
| `molecular/geo2r/string_input.txt`, `background.txt` | DEG symbols for STRING; all tested genes (enrichment background) |
| `molecular/string/*.tsv` | STRING ID mapping, network edges, PPI-enrichment statistics (web-tool exports) |
| `molecular/enrichment/enrichment_<set>.{tsv,png}`, `_info.json` | g:Profiler terms for `deg_both`, `deg_up`, `padj_both`, with the settings used |

## Methodological notes

- **No test leakage**: features are standardised with training-split statistics; early stopping and
  configuration selection use validation QWK only.
- **Effective parameters** count only weights that can be non-zero (masked dendritic connections are excluded).
- **Multiple testing**: per-unit tests are corrected with Benjamini–Hochberg across units; model
  comparisons are corrected across all pairs.
- **Two sources of variability** are reported separately: training randomness (seeds; Wilcoxon,
  t-interval) and finite test set (bootstrap over test images).
- **Dead units**: ReLU units that never activate on the analysed split are reported (`frac_dead`)
  and excluded from FDR and selectivity.
- **Windows**: DataLoader workers default to 0 on Windows; pass `--num-workers N` to override.
  Line endings are normalised by `.gitattributes`.

## References

- Chavlis, S. and Poirazi, P. (2025). Dendrites endow artificial neural networks with accurate, robust and
  parameter-efficient learning. *Nature Communications*. doi:10.1038/s41467-025-56297-9
- Asia Pacific Tele-Ophthalmology Society (2019). APTOS 2019 Blindness Detection [dataset]. Kaggle.
  https://www.kaggle.com/competitions/aptos2019-blindness-detection
- Xiao, H., Rasul, K. and Vollgraf, R. (2017). Fashion-MNIST: a novel image dataset for benchmarking machine
  learning algorithms. arXiv:1708.07747
- He, K., Zhang, X., Ren, S. and Sun, J. (2016). Deep residual learning for image recognition. *CVPR*.
- Cao, W., Mirjalili, V. and Raschka, S. (2020). Rank consistent ordinal regression for neural networks with
  application to age estimation. *Pattern Recognition Letters*, 140, 325–331.
- Paszke, A. et al. (2019). PyTorch: an imperative style, high-performance deep learning library. *NeurIPS*.
- Pedregosa, F. et al. (2011). Scikit-learn: machine learning in Python. *JMLR*, 12, 2825–2830.
- Virtanen, P. et al. (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python. *Nature Methods*, 17, 261–272.
- Benjamini, Y. and Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach
  to multiple testing. *J. R. Stat. Soc. B*, 57, 289–300.
- Efron, B. and Tibshirani, R. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- Becker, K. et al. (2021). In-depth transcriptomic analysis of human retina reveals molecular mechanisms underlying
  diabetic retinopathy. *Scientific Reports*, 11, 10494. (GSE160306)
- Love, M. I., Huber, W. and Anders, S. (2014). Moderated estimation of fold change and dispersion for RNA-seq data
  with DESeq2. *Genome Biology*, 15, 550.
- Muzellec, B. et al. (2023). PyDESeq2: a python package for bulk RNA-seq differential expression analysis.
  *Bioinformatics*, 39, btad547.
- Kolberg, L. et al. (2023). g:Profiler: interoperable web service for functional enrichment analysis and gene
  identifier mapping (2023 update). *Nucleic Acids Research*, 51, W207–W212.
- Szklarczyk, D. et al. (2023). The STRING database in 2023. *Nucleic Acids Research*, 51, D638–D646.
- Molecular-analysis tools (GEO, STRING): see `molecular/README.md`.
