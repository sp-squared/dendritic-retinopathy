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

- Python 3.10+ (tested with 3.13)
- PyTorch 2.2+ and torchvision (GPU optional; cached-feature heads train quickly on CPU)
- Roughly 10 GB free disk for APTOS raw + processed images

## Installation

PyTorch first: pick the right command for your OS/GPU at <https://pytorch.org/get-started/locally/>.

**Linux / macOS**
```bash
git clone <your-repo-url> dendritic-dr
cd dendritic-dr
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision          # or the CUDA-specific command from pytorch.org
pip install -e ".[dev]"
pytest                                 # should report all tests passing
```

**Windows (PowerShell)**
```powershell
git clone <your-repo-url> dendritic-dr
cd dendritic-dr
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
dendritic-dr/
├── README.md
├── pyproject.toml            package + dependencies
├── requirements.txt
├── data/                     (git-ignored except README) raw, processed, cached features
├── results/                  CSV logs of every run; figures/ for report plots
├── scripts/
│   ├── 01_replicate_fmnist.py   dANN vs. vanilla on Fashion-MNIST
│   ├── 02_prepare_aptos.py      crop/resize fundus images, stratified splits
│   ├── 03_extract_features.py   frozen CNN backbone features -> .npz
│   ├── 04_train_heads.py        dendritic / dense heads on features
│   ├── 05_plot_results.py       report figures (QWK vs. params, confusion matrices)
│   ├── 06_activation_analysis.py  differential activation, clustering, PCA, selectivity
│   └── 07_stats.py              bootstrap CIs, paired Wilcoxon, FDR across comparisons
├── molecular/                GEO2R / STRING / enrichment exports + protocol (manual web-tool steps)
├── src/dendritic_dr/
│   ├── masks.py              dendritic connectivity masks (random, local receptive field, soma)
│   ├── models.py             MaskedLinear, DANN, VanillaMLP, effective_params
│   ├── metrics.py            QWK, ordinal (cumulative-link) loss
│   ├── analysis.py           FDR, differential activation, selectivity, PCA, bootstrap
│   ├── features.py           loading/standardising cached features
│   ├── data.py               fundus preprocessing, splits, Dataset
│   └── utils.py              paths, seeding, device, CSV logging
└── tests/                    unit tests (pytest)
```

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

**8. Molecular context**: follow the protocol in [`molecular/README.md`](molecular/README.md)
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
| `results/models/<run_id>.pt` | best checkpoint, model config, feature standardisation stats |
| `results/predictions/<run_id>.csv` | per-image test predictions (`id_code`, `true`, `pred`) |
| `results/analysis/activation_summary*.csv` | per run / per config: fraction of FDR-significant units, selectivity, progression score |
| `results/analysis/units/<run_id>.csv` | per-unit "DE table": Kruskal–Wallis and Spearman p/q values, grade means, selectivity |
| `results/analysis/fig_heatmap_<config>.png` | clustered heatmap of grade-associated units, images sorted by grade |
| `results/analysis/fig_pca_<config>.png` | PCA of hidden activations coloured by grade, with progression score |
| `results/analysis/fig_selectivity.png` | number of grades each unit responds to, dANN vs. dense |
| `results/analysis/stats_*.csv` | bootstrap CIs per run, per-config seed CIs, paired comparisons with BH q-values |

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
- Molecular-analysis tools (GEO, GEO2R/limma, STRING, Enrichr): see `molecular/README.md`.
