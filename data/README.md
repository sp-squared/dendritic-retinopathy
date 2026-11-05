# Data

Nothing in this folder is committed except this file.

## APTOS 2019 Blindness Detection (required)

1. Create a Kaggle account, open the competition page
   (https://www.kaggle.com/competitions/aptos2019-blindness-detection) and accept the rules.
2. Download with the Kaggle CLI (`pip install kaggle`, API token in `~/.kaggle/kaggle.json`
   on Linux or `C:\Users\<you>\.kaggle\kaggle.json` on Windows):

   ```
   kaggle competitions download -c aptos2019-blindness-detection -p data/raw/aptos
   ```

3. Unzip so these paths exist:

   ```
   data/raw/aptos/train.csv
   data/raw/aptos/train_images/<id_code>.png
   ```

Only the labelled training set (3,662 images) is used; the competition test set has no public labels.

## IDRiD lesion masks (step 4g, optional)

Download `A. Segmentation.zip` from https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid
(free IEEE account; CC BY 4.0) and unzip it anywhere under `data/raw/idrid/`. Script 17 finds the files by name:
`IDRiD_xx.jpg` and the masks `IDRiD_xx_MA.tif`, `_HE.tif`, `_EX.tif`, `_SE.tif`, `_OD.tif` (81 images).

## DDR lesion masks (step 4i, optional)

DDR (Li et al., 2019; CC BY 4.0) has 757 images with MA, HE, EX and SE masks (383 train, 149 validation,
225 test). Download only the lesion-segmentation part (~800 MB) from the Hugging Face mirror:

```bash
pip install -U huggingface_hub
hf download ctmedtech/DDR-dataset --repo-type dataset --include "lesion_segmentation/*" --local-dir data/raw/ddr
```

Script 20 finds images (`.jpg`) and masks (`.tif` in folders named `EX`, `HE`, `MA`, `SE`) anywhere under
`data/raw/ddr/`, and reads the split from folder names (`train`, `val`/`valid`, `test`), so the original
release's layout (`lesion_segmentation/<split>/image`, `.../label/<lesion>`) works too.

## Fashion-MNIST (replication step)

Downloaded automatically by torchvision into `data/fmnist/` the first time you run
`scripts/01_replicate_fmnist.py`.

## Generated folders

| Folder | Created by |
|---|---|
| `data/processed/aptos_512/`, `data/processed/splits.csv` | `scripts/02_prepare_aptos.py` |
| `data/features/<backbone>_<size>.npz` | `scripts/03_extract_features.py` |
| `data/features/<backbone>_<size>_spatial.npz` | `scripts/03_extract_features.py --spatial` (layer4 7 x 7 map) |
| `data/features/<backbone>_<size>_spatial_l<layer>_g<grid>.npz` | `scripts/03_extract_features.py --spatial --layer L --grid G` (step 4c) |
| `data/features/pixels_<channel>_<size>.npz` | `scripts/12_extract_pixels.py` (step 4d, raw pixels) |
| `data/features/lesions_<size>/<dataset>/<id>.npz` | `scripts/20_train_fullres.py` (step 4i: green channel + lesion bitfield per image) |
| `data/fmnist/` | torchvision, on the first run of `scripts/01_replicate_fmnist.py` |
