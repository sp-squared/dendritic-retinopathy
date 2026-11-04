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

## Fashion-MNIST (replication step)

Downloaded automatically by torchvision into `data/fmnist/` the first time you run
`scripts/01_replicate_fmnist.py`.

## Generated folders

| Folder | Created by |
|---|---|
| `data/processed/aptos_512/`, `data/processed/splits.csv` | `scripts/02_prepare_aptos.py` |
| `data/features/<backbone>_<size>.npz` | `scripts/03_extract_features.py` |
