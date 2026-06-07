# Data

This folder holds the metadata catalog that points WaveDiT at your MRI volumes. The
actual scans and any catalog of absolute paths are **git-ignored** and must not be
committed; obtain the raw data from the original providers:

* BHB: https://baobablab.github.io/bhb/dataset
* ADNI: https://adni.loni.usc.edu/
* OASIS: https://sites.wustl.edu/oasisbrains/

## Build the catalog (CSV mode, recommended)

```bash
python scripts/prepare_metadata.py \
    --input-dirs /path/to/CN_scans \
    --output-csv ./data/dataset.csv \
    --condition-label CN
```

This produces `data/dataset.csv` with columns `SubjectID, FilePath, Age, Condition`.
Point `data.metadata_csv` in your config at this file. Age is parsed from filenames
matching `[_-]AGE[_-]<number>` (e.g. `sub-001_AGE_65.3.nii.gz`).

See [`example_catalog.csv`](example_catalog.csv) for the exact format: a tiny three-row
sample (one placeholder row per data source) with dummy `/path/to/dataset/...` paths.

## Filename mode (no CSV)

Alternatively, set `data.data_folder` (and leave `data.metadata_csv: null`) to load a
flat folder of `*.nii.gz` files, parsing age directly from each filename. CSV mode is
preferred for anything beyond a single numeric `age` condition.
