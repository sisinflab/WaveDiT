"""Unified 3D brain-MRI dataset.

A single :class:`BrainMRIDataset` covers both supported input modes:

* **CSV mode** (recommended): conditions and absolute file paths come from a
  metadata CSV (:meth:`BrainMRIDataset.from_csv`).
* **Filename mode**: a folder of ``*.nii.gz`` files whose age is parsed from the
  filename (:meth:`BrainMRIDataset.from_folder`).

Both modes build a common *catalog* (a list of ``(filepath, raw_conditions)``)
and share one ``__getitem__``, which loads, normalises, pads, optionally augments,
and returns ``(image, conditions)``:

* ``image``: ``float32`` tensor ``(1, D, H, W)`` in ``[-1, 1]``.
* ``conditions``: ``{name: tensor([value])}`` with numeric values normalised to
  ``[0, 1]`` and categorical values as float class ids.

The 3D wavelet transform is **not** computed here: the model applies it on-device
inside its loss, so doing it per-item on CPU would be redundant work.
"""

from __future__ import annotations

import glob
import os
import re
from collections import defaultdict

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..utils import get_logger
from .preprocessing import pad_to_size, robust_normalize

logger = get_logger(__name__)

# Matches "_AGE_72.5", "-age-30", etc. (case-insensitive). The number is matched
# precisely so a trailing dot (e.g. "..._AGE_30.nii.gz") is not captured.
AGE_PATTERN = re.compile(r"[_-]AGE[_-](\d+(?:\.\d+)?)", re.IGNORECASE)


class BrainMRIDataset(Dataset):
    def __init__(
        self,
        entries: list[dict],
        condition_types: dict[str, str],
        condition_ranges: dict[str, dict[str, float]],
        categorical_maps: dict[str, dict],
        image_size: tuple[int, int, int],
        transform=None,
    ):
        self.entries = entries
        self.condition_types = condition_types
        self.condition_ranges = condition_ranges
        self.categorical_maps = categorical_maps
        self.image_size = tuple(image_size)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        file_path = entry["filepath"]
        try:
            volume = nib.load(file_path).get_fdata(dtype=np.float32)
            volume = robust_normalize(volume)

            image = torch.from_numpy(volume)  # (D, H, W)
            if tuple(image.shape) != self.image_size:
                image = pad_to_size(image, self.image_size)
            image = image.unsqueeze(0)  # (1, D, H, W)

            if self.transform is not None:
                image = self.transform(image).float()
                if not torch.isfinite(image).all():
                    logger.error("Non-finite values after augmentation for %s; clamping.", file_path)
                    image = torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=-1.0)

            if tuple(image.shape[-3:]) != self.image_size:
                raise ValueError(f"Image shape {tuple(image.shape)} != expected spatial {self.image_size}")

            conditions = self._encode_conditions(entry["conditions"])
            return image, conditions
        except Exception as exc:  # noqa: BLE001 - skip unreadable/corrupt files
            logger.error("Error processing %s: %s", file_path, exc, exc_info=True)
            return None

    def _encode_conditions(self, raw_conditions: dict) -> dict[str, torch.Tensor]:
        encoded = {}
        for name, kind in self.condition_types.items():
            raw_value = raw_conditions[name]
            if kind == "numeric":
                value = _normalize_numeric(raw_value, self.condition_ranges.get(name))
            else:  # categorical: already mapped to a float class id
                value = float(raw_value)
            encoded[name] = torch.tensor([value], dtype=torch.float32)
        return encoded

    # ------------------------------------------------------------------ #
    # Catalog builders
    # ------------------------------------------------------------------ #
    @classmethod
    def from_csv(
        cls,
        metadata_csv: str,
        condition_types: dict[str, str],
        image_size: tuple[int, int, int],
        filepath_col: str = "FilePath",
        filter_col: str | None = "Condition",
        filter_value: str | None = "CN",
        transform=None,
    ) -> "BrainMRIDataset":
        if not os.path.exists(metadata_csv):
            raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")
        df = pd.read_csv(metadata_csv)
        logger.info("Loaded metadata CSV with %d rows from %s", len(df), metadata_csv)

        lower_to_actual = {c.lower(): c for c in df.columns}

        def resolve(name: str) -> str:
            actual = lower_to_actual.get(name.lower())
            if actual is None:
                raise ValueError(f"Column '{name}' not found in CSV. Available: {list(df.columns)}")
            return actual

        path_col = resolve(filepath_col)
        if filter_col and filter_value is not None:
            df = _filter_dataframe(df, resolve(filter_col), filter_value)
            logger.info("Filtered to %d rows where %s == %s", len(df), filter_col, filter_value)
        if len(df) == 0:
            raise RuntimeError("No rows left after filtering the metadata CSV.")

        cond_cols = {name: resolve(name) for name in condition_types}
        categorical_maps = _build_categorical_maps(df, condition_types, cond_cols)

        entries: list[dict] = []
        numeric_values: dict[str, list[float]] = defaultdict(list)
        for _, row in df.iterrows():
            file_path = row[path_col]
            if not isinstance(file_path, str) or not os.path.exists(file_path):
                logger.warning("Missing/invalid file path, skipping: %r", file_path)
                continue
            parsed = _parse_row_conditions(row, condition_types, cond_cols, categorical_maps, numeric_values)
            if parsed is not None:
                entries.append({"filepath": file_path, "conditions": parsed})

        if not entries:
            raise RuntimeError("No valid samples after validating file paths and conditions.")
        condition_ranges = _ranges_from(numeric_values)
        logger.info("CSV dataset ready: %d samples. Ranges=%s", len(entries), condition_ranges)
        return cls(entries, condition_types, condition_ranges, categorical_maps, image_size, transform)

    @classmethod
    def from_folder(
        cls,
        data_folder: str,
        condition_types: dict[str, str],
        image_size: tuple[int, int, int],
        require_conditions: bool = True,
        transform=None,
    ) -> "BrainMRIDataset":
        if not os.path.isdir(data_folder):
            raise FileNotFoundError(f"Data folder not found: {data_folder}")
        if set(condition_types) - {"age"} or condition_types.get("age", "numeric") != "numeric":
            raise NotImplementedError(
                "Filename mode only supports a numeric 'age' condition parsed from the "
                "filename. Use CSV mode for other/categorical conditions."
            )

        files = sorted(glob.glob(os.path.join(data_folder, "*.nii.gz")))
        logger.info("Found %d .nii.gz files in %s", len(files), data_folder)

        entries: list[dict] = []
        ages: list[float] = []
        for file_path in files:
            match = AGE_PATTERN.search(os.path.basename(file_path))
            if match is None:
                if require_conditions:
                    logger.warning("No age tag in %s; skipping.", os.path.basename(file_path))
                continue
            age = float(match.group(1))
            entries.append({"filepath": file_path, "conditions": {"age": age}})
            ages.append(age)

        if not entries:
            raise RuntimeError("No files with a parseable age tag were found.")
        condition_ranges = _ranges_from({"age": ages}) if ages else {}
        logger.info("Filename dataset ready: %d samples. Age range=%s", len(entries), condition_ranges)
        return cls(entries, condition_types, condition_ranges, {}, image_size, transform)


# ---------------------------------------------------------------------- #
# Catalog helpers
# ---------------------------------------------------------------------- #
def _normalize_numeric(raw_value: float, value_range: dict | None) -> float:
    if value_range is None:
        return float(raw_value)
    span = value_range["max"] - value_range["min"]
    if span <= 1e-8:
        return 0.5
    return float(np.clip((raw_value - value_range["min"]) / span, 0.0, 1.0))


def _ranges_from(numeric_values: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    ranges = {}
    for name, values in numeric_values.items():
        if values:
            ranges[name] = {"min": float(np.min(values)), "max": float(np.max(values))}
    return ranges


def _build_categorical_maps(df, condition_types, cond_cols) -> dict[str, dict]:
    maps: dict[str, dict] = {}
    for name, kind in condition_types.items():
        if kind != "categorical":
            continue
        categories = sorted(df[cond_cols[name]].dropna().unique(), key=str)
        maps[name] = {cat: i for i, cat in enumerate(categories)}
        logger.info("Categorical '%s' -> %s", name, maps[name])
    return maps


def _parse_row_conditions(row, condition_types, cond_cols, categorical_maps, numeric_values):
    parsed = {}
    for name, kind in condition_types.items():
        raw = row[cond_cols[name]]
        if pd.isna(raw):
            return None
        if kind == "numeric":
            try:
                value = float(raw)
            except (TypeError, ValueError):
                return None
            parsed[name] = value
            numeric_values[name].append(value)
        else:
            class_map = categorical_maps[name]
            key = raw if raw in class_map else (str(raw) if str(raw) in class_map else None)
            if key is None:
                return None
            parsed[name] = int(class_map[key])
    return parsed


def _filter_dataframe(df, column: str, value):
    """Filter ``df`` to rows where ``column`` equals ``value`` (numeric-aware)."""
    as_numeric = pd.to_numeric(df[column], errors="coerce")
    try:
        numeric_value = float(value)
        if not as_numeric.isnull().all() and np.isfinite(numeric_value):
            return df[as_numeric == numeric_value].copy()
    except (TypeError, ValueError):
        pass
    return df[df[column].astype(str) == str(value)].copy()
