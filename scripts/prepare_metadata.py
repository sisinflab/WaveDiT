#!/usr/bin/env python3
"""Build a metadata CSV from folders of NIfTI files.

Recursively scans the given directories for ``*.nii.gz`` files, parses the age from
each filename (pattern ``[_-]AGE[_-]<number>``, case-insensitive) and writes a CSV
with columns ``SubjectID, FilePath, Age, Condition`` for use in CSV training mode.

    python scripts/prepare_metadata.py --input-dirs data/CN --output-csv data/catalog.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import logging
import os
import re

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Kept in sync with wavedit.data.dataset.AGE_PATTERN (duplicated to avoid importing torch).
AGE_PATTERN = re.compile(r"[_-]AGE[_-](\d+(?:\.\d+)?)", re.IGNORECASE)


def extract_age(filename: str) -> float | None:
    match = AGE_PATTERN.search(os.path.basename(filename))
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        logger.warning("Unparseable age '%s' in %s", match.group(1), filename)
        return None


def extract_subject_id(stem: str, regex: re.Pattern | None) -> str:
    if regex:
        match = regex.search(stem)
        if match:
            return match.group(1) if match.groups() else match.group(0)
    return stem


def main():
    parser = argparse.ArgumentParser(description="Create a metadata CSV from NIfTI directories.")
    parser.add_argument("--input-dirs", nargs="+", required=True, help="Directories to scan recursively.")
    parser.add_argument("--output-csv", required=True, help="Output CSV path.")
    parser.add_argument("--subject-id-regex", default=None,
                        help=r"Optional regex; the first group becomes SubjectID (e.g. '^(sub-\d+)_').")
    parser.add_argument("--condition-label", default="CN", help="Value written to the Condition column.")
    args = parser.parse_args()

    id_regex = None
    if args.subject_id_regex:
        try:
            id_regex = re.compile(args.subject_id_regex)
        except re.error as exc:
            logger.error("Invalid subject-id regex (%s); using filename as id.", exc)

    records, with_age = [], 0
    for input_dir in args.input_dirs:
        if not os.path.isdir(input_dir):
            logger.warning("Not a directory, skipping: %s", input_dir)
            continue
        files = glob.glob(os.path.join(input_dir, "**", "*.nii.gz"), recursive=True)
        for file_path in tqdm(files, desc=f"Scanning {os.path.basename(input_dir)}", unit="file"):
            stem = os.path.basename(file_path).replace(".nii.gz", "")
            age = extract_age(file_path)
            with_age += age is not None
            records.append({
                "SubjectID": extract_subject_id(stem, id_regex),
                "FilePath": os.path.abspath(file_path),
                "Age": age if age is not None else "",
                "Condition": args.condition_label,
            })

    if not records:
        logger.warning("No NIfTI files found; nothing written.")
        return

    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["SubjectID", "FilePath", "Age", "Condition"])
        writer.writeheader()
        writer.writerows(records)

    logger.info("Wrote %d records (%d with age) to %s", len(records), with_age, args.output_csv)


if __name__ == "__main__":
    main()
