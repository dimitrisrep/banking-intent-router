"""Checksum-verified, pinned BANKING77 ingestion and duplicate isolation."""

import hashlib
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
from sklearn.model_selection import train_test_split

REVISION = "57ec275d8078af65b7731c2a98be812d844a6d6b"
CHECKSUMS = {
    "train": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
    "test": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
}


def verify(path: Path, expected: str):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(
            f"Checksum mismatch: {path.name}. Download the pinned dataset again."
        )


def download(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    for split, expected in CHECKSUMS.items():
        path = directory / f"{split}.csv"
        if not path.exists():
            url = f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{REVISION}/banking_data/{split}.csv"
            with urlopen(url, timeout=60) as response:
                content = response.read(2_000_000)
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError(f"Checksum mismatch for downloaded {split} data.")
            path.write_bytes(content)
        verify(path, expected)


def text_keys(series):
    return series.str.casefold().str.replace(r"\s+", " ", regex=True).str.strip()


def partition(train: pd.DataFrame, test: pd.DataFrame, seed: int = 42):
    """Remove cross-split text duplicates; never use test labels for selection."""
    train = train.assign(key=text_keys(train.text))
    test_keys = set(text_keys(test.text))
    if train.groupby("key").category.nunique().gt(1).any():
        raise ValueError("Conflicting labels for identical training text.")
    duplicates = int(train.key.duplicated().sum())
    train = train.drop_duplicates("key")
    overlaps = int(train.key.isin(test_keys).sum())
    train = train[~train.key.isin(test_keys)].copy()
    fit, remaining = train_test_split(
        train, test_size=0.4, stratify=train.category, random_state=seed
    )
    calibration, validation = train_test_split(
        remaining, test_size=0.5, stratify=remaining.category, random_state=seed
    )
    splits = {"fit": fit, "calibration": calibration, "validation": validation}
    for name, frame in splits.items():
        for other_name, other in splits.items():
            if name != other_name and set(frame.key) & set(other.key):
                raise AssertionError("Text leakage between development partitions.")
        if set(frame.key) & test_keys:
            raise AssertionError("Test text found in a development partition.")
    return splits, {
        "duplicate_train_rows_removed": duplicates,
        "test_overlap_train_rows_removed": overlaps,
        "development_rows_after_cleaning": len(train),
    }


def load(directory: Path):
    for split, expected in CHECKSUMS.items():
        verify(directory / f"{split}.csv", expected)
    return pd.read_csv(directory / "train.csv"), pd.read_csv(directory / "test.csv")
