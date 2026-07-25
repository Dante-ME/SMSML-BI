"""Automated preprocessing pipeline for the raw dataset in ``data/raw/``.

This script is the programmatic twin of ``experiment/notebooks/Eksperimen.ipynb``:
both use the exact same preprocessing logic, which lives here so the notebook can
simply import it.

Run it from the project root (or anywhere, paths are resolved automatically):

    python experiment/automate.py

Optional overrides::

    python experiment/automate.py --input data/raw/my.csv --target DEATH_EVENT
    python experiment/automate.py --test-size 0.25 --random-state 7

Outputs written to ``data/processed/``:
    * ``heart_failure_preprocessed.csv`` - full preprocessed dataset (features + target)
    * ``train.csv`` / ``test.csv``       - stratified train/test split
    * ``preprocessor.joblib``            - the fitted sklearn transformer (fit on train only)
    * ``metadata.json``                  - target column, feature groups, shapes, seed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Fixed seed everywhere so every run of this script yields identical files.
RANDOM_STATE = 42
TEST_SIZE = 0.2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Names commonly used for a supervised target, checked in priority order.
TARGET_NAME_CANDIDATES = (
    "death_event",
    "target",
    "label",
    "class",
    "outcome",
    "diagnosis",
    "y",
)

# Extensions we know how to read from data/raw/.
SUPPORTED_SUFFIXES = (".csv", ".xlsx", ".xls")


# --------------------------------------------------------------------------- #
# Step 1 - locate and load the raw dataset
# --------------------------------------------------------------------------- #


def find_dataset(raw_dir: Path = RAW_DIR) -> Path:
    """Return the dataset file inside ``raw_dir``, chosen automatically.

    If several candidates exist the largest one is used (the archive/zip and
    other side files are ignored because their suffix is not supported).
    """
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    candidates = sorted(
        (p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES),
        key=lambda p: p.stat().st_size,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No dataset with extension {SUPPORTED_SUFFIXES} found in {raw_dir}"
        )
    if len(candidates) > 1:
        print(f"[info] {len(candidates)} candidate files found, picking the largest one.")
    return candidates[0]


def load_dataset(path: Path) -> pd.DataFrame:
    """Read a csv/excel file into a DataFrame and validate it is usable."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    if df.empty:
        raise ValueError(f"Dataset is empty: {path}")
    if df.shape[1] < 2:
        raise ValueError(
            f"Dataset needs at least one feature and one target column, got {df.shape[1]} column(s)."
        )
    return df


# --------------------------------------------------------------------------- #
# Step 2 - clean the table itself (column names, duplicates, empty columns)
# --------------------------------------------------------------------------- #


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names to lowercase snake_case.

    Only cosmetic: ``DEATH_EVENT`` -> ``death_event``. This keeps every later
    reference (notebook, training, serving) case-insensitive and typo-proof.
    """
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(r"[^0-9a-zA-Z]+", "_", regex=True)
        .str.strip("_")
        .str.lower()
    )
    if df.columns.duplicated().any():
        raise ValueError(f"Duplicated column names after standardisation: {list(df.columns)}")
    return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise missing markers, then drop all-empty columns and duplicate rows."""
    df = df.copy()

    # Text columns: trim whitespace and turn ``""``/``None`` into a real NaN so
    # the imputers downstream recognise them as missing.
    text_cols = df.select_dtypes(include="object").columns
    for col in text_cols:
        df[col] = df[col].astype(object).where(df[col].notna(), np.nan)
        df[col] = df[col].apply(lambda v: v.strip() if isinstance(v, str) else v)
        df[col] = df[col].replace("", np.nan)

    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        print(f"[clean] dropping {len(empty_cols)} all-empty column(s): {empty_cols}")
        df = df.drop(columns=empty_cols)

    n_dupes = int(df.duplicated().sum())
    if n_dupes:
        print(f"[clean] dropping {n_dupes} duplicated row(s)")
        df = df.drop_duplicates().reset_index(drop=True)

    if df.empty:
        raise ValueError("No rows left after cleaning.")
    return df


# --------------------------------------------------------------------------- #
# Step 3 - target detection
# --------------------------------------------------------------------------- #


def detect_target(df: pd.DataFrame, explicit: str | None = None) -> str:
    """Return the name of the target column.

    Detection order:
      1. ``explicit`` if the user passed ``--target`` (validated against the frame);
      2. a column whose standardized name matches a known target keyword;
      3. fallback: the last column, which is the near-universal convention for
         tabular ML datasets.
    """
    if explicit:
        key = explicit.strip().lower()
        if key not in df.columns:
            raise ValueError(f"Target column '{explicit}' not found. Available: {list(df.columns)}")
        return key

    for candidate in TARGET_NAME_CANDIDATES:
        if candidate in df.columns:
            print(f"[target] detected by name match: '{candidate}'")
            return candidate

    fallback = df.columns[-1]
    print(f"[target] no keyword match, falling back to the last column: '{fallback}'")
    return fallback


def is_classification(y: pd.Series) -> bool:
    """Treat the task as classification when the target is discrete and low-cardinality."""
    if y.dtype == object or str(y.dtype) in {"category", "bool"}:
        return True
    return y.nunique(dropna=True) <= 20


# --------------------------------------------------------------------------- #
# Step 4 - feature typing and the preprocessing transformer
# --------------------------------------------------------------------------- #


def split_feature_types(X: pd.DataFrame) -> dict[str, list[str]]:
    """Group feature columns into continuous / binary-numeric / categorical.

    The binary-numeric group matters: columns like ``sex`` or ``smoking`` are
    already 0/1 indicators, so they must be imputed but *not* scaled or encoded.
    """
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]

    binary_numeric = [c for c in numeric if X[c].dropna().nunique() <= 2]
    continuous = [c for c in numeric if c not in binary_numeric]

    return {
        "continuous": continuous,
        "binary": binary_numeric,
        "categorical": categorical,
    }


def build_preprocessor(groups: dict[str, list[str]]) -> ColumnTransformer:
    """Build the ColumnTransformer applied to the features.

    * continuous  -> median imputation (robust to the skewed lab values) + StandardScaler
    * binary 0/1  -> most-frequent imputation only (scaling an indicator adds nothing)
    * categorical -> most-frequent imputation + one-hot encoding
    """
    transformers = []

    if groups["continuous"]:
        transformers.append(
            (
                "continuous",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                groups["continuous"],
            )
        )

    if groups["binary"]:
        transformers.append(
            (
                "binary",
                Pipeline([("impute", SimpleImputer(strategy="most_frequent"))]),
                groups["binary"],
            )
        )

    if groups["categorical"]:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        # drop="if_binary" keeps 2-level categories as a single column;
                        # handle_unknown="ignore" makes inference safe on unseen levels.
                        (
                            "encode",
                            OneHotEncoder(
                                handle_unknown="ignore", drop="if_binary", sparse_output=False
                            ),
                        ),
                    ]
                ),
                groups["categorical"],
            )
        )

    if not transformers:
        raise ValueError("No usable feature columns were found.")

    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def transformed_frame(preprocessor: ColumnTransformer, X: pd.DataFrame) -> pd.DataFrame:
    """Apply a *fitted* preprocessor and return a DataFrame with readable names."""
    array = preprocessor.transform(X)
    names = [str(n) for n in preprocessor.get_feature_names_out()]
    return pd.DataFrame(array, columns=names, index=X.index)


# --------------------------------------------------------------------------- #
# Step 5 - the full pipeline
# --------------------------------------------------------------------------- #


def preprocess(
    df: pd.DataFrame,
    target: str | None = None,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> dict:
    """Run the whole preprocessing pipeline on a raw DataFrame.

    Returns a dict with the train/test frames, the fitted preprocessor and the
    metadata describing what was done.
    """
    df = basic_cleaning(standardize_columns(df))

    target_col = detect_target(df, target)
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # A row without a label cannot be used for supervised learning.
    labelled = y.notna()
    if not labelled.all():
        print(f"[clean] dropping {int((~labelled).sum())} row(s) with a missing target")
        X, y = X.loc[labelled], y.loc[labelled]
    if X.empty:
        raise ValueError("No labelled rows left after cleaning.")

    groups = split_feature_types(X)
    print(
        f"[features] continuous={len(groups['continuous'])} "
        f"binary={len(groups['binary'])} categorical={len(groups['categorical'])}"
    )

    # Stratify only when the task is classification and every class can be split.
    classification = is_classification(y)
    stratify = y if classification and y.value_counts().min() >= 2 else None
    if classification and stratify is None:
        print("[split] a class has fewer than 2 samples, stratification disabled")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    # Fit on the training split ONLY, then apply to both -> no data leakage.
    preprocessor = build_preprocessor(groups)
    preprocessor.fit(X_train)

    train = transformed_frame(preprocessor, X_train)
    test = transformed_frame(preprocessor, X_test)
    train[target_col] = y_train.to_numpy()
    test[target_col] = y_test.to_numpy()

    metadata = {
        "target": target_col,
        "task": "classification" if classification else "regression",
        "random_state": random_state,
        "test_size": test_size,
        "n_rows_raw": int(len(df)),
        "n_rows_train": int(len(train)),
        "n_rows_test": int(len(test)),
        "feature_groups": groups,
        "output_features": [c for c in train.columns if c != target_col],
    }
    return {
        "train": train,
        "test": test,
        "preprocessor": preprocessor,
        "metadata": metadata,
    }


def save_outputs(result: dict, output_dir: Path = PROCESSED_DIR) -> dict[str, Path]:
    """Persist the processed data, the fitted preprocessor and the metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)

    train, test = result["train"], result["test"]
    # Full preprocessed table = train + test, handy as a single training input.
    full = pd.concat([train, test], ignore_index=True)

    paths = {
        "full": output_dir / "heart_failure_preprocessed.csv",
        "train": output_dir / "train.csv",
        "test": output_dir / "test.csv",
        "preprocessor": output_dir / "preprocessor.joblib",
        "metadata": output_dir / "metadata.json",
    }

    full.to_csv(paths["full"], index=False)
    train.to_csv(paths["train"], index=False)
    test.to_csv(paths["test"], index=False)
    joblib.dump(result["preprocessor"], paths["preprocessor"])
    paths["metadata"].write_text(json.dumps(result["metadata"], indent=2), encoding="utf-8")
    return paths


def run(
    input_path: Path | None = None,
    target: str | None = None,
    output_dir: Path = PROCESSED_DIR,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> dict:
    """End-to-end: find -> load -> preprocess -> save. Returns the result dict."""
    dataset_path = Path(input_path) if input_path else find_dataset()
    print(f"[load] {dataset_path}")

    df = load_dataset(dataset_path)
    print(f"[load] shape={df.shape}, missing values={int(df.isna().sum().sum())}")

    result = preprocess(df, target=target, test_size=test_size, random_state=random_state)
    paths = save_outputs(result, output_dir)

    meta = result["metadata"]
    print(
        f"[done] target='{meta['target']}' task={meta['task']} "
        f"train={meta['n_rows_train']} test={meta['n_rows_test']} "
        f"features={len(meta['output_features'])}"
    )
    for name, path in paths.items():
        print(f"[saved] {name}: {path}")

    result["paths"] = paths
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=None, help="Raw dataset file (auto-detected by default).")
    parser.add_argument("--target", type=str, default=None, help="Target column (auto-detected by default).")
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR, help="Where to write the processed files.")
    parser.add_argument("--test-size", type=float, default=TEST_SIZE, help="Test split proportion.")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE, help="Random seed.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0.0 < args.test_size < 1.0:
        print(f"[error] --test-size must be between 0 and 1, got {args.test_size}", file=sys.stderr)
        return 2
    try:
        run(
            input_path=args.input,
            target=args.target,
            output_dir=args.output_dir,
            test_size=args.test_size,
            random_state=args.random_state,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
