"""Data cleaning, splitting, imputation, and scaling for the IDS project."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names, normalize labels, and remove rows with missing labels.

    Infinite values in feature columns become NaN and are imputed later using
    statistics learned from the training split only.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    if "Label" not in df.columns:
        raise ValueError("The dataset must contain a 'Label' target column.")

    # Keep labels as strings, trim whitespace, and treat blank labels as missing.
    df["Label"] = df["Label"].astype("string").str.strip()
    df["Label"] = df["Label"].replace("", pd.NA)
    df = df.dropna(subset=["Label"]).copy()

    # Convert positive/negative infinity in numeric feature columns to NaN.
    numeric_cols = df.drop(columns=["Label"]).select_dtypes(include=[np.number]).columns
    df.loc[:, numeric_cols] = df.loc[:, numeric_cols].replace([np.inf, -np.inf], np.nan)

    return df


def summarize_missing_values(df: pd.DataFrame, top_n: int = 20) -> dict:
    """Summarize missing feature values after normalizing infinities to NaN."""
    if top_n < 0:
        raise ValueError("top_n must be non-negative.")

    features = df.drop(columns=["Label"], errors="ignore").select_dtypes(include=[np.number])
    features = features.replace([np.inf, -np.inf], np.nan)
    std = features.std()
    features = features.loc[:, std[std > 0].index]

    return {
        "shape": features.shape,
        "rows_with_missing": int(features.isna().any(axis=1).sum()),
        "missing_by_column": features.isna().sum().sort_values(ascending=False).head(top_n),
    }


def encode_target(y: pd.Series):
    """Encode target class names as integers; return encoded labels and encoder."""
    encoder = LabelEncoder()
    y_encoded = pd.Series(encoder.fit_transform(y.astype(str)), index=y.index, name="Label")
    return y_encoded, encoder


def split_data(df: pd.DataFrame, test_size: float = 0.15,
               val_size: float = 0.15, random_state: int = 42):
    """Create stratified train/validation/test splits.

    `test_size` and `val_size` are fractions of the complete dataset. The
    default gives 70% train, 15% validation, and 15% test.
    """
    if not 0 < test_size < 1 or not 0 < val_size < 1 or test_size + val_size >= 1:
        raise ValueError("test_size and val_size must be positive and sum to less than 1.")

    df = clean_dataset(df)
    y, label_encoder = encode_target(df["Label"])
    X = df.drop(columns=["Label"]).select_dtypes(include=[np.number]).copy()

    # Drop columns that are entirely missing before splitting. Their removal
    # is deterministic because they contain no usable values in the dataset.
    all_missing = X.columns[X.isna().all()].tolist()
    X = X.drop(columns=all_missing)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size + val_size,
        random_state=random_state, stratify=y
    )
    relative_test_size = test_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test_size,
        random_state=random_state, stratify=y_temp
    )

    # Remove zero-variance / all-missing columns based on training data only.
    std = X_train.std(numeric_only=True)
    valid_columns = std[std > 0].index.tolist()
    X_train = X_train[valid_columns].copy()
    X_val = X_val[valid_columns].copy()
    X_test = X_test[valid_columns].copy()

    return X_train, X_val, X_test, y_train, y_val, y_test, label_encoder


def impute_missing_values(X_train: pd.DataFrame, X_val: pd.DataFrame,
                          X_test: pd.DataFrame):
    """Median-impute numerical features, fitting the imputer on train only."""
    imputer = SimpleImputer(strategy="median")
    train_values = imputer.fit_transform(X_train)
    val_values = imputer.transform(X_val)
    test_values = imputer.transform(X_test)

    X_train_out = pd.DataFrame(train_values, columns=X_train.columns, index=X_train.index)
    X_val_out = pd.DataFrame(val_values, columns=X_train.columns, index=X_val.index)
    X_test_out = pd.DataFrame(test_values, columns=X_train.columns, index=X_test.index)
    return X_train_out, X_val_out, X_test_out, imputer


def scale_features(X_train: pd.DataFrame, X_val: pd.DataFrame,
                   X_test: pd.DataFrame):
    """Fit StandardScaler on training features and transform validation/test."""
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(X_train)
    val_scaled = scaler.transform(X_val)
    test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(train_scaled, columns=X_train.columns, index=X_train.index)
    X_val_scaled = pd.DataFrame(val_scaled, columns=X_train.columns, index=X_val.index)
    X_test_scaled = pd.DataFrame(test_scaled, columns=X_train.columns, index=X_test.index)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def save_preprocessing_artifacts(selected_features: list[str], scaler: StandardScaler,
                                 label_encoder: LabelEncoder,
                                 output_dir: str | Path) -> None:
    """Save selected feature names, fitted scaler, and label encoder."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "selected_features.json").open("w", encoding="utf-8") as file:
        json.dump(selected_features, file, indent=4)
    joblib.dump(scaler, output_dir / "scaler.pkl")
    joblib.dump(label_encoder, output_dir / "label_encoder.pkl")
