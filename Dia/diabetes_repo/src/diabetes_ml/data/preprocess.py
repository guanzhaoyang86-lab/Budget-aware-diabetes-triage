from __future__ import annotations

import json
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler, StandardScaler


def auto_detect_columns(df: pd.DataFrame, target_col: str, threshold: int) -> Tuple[List[str], List[str]]:
    cat_cols, num_cols = [], []
    for col in df.columns:
        if col == target_col:
            continue
        nunq = df[col].nunique(dropna=True)
        if nunq <= threshold:
            cat_cols.append(col)
        else:
            num_cols.append(col)
    return cat_cols, num_cols


def fill_missing(
    df: pd.DataFrame,
    cat_cols: List[str],
    num_cols: List[str],
    cat_missing_token: str = "<MISSING>",
    num_missing_strategy: str = "median",
) -> pd.DataFrame:
    df = df.copy()
    for c in cat_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(cat_missing_token)
    for c in num_cols:
        if df[c].isna().any():
            if num_missing_strategy == "median":
                df[c] = df[c].fillna(df[c].median())
            elif num_missing_strategy == "mean":
                df[c] = df[c].fillna(df[c].mean())
            elif num_missing_strategy == "zero":
                df[c] = df[c].fillna(0.0)
            else:
                raise ValueError(f"Unknown num_missing_strategy: {num_missing_strategy}")
    return df


def encode_categoricals(
    df: pd.DataFrame, cat_cols: List[str]
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]], Dict[str, int]]:
    df = df.copy()
    cat_maps: Dict[str, Dict[str, int]] = {}
    cat_cardinalities: Dict[str, int] = {}

    for c in cat_cols:
        df[c] = df[c].astype("category")
        mapping = {str(cat): int(i) for i, cat in enumerate(df[c].cat.categories)}
        cat_maps[c] = mapping
        df[c] = df[c].cat.codes.astype(int)
        cat_cardinalities[c] = int(df[c].max()) + 1

    return df, cat_maps, cat_cardinalities


def fit_scaler(train_df: pd.DataFrame, num_cols: List[str], scaler_name: str):
    if len(num_cols) == 0 or scaler_name == "none":
        return None
    if scaler_name == "robust":
        scaler = RobustScaler()
    elif scaler_name == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unknown scaler: {scaler_name}")

    scaler.fit(train_df[num_cols])
    return scaler


def apply_scaler(df: pd.DataFrame, num_cols: List[str], scaler) -> pd.DataFrame:
    if scaler is None or len(num_cols) == 0:
        return df
    df = df.copy()
    df[num_cols] = scaler.transform(df[num_cols])
    return df


def split_train_val_test(
    df: pd.DataFrame,
    target_col: str,
    seed: int,
    test_size: float = 0.3,
    val_size_in_temp: float = 0.5,
    stratify: bool = True,
):
    strat = df[target_col] if stratify else None
    train_df, temp_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=strat)

    strat2 = temp_df[target_col] if stratify else None
    val_df, test_df = train_test_split(temp_df, test_size=val_size_in_temp, random_state=seed, stratify=strat2)
    return train_df, val_df, test_df


def make_stratified_kfold(df: pd.DataFrame, target_col: str, seed: int, n_splits: int = 5, shuffle: bool = True):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed if shuffle else None)
    X = np.zeros(len(df))
    y = df[target_col].values
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        yield fold, tr_idx, va_idx


def save_cat_maps(cat_maps: Dict[str, Dict[str, int]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cat_maps, f, ensure_ascii=False, indent=2)
