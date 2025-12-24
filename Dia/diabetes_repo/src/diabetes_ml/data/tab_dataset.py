from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TabDataset(Dataset):
    """返回 cat(int64), num(float32), y(long), groups(dict)."""

    def __init__(
        self,
        data: pd.DataFrame,
        cat_cols: List[str],
        num_cols: List[str],
        target_col: str,
        group_cols: Optional[Dict[str, str]] = None,
    ):
        self.data = data.reset_index(drop=True)
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.target_col = target_col
        self.group_cols = group_cols or {}

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        cat = row[self.cat_cols].values.astype(np.int64) if self.cat_cols else np.zeros(0, dtype=np.int64)
        num = row[self.num_cols].values.astype(np.float32) if self.num_cols else np.zeros(0, dtype=np.float32)
        y = int(row[self.target_col])

        groups = {k: int(row[v]) if v in self.data.columns else -1 for k, v in self.group_cols.items()}
        return {
            "cat": torch.from_numpy(cat),
            "num": torch.from_numpy(num),
            "y": torch.tensor(y, dtype=torch.long),
            "groups": groups,
        }
