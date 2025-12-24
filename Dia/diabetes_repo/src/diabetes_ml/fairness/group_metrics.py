from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np


def group_metrics_ovr(
    y_true_bin: np.ndarray,
    y_prob: np.ndarray,
    groups: np.ndarray,
    tau: float = 0.5
) -> Tuple[Dict[int, Any], Dict[str, float]]:
    """
    OvR（二分类）在不同组上的 TPR/FPR/PPV/PPR + 组间差距 ΔTPR/ΔPPR
    y_true_bin: (N,) 0/1
    y_prob: (N,) 概率
    groups: (N,) 分组编号
    """
    y_pred = (y_prob >= tau).astype(int)
    out: Dict[int, Any] = {}

    for g in np.unique(groups):
        m = (groups == g)
        yt, yh = y_true_bin[m], y_pred[m]

        TP = int(((yh == 1) & (yt == 1)).sum())
        FP = int(((yh == 1) & (yt == 0)).sum())
        TN = int(((yh == 0) & (yt == 0)).sum())
        FN = int(((yh == 0) & (yt == 1)).sum())

        tpr = TP / (TP + FN) if (TP + FN) > 0 else np.nan
        fpr = FP / (FP + TN) if (FP + TN) > 0 else np.nan
        ppv = TP / (TP + FP) if (TP + FP) > 0 else np.nan
        ppr = float((yh == 1).mean())

        out[int(g)] = {
            "TPR": float(tpr),
            "FPR": float(fpr),
            "PPV": float(ppv),
            "PPR": float(ppr),
            "pos": int(yt.sum()),
            "n": int(len(yt)),
        }

    tprs = [v["TPR"] for v in out.values() if not np.isnan(v["TPR"])]
    pprs = [v["PPR"] for v in out.values() if not np.isnan(v["PPR"])]

    gaps = {
        "ΔTPR": float(np.max(tprs) - np.min(tprs)) if len(tprs) > 0 else float("nan"),
        "ΔPPR": float(np.max(pprs) - np.min(pprs)) if len(pprs) > 0 else float("nan"),
    }
    return out, gaps
