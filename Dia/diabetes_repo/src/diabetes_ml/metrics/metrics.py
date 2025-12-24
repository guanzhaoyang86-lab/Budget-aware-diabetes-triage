from __future__ import annotations

from typing import Dict, Any
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def brier_multiclass(y_true: np.ndarray, proba: np.ndarray) -> float:
    n, k = proba.shape
    y_onehot = np.eye(k)[y_true]
    return float(np.mean(np.sum((y_onehot - proba) ** 2, axis=1)))


def ece_binary(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(y_prob, bins) - 1
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        acc = y_true[m].mean()
        conf = y_prob[m].mean()
        ece += (m.sum() / n) * abs(acc - conf)
    return float(ece)


def ece_multiclass_ovr(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    K = proba.shape[1]
    return float(np.mean([ece_binary((y_true == k).astype(int), proba[:, k], n_bins=n_bins) for k in range(K)]))


def macro_auc_ovr(y_true: np.ndarray, proba: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))
    except Exception:
        return float("nan")


def summarize_metrics(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
    """
    汇总所有评估指标
    
    Args:
        y_true: 真实标签
        proba: 预测概率
        n_bins: ECE计算的bins数量
    
    Returns:
        dict: 包含所有评估指标
    """
    y_pred = np.argmax(proba, axis=1)
    
    # 原有指标
    base_metrics = {
        "brier": brier_multiclass(y_true, proba),
        "auc_ovr_macro": macro_auc_ovr(y_true, proba),
        "ece_ovr_avg": ece_multiclass_ovr(y_true, proba, n_bins=n_bins),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    
    # 新增：详细的分类报告（包含 Macro-F1）
    classification_report = calculate_classification_report(y_true, proba)
    
    # 合并结果
    return {**base_metrics, **classification_report}

def calculate_classification_report(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, Any]:
    """
    计算详细的分类报告，包括每个类别的 Precision, Recall, F1
    以及 Macro 平均指标
    
    Args:
        y_true: 真实标签 [n_samples]
        proba: 预测概率 [n_samples, n_classes]
    
    Returns:
        dict: 包含详细分类指标的字典
    """
    y_pred = np.argmax(proba, axis=1)
    n_classes = proba.shape[1]
    
    # 计算每个类别的 P/R/F1
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Macro 平均
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    # Weighted 平均
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # 准确率
    acc = accuracy_score(y_true, y_pred)
    
    # 组装结果
    per_class_metrics = {}
    for k in range(n_classes):
        per_class_metrics[f'class_{k}'] = {
            'precision': float(precision[k]),
            'recall': float(recall[k]),
            'f1': float(f1[k]),
            'support': int(support[k])
        }
    
    return {
        'accuracy': float(acc),
        'macro_precision': float(precision_macro),
        'macro_recall': float(recall_macro),
        'macro_f1': float(f1_macro),  # ⭐ P0 要求的关键指标
        'weighted_precision': float(precision_weighted),
        'weighted_recall': float(recall_weighted),
        'weighted_f1': float(f1_weighted),
        'per_class': per_class_metrics
    }
