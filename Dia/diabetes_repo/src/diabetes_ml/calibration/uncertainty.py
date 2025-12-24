"""
不确定性量化模块
用于识别模型预测不确定的样本，这些样本需要大模型进行二次审核
"""

from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
from scipy.stats import entropy


def predictive_entropy(proba: np.ndarray) -> np.ndarray:
    """
    计算预测熵 - 衡量模型对每个样本的不确定性
    熵越高，模型越不确定
    
    Args:
        proba: 预测概率 [n_samples, n_classes]
    
    Returns:
        entropy_scores: 每个样本的熵值 [n_samples]
    """
    # 避免 log(0)
    proba_safe = np.clip(proba, 1e-10, 1.0)
    return entropy(proba_safe, axis=1)


def margin_confidence(proba: np.ndarray) -> np.ndarray:
    """
    计算边界置信度 - 最高概率和第二高概率的差值
    差值越小，说明模型在两个类别间犹豫不决
    
    Args:
        proba: 预测概率 [n_samples, n_classes]
    
    Returns:
        margins: 每个样本的边界置信度 [n_samples]
    """
    sorted_probs = np.sort(proba, axis=1)
    # 最高概率 - 第二高概率
    return sorted_probs[:, -1] - sorted_probs[:, -2]


def identify_uncertain_samples(
    proba: np.ndarray, 
    uncertainty_threshold: float = 0.3,
    confidence_threshold: float = 0.6
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    识别不确定样本 - 这些样本应该触发大模型审核
    
    不确定性判定规则：
    1. 最高预测概率 < confidence_threshold (模型整体不自信)
    2. 边界置信度 < uncertainty_threshold (两个类别概率接近)
    
    Args:
        proba: 预测概率 [n_samples, n_classes]
        uncertainty_threshold: 边界阈值，默认0.3
        confidence_threshold: 置信度阈值，默认0.6
    
    Returns:
        uncertain_mask: 布尔数组，True表示不确定样本 [n_samples]
        uncertainty_metrics: 包含各种不确定性指标的字典
    """
    # 计算各种不确定性指标
    entropies = predictive_entropy(proba)
    margins = margin_confidence(proba)
    max_probs = np.max(proba, axis=1)
    
    # 判定不确定样本
    # 条件1: 最高概率太低（模型不自信）
    low_confidence_mask = max_probs < confidence_threshold
    
    # 条件2: 边界太小（两个类别概率接近，难以区分）
    low_margin_mask = margins < uncertainty_threshold
    
    # 满足任一条件即为不确定样本
    uncertain_mask = low_confidence_mask | low_margin_mask
    
    # 返回详细的不确定性指标
    uncertainty_metrics = {
        'entropy': entropies,           # 预测熵
        'margin': margins,              # 边界置信度
        'max_prob': max_probs,          # 最高预测概率
        'predictions': np.argmax(proba, axis=1)  # 预测类别
    }
    
    return uncertain_mask, uncertainty_metrics


def calculate_uncertainty_statistics(
    uncertain_mask: np.ndarray,
    uncertainty_metrics: Dict[str, np.ndarray],
    y_true: np.ndarray = None
) -> Dict[str, any]:
    """
    计算不确定性统计信息
    
    Args:
        uncertain_mask: 不确定样本的掩码
        uncertainty_metrics: 不确定性指标
        y_true: 真实标签（可选，用于分析不确定样本的真实类别分布）
    
    Returns:
        dict: 统计信息
    """
    n_total = len(uncertain_mask)
    n_uncertain = int(uncertain_mask.sum())
    
    stats = {
        'n_total_samples': n_total,
        'n_uncertain_samples': n_uncertain,
        'uncertain_ratio': float(n_uncertain / n_total),
        'mean_entropy': float(uncertainty_metrics['entropy'].mean()),
        'mean_margin': float(uncertainty_metrics['margin'].mean()),
        'mean_max_prob': float(uncertainty_metrics['max_prob'].mean()),
        'uncertain_samples': {
            'mean_entropy': float(uncertainty_metrics['entropy'][uncertain_mask].mean()) if n_uncertain > 0 else 0.0,
            'mean_margin': float(uncertainty_metrics['margin'][uncertain_mask].mean()) if n_uncertain > 0 else 0.0,
            'mean_max_prob': float(uncertainty_metrics['max_prob'][uncertain_mask].mean()) if n_uncertain > 0 else 0.0,
        }
    }
    
    # 如果提供了真实标签，分析不确定样本的类别分布
    if y_true is not None:
        uncertain_y = y_true[uncertain_mask]
        certain_y = y_true[~uncertain_mask]
        
        stats['class_distribution'] = {
            'uncertain_samples': {
                f'class_{k}': int((uncertain_y == k).sum()) 
                for k in np.unique(y_true)
            },
            'certain_samples': {
                f'class_{k}': int((certain_y == k).sum()) 
                for k in np.unique(y_true)
            }
        }
    
    return stats