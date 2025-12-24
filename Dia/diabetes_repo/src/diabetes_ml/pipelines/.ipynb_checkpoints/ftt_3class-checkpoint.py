from __future__ import annotations

import os
import json
import time
import pickle
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from diabetes_ml.utils.seed import set_seed
from diabetes_ml.data.preprocess import (
    auto_detect_columns,
    fill_missing,
    encode_categoricals,
    split_train_val_test,
    fit_scaler,
    apply_scaler,
    make_stratified_kfold,
    save_cat_maps,
)
from diabetes_ml.data.tab_dataset import TabDataset
from diabetes_ml.models.ft_transformer import FTTransformer
from diabetes_ml.train.trainer import TrainConfig, train_with_early_stopping, predict_logits_proba
from diabetes_ml.metrics.metrics import summarize_metrics, brier_multiclass
from diabetes_ml.calibration.temperature import fit_temperature
from diabetes_ml.fairness.group_metrics import group_metrics_ovr
from diabetes_ml.calibration.uncertainty import (
    identify_uncertain_samples,
    calculate_uncertainty_statistics
)


def _timestamp():
    return time.strftime("%Y%m%d_%H%M%S")


def _build_loader(ds, batch_size: int, shuffle: bool, num_workers: int, pin_memory: bool):
    kwargs = dict(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(ds, **kwargs)


def run(cfg: Dict[str, Any]) -> str:
    seed = int(cfg["seed"])
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = bool(device.type == "cuda")

    out_dir = os.path.join(cfg["output"]["root_dir"], cfg["output"]["run_name"], _timestamp())
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    # --- read data ---
    data_cfg = cfg["data"]
    df = pd.read_csv(data_cfg["csv_path"])
    target_col = data_cfg["target_col"]
    assert target_col in df.columns, f"Target column {target_col} not found."

    if not np.issubdtype(df[target_col].dtype, np.integer):
        df[target_col] = df[target_col].round().astype(int)

    # --- columns ---
    if bool(data_cfg.get("auto_detect_cols", True)):
        cat_cols, num_cols = auto_detect_columns(df, target_col, int(data_cfg.get("categorical_threshold", 20)))
    else:
        cat_cols = list(data_cfg.get("cat_cols", []))
        num_cols = list(data_cfg.get("num_cols", []))

    group_cols = data_cfg.get("group_cols", {}) or {}
    for g in group_cols.values():
        if isinstance(g, str) and g in df.columns and g != target_col and g not in cat_cols:
            cat_cols.append(g)

    # --- missing ---
    df = fill_missing(
        df,
        cat_cols,
        num_cols,
        cat_missing_token=data_cfg.get("cat_missing_token", "<MISSING>"),
        num_missing_strategy=data_cfg.get("num_missing_strategy", "median"),
    )

    # --- encode categoricals ---
    df, cat_maps, cat_cardinalities = encode_categoricals(df, cat_cols)

    n_classes = int(cfg["model"]["n_classes"])
    assert int(df[target_col].nunique()) == n_classes, "Class count mismatch."

    # --- output: save cat maps now ---
    save_cat_maps(cat_maps, os.path.join(out_dir, "cat_maps.json"))

    # --- choose mode: CV or split ---
    cv_cfg = cfg.get("cv", {}) or {}
    use_cv = bool(cv_cfg.get("enabled", False))

    if not use_cv:
        return _run_single_split(cfg, df, cat_cols, num_cols, cat_cardinalities, group_cols, out_dir, device, pin_memory)

    return _run_cv_oof(cfg, df, cat_cols, num_cols, cat_cardinalities, group_cols, out_dir, device, pin_memory)


def _make_model(cfg, cat_cardinalities, num_cols):
    mcfg = cfg["model"]
    return FTTransformer(
        cat_cardinalities,
        num_cols,
        d_model=int(mcfg["d_model"]),
        n_heads=int(mcfg["n_heads"]),
        n_layers=int(mcfg["n_layers"]),
        ff_mult=int(mcfg["ff_mult"]),
        dropout=float(mcfg["dropout"]),
        n_classes=int(mcfg["n_classes"]),
    )


def _train_cfg(cfg) -> TrainConfig:
    tcfg = cfg["train"]
    return TrainConfig(
        lr=float(tcfg["lr"]),
        batch_size=int(tcfg["batch_size"]),
        epochs=int(tcfg["epochs"]),
        patience=int(tcfg["patience"]),
        weight_decay=float(tcfg["weight_decay"]),
        label_smoothing=float(tcfg["label_smoothing"]),
        grad_clip=float(tcfg["grad_clip"]),
        amp=bool(tcfg["amp"]),
    )


def _run_single_split(cfg, df, cat_cols, num_cols, cat_cardinalities, group_cols, out_dir, device, pin_memory) -> str:
    data_cfg = cfg["data"]
    train_df, val_df, test_df = split_train_val_test(
        df,
        data_cfg["target_col"],
        int(cfg["seed"]),
        test_size=float(data_cfg["split"]["test_size"]),
        val_size_in_temp=float(data_cfg["split"]["val_size_in_temp"]),
        stratify=bool(data_cfg["split"].get("stratify", True)),
    )

    # scaler only fit on train
    scaler = fit_scaler(train_df, num_cols, cfg["preprocess"]["scaler"])
    train_df = apply_scaler(train_df, num_cols, scaler)
    val_df = apply_scaler(val_df, num_cols, scaler)
    test_df = apply_scaler(test_df, num_cols, scaler)

    # save scaler
    if scaler is not None:
        import joblib
        joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))

    # loaders
    tcfg = cfg["train"]
    num_workers = int(tcfg["num_workers"])
    bs = int(tcfg["batch_size"])

    tr_ds = TabDataset(train_df, cat_cols, num_cols, data_cfg["target_col"], group_cols)
    va_ds = TabDataset(val_df, cat_cols, num_cols, data_cfg["target_col"], group_cols)
    te_ds = TabDataset(test_df, cat_cols, num_cols, data_cfg["target_col"], group_cols)

    tr_loader = _build_loader(tr_ds, bs, True, num_workers, pin_memory)
    va_loader = _build_loader(va_ds, bs, False, num_workers, pin_memory)
    te_loader = _build_loader(te_ds, bs, False, num_workers, pin_memory)

    # model + train
    model = _make_model(cfg, cat_cardinalities, num_cols).to(device)
    scheduler_factory = lambda opt: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(tcfg["epochs"]))

    model, _ = train_with_early_stopping(
        model,
        tr_loader,
        va_loader,
        device,
        _train_cfg(cfg),
        int(cfg["model"]["n_classes"]),
        y_train=train_df[data_cfg["target_col"]].values,
        metric_fn=lambda y, p: brier_multiclass(y, p),
        scheduler_factory=scheduler_factory,
    )

    # save model
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))

    # logits/proba
    val_logits, val_y, val_proba = predict_logits_proba(model, va_loader, device)
    test_logits, test_y, test_proba = predict_logits_proba(model, te_loader, device)

    # calibration (temperature)
    cal_cfg = cfg.get("calibration", {}) or {}
    if cal_cfg.get("method", "temperature") == "temperature":
        T_scaler = fit_temperature(
            torch.tensor(val_logits, device=device),
            torch.tensor(val_y, device=device, dtype=torch.long),
            lbfgs_steps=int(cal_cfg.get("lbfgs_steps", 10)),
        )
        T = float(torch.exp(T_scaler.log_t).detach().cpu().item())
        print("Fitted temperature T =", T)
        with open(os.path.join(out_dir, "temperature.pkl"), "wb") as f:
            pickle.dump({"log_t": T_scaler.log_t.detach().cpu().numpy()}, f)

        with torch.no_grad():
            scaled = T_scaler(torch.tensor(test_logits, device=device)).cpu()
        test_proba = torch.softmax(scaled, dim=1).numpy()

    # metrics
    ece_bins = int(cfg["eval"]["ece_bins"])
    metrics = {
        "val": summarize_metrics(val_y, val_proba, n_bins=ece_bins),
        "test": summarize_metrics(test_y, test_proba, n_bins=ece_bins),
    }

    # save proba csv (必须先创建 out_df)
    out_df = test_df[[data_cfg["target_col"]]].copy().reset_index(drop=True)
    for k in range(int(cfg["model"]["n_classes"])):
        out_df[f"p_{k}"] = test_proba[:, k]
    for gname, col in group_cols.items():
        if isinstance(col, str) and col in test_df.columns:
            out_df[gname] = test_df[col].values
    out_df.to_csv(os.path.join(out_dir, "test_proba_with_groups.csv"), index=False)

    # ⭐ 新增：不确定性分析 (必须在 out_df 创建之后)
    uncertainty_cfg = cfg.get("uncertainty", {}) or {}
    uncertainty_threshold = float(uncertainty_cfg.get("margin_threshold", 0.3))
    confidence_threshold = float(uncertainty_cfg.get("confidence_threshold", 0.6))

    # 在测试集上进行不确定性分析
    test_uncertain_mask, test_uncertainty_metrics = identify_uncertain_samples(
        test_proba, 
        uncertainty_threshold=uncertainty_threshold,
        confidence_threshold=confidence_threshold
    )

    # 计算不确定性统计
    uncertainty_stats = calculate_uncertainty_statistics(
        test_uncertain_mask,
        test_uncertainty_metrics,
        test_y
    )

    # 保存到metrics
    metrics['uncertainty'] = uncertainty_stats

    # 保存不确定样本的详细信息（供大模型使用）
    uncertain_samples_df = out_df[test_uncertain_mask].copy()
    uncertain_samples_df['entropy'] = test_uncertainty_metrics['entropy'][test_uncertain_mask]
    uncertain_samples_df['margin'] = test_uncertainty_metrics['margin'][test_uncertain_mask]
    uncertain_samples_df['max_prob'] = test_uncertainty_metrics['max_prob'][test_uncertain_mask]
    uncertain_samples_df.to_csv(
        os.path.join(out_dir, "uncertain_samples_for_llm.csv"), 
        index=False
    )

    # fairness (focus classes)
    fairness = {}
    tau = float(cfg["eval"].get("ovr_threshold", 0.5))
    focus_classes: List[int] = list(cfg["eval"].get("focus_classes", [1, 2]))
    for k in focus_classes:
        yk = (out_df[data_cfg["target_col"]].values == k).astype(int)
        pk = out_df[f"p_{k}"].values
        for gname in group_cols.keys():
            if gname in out_df.columns:
                gm, gaps = group_metrics_ovr(yk, pk, out_df[gname].values, tau=tau)
                fairness[f"class_{k}_by_{gname}"] = {"groups": gm, "gaps": gaps}
    metrics["fairness"] = fairness

    # ⭐ 新增：生成人类可读的评估报告
    _generate_readable_report(metrics, out_dir, "test")

    # 保存 metrics.json
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("[DONE] outputs saved to:", out_dir)
    return out_dir


def _run_cv_oof(cfg, df, cat_cols, num_cols, cat_cardinalities, group_cols, out_dir, device, pin_memory) -> str:
    data_cfg = cfg["data"]
    target_col = data_cfg["target_col"]
    n_classes = int(cfg["model"]["n_classes"])

    cv_cfg = cfg["cv"]
    n_splits = int(cv_cfg.get("n_splits", 5))
    shuffle = bool(cv_cfg.get("shuffle", True))

    oof_logits = np.zeros((len(df), n_classes), dtype=np.float32)
    oof_y = df[target_col].values.copy()
    oof_mask = np.zeros(len(df), dtype=bool)

    tcfg = cfg["train"]
    num_workers = int(tcfg["num_workers"])
    bs = int(tcfg["batch_size"])
    scheduler_factory = lambda opt: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(tcfg["epochs"]))

    for fold, tr_idx, va_idx in make_stratified_kfold(df, target_col, int(cfg["seed"]), n_splits=n_splits, shuffle=shuffle):
        fold_dir = os.path.join(out_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        train_df = df.iloc[tr_idx].copy()
        val_df = df.iloc[va_idx].copy()

        scaler = fit_scaler(train_df, num_cols, cfg["preprocess"]["scaler"])
        train_df = apply_scaler(train_df, num_cols, scaler)
        val_df = apply_scaler(val_df, num_cols, scaler)

        tr_ds = TabDataset(train_df, cat_cols, num_cols, target_col, group_cols)
        va_ds = TabDataset(val_df, cat_cols, num_cols, target_col, group_cols)

        tr_loader = _build_loader(tr_ds, bs, True, num_workers, pin_memory)
        va_loader = _build_loader(va_ds, bs, False, num_workers, pin_memory)

        model = _make_model(cfg, cat_cardinalities, num_cols).to(device)
        model, _ = train_with_early_stopping(
            model,
            tr_loader,
            va_loader,
            device,
            _train_cfg(cfg),
            n_classes,
            y_train=train_df[target_col].values,
            metric_fn=lambda y, p: brier_multiclass(y, p),
            scheduler_factory=scheduler_factory,
        )

        val_logits, val_y, _ = predict_logits_proba(model, va_loader, device)
        oof_logits[va_idx] = val_logits
        oof_mask[va_idx] = True

        torch.save(model.state_dict(), os.path.join(fold_dir, "model.pt"))
        np.save(os.path.join(fold_dir, "val_logits.npy"), val_logits)
        np.save(os.path.join(fold_dir, "val_y.npy"), val_y)

    assert oof_mask.all(), "OOF mask incomplete."

    # OOF temperature
    cal_cfg = cfg.get("calibration", {}) or {}
    if cal_cfg.get("method", "temperature") == "temperature":
        T_scaler = fit_temperature(
            torch.tensor(oof_logits, device=device),
            torch.tensor(oof_y, device=device, dtype=torch.long),
            lbfgs_steps=int(cal_cfg.get("lbfgs_steps", 10)),
        )
        T = float(torch.exp(T_scaler.log_t).detach().cpu().item())
        print("Fitted OOF temperature T =", T)
        with open(os.path.join(out_dir, "temperature.pkl"), "wb") as f:
            pickle.dump({"log_t": T_scaler.log_t.detach().cpu().numpy()}, f)

        with torch.no_grad():
            scaled = T_scaler(torch.tensor(oof_logits, device=device)).cpu()
        oof_proba = torch.softmax(scaled, dim=1).numpy()
    else:
        oof_proba = torch.softmax(torch.tensor(oof_logits), dim=1).numpy()

    # 基础 metrics
    metrics = {"oof": summarize_metrics(oof_y, oof_proba, n_bins=int(cfg["eval"]["ece_bins"]))}
    
    # ⭐ 新增：OOF 不确定性分析
    uncertainty_cfg = cfg.get("uncertainty", {}) or {}
    uncertainty_threshold = float(uncertainty_cfg.get("margin_threshold", 0.3))
    confidence_threshold = float(uncertainty_cfg.get("confidence_threshold", 0.6))

    oof_uncertain_mask, oof_uncertainty_metrics = identify_uncertain_samples(
        oof_proba,
        uncertainty_threshold=uncertainty_threshold,
        confidence_threshold=confidence_threshold
    )

    uncertainty_stats = calculate_uncertainty_statistics(
        oof_uncertain_mask,
        oof_uncertainty_metrics,
        oof_y
    )

    metrics['uncertainty'] = uncertainty_stats

    # 保存 OOF 不确定样本
    oof_df = df[[target_col]].copy()
    for k in range(n_classes):
        oof_df[f'p_{k}'] = oof_proba[:, k]
    oof_df['uncertain'] = oof_uncertain_mask
    oof_df['entropy'] = oof_uncertainty_metrics['entropy']
    oof_df['margin'] = oof_uncertainty_metrics['margin']
    oof_df['max_prob'] = oof_uncertainty_metrics['max_prob']

    uncertain_oof_df = oof_df[oof_uncertain_mask]
    uncertain_oof_df.to_csv(
        os.path.join(out_dir, "oof_uncertain_samples_for_llm.csv"),
        index=False
    )

    # 保存所有的 OOF 结果
    np.save(os.path.join(out_dir, "oof_logits.npy"), oof_logits)
    np.save(os.path.join(out_dir, "oof_y.npy"), oof_y)
    np.save(os.path.join(out_dir, "oof_proba.npy"), oof_proba)

    # ⭐ 新增：生成人类可读的报告
    _generate_readable_report(metrics, out_dir, "oof")

    # 保存 metrics.json
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("[DONE] CV outputs saved to:", out_dir)
    return out_dir


def _generate_readable_report(metrics: Dict[str, Any], out_dir: str, split_name: str = "test"):
    """
    生成人类可读的评估报告（满足 DoD 要求）
    
    Args:
        metrics: 评估指标字典
        out_dir: 输出目录
        split_name: 数据集名称（test/oof/val）
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append(f"模型评估报告 - {split_name.upper()} SET")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # 获取对应的metrics
    if split_name in metrics:
        m = metrics[split_name]
    else:
        m = metrics
    
    # 1. 整体性能指标
    report_lines.append("【整体性能指标】")
    report_lines.append(f"  准确率 (Accuracy):        {m.get('accuracy', 0):.4f}")
    report_lines.append(f"  Macro-F1:                 {m.get('macro_f1', 0):.4f}")
    report_lines.append(f"  Macro-Precision:          {m.get('macro_precision', 0):.4f}")
    report_lines.append(f"  Macro-Recall:             {m.get('macro_recall', 0):.4f}")
    report_lines.append(f"  AUC (OvR):                {m.get('auc_ovr_macro', 0):.4f}")
    report_lines.append(f"  Brier Score:              {m.get('brier', 0):.4f}")
    report_lines.append(f"  ECE (Expected Cal Error): {m.get('ece_ovr_avg', 0):.4f}")
    report_lines.append("")
    
    # 2. 每个类别的详细指标
    if 'per_class' in m:
        report_lines.append("【各类别详细指标】")
        report_lines.append(f"{'类别':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        report_lines.append("-" * 60)
        
        for class_key, class_metrics in m['per_class'].items():
            class_id = class_key.split('_')[1]
            class_name = f"Class {class_id}"
            report_lines.append(
                f"{class_name:<10} "
                f"{class_metrics['precision']:<12.4f} "
                f"{class_metrics['recall']:<12.4f} "
                f"{class_metrics['f1']:<12.4f} "
                f"{class_metrics['support']:<10}"
            )
        report_lines.append("")
    
    # 3. 混淆矩阵
    if 'confusion_matrix' in m:
        report_lines.append("【混淆矩阵】")
        cm = np.array(m['confusion_matrix'])
        n_classes = cm.shape[0]
        
        # 表头
        header = "真实\\预测 "
        for i in range(n_classes):
            header += f"Class_{i:<8}"
        report_lines.append(header)
        report_lines.append("-" * (12 + 13 * n_classes))
        
        # 每一行
        for i in range(n_classes):
            row = f"Class_{i:<6}"
            for j in range(n_classes):
                row += f"{cm[i, j]:<13}"
            report_lines.append(row)
        report_lines.append("")
    
    # 4. 不确定性分析
    if 'uncertainty' in metrics:
        unc = metrics['uncertainty']
        report_lines.append("【不确定性分析】")
        report_lines.append(f"  总样本数:                 {unc['n_total_samples']}")
        report_lines.append(f"  不确定样本数:             {unc['n_uncertain_samples']}")
        report_lines.append(f"  不确定样本比例:           {unc['uncertain_ratio']:.2%}")
        report_lines.append(f"  平均预测熵:               {unc['mean_entropy']:.4f}")
        report_lines.append(f"  平均边界置信度:           {unc['mean_margin']:.4f}")
        report_lines.append(f"  平均最高预测概率:         {unc['mean_max_prob']:.4f}")
        report_lines.append("")
        report_lines.append("  不确定样本的统计:")
        unc_samples = unc['uncertain_samples']
        report_lines.append(f"    平均熵:                 {unc_samples['mean_entropy']:.4f}")
        report_lines.append(f"    平均边界:               {unc_samples['mean_margin']:.4f}")
        report_lines.append(f"    平均最高概率:           {unc_samples['mean_max_prob']:.4f}")
        
        if 'class_distribution' in unc:
            report_lines.append("")
            report_lines.append("  不确定样本的类别分布:")
            for class_key, count in unc['class_distribution']['uncertain_samples'].items():
                report_lines.append(f"    {class_key}: {count}")
        report_lines.append("")
    
    # 5. 结论和建议
    report_lines.append("【结论与建议】")
    
    # 根据指标给出建议
    if m.get('macro_f1', 0) >= 0.8:
        report_lines.append("  ✓ 模型整体性能良好 (Macro-F1 >= 0.8)")
    elif m.get('macro_f1', 0) >= 0.6:
        report_lines.append("  ⚠ 模型性能中等 (0.6 <= Macro-F1 < 0.8)，建议进一步优化")
    else:
        report_lines.append("  ✗ 模型性能较差 (Macro-F1 < 0.6)，需要重新设计")
    
    if 'uncertainty' in metrics:
        ratio = metrics['uncertainty']['uncertain_ratio']
        if ratio > 0.3:
            report_lines.append(f"  ⚠ 不确定样本比例较高 ({ratio:.1%})，大模型介入可显著提升可靠性")
        else:
            report_lines.append(f"  ✓ 不确定样本比例合理 ({ratio:.1%})")
    
    if m.get('ece_ovr_avg', 1) < 0.05:
        report_lines.append("  ✓ 模型校准良好 (ECE < 0.05)，预测概率可信")
    elif m.get('ece_ovr_avg', 1) < 0.10:
        report_lines.append("  ⚠ 模型校准一般 (ECE < 0.10)，建议使用温度缩放")
    else:
        report_lines.append("  ✗ 模型校准较差 (ECE >= 0.10)，预测概率不够可靠")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    # 保存报告
    report_path = os.path.join(out_dir, f"evaluation_report_{split_name}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    # 同时打印到控制台
    print("\n".join(report_lines))