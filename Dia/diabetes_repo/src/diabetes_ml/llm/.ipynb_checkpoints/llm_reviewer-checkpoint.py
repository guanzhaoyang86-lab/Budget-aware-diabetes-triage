"""
使用大模型审核不确定样本
"""

import os
import sys
import json
import glob
from pathlib import Path

import pandas as pd

# ============ 路径配置 ============
# 项目根目录：.../糖尿病/Code open
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 确保 src 在 Python 搜索路径中
src_path = PROJECT_ROOT / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# 大模型路径：换成 32B 版本（老师说“你换32B”）
MODEL_PATH = "/hy-tmp/models/Qwen/Qwen2___5-32B-Instruct"

# 输出目录（大模型审核结果会存这里）
LLM_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "llm_reviews"
# ==================================

from src.diabetes_ml.llm.llm_reviewer import DiabetesLLMReviewer


def find_uncertain_samples_file() -> str | None:
    """
    在 outputs 目录里自动查找最近一次的
    uncertain_samples_for_llm.csv 文件
    """
    outputs_dir = PROJECT_ROOT / "outputs"

    if not outputs_dir.exists():
        print(f"❌ outputs 目录不存在: {outputs_dir}")
        return None

    # 匹配所有子目录里的 uncertain_samples_for_llm.csv
    pattern = str(outputs_dir / "**" / "uncertain_samples_for_llm.csv")
    files = glob.glob(pattern, recursive=True)

    if not files:
        print("❌ 未找到 uncertain_samples_for_llm.csv 文件")
        print(f"   查找模式: {pattern}")
        print("\n请先运行训练（python main.py --pipeline ftt_3class）生成不确定样本。")
        return None

    # 选修改时间最新的那一个
    latest_file = max(files, key=os.path.getmtime)
    print(f"✓ 使用最近的文件: {latest_file}")
    return latest_file


def run_llm_review(max_samples: int = 5) -> None:
    """主函数：运行大模型，对不确定样本做审核"""

    print("=" * 60)
    print("大模型审核系统 - Diabetes LLM Review System")
    print("=" * 60)

    # 1. 找不确定样本文件
    print("\n[1/5] 查找不确定样本文件...")
    csv_file = find_uncertain_samples_file()
    if csv_file is None:
        return

    # 2. 读取样本
    print("\n[2/5] 读取样本数据...")
    df = pd.read_csv(csv_file)
    print(f"✓ 读取到 {len(df)} 个不确定样本")

    if len(df) == 0:
        print("⚠ 不确定样本数量为 0，无需大模型审核。")
        return

    # 3. 检查并加载大模型
    print("\n[3/5] 检查并加载大模型...")
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 模型路径不存在: {MODEL_PATH}")
        print("请确认 32B 模型已经下载到这个目录。")
        return

    print(f"正在加载大模型: {MODEL_PATH}")
    reviewer = DiabetesLLMReviewer(model_path=MODEL_PATH)

    # 4. 处理样本
    n_samples = min(max_samples, len(df))
    print(f"\n[4/5] 开始审核 {n_samples} 个样本...\n")

    results = []

    for idx in range(n_samples):
        sample = df.iloc[idx].to_dict()

        print("\n" + "=" * 60)
        print(f"样本 {idx + 1}/{n_samples}")
        print("=" * 60)

        # 显示学生模型信息
        print(f"学生模型预测: 类别 {sample.get('prediction', 'N/A')}")
        print(
            "概率分布: "
            f"[{sample.get('p_0', 0):.2%}, "
            f"{sample.get('p_1', 0):.2%}, "
            f"{sample.get('p_2', 0):.2%}]"
        )
        print(f"不确定性: 熵 = {sample.get('entropy', 0):.3f}")

        # 审核
        print("\n正在请教大模型（Qwen2.5-32B-Instruct）...")
        review = reviewer.review_sample(sample)

        # 打印结果
        print("\n【大模型审核结果】")
        print(json.dumps(review, ensure_ascii=False, indent=2))

        # 收集结果
        results.append(
            {
                "sample_id": int(idx),
                "student_prediction": {
                    "class": int(sample.get("prediction", 0)),
                    "probabilities": [
                        float(sample.get("p_0", 0)),
                        float(sample.get("p_1", 0)),
                        float(sample.get("p_2", 0)),
                    ],
                    "entropy": float(sample.get("entropy", 0)),
                },
                "llm_review": review,
            }
        )

    # 5. 保存结果
    print("\n[5/5] 保存结果...")
    LLM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = LLM_OUTPUT_DIR / "reviews.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("✓ 审核完成！")
    print(f"✓ 结果已保存到: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    # 直接运行脚本的入口：默认审核前 5 个样本
    run_llm_review(max_samples=5)
