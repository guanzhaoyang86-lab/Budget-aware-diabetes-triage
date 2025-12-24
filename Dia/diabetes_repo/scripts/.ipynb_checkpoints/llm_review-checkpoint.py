"""
大模型审核脚本 - 简化版
为不确定样本生成医学审核报告

使用方法：
1. 确保已经运行过训练，生成了 uncertain_samples_for_llm.csv
2. 确保已经下载了大模型
3. 运行: python scripts/llm_review.py
"""

import os
import sys
import pandas as pd
import json
from pathlib import Path
import glob

print("="*70)
print("       大模型审核系统 - Diabetes LLM Review System")
print("="*70)

# ============ 配置区 ============
# 1. 大模型路径（修改为你下载后的实际路径）
MODEL_PATH = "/hy-tmp/models/Qwen/Qwen2.5-32B-Instruct"

# 2. 项目根目录
PROJECT_ROOT = "/hy-tmp/糖尿病/Code open"

# 3. 输出目录
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs/llm_reviews")
# ================================


def find_uncertain_samples_file():
    """查找不确定样本文件"""
    outputs_dir = Path("/hy-tmp/糖尿病/Code open/outputs")

    if not outputs_dir.exists():
        print(f"❌ outputs 目录不存在: {outputs_dir}")
        return None

    pattern = str(outputs_dir / "**/uncertain_samples_for_llm.csv")
    files = glob.glob(pattern, recursive=True)

    if not files:
        print("❌ 未找到 uncertain_samples_for_llm.csv 文件")
        print(f"   查找路径: {pattern}")

        print("\noutputs 目录内容:")
        for item in outputs_dir.iterdir():
            print(f"  {item}")

        return None

    latest_file = max(files, key=os.path.getmtime)
    print(f"✓ 找到文件: {latest_file}")
    return latest_file


def check_model_exists():
    """检查大模型是否存在"""
    print("\n[2/5] 正在检查大模型...")

    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误：找不到大模型！")
        print(f"   期望路径: {MODEL_PATH}")
        sys.exit(1)

    print("✓ 大模型路径正确")


def load_model():
    """加载大模型"""
    print("\n[3/5] 正在加载大模型（这可能需要1-2分钟）...")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        print("   - 加载 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True
        )

        print("   - 加载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

        print("✓ 大模型加载成功！")
        return model, tokenizer

    except ImportError as e:
        print("❌ 缺少必要的包！具体错误信息：")
        print(e)
        import traceback
        traceback.print_exc()
        print("\n请运行（如果还没运行过）：")
        print('pip install --upgrade "transformers>=4.40.0" "accelerate>=0.30.0" protobuf sentencepiece --break-system-packages')
        sys.exit(1)

    except Exception as e:
        print(f"❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def get_pred_from_row(sample_row):
    """
    从一行样本里得到模型预测的类别：
    1. 如果有 prediction 列，优先用它
    2. 否则用 p_0, p_1, p_2 里最大概率的那个当预测
    """
    try:
        if "prediction" in sample_row and pd.notna(sample_row["prediction"]):
            return int(sample_row["prediction"])
    except Exception:
        pass

    prob_keys = [k for k in ["p_0", "p_1", "p_2"] if k in sample_row]
    if prob_keys:
        probs = [float(sample_row[k]) for k in prob_keys]
        pred_idx = max(range(len(probs)), key=lambda i: probs[i])
        return int(pred_idx)

    return 0


def create_medical_prompt(sample_row):
    """为医学审核创建提示词"""
    pred = get_pred_from_row(sample_row)
    p0 = float(sample_row.get('p_0', 0))
    p1 = float(sample_row.get('p_1', 0))
    p2 = float(sample_row.get('p_2', 0))
    entropy = float(sample_row.get('entropy', 0))
    margin = float(sample_row.get('margin', 0))
    max_prob = float(sample_row.get('max_prob', 0))

    class_map = {0: "无糖尿病", 1: "糖尿病前期", 2: "糖尿病中期"}

    prompt = f"""你是一位经验丰富的内分泌科医生。以下是一个糖尿病诊断病例，AI模型对此病例的判断存在不确定性，需要你的专业审核。

【AI模型诊断】
- AI预测: {class_map.get(pred, str(pred))}
- 概率分布:
  * 无糖尿病: {p0:.1%}
  * 糖尿病前期: {p1:.1%}
  * 糖尿病中期: {p2:.1%}

【不确定性指标】
- 预测熵: {entropy:.3f} (数值越高表示越不确定)
- 边界置信度: {margin:.3f} (数值越低表示类别越难区分)
- 最高概率: {max_prob:.1%} (低于60%表示模型不自信)

请作为专业医生，提供简洁的审核意见：
1. 你的诊断结论（选择：无糖尿病/糖尿病前期/糖尿病中期）
2. 是否同意AI的判断？（同意/部分同意/不同意）
3. 一句话建议

请用中文简洁回答，总共不超过100字。"""

    return prompt


def review_one_sample(model, tokenizer, sample_row, idx):
    """审核单个样本"""

    print(f"\n{'─'*60}")
    print(f"样本 #{idx}")
    print(f"{'─'*60}")

    # ✅ 关键：在本函数里定义 pred
    pred = get_pred_from_row(sample_row)

    # 创建提示
    prompt = create_medical_prompt(sample_row)

    # 打印样本信息
    print(f"AI预测: 类别 {pred}")
    print(f"概率: [{float(sample_row.get('p_0', 0)):.2%}, {float(sample_row.get('p_1', 0)):.2%}, {float(sample_row.get('p_2', 0)):.2%}]")
    print(f"不确定性: 熵={float(sample_row.get('entropy', 0)):.3f}")

    print("\n正在请教大模型...")

    messages = [
        {"role": "system", "content": "你是一位专业的内分泌科医生，擅长糖尿病诊断。"},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        do_sample=True,
        top_p=0.9
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\n【大模型审核意见】")
    print("─" * 60)
    print(response)
    print("─" * 60)

    return {
        'sample_index': idx,
        'ai_prediction': pred,
        'probabilities': [
            float(sample_row.get('p_0', 0)),
            float(sample_row.get('p_1', 0)),
            float(sample_row.get('p_2', 0))
        ],
        'uncertainty_metrics': {
            'entropy': float(sample_row.get('entropy', 0)),
            'margin': float(sample_row.get('margin', 0)),
            'max_prob': float(sample_row.get('max_prob', 0))
        },
        'llm_review': response
    }


def save_results(results, output_dir):
    """保存审核结果"""
    print(f"\n[5/5] 正在保存结果...")

    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "llm_reviews.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✓ JSON 结果: {json_path}")

    txt_path = os.path.join(output_dir, "review_summary.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("大模型审核汇总报告\n")
        f.write("Diabetes LLM Review Summary\n")
        f.write("="*70 + "\n\n")
        f.write(f"审核样本总数: {len(results)}\n")
        f.write(f"生成时间: {pd.Timestamp.now()}\n\n")

        for r in results:
            f.write(f"\n{'─'*70}\n")
            f.write(f"样本 #{r['sample_index']}\n")
            f.write(f"{'─'*70}\n")
            f.write(f"AI预测: 类别 {r['ai_prediction']}\n")
            f.write(f"概率分布: {r['probabilities']}\n")
            f.write(f"不确定性: 熵={r['uncertainty_metrics']['entropy']:.3f}, ")
            f.write(f"边界={r['uncertainty_metrics']['margin']:.3f}\n\n")
            f.write("【大模型审核意见】\n")
            f.write(r['llm_review'])
            f.write("\n\n")

    print(f"✓ 文本报告: {txt_path}")


def main():
    """主函数"""

    try:
        csv_path = find_uncertain_samples_file()
        if csv_path is None:
            sys.exit(1)

        check_model_exists()
        model, tokenizer = load_model()

        print("\n[4/5] 正在审核样本...")
        df = pd.read_csv(csv_path)
        print(f"✓ 共发现 {len(df)} 个不确定样本")

        default_n = min(5, len(df))
        print(f"\n建议先审核 {default_n} 个样本测试")
        user_input = input(f"要审核多少个？（直接回车使用默认值 {default_n}）: ").strip()

        if user_input == "":
            n_samples = default_n
        else:
            try:
                n_samples = int(user_input)
                n_samples = min(n_samples, len(df))
            except:
                n_samples = default_n

        print(f"\n将审核 {n_samples} 个样本\n")
        print("="*70)

        results = []
        for i in range(n_samples):
            sample = df.iloc[i]
            result = review_one_sample(model, tokenizer, sample, i + 1)
            results.append(result)

        save_results(results, OUTPUT_DIR)

        print("\n" + "="*70)
        print("✓ 审核完成！")
        print("="*70)
        print(f"\n查看结果:")
        print(f"1. JSON格式: {OUTPUT_DIR}/llm_reviews.json")
        print(f"2. 文本报告: {OUTPUT_DIR}/review_summary.txt")
        print("\n")

    except KeyboardInterrupt:
        print("\n\n用户中断，程序退出")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
