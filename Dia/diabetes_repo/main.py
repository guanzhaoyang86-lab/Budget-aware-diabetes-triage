from __future__ import annotations

import argparse
import os
import sys


def load_yaml(path: str):
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser("Diabetes ML Main Entry")
    parser.add_argument("--pipeline", type=str, default="ftt_3class")
    parser.add_argument("--config", type=str, default="configs/ftt_3class.yaml")
    args = parser.parse_args()

    # 允许直接在 repo root 跑，不用 pip install -e .
    repo_root = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(repo_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    cfg = load_yaml(args.config)

    if args.pipeline == "ftt_3class":
        from diabetes_ml.pipelines.ftt_3class import run
        out_dir = run(cfg)
        print("Run finished. Outputs:", out_dir)
    else:
        raise ValueError(f"Unknown pipeline: {args.pipeline}")


if __name__ == "__main__":
    # Windows 多进程入口保护（你原脚本的关键点）
    import multiprocessing as mp
    mp.freeze_support()
    try:
        import torch.multiprocessing
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    main()
