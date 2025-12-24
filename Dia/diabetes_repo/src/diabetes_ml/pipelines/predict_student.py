"""
学生模型独立预测接口
用于连接大模型：当学生模型不确定时，触发大模型审核
"""

from __future__ import annotations
import os
import json
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
import torch
import joblib

from diabetes_ml.models.ft_transformer import FTTransformer
from diabetes_ml.calibration.uncertainty import identify_uncertain_samples
from diabetes_ml.calibration.temperature import TemperatureScaler


def load_model_artifacts(model_dir: str) -> Dict[str, Any]:
    """
    加载模型及相关资源
    
    Args:
        model_dir: 模型输出目录
    
    Returns:
        dict: 包含model, scaler, temperature, config等
    """
    # 加载配置
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        config = json.load(f)
    
    # 加载类别映射
    with open(os.path.join(model_dir, "cat_maps.json"), "r") as f:
        cat_maps = json.load(f)
    
    # 加载 scaler
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    
    # 加载模型
    model_path = os.path.join(model_dir, "model.pt")
    
    # 重建模型结构
    # 需要从config中获取cat_cardinalities
    # 这里简化处理，实际使用时需要完整实现
    state_dict = torch.load(model_path, map_location='cpu')
    
    # 加载温度参数
    temp_path = os.path.join(model_dir, "temperature.pkl")
    temperature = None
    if os.path.exists(temp_path):
        import pickle
        with open(temp_path, "rb") as f:
            temp_data = pickle.load(f)
            temperature = np.exp(temp_data['log_t'])
    
    return {
        'config': config,
        'cat_maps': cat_maps,
        'scaler': scaler,
        'state_dict': state_dict,
        'temperature': temperature
    }


class StudentModelPredictor:
    """
    学生模型预测器
    提供单样本和批量预测接口
    """
    
    def __init__(
        self, 
        model_dir: str,
        uncertainty_threshold: float = 0.3,
        confidence_threshold: float = 0.6,
        device: Optional[str] = None
    ):
        """
        初始化预测器
        
        Args:
            model_dir: 模型目录
            uncertainty_threshold: 不确定性阈值
            confidence_threshold: 置信度阈值
            device: 计算设备
        """
        self.model_dir = model_dir
        self.uncertainty_threshold = uncertainty_threshold
        self.confidence_threshold = confidence_threshold
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # 加载模型资源
        self.artifacts = load_model_artifacts(model_dir)
        self.config = self.artifacts['config']
        self.scaler = self.artifacts['scaler']
        self.temperature = self.artifacts['temperature']
        
        # 加载模型（这里需要完整实现，简化处理）
        # self.model = ... 加载实际模型
        # self.model.load_state_dict(self.artifacts['state_dict'])
        # self.model.to(self.device)
        # self.model.eval()
        
        print(f"✓ 模型加载完成 (device: {self.device})")
        print(f"✓ 不确定性阈值: {uncertainty_threshold}")
        print(f"✓ 置信度阈值: {confidence_threshold}")
    
    def preprocess_input(self, sample_data: Dict[str, Any]) -> torch.Tensor:
        """
        预处理输入数据
        
        Args:
            sample_data: 特征字典
        
        Returns:
            tensor: 预处理后的输入张量
        """
        # 这里需要根据实际数据格式实现
        # 1. 处理缺失值
        # 2. 类别编码
        # 3. 数值归一化
        # 4. 转换为tensor
        pass
    
    def predict_single(self, sample_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        单样本预测 - 核心接口，供大模型调用
        
        Args:
            sample_data: 包含所有特征的字典
        
        Returns:
            dict: {
                'prediction': int,  # 预测类别 (0/1/2)
                'probabilities': list,  # 各类别概率 [p0, p1, p2]
                'uncertainty': {
                    'entropy': float,
                    'margin': float,
                    'max_prob': float
                },
                'need_llm_review': bool,  # 是否需要大模型审核
                'confidence_level': str,  # 'high' / 'medium' / 'low'
                'llm_prompt_ready': bool  # 是否已准备好LLM提示
            }
        """
        # 1. 预处理
        x_tensor = self.preprocess_input(sample_data)
        
        # 2. 模型预测
        with torch.no_grad():
            x_tensor = x_tensor.to(self.device)
            logits = self.model(x_tensor)
            
            # 应用温度缩放
            if self.temperature is not None:
                logits = logits / self.temperature
            
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        
        # 3. 不确定性评估
        uncertain_mask, uncertainty_metrics = identify_uncertain_samples(
            probs,
            uncertainty_threshold=self.uncertainty_threshold,
            confidence_threshold=self.confidence_threshold
        )
        
        # 4. 组装结果
        prediction = int(np.argmax(probs[0]))
        max_prob = float(uncertainty_metrics['max_prob'][0])
        
        # 判断置信度等级
        if max_prob >= 0.8:
            confidence_level = 'high'
        elif max_prob >= 0.6:
            confidence_level = 'medium'
        else:
            confidence_level = 'low'
        
        result = {
            'prediction': prediction,
            'probabilities': probs[0].tolist(),
            'uncertainty': {
                'entropy': float(uncertainty_metrics['entropy'][0]),
                'margin': float(uncertainty_metrics['margin'][0]),
                'max_prob': max_prob
            },
            'need_llm_review': bool(uncertain_mask[0]),
            'confidence_level': confidence_level,
            'llm_prompt_ready': bool(uncertain_mask[0])
        }
        
        return result
    
    def predict_batch(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        批量预测
        
        Args:
            samples: 样本列表
        
        Returns:
            list: 预测结果列表
        """
        return [self.predict_single(sample) for sample in samples]
    
    def prepare_llm_prompt(self, sample_data: Dict[str, Any], prediction_result: Dict[str, Any]) -> str:
        """
        为不确定样本准备大模型提示词
        
        Args:
            sample_data: 原始特征数据
            prediction_result: 学生模型的预测结果
        
        Returns:
            str: LLM提示词
        """
        if not prediction_result['need_llm_review']:
            return ""
        
        prompt = f"""
请基于以下信息，对糖尿病诊断进行专业分析：

【患者特征】
{json.dumps(sample_data, indent=2, ensure_ascii=False)}

【初步AI模型预测】
- 预测类别: {prediction_result['prediction']} (0=无糖尿病, 1=前期, 2=中期)
- 各类别概率: {prediction_result['probabilities']}
- 模型不确定性:
  * 预测熵: {prediction_result['uncertainty']['entropy']:.4f}
  * 边界置信度: {prediction_result['uncertainty']['margin']:.4f}
  * 最高概率: {prediction_result['uncertainty']['max_prob']:.4f}

【注意】
模型对此样本的判断存在不确定性，请您：
1. 综合患者特征，给出您的专业判断
2. 解释关键特征对诊断的影响
3. 提供后续检查或治疗建议
4. 评估该预测的可信度

请生成结构化报告。
"""
        return prompt.strip()


# ========== 使用示例 ==========

if __name__ == "__main__":
    # 示例：加载模型并预测
    model_dir = "outputs/your_run_name/20241220_123456"
    
    predictor = StudentModelPredictor(
        model_dir=model_dir,
        uncertainty_threshold=0.3,
        confidence_threshold=0.6
    )
    
    # 单样本预测
    sample = {
        'age': 45,
        'bmi': 28.5,
        'blood_glucose': 120,
        # ... 其他特征
    }
    
    result = predictor.predict_single(sample)
    
    print("预测结果:", result)
    
    # 如果需要大模型介入
    if result['need_llm_review']:
        prompt = predictor.prepare_llm_prompt(sample, result)
        print("\n=== LLM Prompt ===")
        print(prompt)
        
        # 这里可以调用大模型API
        # llm_response = call_llm_api(prompt)