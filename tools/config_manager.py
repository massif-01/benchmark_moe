#!/usr/bin/env python3
"""
MoE 模型配置管理脚本
用于管理多个 MoE 模型的优化配置
"""

import json
import os
import argparse
from pathlib import Path

class MoEConfigManager:
    def __init__(self, config_root="./configs"):
        self.config_root = Path(config_root)
        self.config_root.mkdir(exist_ok=True)
        
        # 模型配置映射
        self.model_configs = {
            "qwen3_30b": {
                "model_path": "/home/rm01/models/dev/llm/Qwen3-30B-A3B-Instruct-2507-AWQ",
                "experts": 64,
                "intermediate_size": 18944,
                "hidden_size": 3584,
                "topk": 4,
                "category": "medium_moe"
            },
            "mixtral_8x7b": {
                "model_path": "mistralai/Mixtral-8x7B-Instruct-v0.1", 
                "experts": 8,
                "intermediate_size": 14336,
                "hidden_size": 4096,
                "topk": 2,
                "category": "small_moe"
            },
            "deepseek_v2": {
                "model_path": "deepseek-ai/DeepSeek-V2-Chat",
                "experts": 160,
                "intermediate_size": 12288,
                "hidden_size": 5120,
                "topk": 6,
                "category": "large_moe"
            }
        }
    
    def tune_model(self, model_name, tp_size=1, dtype="auto", batch_sizes=None):
        """为指定模型运行调优"""
        if model_name not in self.model_configs:
            print(f"错误: 未知模型 {model_name}")
            print(f"支持的模型: {list(self.model_configs.keys())}")
            return False
            
        config = self.model_configs[model_name]
        model_path = config["model_path"]
        
        # 设置默认批次大小
        if batch_sizes is None:
            if config["category"] == "small_moe":
                batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
            elif config["category"] == "medium_moe":
                batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
            else:  # large_moe
                batch_sizes = [1, 2, 4, 8, 16, 32, 64]
        
        # 创建模型专用目录
        save_dir = self.config_root / model_name
        save_dir.mkdir(exist_ok=True)
        
        batch_str = " ".join(map(str, batch_sizes))
        
        # 构建命令
        cmd = f"""python benchmark_moe.py \\
  --model {model_path} \\
  --tp-size {tp_size} \\
  --dtype {dtype} \\
  --batch-size {batch_str} \\
  --tune \\
  --save-dir {save_dir} \\
  --trust-remote-code \\
  --seed 42"""
        
        print(f"为模型 {model_name} 运行调优...")
        print(f"命令: {cmd}")
        print()
        
        # 执行调优
        exit_code = os.system(cmd)
        return exit_code == 0
    
    def find_compatible_config(self, model_name, target_experts, target_intermediate):
        """寻找兼容的配置文件"""
        compatible_configs = []
        
        for config_dir in self.config_root.iterdir():
            if not config_dir.is_dir():
                continue
                
            for config_file in config_dir.glob("*.json"):
                # 解析文件名 E{experts}N{intermediate}_tp{tp}_{dtype}.json
                filename = config_file.stem
                try:
                    parts = filename.split("_")
                    e_part = parts[0]  # E64N9472
                    experts = int(e_part.split("N")[0][1:])  # 去掉 'E'
                    intermediate = int(e_part.split("N")[1])
                    
                    # 检查兼容性 (允许一定误差)
                    experts_ratio = abs(experts - target_experts) / target_experts
                    intermediate_ratio = abs(intermediate - target_intermediate) / target_intermediate
                    
                    if experts_ratio < 0.2 and intermediate_ratio < 0.2:  # 20% 误差范围
                        compatible_configs.append({
                            "file": config_file,
                            "experts": experts,
                            "intermediate": intermediate,
                            "similarity": 1 - (experts_ratio + intermediate_ratio) / 2
                        })
                except (IndexError, ValueError):
                    continue
        
        # 按相似度排序
        compatible_configs.sort(key=lambda x: x["similarity"], reverse=True)
        return compatible_configs
    
    def recommend_config(self, model_name):
        """为模型推荐配置"""
        if model_name not in self.model_configs:
            print(f"未知模型: {model_name}")
            return
            
        config = self.model_configs[model_name]
        experts = config["experts"]
        intermediate = config["intermediate_size"]
        
        print(f"为模型 {model_name} 推荐配置:")
        print(f"  专家数: {experts}")
        print(f"  中间层维度: {intermediate}")
        print()
        
        # 查找专用配置
        model_config_dir = self.config_root / model_name
        if model_config_dir.exists():
            configs = list(model_config_dir.glob("*.json"))
            if configs:
                print("✅ 找到专用配置:")
                for config_file in configs:
                    print(f"  - {config_file}")
                return
        
        # 查找兼容配置
        compatible = self.find_compatible_config(model_name, experts, intermediate)
        if compatible:
            print("🔄 找到兼容配置:")
            for cfg in compatible[:3]:  # 显示前3个最相似的
                print(f"  - {cfg['file']} (相似度: {cfg['similarity']:.2f})")
        else:
            print("❌ 未找到兼容配置，建议运行调优:")
            print(f"  python config_manager.py tune {model_name}")

def main():
    parser = argparse.ArgumentParser(description="MoE 配置管理工具")
    parser.add_argument("action", choices=["tune", "recommend", "list"], 
                       help="操作: tune(调优), recommend(推荐), list(列出)")
    parser.add_argument("model", nargs="?", help="模型名称")
    parser.add_argument("--tp-size", type=int, default=1, help="张量并行度")
    parser.add_argument("--dtype", default="auto", help="数据类型")
    parser.add_argument("--batch-sizes", nargs="+", type=int, help="批次大小列表")
    
    args = parser.parse_args()
    
    manager = MoEConfigManager()
    
    if args.action == "list":
        print("支持的模型:")
        for name, config in manager.model_configs.items():
            print(f"  {name}: {config['experts']}专家, {config['category']}")
    
    elif args.action == "recommend":
        if not args.model:
            print("请指定模型名称")
            return
        manager.recommend_config(args.model)
    
    elif args.action == "tune":
        if not args.model:
            print("请指定模型名称")
            return
        success = manager.tune_model(args.model, args.tp_size, args.dtype, args.batch_sizes)
        if success:
            print(f"✅ 模型 {args.model} 调优完成")
        else:
            print(f"❌ 模型 {args.model} 调优失败")

if __name__ == "__main__":
    main()