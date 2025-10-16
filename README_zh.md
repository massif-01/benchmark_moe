# benchmark_moe

[🇺🇸 English](README.md) | [🇨🇳 中文](README_zh.md)

基于 vLLM 的 MoE (Mixture of Experts) 模型内核性能优化工具

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![vLLM](https://img.shields.io/badge/vLLM-0.10.0+-green.svg)](https://github.com/vllm-project/vllm)

一个专门用于优化 vLLM 框架中 MoE 模型推理性能的工具集，通过自动化调优 Triton 内核参数，为不同的模型架构和硬件配置找到最优的执行配置。

## 🎯 主要功能

- **🔧 自动化内核调优**: 使用 Ray 分布式框架自动搜索最优的 Triton 内核配置
- **📊 多模型支持**: 支持 Mixtral、DeepSeek、Qwen、Jamba 等主流 MoE 模型
- **⚡ 性能优化**: 针对不同批次大小和硬件配置进行专门优化
- **🛠️ 故障诊断**: 提供完善的环境检查和问题排查工具
- **📈 结果分析**: 生成详细的性能报告和配置推荐

## 🚀 快速开始

### 前置要求

- **硬件**: NVIDIA GPU (推荐 A100/H100)
- **软件**: Ubuntu 18.04+, Python 3.11+, CUDA 11.8+
- **依赖**: vLLM 0.10.0+, PyTorch 2.0+, Ray

### 安装步骤

1. **克隆项目**
   ```bash
   git clone https://github.com/your_username/benchmark_moe.git
   cd benchmark_moe
   ```

2. **环境检查**
   ```bash
   bash scripts/server_check.sh
   ```

3. **运行单个模型调优**
   ```bash
   # 基本调优 - Qwen3 模型
   python benchmark_moe.py \
     --model /path/to/your/qwen3-model \
     --tp-size 1 \
     --dtype auto \
     --batch-size 1 2 4 8 16 32 64 128 \
     --tune \
     --save-dir ./optimized_configs \
     --trust-remote-code
   ```

4. **查看结果**
   ```bash
   ls ./optimized_configs/
   # 输出: E64N9472_tp1_fp16.json (示例配置文件)
   ```

## 📋 详细使用指南

### 环境准备

#### 检查系统环境
```bash
# 运行环境检查脚本
bash scripts/server_check.sh

# 检查 GPU 状态
nvidia-smi

# 检查 Python 依赖
python -c "import vllm, ray, torch, triton; print('环境检查通过')"
```

#### 处理常见环境问题

**问题 1: Triton 缓存损坏**
```bash
# 清理 Triton 缓存（如果遇到 JSONDecodeError）
rm -rf ~/.triton/cache/*

# 或设置新的缓存目录
export TRITON_CACHE_DIR=/tmp/triton_cache_$(date +%s)
mkdir -p $TRITON_CACHE_DIR
```

**问题 2: libstdc++ 版本问题**
```bash
# 更新 conda 环境中的 libstdc++
conda install -c conda-forge libstdcxx-ng

# 或使用系统库
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

**问题 3: Ray 警告**
```bash
# 消除 Ray 相关警告
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export RAY_DISABLE_IMPORT_WARNING=1
```

### 调优参数说明

#### 基本参数
- `--model`: 模型路径或 HuggingFace 模型名
- `--tp-size`: 张量并行度（根据 GPU 数量设置）
- `--dtype`: 数据类型 (`auto`, `fp8_w8a8`, `int8_w8a16`)
- `--batch-size`: 要测试的批次大小列表
- `--tune`: 启用调优模式
- `--save-dir`: 配置文件保存目录

#### 高级参数
- `--use-deep-gemm`: 启用 DeepGEMM 优化
- `--enable-expert-parallel`: 启用专家并行（适用于大型模型）
- `--seed`: 随机种子（保证结果可重现）

### 支持的模型类型

| 模型系列 | 专家数 | Top-K | 推荐显存 | 示例命令 |
|---------|--------|-------|----------|----------|
| **Qwen3-30B-A3B** | 64 | 4 | 64GB+ | `--model path/to/qwen3 --tp-size 1` |
| **Mixtral-8x7B** | 8 | 2 | 45GB+ | `--model mistralai/Mixtral-8x7B-Instruct-v0.1 --tp-size 2` |
| **DeepSeek-V2** | 160 | 6 | 80GB+ | `--model deepseek-ai/DeepSeek-V2-Chat --tp-size 4` |
| **DeepSeek-V3** | 256 | 8 | 120GB+ | `--model deepseek-ai/DeepSeek-V3-Base --tp-size 8` |

### 批量调优脚本

#### 使用配置管理工具
```bash
# 列出支持的模型
python tools/config_manager.py list

# 为特定模型调优
python tools/config_manager.py tune qwen3_30b

# 查看配置推荐
python tools/config_manager.py recommend qwen3_30b
```

#### 安全的批量调优
```bash
# 使用安全脚本逐个测试批次大小
bash scripts/run_benchmark_safe.sh
```

## 📊 结果解读

### 配置文件格式
```json
{
  "triton_version": "2.1.0",
  "1": {                    // batch_size=1 的最优配置
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 64, 
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 1,
    "num_warps": 4,
    "num_stages": 3
  },
  "64": {                   // batch_size=64 的最优配置
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 256,
    "GROUP_SIZE_M": 32,
    "num_warps": 8,
    "num_stages": 4
  }
}
```

### 性能分析
```bash
# 运行性能基准测试（不加 --tune）
python benchmark_moe.py \
  --model your_model \
  --tp-size 1 \
  --batch-size 1 2 4 8 16 32 64 128

# 输出示例:
# Batch size: 1, Kernel time: 45.23 us
# Batch size: 64, Kernel time: 892.15 us
```

## 🛠️ 故障排除

### 常见问题及解决方案

#### 1. 内存不足错误
```bash
# 症状: CUDA out of memory
# 解决方案:
# - 减少批次大小
--batch-size 1 2 4 8 16 32

# - 使用量化
--dtype fp8_w8a8

# - 增加张量并行度（如果有多GPU）
--tp-size 2
```

#### 2. Triton 编译错误
```bash
# 症状: JSONDecodeError, OutOfResources
# 解决方案:
rm -rf ~/.triton/cache/*
export TRITON_CACHE_DIR=/tmp/triton_cache_new
```

#### 3. 模型加载失败
```bash
# 症状: Model not found, Permission denied
# 解决方案:
# - 检查模型路径
ls /path/to/your/model

# - 添加访问权限
--trust-remote-code

# - 预下载模型
huggingface-cli download model_name --local-dir ./models/
```

#### 4. Ray 初始化问题
```bash
# 症状: Ray 无法启动
# 解决方案:
export RAY_DISABLE_IMPORT_WARNING=1
ray stop  # 停止现有实例
ray start --head  # 重新启动
```

### 性能调优建议

#### 针对不同使用场景

**低延迟场景（在线推理）**
```bash
# 优化小批次性能
--batch-size 1 2 4 8 16
--dtype fp8_w8a8  # 减少内存访问延迟
```

**高吞吐场景（批量处理）**
```bash
# 优化大批次性能  
--batch-size 64 128 256 512 1024
--dtype auto  # 平衡精度和性能
```

**内存受限场景**
```bash
# 最大化内存利用率
--dtype fp8_w8a8
--use-deep-gemm
--enable-expert-parallel
```

```
benchmark_moe/
├── README.md                   # 项目说明文档
├── benchmark_moe.py           # vLLM MoE 基准测试核心脚本
├── scripts/                   # 运行脚本目录
│   ├── server_check.sh        # 服务器环境检查脚本
│   ├── tune_mixtral.sh        # Mixtral 模型调优脚本
│   ├── tune_deepseek.sh       # DeepSeek 模型调优脚本
│   └── run_benchmark_safe.sh  # 安全的批量调优脚本
├── configs/                   # 配置文件目录
│   ├── models.json            # 支持的模型配置
│   └── tuning_params.json     # 调优参数配置
├── tools/                     # 分析工具目录
│   └── config_manager.py      # 配置管理工具
├── results/                   # 结果输出目录
│   ├── tuned_configs/         # 调优后的配置文件
│   └── performance_reports/   # 性能测试报告
└── deployment/                # 部署相关文件
    ├── requirements.txt       # Python依赖列表
    └── DEPLOYMENT_GUIDE.md    # 详细部署指南
```

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

### 开发环境设置
```bash
git clone https://github.com/your_username/benchmark_moe.git
cd benchmark_moe
pip install -r deployment/requirements.txt
```

### 提交代码
1. Fork 这个项目
2. 创建您的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的改动 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启一个 Pull Request

## 📄 许可证

此项目采用 Apache-2.0 许可证。详情请见 [LICENSE](LICENSE) 文件。

## � 致谢

- [vLLM](https://github.com/vllm-project/vllm) - 高性能 LLM 推理引擎
- [Ray](https://github.com/ray-project/ray) - 分布式计算框架
- [Triton](https://github.com/openai/triton) - GPU 编程语言

## 📮 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 [GitHub Issue](https://github.com/your_username/benchmark_moe/issues)
- 发起 [Discussion](https://github.com/your_username/benchmark_moe/discussions)

---

**⭐ 如果这个项目对您有帮助，请给我们一个 Star！**