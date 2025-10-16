# vLLM MoE Benchmark 部署指南

[🇺🇸 English](DEPLOYMENT_GUIDE.md) | [🇨🇳 中文](DEPLOYMENT_GUIDE_zh.md)

本文档提供了详细的部署和故障排除指南。

## 系统要求

### 硬件要求
- **GPU**: NVIDIA GPU with CUDA Compute Capability 7.0+ (推荐 A100/H100)
- **显存**: 至少 40GB (根据模型大小调整)
- **内存**: 至少 32GB RAM
- **存储**: 至少 100GB 可用空间

### 软件要求
- **操作系统**: Ubuntu 18.04+ / CentOS 7+ / RHEL 7+
- **Python**: 3.11+
- **CUDA**: 11.8+ (推荐 12.1+)
- **Docker**: 可选，用于容器化部署

## 安装步骤

### 1. 环境准备

#### 检查 CUDA 环境
```bash
nvidia-smi
nvcc --version
```

#### 创建 Python 环境
```bash
# 使用 conda (推荐)
conda create -n benchmark_moe python=3.11
conda activate benchmark_moe

# 或使用 venv
python3.11 -m venv benchmark_moe_env
source benchmark_moe_env/bin/activate
```

### 2. 安装依赖

#### 基础依赖
```bash
# 安装 PyTorch (CUDA 版本)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装 vLLM
pip install vllm

# 安装其他依赖
pip install -r deployment/requirements.txt
```

#### 验证安装
```bash
python -c "
import torch
import vllm
import ray
import triton
print('✅ 所有依赖安装成功')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')
print(f'vLLM: {vllm.__version__}')
"
```

### 3. 模型准备

#### 下载模型
```bash
# 使用 huggingface-cli 下载
huggingface-cli download mistralai/Mixtral-8x7B-Instruct-v0.1 --local-dir ./models/mixtral-8x7b

# 或者使用 Python 脚本
python -c "
from transformers import AutoTokenizer, AutoConfig
model_name = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='./models')
config = AutoConfig.from_pretrained(model_name, cache_dir='./models')
print('模型配置下载完成')
"
```

## 配置优化

### 环境变量设置
```bash
# 添加到 ~/.bashrc 或 ~/.zshrc
export CUDA_VISIBLE_DEVICES=0  # 根据可用 GPU 调整
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export RAY_DISABLE_IMPORT_WARNING=1
export TRITON_CACHE_DIR=/tmp/triton_cache
```

### 内存优化
```bash
# 设置 GPU 内存分配策略
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 启用内存映射
export VLLM_USE_MODELSCOPE=0
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1
```

## 故障排除

### 常见问题

#### 1. CUDA 相关错误

**错误**: `CUDA out of memory`
```bash
# 解决方案:
# - 减少批次大小
python benchmark_moe.py --batch-size 1 2 4 8

# - 使用量化
python benchmark_moe.py --dtype fp8_w8a8

# - 清理 GPU 内存
nvidia-smi --gpu-reset
```

**错误**: `CUDA driver version is insufficient`
```bash
# 检查驱动版本
nvidia-smi

# 更新 NVIDIA 驱动
sudo ubuntu-drivers autoinstall
sudo reboot
```

#### 2. Triton 编译错误

**错误**: `JSONDecodeError: Expecting value`
```bash
# 清理 Triton 缓存
rm -rf ~/.triton/cache/*
rm -rf /tmp/triton_cache*

# 设置新的缓存目录
export TRITON_CACHE_DIR=/tmp/triton_cache_$(date +%s)
mkdir -p $TRITON_CACHE_DIR
```

**错误**: `OutOfResources`
```bash
# 减少搜索空间
python benchmark_moe.py --batch-size 1 2 4  # 较小的批次范围

# 或使用快速模式
bash scripts/run_benchmark_safe.sh --batch-sizes 1,2,4,8
```

#### 3. Ray 相关问题

**错误**: `Ray cluster initialization failed`
```bash
# 停止现有 Ray 进程
ray stop

# 重新启动
ray start --head --port=6379

# 或者清理 Ray 临时文件
rm -rf /tmp/ray*
```

#### 4. 模型加载错误

**错误**: `Model not found` 或 `Permission denied`
```bash
# 检查模型路径
ls -la /path/to/your/model

# 设置正确权限
chmod -R 755 /path/to/your/model

# 使用信任远程代码
python benchmark_moe.py --trust-remote-code
```

**错误**: `Tokenizer not found`
```bash
# 预下载 tokenizer
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('your_model_name')
"
```

#### 5. 网络相关问题

**错误**: `Connection timeout` (访问 HuggingFace)
```bash
# 设置代理 (如需要)
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080

# 或使用镜像
export HF_ENDPOINT=https://hf-mirror.com
```

### 性能调优建议

#### 针对不同硬件的优化

**单 GPU (A100 80GB)**
```bash
python benchmark_moe.py \
  --tp-size 1 \
  --dtype fp8_w8a8 \
  --batch-size 1 2 4 8 16 32 64 128
```

**多 GPU (2x A100)**
```bash
python benchmark_moe.py \
  --tp-size 2 \
  --dtype auto \
  --batch-size 16 32 64 128 256 512
```

**内存受限环境**
```bash
python benchmark_moe.py \
  --tp-size 1 \
  --dtype fp8_w8a8 \
  --batch-size 1 2 4 8 \
  --use-deep-gemm
```

## 监控和调试

### 性能监控
```bash
# GPU 使用监控
watch -n 1 nvidia-smi

# 内存使用监控
htop

# 网络 I/O 监控
iotop
```

### 调试日志
```bash
# 启用详细日志
export VLLM_LOGGING_LEVEL=DEBUG
export RAY_VERBOSE=1

# 运行调试模式
python benchmark_moe.py --batch-size 1 --tune 2>&1 | tee debug.log
```

## 容器化部署

### Dockerfile 示例
```dockerfile
FROM nvidia/cuda:12.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-pip python3.11-dev \
    git wget curl

WORKDIR /app
COPY . .

RUN pip install -r deployment/requirements.txt

CMD ["python", "benchmark_moe.py", "--help"]
```

### Docker 运行
```bash
# 构建镜像
docker build -t benchmark_moe .

# 运行容器
docker run --gpus all -v /path/to/models:/app/models benchmark_moe \
  python benchmark_moe.py --model /app/models/your_model --tune
```

## 最佳实践

### 1. 资源管理
- 使用 `nvidia-smi` 定期检查 GPU 使用情况
- 设置合适的批次大小避免内存溢出
- 定期清理 Triton 缓存避免磁盘空间不足

### 2. 调优策略
- 从小批次开始逐步增加
- 优先测试常用的批次大小
- 保存调优结果以供后续使用

### 3. 生产部署
- 使用稳定的模型版本
- 设置合适的超时时间
- 建立监控和告警机制

## 技术支持

如果遇到问题，请按以下顺序排查：

1. **检查环境**: 运行 `bash scripts/server_check.sh`
2. **查看日志**: 检查错误信息和堆栈跟踪
3. **搜索已知问题**: 查看项目 Issues 页面
4. **创建新 Issue**: 提供详细的错误信息和环境配置

### 报告问题时请包含：
- 操作系统和版本
- GPU 型号和驱动版本
- Python 和依赖包版本
- 完整的错误日志
- 重现步骤