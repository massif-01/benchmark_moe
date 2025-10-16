#!/bin/bash

# vLLM MoE Benchmark 服务器环境检查脚本
# 检查服务器是否具备运行 MoE 基准测试的条件

echo "============================================"
echo "vLLM MoE Benchmark 服务器环境检查"
echo "============================================"

# 检查函数
check_command() {
    if command -v $1 &> /dev/null; then
        echo "✅ $1 已安装"
        return 0
    else
        echo "❌ $1 未安装"
        return 1
    fi
}

check_python_package() {
    if python -c "import $1" &> /dev/null; then
        version=$(python -c "import $1; print($1.__version__)" 2>/dev/null)
        echo "✅ $1 已安装 (版本: $version)"
        return 0
    else
        echo "❌ $1 未安装"
        return 1
    fi
}

# 系统信息检查
echo ""
echo "=== 系统信息 ==="
echo "操作系统: $(uname -s -r)"
echo "主机名: $(hostname)"
echo "当前用户: $(whoami)"
echo "Python 版本: $(python --version 2>&1)"

# CUDA 环境检查
echo ""
echo "=== CUDA 环境检查 ==="
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA 驱动已安装"
    echo "GPU 信息:"
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits | \
    awk -F', ' '{printf "  - %s: 总计 %dGB, 已用 %dGB, 空闲 %dGB\n", $1, $2/1024, $3/1024, $4/1024}'
    
    echo ""
    echo "CUDA 版本:"
    nvidia-smi | grep "CUDA Version" | awk '{print "  - Driver: " $NF}'
    
    if command -v nvcc &> /dev/null; then
        echo "  - Runtime: $(nvcc --version | grep -oP 'release \K[0-9.]+')"
    else
        echo "  - Runtime: nvcc 未找到"
    fi
else
    echo "❌ NVIDIA 驱动未安装或不可用"
fi

# Python 依赖检查
echo ""
echo "=== Python 依赖检查 ==="
check_python_package "torch"
if python -c "import torch; print('CUDA 可用:', torch.cuda.is_available())" 2>/dev/null; then
    python -c "import torch; print('GPU 数量:', torch.cuda.device_count())"
fi

check_python_package "vllm"
check_python_package "ray"
check_python_package "transformers"
check_python_package "triton"

# vLLM 特定检查
echo ""
echo "=== vLLM 特定检查 ==="
if python -c "import vllm" &> /dev/null; then
    echo "检查 vLLM MoE 支持..."
    python -c "
try:
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
    print('✅ MoE 层支持正常')
except ImportError as e:
    print('❌ MoE 层导入失败:', e)

try:
    from vllm.platforms import current_platform
    platform = current_platform
    print('✅ 平台检测:', type(platform).__name__)
    print('  - FP8 数据类型:', platform.fp8_dtype())
    print('  - 是否 ROCm:', platform.is_rocm())
except Exception as e:
    print('❌ 平台检测失败:', e)
"
fi

# 环境变量检查
echo ""
echo "=== 环境变量检查 ==="
env_vars=("CUDA_VISIBLE_DEVICES" "HIP_VISIBLE_DEVICES" "VLLM_USE_TRITON_FLASH_ATTN")
for var in "${env_vars[@]}"; do
    if [[ -n "${!var}" ]]; then
        echo "✅ $var = ${!var}"
    else
        echo "ℹ️  $var 未设置"
    fi
done

# 磁盘空间检查
echo ""
echo "=== 磁盘空间检查 ==="
echo "当前目录磁盘使用情况:"
df -h . | tail -n1 | awk '{printf "  - 已用: %s / 总计: %s (使用率: %s)\n", $3, $2, $5}'

# 内存检查
echo ""
echo "=== 内存使用情况 ==="
if command -v free &> /dev/null; then
    free -h | grep -E "Mem|Swap" | awk '{printf "  - %s: 已用 %s / 总计 %s\n", $1, $3, $2}'
fi

# 网络连接检查
echo ""
echo "=== 网络连接检查 ==="
echo "检查 Hugging Face 连接..."
if curl -s --max-time 10 https://huggingface.co &> /dev/null; then
    echo "✅ Hugging Face 连接正常"
else
    echo "❌ Hugging Face 连接失败，可能影响模型下载"
fi

# 生成建议
echo ""
echo "=== 环境建议 ==="
suggestions=""

if ! command -v nvidia-smi &> /dev/null; then
    suggestions+="- 需要安装 NVIDIA 驱动和 CUDA\n"
fi

if ! python -c "import vllm" &> /dev/null; then
    suggestions+="- 需要安装 vLLM: pip install vllm\n"
fi

if ! python -c "import ray" &> /dev/null; then
    suggestions+="- 需要安装 Ray: pip install 'ray[default]'\n"
fi

if [[ -n "$suggestions" ]]; then
    echo -e "$suggestions"
else
    echo "✅ 环境检查通过，可以运行 MoE 基准测试"
fi

echo ""
echo "=== 推荐的运行命令 ==="
echo "环境检查完成后，可以使用以下命令运行基准测试："
echo ""
echo "# 调优 Mixtral 模型"
echo "python benchmark_moe.py --model mistralai/Mixtral-8x7B-Instruct-v0.1 --tune --tp-size 2"
echo ""
echo "# 运行性能基准测试"
echo "python benchmark_moe.py --model mistralai/Mixtral-8x7B-Instruct-v0.1 --tp-size 2"
echo ""

echo "============================================"
echo "环境检查完成"
echo "============================================"