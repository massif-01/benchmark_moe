#!/bin/bash

# DeepSeek V2/V3 模型调优脚本
# 适用于高端服务器环境（需要大量显存）

set -e

echo "==============================================="
echo "DeepSeek MoE 内核调优脚本"
echo "==============================================="

# 默认参数
MODEL="deepseek-ai/DeepSeek-V2-Chat"
TP_SIZE=4
DTYPE="auto"
SAVE_DIR="./results/tuned_configs"
SCENARIO="standard"
MODEL_VERSION="v2"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-version)
            MODEL_VERSION="$2"
            case $MODEL_VERSION in
                "v2")
                    MODEL="deepseek-ai/DeepSeek-V2-Chat"
                    ;;
                "v3")
                    MODEL="deepseek-ai/DeepSeek-V3-Base"
                    TP_SIZE=8  # V3 需要更大的并行度
                    ;;
                *)
                    echo "不支持的模型版本: $MODEL_VERSION"
                    echo "支持的版本: v2, v3"
                    exit 1
                    ;;
            esac
            shift 2
            ;;
        --tp-size)
            TP_SIZE="$2"
            shift 2
            ;;
        --dtype)
            DTYPE="$2"
            shift 2
            ;;
        --save-dir)
            SAVE_DIR="$2"
            shift 2
            ;;
        --scenario)
            SCENARIO="$2"
            shift 2
            ;;
        --quick)
            SCENARIO="quick"
            shift
            ;;
        --comprehensive)
            SCENARIO="comprehensive"
            shift
            ;;
        --help)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --model-version VER   模型版本 [v2|v3] (默认: v2)"
            echo "  --tp-size SIZE        张量并行大小 (默认: v2=4, v3=8)"
            echo "  --dtype TYPE          数据类型 [auto|fp8_w8a8|int8_w8a16] (默认: auto)"
            echo "  --save-dir DIR        保存目录 (默认: ./results/tuned_configs)"
            echo "  --scenario SCENARIO   调优场景 [quick|standard|comprehensive] (默认: standard)"
            echo "  --quick               快速调优"
            echo "  --comprehensive       全面调优"
            echo "  --help                显示此帮助信息"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 创建保存目录
mkdir -p "$SAVE_DIR"

# 根据场景设置批次大小
case $SCENARIO in
    "quick")
        BATCH_SIZES="1 2 4 8 16"
        echo "🚀 快速调优模式 - 批次大小: $BATCH_SIZES"
        ;;
    "comprehensive")
        BATCH_SIZES="1 2 4 8 16 24 32 48 64 96 128 256 512 1024"
        echo "🔬 全面调优模式 - 批次大小: $BATCH_SIZES"
        ;;
    *)
        BATCH_SIZES="4 8 16 32 64 128 256"
        echo "⚖️  标准调优模式 - 批次大小: $BATCH_SIZES"
        ;;
esac

echo ""
echo "调优配置:"
echo "  模型: $MODEL (DeepSeek $MODEL_VERSION)"
echo "  张量并行度: $TP_SIZE"
echo "  数据类型: $DTYPE"
echo "  保存目录: $SAVE_DIR"
echo "  调优场景: $SCENARIO"

# 模型特定的警告和建议
case $MODEL_VERSION in
    "v2")
        echo ""
        echo "📋 DeepSeek V2 特性:"
        echo "  - 专家数量: 160"
        echo "  - Top-K: 6"
        echo "  - 推荐显存: 80GB+"
        echo "  - 推荐 TP: 4-8"
        MIN_MEMORY=80
        ;;
    "v3") 
        echo ""
        echo "📋 DeepSeek V3 特性:"
        echo "  - 专家数量: 256"
        echo "  - Top-K: 8"
        echo "  - 推荐显存: 120GB+"
        echo "  - 推荐 TP: 8-16"
        MIN_MEMORY=120
        ;;
esac

echo ""

# 检查 GPU 资源
echo "检查 GPU 资源..."
if command -v nvidia-smi &> /dev/null; then
    gpu_count=$(nvidia-smi --list-gpus | wc -l)
    echo "可用 GPU 数量: $gpu_count"
    
    if [[ $gpu_count -lt $TP_SIZE ]]; then
        echo "❌ 错误: 需要至少 $TP_SIZE 个 GPU，但只检测到 $gpu_count 个"
        echo "请调整 --tp-size 参数或增加 GPU 资源"
        exit 1
    fi
    
    # 检查总内存
    total_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{sum+=$1} END {print sum/1024}')
    echo "总 GPU 内存: ${total_memory}GB"
    
    if [[ $(echo "$total_memory < $MIN_MEMORY" | bc -l) -eq 1 ]]; then
        echo "⚠️  警告: 总 GPU 内存可能不足 (推荐至少 ${MIN_MEMORY}GB)"
        echo "建议使用更多 GPU 或更大的张量并行度"
    fi
    
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits | \
    awk -F', ' '{printf "GPU %d: %s, 总计 %dGB, 可用 %dGB\n", $1, $2, $3/1024, $4/1024}'
    
else
    echo "⚠️  无法检查 GPU 状态，请确保 NVIDIA 驱动正常"
fi

echo ""
echo "开始调优..."

# 设置环境变量
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}

# 为 DeepSeek 模型设置特殊参数
extra_args=""
if [[ "$MODEL_VERSION" == "v3" ]]; then
    extra_args="--enable-expert-parallel"
fi

# 运行调优
python benchmark_moe.py \
    --model "$MODEL" \
    --tp-size $TP_SIZE \
    --dtype "$DTYPE" \
    --batch-size $BATCH_SIZES \
    --tune \
    --save-dir "$SAVE_DIR" \
    --trust-remote-code \
    --seed 42 \
    $extra_args

# 检查调优结果
if [[ $? -eq 0 ]]; then
    echo ""
    echo "✅ 调优完成！"
    echo ""
    echo "生成的配置文件:"
    find "$SAVE_DIR" -name "*deepseek*${MODEL_VERSION}*tp${TP_SIZE}*${DTYPE}*" -type f | head -5
    
    echo ""
    echo "性能优化建议:"
    case $MODEL_VERSION in
        "v2")
            echo "1. DeepSeek V2 具有大量专家(160个)，建议使用 FP8 量化以节省内存"
            echo "2. 对于在线服务，推荐使用较小的批次大小 (1-32)"
            echo "3. 考虑启用专家并行以进一步优化性能"
            ;;
        "v3")
            echo "1. DeepSeek V3 是最大的 MoE 模型，强烈建议使用 FP8 量化"
            echo "2. 推荐使用高端 GPU 如 H100 以获得最佳性能"
            echo "3. 考虑增加张量并行度到 16 以减少单 GPU 内存压力"
            ;;
    esac
    
    echo ""
    echo "下一步操作:"
    echo "1. 运行基准测试: bash scripts/benchmark_performance.sh --model deepseek_${MODEL_VERSION}"
    echo "2. 分析结果: python tools/analyze_results.py --model deepseek_${MODEL_VERSION}"
    
else
    echo ""
    echo "❌ 调优失败，请检查错误信息"
    echo ""
    echo "DeepSeek 模型常见问题:"
    echo "1. 内存不足 - 尝试增加 TP 大小或使用量化"
    echo "2. 网络问题 - 确保能访问 Hugging Face"
    echo "3. 权限问题 - 某些 DeepSeek 模型需要申请访问权限"
    exit 1
fi

echo ""
echo "==============================================="
echo "DeepSeek $MODEL_VERSION 调优完成"
echo "==============================================="