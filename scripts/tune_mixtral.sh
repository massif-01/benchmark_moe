#!/bin/bash

# Mixtral 8x7B 模型调优脚本
# 适用于具有充足显存的服务器环境

set -e

echo "==============================================="
echo "Mixtral 8x7B MoE 内核调优脚本"
echo "==============================================="

# 默认参数
MODEL="mistralai/Mixtral-8x7B-Instruct-v0.1"
TP_SIZE=2
DTYPE="auto"
SAVE_DIR="./results/tuned_configs"
SCENARIO="standard"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
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
            echo "  --tp-size SIZE        张量并行大小 (默认: 2)"
            echo "  --dtype TYPE          数据类型 [auto|fp8_w8a8|int8_w8a16] (默认: auto)"
            echo "  --save-dir DIR        保存目录 (默认: ./results/tuned_configs)"
            echo "  --scenario SCENARIO   调优场景 [quick|standard|comprehensive] (默认: standard)"
            echo "  --quick               快速调优 (等同于 --scenario quick)"
            echo "  --comprehensive       全面调优 (等同于 --scenario comprehensive)"
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
        BATCH_SIZES="1 2 4 8 16 32"
        echo "🚀 快速调优模式 - 批次大小: $BATCH_SIZES"
        ;;
    "comprehensive")
        BATCH_SIZES="1 2 4 8 16 24 32 48 64 96 128 256 512 1024 1536 2048"
        echo "🔬 全面调优模式 - 批次大小: $BATCH_SIZES"
        ;;
    *)
        BATCH_SIZES="8 16 32 64 128 256 512"
        echo "⚖️  标准调优模式 - 批次大小: $BATCH_SIZES"
        ;;
esac

echo ""
echo "调优配置:"
echo "  模型: $MODEL"
echo "  张量并行度: $TP_SIZE"
echo "  数据类型: $DTYPE"
echo "  保存目录: $SAVE_DIR"
echo "  调优场景: $SCENARIO"
echo ""

# 检查 GPU 内存
echo "检查 GPU 内存..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits | \
    awk -F', ' '{printf "GPU: %s, 总内存: %dGB, 可用: %dGB\n", $1, $2/1024, $3/1024}'
    
    # 检查内存是否足够
    free_memory=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [[ $free_memory -lt 40000 ]]; then
        echo "⚠️  警告: GPU 内存可能不足 (推荐至少 40GB 用于 Mixtral 8x7B)"
        echo "考虑使用更大的张量并行度或释放 GPU 内存"
    fi
else
    echo "⚠️  无法检查 GPU 状态，请确保 NVIDIA 驱动正常"
fi

echo ""
echo "开始调优..."

# 设置环境变量
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}

# 运行调优
python benchmark_moe.py \
    --model "$MODEL" \
    --tp-size $TP_SIZE \
    --dtype "$DTYPE" \
    --batch-size $BATCH_SIZES \
    --tune \
    --save-dir "$SAVE_DIR" \
    --trust-remote-code \
    --seed 42

# 检查调优结果
if [[ $? -eq 0 ]]; then
    echo ""
    echo "✅ 调优完成！"
    echo ""
    echo "生成的配置文件:"
    find "$SAVE_DIR" -name "*mixtral*tp${TP_SIZE}*${DTYPE}*" -type f | head -5
    
    echo ""
    echo "下一步建议:"
    echo "1. 运行性能基准测试验证调优效果:"
    echo "   bash scripts/benchmark_performance.sh --model mixtral_8x7b --tp-size $TP_SIZE"
    echo ""
    echo "2. 查看调优结果分析:"
    echo "   python tools/analyze_results.py --config-dir $SAVE_DIR"
    echo ""
    echo "3. 在生产环境中应用优化配置"
    
else
    echo ""
    echo "❌ 调优失败，请检查错误信息"
    echo ""
    echo "常见问题排查:"
    echo "1. 检查 GPU 内存是否足够"
    echo "2. 确认 vLLM 安装正确"
    echo "3. 验证模型名称和参数"
    echo "4. 检查网络连接是否正常"
    exit 1
fi

echo ""
echo "==============================================="
echo "Mixtral 8x7B 调优完成"
echo "==============================================="