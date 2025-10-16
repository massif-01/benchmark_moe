#!/bin/bash

# 安全的 MoE 基准测试运行脚本
# 逐个测试不同的批次大小，避免缓存问题

set -e

echo "==============================================="
echo "MoE 模型安全批量基准测试脚本"
echo "==============================================="

# 默认参数
MODEL_PATH="/home/rm01/models/dev/llm/Qwen3-30B-A3B-Instruct-2507-AWQ"
TP_SIZE=1
DTYPE="auto"
SAVE_DIR="./optimized_configs"
BATCH_SIZES=(1 2 4 8 16 32 64 128)

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
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
        --batch-sizes)
            IFS=',' read -ra BATCH_SIZES <<< "$2"
            shift 2
            ;;
        --help)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --model PATH          模型路径"
            echo "  --tp-size SIZE        张量并行大小"
            echo "  --dtype TYPE          数据类型 [auto|fp8_w8a8|int8_w8a16]"
            echo "  --save-dir DIR        保存目录"
            echo "  --batch-sizes SIZES   批次大小列表（逗号分隔）"
            echo "  --help                显示此帮助"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 设置环境变量
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export RAY_DISABLE_IMPORT_WARNING=1

# 创建新的 Triton 缓存目录
export TRITON_CACHE_DIR=/tmp/triton_cache_$(date +%s)
mkdir -p $TRITON_CACHE_DIR
echo "使用 Triton 缓存目录: $TRITON_CACHE_DIR"

# 创建保存目录
mkdir -p "$SAVE_DIR"

echo ""
echo "运行配置:"
echo "  模型: $MODEL_PATH"
echo "  张量并行度: $TP_SIZE"
echo "  数据类型: $DTYPE"
echo "  保存目录: $SAVE_DIR"
echo "  批次大小: ${BATCH_SIZES[*]}"
echo ""

# 检查 GPU 状态
echo "检查 GPU 状态..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits | \
    awk -F', ' '{printf "GPU: %s, 总内存: %dGB, 可用: %dGB\n", $1, $2/1024, $3/1024}'
else
    echo "⚠️  无法检查 GPU 状态"
fi

echo ""
echo "开始逐个批次大小调优..."

# 成功和失败的计数
success_count=0
total_count=${#BATCH_SIZES[@]}
failed_batches=()

# 逐个测试批次大小
for batch_size in "${BATCH_SIZES[@]}"; do
    echo ""
    echo "========================================="
    echo "正在测试 batch_size: $batch_size"
    echo "进度: $((success_count + ${#failed_batches[@]} + 1))/$total_count"
    echo "========================================="
    
    # 运行调优
    python benchmark_moe.py \
        --model "$MODEL_PATH" \
        --tp-size $TP_SIZE \
        --dtype "$DTYPE" \
        --batch-size $batch_size \
        --tune \
        --save-dir "$SAVE_DIR" \
        --trust-remote-code \
        --seed 42
    
    # 检查结果
    if [ $? -eq 0 ]; then
        echo "✅ batch_size $batch_size 调优成功"
        ((success_count++))
    else
        echo "❌ batch_size $batch_size 调优失败"
        failed_batches+=($batch_size)
        
        # 清理可能损坏的缓存
        echo "清理 Triton 缓存..."
        rm -rf $TRITON_CACHE_DIR/*
    fi
    
    # 短暂休息
    echo "等待 10 秒后继续..."
    sleep 10
done

echo ""
echo "==============================================="
echo "批量调优完成"
echo "==============================================="
echo "成功: $success_count/$total_count"

if [ ${#failed_batches[@]} -gt 0 ]; then
    echo "失败的批次大小: ${failed_batches[*]}"
    echo ""
    echo "建议："
    echo "1. 检查失败批次的显存使用情况"
    echo "2. 尝试使用量化: --dtype fp8_w8a8"
    echo "3. 单独重试失败的批次大小"
else
    echo "🎉 所有批次大小调优成功！"
fi

echo ""
echo "生成的配置文件:"
find "$SAVE_DIR" -name "*.json" -type f | head -10

echo ""
echo "下一步建议:"
echo "1. 运行性能基准测试验证配置效果"
echo "2. 分析调优结果: python tools/config_manager.py recommend"
echo "3. 在生产环境中应用最优配置"

# 清理临时缓存目录
echo ""
echo "清理临时缓存目录: $TRITON_CACHE_DIR"
rm -rf $TRITON_CACHE_DIR

echo "脚本执行完成！"