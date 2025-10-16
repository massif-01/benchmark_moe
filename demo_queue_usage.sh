#!/bin/bash

# 🎯 vLLM MoE 调优队列管理演示脚本
# 演示如何使用队列系统避免资源竞争，获得最佳调优结果

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}🎯 vLLM MoE 调优队列管理演示${NC}"
echo -e "${CYAN}===============================================${NC}"
echo ""

# 检查脚本是否存在
QUEUE_SCRIPT="./scripts/queue_manager.sh"
if [[ ! -f "$QUEUE_SCRIPT" ]]; then
    echo -e "${RED}❌ 队列管理脚本不存在: $QUEUE_SCRIPT${NC}"
    exit 1
fi

echo -e "${BLUE}📋 步骤1: 查看当前队列状态${NC}"
echo "运行命令: bash $QUEUE_SCRIPT status"
bash "$QUEUE_SCRIPT" status
echo ""

echo -e "${BLUE}📦 步骤2: 添加调优任务到队列${NC}"
echo "我们将演示添加3个不同的模型调优任务："
echo ""

# 任务1: Qwen3-30B 全面调优
echo -e "${YELLOW}📌 任务1: Qwen3-30B 全面调优${NC}"
echo "特点: 高优先级，全批次测试，预计3.2小时"
echo "运行命令: bash $QUEUE_SCRIPT add configs/tasks/qwen3_30b_comprehensive_task.json"

if [[ -f "configs/tasks/qwen3_30b_comprehensive_task.json" ]]; then
    bash "$QUEUE_SCRIPT" add configs/tasks/qwen3_30b_comprehensive_task.json
else
    echo -e "${YELLOW}⚠️ 任务文件不存在，跳过添加${NC}"
fi
echo ""

# 任务2: Mixtral-8x7B 优化
echo -e "${YELLOW}📌 任务2: Mixtral-8x7B 优化${NC}"
echo "特点: 中等优先级，FP8量化，预计1.8小时"
echo "运行命令: bash $QUEUE_SCRIPT add configs/tasks/mixtral_8x7b_task.json"

if [[ -f "configs/tasks/mixtral_8x7b_task.json" ]]; then
    bash "$QUEUE_SCRIPT" add configs/tasks/mixtral_8x7b_task.json
else
    echo -e "${YELLOW}⚠️ 任务文件不存在，跳过添加${NC}"
fi
echo ""

# 任务3: DeepSeek-V2 快速测试
echo -e "${YELLOW}📌 任务3: DeepSeek-V2 快速测试${NC}"
echo "特点: 低优先级，快速验证，预计0.8小时"
echo "运行命令: bash $QUEUE_SCRIPT add configs/tasks/deepseek_v2_task.json"

if [[ -f "configs/tasks/deepseek_v2_task.json" ]]; then
    bash "$QUEUE_SCRIPT" add configs/tasks/deepseek_v2_task.json
else
    echo -e "${YELLOW}⚠️ 任务文件不存在，跳过添加${NC}"
fi
echo ""

echo -e "${BLUE}📊 步骤3: 查看更新后的队列状态${NC}"
echo "运行命令: bash $QUEUE_SCRIPT status"
bash "$QUEUE_SCRIPT" status
echo ""

echo -e "${BLUE}⏱️ 步骤4: 估算任务完成时间${NC}"
for task_file in configs/tasks/*.json; do
    if [[ -f "$task_file" ]]; then
        task_name=$(basename "$task_file" .json)
        echo -e "${CYAN}📋 $task_name:${NC}"
        echo "运行命令: bash $QUEUE_SCRIPT estimate $task_file"
        bash "$QUEUE_SCRIPT" estimate "$task_file" 2>/dev/null || echo "无法估算"
        echo ""
    fi
done

echo -e "${PURPLE}🚀 步骤5: 如何启动队列处理${NC}"
echo "要开始自动处理队列中的所有任务，运行："
echo -e "${GREEN}bash $QUEUE_SCRIPT start${NC}"
echo ""
echo "这将会："
echo "• ✅ 按优先级顺序执行任务"
echo "• ✅ 确保每次只运行一个任务（避免资源竞争）"
echo "• ✅ 自动处理错误和重试"
echo "• ✅ 生成详细的执行日志"
echo "• ✅ 保存优化后的配置文件"
echo ""

echo -e "${PURPLE}📊 步骤6: 如何监控进度${NC}"
echo "在另一个终端中实时监控任务进度："
echo -e "${GREEN}bash $QUEUE_SCRIPT monitor${NC}"
echo ""
echo "监控功能包括："
echo "• 📈 实时GPU使用情况"
echo "• 📋 当前任务状态"
echo "• 📊 最新执行日志"
echo "• ⏱️ 剩余任务数量"
echo ""

echo -e "${BLUE}🔧 步骤7: 队列管理操作${NC}"
echo "其他有用的队列管理命令："
echo ""
echo -e "${CYAN}查看状态:${NC} bash $QUEUE_SCRIPT status"
echo -e "${CYAN}清空队列:${NC} bash $QUEUE_SCRIPT clear"
echo -e "${CYAN}添加任务:${NC} bash $QUEUE_SCRIPT add <task_file.json>"
echo -e "${CYAN}监控进度:${NC} bash $QUEUE_SCRIPT monitor"
echo -e "${CYAN}估算时间:${NC} bash $QUEUE_SCRIPT estimate <task_file.json>"
echo ""

echo -e "${GREEN}✨ 示例任务配置文件结构:${NC}"
echo ""
cat << 'EOF'
{
  "name": "模型名称",
  "model": "模型路径或HuggingFace名称",
  "tp_size": 1,
  "dtype": "auto|fp8_w8a8|int8_w8a16",
  "batch_sizes": [1, 2, 4, 8, 16, 32, 64],
  "save_dir": "./results/tuned_configs/model_name",
  "priority": 1,
  "estimated_time_hours": 2.5,
  "description": "任务描述",
  "additional_args": ["--trust-remote-code", "--use-deep-gemm"]
}
EOF
echo ""

echo -e "${PURPLE}🎯 为什么使用队列系统？${NC}"
echo ""
echo -e "${RED}❌ 同时运行多个调优的问题:${NC}"
echo "• GPU内存不足 (OOM错误)"
echo "• Triton编译缓存冲突"
echo "• Ray集群资源争夺"
echo "• 调优结果不准确"
echo "• 系统不稳定"
echo ""
echo -e "${GREEN}✅ 队列系统的优势:${NC}"
echo "• 确保资源独占使用"
echo "• 获得真正的最优配置"
echo "• 系统稳定运行"
echo "• 自动错误处理"
echo "• 详细的执行记录"
echo ""

echo -e "${BLUE}📚 更多信息${NC}"
echo "• 部署指南: deployment/DEPLOYMENT_GUIDE.md"
echo "• 配置示例: configs/tasks/"
echo "• 结果目录: results/tuned_configs/"
echo "• 执行日志: results/tuning_logs/"
echo ""

echo -e "${CYAN}🎉 演示完成！现在您可以安全地进行多模型调优了。${NC}"
echo -e "${YELLOW}💡 提示: 建议先运行小批次测试验证环境，再进行完整调优。${NC}"