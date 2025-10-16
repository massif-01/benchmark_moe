#!/bin/bash

# 智能队列管理脚本 - 避免资源竞争的模型调优
# 确保每次只运行一个调优任务，获得最佳配置结果

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 配置文件
QUEUE_DIR="/tmp/benchmark_moe_queue"
LOCK_FILE="/tmp/benchmark_moe.lock"
LOG_DIR="./results/tuning_logs"

# 创建必要目录
mkdir -p "$QUEUE_DIR" "$LOG_DIR"

# 帮助信息
show_help() {
    echo "使用方法: $0 [选项] [任务文件]"
    echo ""
    echo "选项:"
    echo "  add <task_file>     添加调优任务到队列"
    echo "  start              开始处理队列中的任务"
    echo "  status             查看队列状态"
    echo "  clear              清空队列"
    echo "  monitor            实时监控任务进度"
    echo "  estimate <task>    估算任务完成时间"
    echo ""
    echo "任务文件格式示例 (JSON):"
    echo "{"
    echo "  \"name\": \"qwen3_30b_tuning\","
    echo "  \"model\": \"/path/to/qwen3-30b\","
    echo "  \"tp_size\": 1,"
    echo "  \"dtype\": \"auto\","
    echo "  \"batch_sizes\": [1, 2, 4, 8, 16, 32, 64],"
    echo "  \"priority\": 1,"
    echo "  \"estimated_time_hours\": 2.5"
    echo "}"
}

# 检查是否有任务正在运行
is_tuning_running() {
    if [[ -f "$LOCK_FILE" ]]; then
        PID=$(cat "$LOCK_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            return 0  # 正在运行
        else
            rm -f "$LOCK_FILE"  # 清理无效的锁文件
            return 1  # 没有运行
        fi
    fi
    return 1  # 没有运行
}

# 估算任务完成时间
estimate_task_time() {
    local task_file="$1"
    if [[ ! -f "$task_file" ]]; then
        echo "任务文件不存在: $task_file"
        return 1
    fi
    
    local batch_count=$(jq '.batch_sizes | length' "$task_file")
    local model_name=$(jq -r '.model' "$task_file" | xargs basename)
    
    # 基于经验的时间估算 (分钟)
    local base_time=10  # 每个batch size的基础时间
    local model_factor=1
    
    case "$model_name" in
        *"qwen3"*|*"Qwen3"*) model_factor=1.2 ;;
        *"mixtral"*|*"Mixtral"*) model_factor=1.0 ;;
        *"deepseek"*|*"DeepSeek"*) model_factor=1.5 ;;
        *) model_factor=1.0 ;;
    esac
    
    local estimated_minutes=$((batch_count * base_time))
    case "$model_name" in
        *"qwen3"*|*"Qwen3"*) estimated_minutes=$((estimated_minutes * 12 / 10)) ;;
        *"mixtral"*|*"Mixtral"*) estimated_minutes=$((estimated_minutes * 10 / 10)) ;;
        *"deepseek"*|*"DeepSeek"*) estimated_minutes=$((estimated_minutes * 15 / 10)) ;;
        *) estimated_minutes=$((estimated_minutes * 10 / 10)) ;;
    esac
    
    local estimated_hours=$((estimated_minutes / 60))
    local remaining_minutes=$((estimated_minutes % 60))
    
    echo "${estimated_hours}小时${remaining_minutes}分钟"
}

# 添加任务到队列
add_task() {
    local task_file="$1"
    if [[ ! -f "$task_file" ]]; then
        echo -e "${RED}❌ 任务文件不存在: $task_file${NC}"
        return 1
    fi
    
    # 验证JSON格式
    if ! jq empty "$task_file" 2>/dev/null; then
        echo -e "${RED}❌ 任务文件格式错误，请检查JSON语法${NC}"
        return 1
    fi
    
    local task_name=$(jq -r '.name' "$task_file")
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local queue_file="$QUEUE_DIR/${timestamp}_${task_name}.json"
    
    # 添加时间戳和状态
    jq ". + {\"queued_at\": \"$(date -Iseconds)\", \"status\": \"queued\"}" "$task_file" > "$queue_file"
    
    local estimated_time=$(estimate_task_time "$task_file")
    echo -e "${GREEN}✅ 任务已添加到队列: $task_name${NC}"
    echo -e "${BLUE}📊 预估完成时间: $estimated_time${NC}"
    echo -e "${BLUE}📋 队列文件: $queue_file${NC}"
}

# 查看队列状态
show_status() {
    echo -e "${BLUE}📋 当前队列状态:${NC}"
    
    if is_tuning_running; then
        local running_pid=$(cat "$LOCK_FILE")
        echo -e "${YELLOW}🔄 正在运行: PID $running_pid${NC}"
        
        # 尝试从日志中获取当前任务信息
        local current_log=$(ls -t "$LOG_DIR"/*.log 2>/dev/null | head -1)
        if [[ -n "$current_log" ]]; then
            echo -e "${BLUE}📊 当前日志: $(basename "$current_log")${NC}"
            tail -3 "$current_log" | sed 's/^/   /'
        fi
    else
        echo -e "${GREEN}✅ 当前没有任务运行${NC}"
    fi
    
    local queued_tasks=($(ls "$QUEUE_DIR"/*.json 2>/dev/null | sort))
    if [[ ${#queued_tasks[@]} -eq 0 ]]; then
        echo -e "${GREEN}📋 队列为空${NC}"
    else
        echo -e "${BLUE}📋 队列中的任务 (${#queued_tasks[@]}个):${NC}"
        for i in "${!queued_tasks[@]}"; do
            local task_file="${queued_tasks[$i]}"
            local task_name=$(jq -r '.name' "$task_file")
            local queued_at=$(jq -r '.queued_at' "$task_file")
            local estimated_time=$(estimate_task_time "$task_file")
            
            echo -e "  $((i+1)). ${task_name} (排队时间: $queued_at, 预估: $estimated_time)"
        done
    fi
}

# 执行单个调优任务
run_task() {
    local task_file="$1"
    local task_name=$(jq -r '.name' "$task_file")
    local log_file="$LOG_DIR/${task_name}_$(date +%Y%m%d_%H%M%S).log"
    
    echo -e "${BLUE}🚀 开始执行任务: $task_name${NC}"
    echo "$$" > "$LOCK_FILE"
    
    # 更新任务状态
    jq '. + {"status": "running", "started_at": "'$(date -Iseconds)'"}' "$task_file" > "${task_file}.tmp"
    mv "${task_file}.tmp" "$task_file"
    
    # 构建benchmark命令
    local model=$(jq -r '.model' "$task_file")
    local tp_size=$(jq -r '.tp_size // 1' "$task_file")
    local dtype=$(jq -r '.dtype // "auto"' "$task_file")
    local batch_sizes=($(jq -r '.batch_sizes[]' "$task_file"))
    local save_dir=$(jq -r '.save_dir // "./results/tuned_configs"' "$task_file")
    local additional_args=($(jq -r '.additional_args[]? // empty' "$task_file"))
    
    local cmd="python benchmark_moe.py \
        --model '$model' \
        --tp-size $tp_size \
        --dtype $dtype \
        --batch-size ${batch_sizes[*]} \
        --tune \
        --save-dir '$save_dir' \
        --seed 42"
    
    # 添加额外参数
    if [[ ${#additional_args[@]} -gt 0 ]]; then
        cmd="$cmd ${additional_args[*]}"
    fi
    
    echo -e "${BLUE}📋 执行命令: $cmd${NC}"
    echo "开始时间: $(date)" > "$log_file"
    echo "任务: $task_name" >> "$log_file"
    echo "命令: $cmd" >> "$log_file"
    echo "===================" >> "$log_file"
    
    # 执行调优
    if eval "$cmd" >> "$log_file" 2>&1; then
        echo -e "${GREEN}✅ 任务完成: $task_name${NC}"
        jq '. + {"status": "completed", "completed_at": "'$(date -Iseconds)'"}' "$task_file" > "${task_file}.tmp"
        mv "${task_file}.tmp" "$task_file"
        
        # 移动到完成目录
        mkdir -p "$QUEUE_DIR/completed"
        mv "$task_file" "$QUEUE_DIR/completed/"
    else
        echo -e "${RED}❌ 任务失败: $task_name${NC}"
        jq '. + {"status": "failed", "failed_at": "'$(date -Iseconds)'"}' "$task_file" > "${task_file}.tmp"
        mv "${task_file}.tmp" "$task_file"
        
        # 移动到失败目录
        mkdir -p "$QUEUE_DIR/failed"
        mv "$task_file" "$QUEUE_DIR/failed/"
    fi
    
    echo "结束时间: $(date)" >> "$log_file"
    rm -f "$LOCK_FILE"
}

# 处理队列中的所有任务
process_queue() {
    echo -e "${BLUE}🔄 开始处理队列...${NC}"
    
    while true; do
        if is_tuning_running; then
            echo -e "${YELLOW}⏳ 检测到任务正在运行，等待完成...${NC}"
            sleep 30
            continue
        fi
        
        # 获取下一个任务
        local next_task=($(ls "$QUEUE_DIR"/*.json 2>/dev/null | sort | head -1))
        if [[ ${#next_task[@]} -eq 0 ]]; then
            echo -e "${GREEN}🎉 队列处理完成，所有任务已执行${NC}"
            break
        fi
        
        run_task "${next_task[0]}"
        sleep 5  # 任务间短暂休息
    done
}

# 实时监控
monitor_progress() {
    echo -e "${BLUE}📊 实时监控模式 (Ctrl+C 退出)${NC}"
    
    while true; do
        clear
        echo "=== vLLM MoE 调优任务监控 ==="
        echo "时间: $(date)"
        echo ""
        
        show_status
        
        if is_tuning_running; then
            echo ""
            echo -e "${BLUE}🔍 GPU使用情况:${NC}"
            nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | while IFS=',' read gpu_util mem_used mem_total temp; do
                echo "  GPU利用率: ${gpu_util}%, 显存: ${mem_used}MB/${mem_total}MB, 温度: ${temp}°C"
            done
            
            echo ""
            echo -e "${BLUE}📊 最新日志 (最后5行):${NC}"
            local current_log=$(ls -t "$LOG_DIR"/*.log 2>/dev/null | head -1)
            if [[ -n "$current_log" ]]; then
                tail -5 "$current_log" | sed 's/^/  /'
            fi
        fi
        
        sleep 10
    done
}

# 清空队列
clear_queue() {
    if is_tuning_running; then
        echo -e "${RED}❌ 有任务正在运行，无法清空队列${NC}"
        return 1
    fi
    
    read -p "确认清空队列? (y/N): " confirm
    if [[ "$confirm" == "y" || "$confirm" == "Y" ]]; then
        rm -f "$QUEUE_DIR"/*.json
        echo -e "${GREEN}✅ 队列已清空${NC}"
    fi
}

# 主逻辑
case "${1:-help}" in
    "add")
        if [[ -z "$2" ]]; then
            echo -e "${RED}❌ 请指定任务文件${NC}"
            show_help
            exit 1
        fi
        add_task "$2"
        ;;
    "start")
        process_queue
        ;;
    "status")
        show_status
        ;;
    "monitor")
        monitor_progress
        ;;
    "clear")
        clear_queue
        ;;
    "estimate")
        if [[ -z "$2" ]]; then
            echo -e "${RED}❌ 请指定任务文件${NC}"
            exit 1
        fi
        echo "预估完成时间: $(estimate_task_time "$2")"
        ;;
    "help"|*)
        show_help
        ;;
esac