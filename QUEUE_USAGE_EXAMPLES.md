# 🎯 队列管理系统使用示例

## 📋 快速开始

### 1. 环境测试（推荐第一步）
```bash
# 添加快速测试任务验证环境
bash scripts/queue_manager.sh add configs/tasks/environment_test.json

# 启动队列处理
bash scripts/queue_manager.sh start
```

### 2. 添加多个调优任务
```bash
# 高优先级：Qwen3-30B 全面调优
bash scripts/queue_manager.sh add configs/tasks/qwen3_30b_comprehensive_task.json

# 中等优先级：Mixtral-8x7B 优化
bash scripts/queue_manager.sh add configs/tasks/mixtral_8x7b_task.json

# 低优先级：DeepSeek-V2 快速测试
bash scripts/queue_manager.sh add configs/tasks/deepseek_v2_task.json
```

### 3. 查看队列状态
```bash
bash scripts/queue_manager.sh status
```

### 4. 启动自动处理
```bash
# 在主终端启动队列处理
bash scripts/queue_manager.sh start
```

### 5. 监控进度（另一个终端）
```bash
# 实时监控任务进度
bash scripts/queue_manager.sh monitor
```

## 📊 实际使用场景示例

### 场景1: 单个模型深度优化
```bash
# 创建任务配置
cat > my_model_task.json << EOF
{
  "name": "my_qwen3_optimization",
  "model": "/path/to/your/qwen3-model",
  "tp_size": 1,
  "dtype": "auto",
  "batch_sizes": [1, 2, 4, 8, 16, 32, 64, 128, 256],
  "save_dir": "./results/my_qwen3_configs",
  "priority": 1,
  "estimated_time_hours": 4.0,
  "description": "深度优化Qwen3模型，测试大批次性能",
  "additional_args": ["--trust-remote-code", "--use-deep-gemm"]
}
EOF

# 添加到队列并执行
bash scripts/queue_manager.sh add my_model_task.json
bash scripts/queue_manager.sh start
```

### 场景2: 多模型批量测试
```bash
# 批量添加任务
for model in qwen3_30b mixtral_8x7b deepseek_v2; do
    bash scripts/queue_manager.sh add configs/tasks/${model}_task.json
done

# 查看队列
bash scripts/queue_manager.sh status

# 开始批量处理
bash scripts/queue_manager.sh start
```

### 场景3: 不同量化方案对比
```bash
# 创建FP8量化任务
cat > qwen3_fp8_task.json << EOF
{
  "name": "qwen3_fp8_quantization",
  "model": "/path/to/qwen3",
  "dtype": "fp8_w8a8",
  "batch_sizes": [1, 4, 16, 64],
  "save_dir": "./results/qwen3_fp8"
}
EOF

# 创建INT8量化任务
cat > qwen3_int8_task.json << EOF
{
  "name": "qwen3_int8_quantization", 
  "model": "/path/to/qwen3",
  "dtype": "int8_w8a16",
  "batch_sizes": [1, 4, 16, 64],
  "save_dir": "./results/qwen3_int8"
}
EOF

# 添加对比任务
bash scripts/queue_manager.sh add qwen3_fp8_task.json
bash scripts/queue_manager.sh add qwen3_int8_task.json
bash scripts/queue_manager.sh start
```

## 🔍 监控和调试

### 实时监控输出示例：
```
=== vLLM MoE 调优任务监控 ===
时间: 2025-10-16 18:30:00

📋 当前队列状态:
🔄 正在运行: PID 12345
📊 当前日志: qwen3_30b_comprehensive_20251016_183000.log

🔍 GPU使用情况:
  GPU利用率: 95%, 显存: 75GB/80GB, 温度: 78°C

📊 最新日志 (最后5行):
  Batch size: 16, Testing configuration 45/120
  Current best: BLOCK_SIZE_M=128, BLOCK_SIZE_N=64, time=245.6us
  Progress: [███████████████████████▒▒▒▒▒▒▒] 75%
  ETA: 32 minutes remaining
  Memory usage: 75.2GB/80GB
```

### 查看详细日志：
```bash
# 查看最新日志
tail -f results/tuning_logs/$(ls -t results/tuning_logs/*.log | head -1)

# 查看特定任务日志
ls results/tuning_logs/qwen3_*
```

## 📁 结果文件结构

执行完成后，文件结构如下：
```
results/
├── tuned_configs/           # 优化配置文件
│   ├── qwen3_30b/
│   │   └── config_qwen3.json
│   ├── mixtral_8x7b/
│   │   └── config_mixtral.json
│   └── deepseek_v2/
│       └── config_deepseek.json
├── tuning_logs/             # 执行日志
│   ├── qwen3_30b_20251016_183000.log
│   ├── mixtral_8x7b_20251016_190000.log
│   └── deepseek_v2_20251016_193000.log
└── performance_reports/     # 性能报告（如果生成）
```

## ⚠️ 注意事项

### 资源管理
- ✅ **推荐**: 使用队列系统逐个执行
- ❌ **避免**: 同时运行多个调优脚本
- 💡 **建议**: 先运行小规模测试验证环境

### 任务优先级
- `priority: 0` - 测试任务
- `priority: 1` - 高优先级（重要模型）
- `priority: 2` - 中等优先级
- `priority: 3` - 低优先级（实验性）

### 最佳实践
1. 从环境测试开始
2. 使用合适的批次大小范围
3. 根据GPU显存调整量化设置
4. 保存重要的配置文件
5. 定期清理Triton缓存

## 🆘 常见问题

### Q: 如何停止正在运行的任务？
```bash
# 查找进程ID
bash scripts/queue_manager.sh status

# 终止进程
kill <PID>

# 清理锁文件
rm -f /tmp/benchmark_moe.lock
```

### Q: 任务失败了怎么办？
```bash
# 查看失败的任务
ls /tmp/benchmark_moe_queue/failed/

# 重新添加到队列
bash scripts/queue_manager.sh add /tmp/benchmark_moe_queue/failed/task_name.json
```

### Q: 如何清空队列？
```bash
bash scripts/queue_manager.sh clear
```

这个队列系统确保您能安全、高效地进行多模型调优，避免资源竞争，获得最佳的配置结果！