# benchmark_moe

[🇨🇳 中文](README_zh.md) | [🇺🇸 English](README.md)

A high-performance optimization tool for vLLM MoE (Mixture of Experts) model kernel tuning

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![vLLM](https://img.shields.io/badge/vLLM-0.10.0+-green.svg)](https://github.com/vllm-project/vllm)

A specialized toolkit for optimizing MoE model inference performance in the vLLM framework through automated Triton kernel parameter tuning, finding optimal execution configurations for different model architectures and hardware setups.

## 🎯 Key Features

- **🔧 Automated Kernel Tuning**: Use Ray distributed framework to automatically search for optimal Triton kernel configurations
- **📊 Multi-Model Support**: Support mainstream MoE models including Mixtral, DeepSeek, Qwen, Jamba, etc.
- **⚡ Performance Optimization**: Specialized optimization for different batch sizes and hardware configurations
- **🛠️ Fault Diagnosis**: Complete environment checking and troubleshooting tools
- **📈 Result Analysis**: Generate detailed performance reports and configuration recommendations

## 🚀 Quick Start

### Prerequisites

- **Hardware**: NVIDIA GPU (recommended A100/H100)
- **Software**: Ubuntu 18.04+, Python 3.11+, CUDA 11.8+
- **Dependencies**: vLLM 0.10.0+, PyTorch 2.0+, Ray

### Installation

1. **Clone the project**
   ```bash
   git clone https://github.com/massif-01/benchmark_moe.git
   cd benchmark_moe
   ```

2. **Environment check**
   ```bash
   bash scripts/server_check.sh
   ```

3. **Run single model tuning**
   ```bash
   # Basic tuning - Qwen3 model
   python benchmark_moe.py \
     --model /path/to/your/qwen3-model \
     --tp-size 1 \
     --dtype auto \
     --batch-size 1 2 4 8 16 32 64 128 \
     --tune \
     --save-dir ./optimized_configs \
     --trust-remote-code
   ```

4. **View results**
   ```bash
   ls ./optimized_configs/
   # Output: E64N9472_tp1_fp16.json (example config file)
   ```

## 📋 Detailed Usage Guide

### Environment Setup

#### System Environment Check
```bash
# Run environment check script
bash scripts/server_check.sh

# Check GPU status
nvidia-smi

# Check Python dependencies
python -c "import vllm, ray, torch, triton; print('Environment check passed')"
```

#### Common Environment Issues

**Issue 1: Triton Cache Corruption**
```bash
# Clear Triton cache (if encountering JSONDecodeError)
rm -rf ~/.triton/cache/*

# Or set new cache directory
export TRITON_CACHE_DIR=/tmp/triton_cache_$(date +%s)
mkdir -p $TRITON_CACHE_DIR
```

**Issue 2: libstdc++ Version Issues**
```bash
# Update libstdc++ in conda environment
conda install -c conda-forge libstdcxx-ng

# Or use system library
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

**Issue 3: Ray Warnings**
```bash
# Suppress Ray-related warnings
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export RAY_DISABLE_IMPORT_WARNING=1
```

### Tuning Parameters

#### Basic Parameters
- `--model`: Model path or HuggingFace model name
- `--tp-size`: Tensor parallelism degree (set according to GPU count)
- `--dtype`: Data type (`auto`, `fp8_w8a8`, `int8_w8a16`)
- `--batch-size`: List of batch sizes to test
- `--tune`: Enable tuning mode
- `--save-dir`: Configuration file save directory

#### Advanced Parameters
- `--use-deep-gemm`: Enable DeepGEMM optimization
- `--enable-expert-parallel`: Enable expert parallelism (for large models)
- `--seed`: Random seed (ensures reproducible results)

### Supported Model Types

| Model Series | Experts | Top-K | Recommended VRAM | Example Command |
|-------------|---------|-------|------------------|-----------------|
| **Qwen3-30B-A3B** | 64 | 4 | 64GB+ | `--model path/to/qwen3 --tp-size 1` |
| **Mixtral-8x7B** | 8 | 2 | 45GB+ | `--model mistralai/Mixtral-8x7B-Instruct-v0.1 --tp-size 2` |
| **DeepSeek-V2** | 160 | 6 | 80GB+ | `--model deepseek-ai/DeepSeek-V2-Chat --tp-size 4` |
| **DeepSeek-V3** | 256 | 8 | 120GB+ | `--model deepseek-ai/DeepSeek-V3-Base --tp-size 8` |

### Batch Tuning Scripts

#### Using Configuration Management Tool
```bash
# List supported models
python tools/config_manager.py list

# Tune specific model
python tools/config_manager.py tune qwen3_30b

# View configuration recommendations
python tools/config_manager.py recommend qwen3_30b
```

#### Safe Batch Tuning
```bash
# Use safe script to test batch sizes one by one
bash scripts/run_benchmark_safe.sh
```

## 📊 Result Interpretation

### Configuration File Format
```json
{
  "triton_version": "2.1.0",
  "1": {                    // Optimal config for batch_size=1
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 64, 
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 1,
    "num_warps": 4,
    "num_stages": 3
  },
  "64": {                   // Optimal config for batch_size=64
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 256,
    "GROUP_SIZE_M": 32,
    "num_warps": 8,
    "num_stages": 4
  }
}
```

### Performance Analysis
```bash
# Run performance benchmark (without --tune)
python benchmark_moe.py \
  --model your_model \
  --tp-size 1 \
  --batch-size 1 2 4 8 16 32 64 128

# Example output:
# Batch size: 1, Kernel time: 45.23 us
# Batch size: 64, Kernel time: 892.15 us
```

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### 1. Out of Memory Errors
```bash
# Symptom: CUDA out of memory
# Solutions:
# - Reduce batch size
--batch-size 1 2 4 8 16 32

# - Use quantization
--dtype fp8_w8a8

# - Increase tensor parallelism (if multi-GPU)
--tp-size 2
```

#### 2. Triton Compilation Errors
```bash
# Symptom: JSONDecodeError, OutOfResources
# Solution:
rm -rf ~/.triton/cache/*
export TRITON_CACHE_DIR=/tmp/triton_cache_new
```

#### 3. Model Loading Failures
```bash
# Symptom: Model not found, Permission denied
# Solutions:
# - Check model path
ls /path/to/your/model

# - Add access permissions
--trust-remote-code

# - Pre-download model
huggingface-cli download model_name --local-dir ./models/
```

#### 4. Ray Initialization Issues
```bash
# Symptom: Ray cannot start
# Solution:
export RAY_DISABLE_IMPORT_WARNING=1
ray stop  # Stop existing instance
ray start --head  # Restart
```

### Performance Tuning Recommendations

#### For Different Use Cases

**Low Latency Scenarios (Online Inference)**
```bash
# Optimize small batch performance
--batch-size 1 2 4 8 16
--dtype fp8_w8a8  # Reduce memory access latency
```

**High Throughput Scenarios (Batch Processing)**
```bash
# Optimize large batch performance  
--batch-size 64 128 256 512 1024
--dtype auto  # Balance precision and performance
```

**Memory-Constrained Scenarios**
```bash
# Maximize memory utilization
--dtype fp8_w8a8
--use-deep-gemm
--enable-expert-parallel
```

## 📁 Project Structure

```
benchmark_moe/
├── README.md                   # English documentation
├── README_zh.md               # Chinese documentation  
├── LICENSE                     # Apache-2.0 license
├── .gitignore                  # Git ignore rules
├── benchmark_moe.py           # vLLM MoE benchmark core script
├── scripts/                   # Script directory
│   ├── server_check.sh        # Server environment check script
│   ├── tune_mixtral.sh        # Mixtral model tuning script
│   ├── tune_deepseek.sh       # DeepSeek model tuning script
│   └── run_benchmark_safe.sh  # Safe batch tuning script
├── configs/                   # Configuration directory
│   ├── models.json            # Supported model configurations
│   └── tuning_params.json     # Tuning parameter configurations
├── tools/                     # Analysis tools directory
│   └── config_manager.py      # Configuration management tool
└── deployment/                # Deployment related files
    ├── requirements.txt       # Python dependencies list
    └── DEPLOYMENT_GUIDE.md    # Detailed deployment guide
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit Issues and Pull Requests.

### Development Environment Setup
```bash
git clone https://github.com/massif-01/benchmark_moe.git
cd benchmark_moe
pip install -r deployment/requirements.txt
```

### Contributing Code
1. Fork this project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the Apache-2.0 License. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [vLLM](https://github.com/vllm-project/vllm) - High-performance LLM inference engine
- [Ray](https://github.com/ray-project/ray) - Distributed computing framework
- [Triton](https://github.com/openai/triton) - GPU programming language

## 📮 Contact

For questions or suggestions, please contact us via:

- Submit [GitHub Issue](https://github.com/massif-01/benchmark_moe/issues)
- Start a [Discussion](https://github.com/massif-01/benchmark_moe/discussions)

---

**⭐ If this project helps you, please give us a Star!**