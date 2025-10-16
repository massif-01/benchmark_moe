# vLLM MoE Benchmark Deployment Guide

[🇺🇸 English](DEPLOYMENT_GUIDE.md) | [🇨🇳 中文](DEPLOYMENT_GUIDE_zh.md)

This document provides detailed deployment and troubleshooting guidelines.

## System Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA Compute Capability 7.0+ (recommended A100/H100)
- **VRAM**: At least 40GB (adjust according to model size)
- **RAM**: At least 32GB RAM
- **Storage**: At least 100GB available space

### Software Requirements
- **OS**: Ubuntu 18.04+ / CentOS 7+ / RHEL 7+
- **Python**: 3.11+
- **CUDA**: 11.8+ (recommended 12.1+)
- **Docker**: Optional, for containerized deployment

## Installation Steps

### 1. Environment Preparation

#### Check CUDA Environment
```bash
nvidia-smi
nvcc --version
```

#### Create Python Environment
```bash
# Using conda (recommended)
conda create -n benchmark_moe python=3.11
conda activate benchmark_moe

# Or using venv
python3.11 -m venv benchmark_moe_env
source benchmark_moe_env/bin/activate
```

### 2. Install Dependencies

#### Basic Dependencies
```bash
# Install PyTorch (CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install vLLM
pip install vllm

# Install other dependencies
pip install -r deployment/requirements.txt
```

#### Verify Installation
```bash
python -c "
import torch
import vllm
import ray
import triton
print('✅ All dependencies installed successfully')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'vLLM: {vllm.__version__}')
"
```

### 3. Model Preparation

#### Download Models
```bash
# Using huggingface-cli
huggingface-cli download mistralai/Mixtral-8x7B-Instruct-v0.1 --local-dir ./models/mixtral-8x7b

# Or using Python script
python -c "
from transformers import AutoTokenizer, AutoConfig
model_name = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='./models')
config = AutoConfig.from_pretrained(model_name, cache_dir='./models')
print('Model configuration downloaded')
"
```

## Configuration Optimization

### Environment Variables
```bash
# Add to ~/.bashrc or ~/.zshrc
export CUDA_VISIBLE_DEVICES=0  # Adjust according to available GPUs
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export RAY_DISABLE_IMPORT_WARNING=1
export TRITON_CACHE_DIR=/tmp/triton_cache
```

### Memory Optimization
```bash
# Set GPU memory allocation strategy
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Enable memory mapping
export VLLM_USE_MODELSCOPE=0
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1
```

## Troubleshooting

### Common Issues

#### 1. CUDA Related Errors

**Error**: `CUDA out of memory`
```bash
# Solutions:
# - Reduce batch size
python benchmark_moe.py --batch-size 1 2 4 8

# - Use quantization
python benchmark_moe.py --dtype fp8_w8a8

# - Clear GPU memory
nvidia-smi --gpu-reset
```

**Error**: `CUDA driver version is insufficient`
```bash
# Check driver version
nvidia-smi

# Update NVIDIA driver
sudo ubuntu-drivers autoinstall
sudo reboot
```

#### 2. Triton Compilation Errors

**Error**: `JSONDecodeError: Expecting value`
```bash
# Clear Triton cache
rm -rf ~/.triton/cache/*
rm -rf /tmp/triton_cache*

# Set new cache directory
export TRITON_CACHE_DIR=/tmp/triton_cache_$(date +%s)
mkdir -p $TRITON_CACHE_DIR
```

**Error**: `OutOfResources`
```bash
# Reduce search space
python benchmark_moe.py --batch-size 1 2 4  # Smaller batch range

# Or use quick mode
bash scripts/run_benchmark_safe.sh --batch-sizes 1,2,4,8
```

#### 3. Ray Related Issues

**Error**: `Ray cluster initialization failed`
```bash
# Stop existing Ray processes
ray stop

# Restart
ray start --head --port=6379

# Or clean Ray temp files
rm -rf /tmp/ray*
```

#### 4. Model Loading Errors

**Error**: `Model not found` or `Permission denied`
```bash
# Check model path
ls -la /path/to/your/model

# Set correct permissions
chmod -R 755 /path/to/your/model

# Use trust remote code
python benchmark_moe.py --trust-remote-code
```

**Error**: `Tokenizer not found`
```bash
# Pre-download tokenizer
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('your_model_name')
"
```

#### 5. Network Related Issues

**Error**: `Connection timeout` (accessing HuggingFace)
```bash
# Set proxy (if needed)
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080

# Or use mirror
export HF_ENDPOINT=https://hf-mirror.com
```

### Performance Tuning Recommendations

#### Optimization for Different Hardware

**Single GPU (A100 80GB)**
```bash
python benchmark_moe.py \
  --tp-size 1 \
  --dtype fp8_w8a8 \
  --batch-size 1 2 4 8 16 32 64 128
```

**Multi GPU (2x A100)**
```bash
python benchmark_moe.py \
  --tp-size 2 \
  --dtype auto \
  --batch-size 16 32 64 128 256 512
```

**Memory-Constrained Environment**
```bash
python benchmark_moe.py \
  --tp-size 1 \
  --dtype fp8_w8a8 \
  --batch-size 1 2 4 8 \
  --use-deep-gemm
```

## Monitoring and Debugging

### Performance Monitoring
```bash
# GPU usage monitoring
watch -n 1 nvidia-smi

# Memory usage monitoring
htop

# Network I/O monitoring
iotop
```

### Debug Logging
```bash
# Enable verbose logging
export VLLM_LOGGING_LEVEL=DEBUG
export RAY_VERBOSE=1

# Run debug mode
python benchmark_moe.py --batch-size 1 --tune 2>&1 | tee debug.log
```

## Containerized Deployment

### Dockerfile Example
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

### Docker Run
```bash
# Build image
docker build -t benchmark_moe .

# Run container
docker run --gpus all -v /path/to/models:/app/models benchmark_moe \
  python benchmark_moe.py --model /app/models/your_model --tune
```

## Best Practices

### 1. Resource Management
- Use `nvidia-smi` to regularly check GPU usage
- Set appropriate batch sizes to avoid memory overflow
- Regularly clean Triton cache to avoid disk space issues

### 2. Tuning Strategy
- Start with small batches and gradually increase
- Prioritize testing commonly used batch sizes
- Save tuning results for future use

### 3. Production Deployment
- Use stable model versions
- Set appropriate timeout values
- Establish monitoring and alerting mechanisms

## Technical Support

If you encounter issues, please troubleshoot in the following order:

1. **Check Environment**: Run `bash scripts/server_check.sh`
2. **Review Logs**: Check error messages and stack traces
3. **Search Known Issues**: Check project Issues page
4. **Create New Issue**: Provide detailed error information and environment configuration

### When Reporting Issues, Please Include:
- Operating system and version
- GPU model and driver version
- Python and dependency package versions
- Complete error logs
- Steps to reproduce