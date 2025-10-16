#!/bin/bash

# libstdc++ 兼容性修复脚本
# 解决 GLIBCXX 版本不兼容问题

set -e

echo "🔧 检测并修复 libstdc++ 兼容性问题..."

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 检查当前环境
echo -e "${BLUE}📋 当前环境信息:${NC}"
echo "Conda环境: $(conda info --envs | grep '*' | awk '{print $1}')"
echo "Python版本: $(python --version)"
echo "PyTorch版本: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo '未安装')"

# 检查libstdc++版本
echo -e "\n${BLUE}🔍 检查libstdc++版本...${NC}"

# 检查conda环境中的libstdc++
LIBSTDCXX_VERSION=$(conda list libstdcxx-ng 2>/dev/null | grep libstdcxx-ng | awk '{print $2}' || echo "未安装")
echo "Conda libstdcxx-ng版本: $LIBSTDCXX_VERSION"

# 检查系统libstdc++
SYSTEM_LIBSTD=$(find /usr/lib* -name "libstdc++.so.6*" 2>/dev/null | head -1)
if [[ -n "$SYSTEM_LIBSTD" ]]; then
    echo "系统libstdc++路径: $SYSTEM_LIBSTD"
    SYSTEM_GLIBCXX=$(strings "$SYSTEM_LIBSTD" 2>/dev/null | grep GLIBCXX | tail -1 || echo "无法检测")
    echo "系统GLIBCXX版本: $SYSTEM_GLIBCXX"
fi

# 测试PyTorch导入
echo -e "\n${BLUE}🧪 测试PyTorch导入...${NC}"
if python -c "import torch; print('✅ PyTorch导入成功')" 2>/dev/null; then
    echo -e "${GREEN}✅ 当前环境正常，无需修复${NC}"
    exit 0
else
    echo -e "${RED}❌ PyTorch导入失败，开始修复...${NC}"
fi

# 修复选项
echo -e "\n${YELLOW}🛠️ 选择修复方案:${NC}"
echo "1) 更新conda环境中的libstdc++ (推荐)"
echo "2) 使用系统libstdc++"
echo "3) 安装完整的GCC工具链"
echo "4) 创建新的clean环境"
echo "5) 手动设置环境变量"

read -p "请选择方案 (1-5): " choice

case $choice in
    1)
        echo -e "${BLUE}📦 更新conda libstdc++...${NC}"
        conda install -c conda-forge libstdcxx-ng=12 -y
        echo "✅ libstdc++更新完成"
        ;;
    2)
        echo -e "${BLUE}🔗 配置使用系统libstdc++...${NC}"
        if [[ -n "$SYSTEM_LIBSTD" ]]; then
            SYSTEM_LIB_DIR=$(dirname "$SYSTEM_LIBSTD")
            echo "export LD_LIBRARY_PATH=$SYSTEM_LIB_DIR:\$LD_LIBRARY_PATH" >> ~/.bashrc
            export LD_LIBRARY_PATH=$SYSTEM_LIB_DIR:$LD_LIBRARY_PATH
            echo "✅ 系统库路径已添加到LD_LIBRARY_PATH"
        else
            echo -e "${RED}❌ 未找到系统libstdc++${NC}"
            exit 1
        fi
        ;;
    3)
        echo -e "${BLUE}🔨 安装完整GCC工具链...${NC}"
        conda install -c conda-forge gcc_linux-64=12 gxx_linux-64=12 libstdcxx-ng=12 -y
        echo "✅ GCC工具链安装完成"
        ;;
    4)
        echo -e "${BLUE}🆕 创建新环境...${NC}"
        ENV_NAME="vllm_fixed_$(date +%Y%m%d)"
        echo "环境名称: $ENV_NAME"
        
        conda create -n $ENV_NAME python=3.11 -y
        echo "请运行以下命令切换到新环境:"
        echo -e "${GREEN}conda activate $ENV_NAME${NC}"
        echo -e "${GREEN}conda install -c conda-forge libstdcxx-ng=12 gcc_linux-64=12 -y${NC}"
        echo -e "${GREEN}pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121${NC}"
        echo -e "${GREEN}pip install vllm ray${NC}"
        exit 0
        ;;
    5)
        echo -e "${BLUE}⚙️ 设置环境变量...${NC}"
        echo "# libstdc++ 兼容性修复" >> ~/.bashrc
        echo "export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH" >> ~/.bashrc
        echo "export CONDA_DLL_SEARCH_MODIFICATION_ENABLE=1" >> ~/.bashrc
        
        export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
        export CONDA_DLL_SEARCH_MODIFICATION_ENABLE=1
        
        echo "✅ 环境变量已设置"
        echo "请运行: source ~/.bashrc"
        ;;
    *)
        echo -e "${RED}❌ 无效选择${NC}"
        exit 1
        ;;
esac

# 验证修复结果
echo -e "\n${BLUE}🧪 验证修复结果...${NC}"
if python -c "import torch; print(f'✅ PyTorch {torch.__version__} 导入成功')" 2>/dev/null; then
    echo -e "${GREEN}🎉 修复成功！可以正常使用PyTorch了${NC}"
    
    # 测试vLLM
    if python -c "import vllm; print(f'✅ vLLM {vllm.__version__} 导入成功')" 2>/dev/null; then
        echo -e "${GREEN}🎉 vLLM也正常工作！${NC}"
    else
        echo -e "${YELLOW}⚠️ vLLM可能需要重新安装${NC}"
        echo "运行: pip install --force-reinstall vllm"
    fi
else
    echo -e "${RED}❌ 修复失败，请尝试其他方案或联系技术支持${NC}"
    exit 1
fi

echo -e "\n${GREEN}🎯 修复完成！现在可以运行benchmark_moe.py了${NC}"