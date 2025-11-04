#!/bin/bash

# MCP工具系统启动脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# 显示帮助信息
show_help() {
    echo "MCP工具系统启动脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help          显示此帮助信息"
    echo "  -d, --demo          运行演示模式"
    echo "  -s, --server        启动服务器模式"
    echo "  -t, --test          运行测试"
    echo "  -i, --install       安装依赖"
    echo "  -c, --check         检查系统状态"
    echo "  --init              初始化系统"
    echo ""
    echo "示例:"
    echo "  $0 --demo           # 运行演示"
    echo "  $0 --server         # 启动服务器"
    echo "  $0 --init           # 初始化系统"
}

# 检查Python环境
check_python() {
    log_step "检查Python环境..."
    
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 未安装"
        exit 1
    fi
    
    python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
    log_info "Python版本: $python_version"
    
    # 检查Python版本 (需要3.8+)
    if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
        log_error "需要Python 3.8或更高版本"
        exit 1
    fi
}

# 安装依赖
install_dependencies() {
    log_step "安装依赖包..."
    
    # 检查requirements.txt是否存在
    if [ ! -f "requirements.txt" ]; then
        log_warn "requirements.txt 不存在，创建基础依赖文件..."
        cat > requirements.txt << EOF
# MCP工具系统依赖
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.0.0
psutil>=5.9.0
pyyaml>=6.0
asyncio-mqtt>=0.16.0
aiofiles>=23.0.0
python-multipart>=0.0.6
jinja2>=3.1.0
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4
redis>=5.0.0
sqlalchemy>=2.0.0
alembic>=1.12.0
httpx>=0.25.0
websockets>=12.0
EOF
    fi
    
    # 安装依赖
    if command -v pip3 &> /dev/null; then
        pip3 install -r requirements.txt
    elif command -v pip &> /dev/null; then
        pip install -r requirements.txt
    else
        log_error "未找到pip包管理器"
        exit 1
    fi
    
    log_info "依赖安装完成"
}

# 初始化系统
init_system() {
    log_step "初始化MCP系统..."
    
    # 设置Python路径
    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
    
    # 运行初始化
    python3 -c "
import sys
sys.path.append('.')
from mcp.config.config_manager import config_manager
from mcp.core.tool_registry import global_registry

# 初始化配置
config_manager.load_config()
print('✓ 配置初始化完成')

# 初始化工具注册
import asyncio
from mcp.core.mcp_server import initialize_tools, initialize_skills

async def init():
    await initialize_tools()
    await initialize_skills()
    print('✓ 工具和技能初始化完成')

asyncio.run(init())
"
    
    log_info "系统初始化完成"
}

# 检查系统状态
check_system() {
    log_step "检查系统状态..."
    
    # 检查Python环境
    check_python
    
    # 检查依赖
    log_info "检查依赖包..."
    python3 -c "
import sys
missing_packages = []
required_packages = ['fastapi', 'uvicorn', 'pydantic', 'psutil', 'yaml']

for package in required_packages:
    try:
        if package == 'yaml':
            import yaml
        else:
            __import__(package)
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    print(f'缺少依赖包: {missing_packages}')
    sys.exit(1)
else:
    print('✓ 所有依赖包已安装')
"
    
    if [ $? -ne 0 ]; then
        log_error "依赖检查失败，请运行: $0 --install"
        exit 1
    fi
    
    # 检查配置
    log_info "检查配置文件..."
    if [ ! -f "config/mcp_config.yaml" ]; then
        log_warn "配置文件不存在，将使用默认配置"
    else
        log_info "✓ 配置文件存在"
    fi
    
    # 检查工具注册
    log_info "检查工具注册状态..."
    python3 -c "
import sys
sys.path.append('.')
from mcp.core.tool_registry import global_registry
from mcp.config.config_manager import config_manager

config_manager.load_config()
tools = global_registry.list_tools()
print(f'✓ 已注册 {len(tools)} 个工具')
"
    
    log_info "系统状态检查完成"
}

# 运行演示
run_demo() {
    log_step "运行MCP系统演示..."
    
    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
    
    python3 src/mcp/example.py
}

# 启动服务器
start_server() {
    log_step "启动MCP服务器..."
    
    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
    
    # 检查端口是否被占用
    if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
        log_warn "端口8000已被占用，尝试终止现有进程..."
        lsof -ti:8000 | xargs kill -9 2>/dev/null || true
        sleep 2
    fi
    
    # 启动服务器
    python3 -c "
import uvicorn
import sys
sys.path.append('.')
from mcp.core.mcp_server import mcp_server

print('启动MCP服务器...')
print('访问地址: http://localhost:8000')
print('API文档: http://localhost:8000/docs')
print('按 Ctrl+C 停止服务器')

uvicorn.run(mcp_server, host='0.0.0.0', port=8000, log_level='info')
"
}

# 运行测试
run_tests() {
    log_step "运行系统测试..."
    
    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
    
    # 创建简单的测试
    python3 -c "
import sys
sys.path.append('.')
import asyncio

async def test_basic_functionality():
    print('开始基础功能测试...')
    
    # 测试工具注册
    from mcp.core.tool_registry import global_registry
    tools = global_registry.list_tools()
    print(f'✓ 工具注册测试: {len(tools)} 个工具')
    
    # 测试权限管理
    from mcp.security.permission_manager import permission_manager
    users = len(permission_manager.users)
    print(f'✓ 权限管理测试: {users} 个用户')
    
    # 测试技能系统
    from mcp.skills import skill_manager
    skills = len(skill_manager.list_skills())
    print(f'✓ 技能系统测试: {skills} 个技能')
    
    # 测试配置管理
    from mcp.config.config_manager import config_manager
    config = config_manager.get_config()
    print(f'✓ 配置管理测试: {config.environment} 环境')
    
    print('所有基础测试通过!')

asyncio.run(test_basic_functionality())
"
    
    if [ $? -eq 0 ]; then
        log_info "系统测试完成 - 全部通过"
    else
        log_error "系统测试失败"
        exit 1
    fi
}

# 主函数
main() {
    # 设置脚本目录
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "$SCRIPT_DIR"
    
    log_info "MCP工具系统启动脚本"
    log_info "当前目录: $(pwd)"
    
    # 解析命令行参数
    case "${1:-}" in
        -h|--help)
            show_help
            exit 0
            ;;
        -i|--install)
            check_python
            install_dependencies
            ;;
        --init)
            check_python
            init_system
            ;;
        -c|--check)
            check_system
            ;;
        -t|--test)
            check_system
            run_tests
            ;;
        -d|--demo)
            check_system
            run_demo
            ;;
        -s|--server)
            check_system
            start_server
            ;;
        "")
            log_info "未指定参数，显示帮助信息:"
            show_help
            ;;
        *)
            log_error "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@"