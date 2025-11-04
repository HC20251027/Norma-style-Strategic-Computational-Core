# 诺玛式战略计算中枢 (Norma-style Strategic Computational Core)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()

## 项目简介

诺玛式战略计算中枢是一个先进的AI智能体系统，集成了多模态处理、战略分析、决策优化和知识管理等核心功能。该系统采用模块化架构设计，支持高并发处理和智能协作。

## 核心特性

### 🚀 多模态能力
- **文本处理**: 自然语言理解与生成
- **语音服务**: 语音识别、合成与处理
- **图像处理**: 图像分析、编辑与生成
- **视频处理**: 视频生成与分析

### 🧠 智能架构
- **五层智能体系统**: 专业化分工的智能体团队
- **知识库与记忆系统**: 持续学习与知识积累
- **多模态集成**: 统一的多模态处理接口
- **实时交互**: WebSocket支持的低延迟通信

### 🎯 专业功能
- **团队协作**: 多智能体协作模式
- **性能监控**: 实时系统健康监控
- **品牌一致性**: 统一的品牌形象与交互风格
- **部署支持**: 多种云平台部署方案

## 项目结构

```
norma-agent/
├── src/                    # 源代码
│   ├── core/              # 核心组件
│   ├── agents/            # 智能体模块
│   ├── ui/                # 用户界面
│   ├── utils/             # 工具函数
│   └── config/            # 配置文件
├── tests/                 # 测试文件
├── docs/                  # 文档
├── deployment/            # 部署配置
├── scripts/               # 部署脚本
└── examples/              # 示例代码
```

## 快速开始

### 环境要求
- Python 3.12+
- Node.js 18+
- Docker (可选)

### 安装步骤

1. **克隆项目**
```bash
git clone <repository-url>
cd norma-agent
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **配置环境**
```bash
cp .env.example .env
# 编辑 .env 文件配置必要的API密钥
```

4. **启动服务**
```bash
python src/main.py
```

### Docker部署

```bash
docker-compose up -d
```

## 核心模块

### 1. 核心引擎 (Core Engine)
- `norma_agent.py`: 主要智能体实现
- `conversation_engine.py`: 对话管理引擎
- `memory_manager.py`: 记忆管理系统
- `event_system.py`: 事件驱动系统

### 2. 多模态接口 (Multimodal Interface)
- `multimodal_interface.py`: 统一多模态接口
- `speech_services/`: 语音处理服务
- `voice_pipeline/`: 语音处理管道

### 3. 用户界面 (User Interface)
- `chat_interface.py`: 聊天界面
- `conversation_view.py`: 对话视图
- AG-UI集成支持

### 4. 智能体团队 (Agent Team)
- 专业智能体池
- 任务分配与协调
- 协作模式管理

## API文档

### 核心API端点

#### 聊天接口
```http
POST /api/chat
Content-Type: application/json

{
  "message": "你好，诺玛",
  "user_id": "user123",
  "session_id": "session456"
}
```

#### 多模态处理
```http
POST /api/multimodal/process
Content-Type: multipart/form-data

{
  "type": "image",
  "data": "<binary_data>",
  "options": {}
}
```

## 配置说明

### 环境变量
```bash
# API配置
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# 数据库配置
DATABASE_URL=sqlite:///./data/norma.db

# 服务配置
HOST=0.0.0.0
PORT=8000
DEBUG=false
```

### 配置文件
- `config/settings.py`: 主配置文件
- `config/norma_config.json`: 诺玛特定配置
- `monitoring_config.json`: 监控配置

## 监控与日志

### 健康检查
```bash
curl http://localhost:8000/health
```

### 性能监控
- 实时性能指标
- 系统资源使用情况
- 用户交互统计

### 日志管理
- 结构化日志记录
- 错误追踪与报告
- 审计日志

## 部署指南

### 支持的平台
- **Railway**: 一键部署
- **Render**: 简单部署
- **Heroku**: 传统部署
- **Docker**: 容器化部署
- **本地部署**: 开发环境

### 部署脚本
```bash
# Railway部署
./scripts/deploy_railway.sh

# Render部署
./scripts/deploy_render.sh

# Heroku部署
./scripts/deploy_heroku.sh
```

## 开发指南

### 开发环境设置
```bash
# 创建虚拟环境
python -m venv agno_env
source agno_env/bin/activate  # Linux/Mac
# agno_env\Scripts\activate  # Windows

# 安装开发依赖
pip install -r requirements-dev.txt
```

### 代码规范
- 使用Black进行代码格式化
- 使用Flake8进行代码检查
- 遵循PEP 8标准
- 添加适当的类型注解

### 测试
```bash
# 运行所有测试
python -m pytest tests/

# 运行特定测试
python -m pytest tests/test_norma_core.py

# 生成覆盖率报告
python -m pytest --cov=src tests/
```

## 贡献指南

1. Fork项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

## 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 联系方式

- 项目作者: 诺玛开发团队
- 邮箱: norma-team@example.com
- 项目主页: https://github.com/your-org/norma-agent

## 更新日志

### v2.0.0 (2025-11-01)
- ✨ 全新五层智能体系统架构
- 🚀 增强的多模态处理能力
- 📊 实时性能监控系统
- 🎨 优化的用户界面体验
- 🔧 简化的部署流程

### v1.5.0
- 添加知识库与记忆系统
- 改进团队协作模式
- 优化语音处理流程

### v1.0.0
- 初始版本发布
- 基础多模态功能
- AG-UI集成

## 致谢

感谢以下开源项目：
- [Agno](https://github.com/agno-agi/agno) - AGI框架
- [AG-UI](https://github.com/agno-agi/ag-ui) - UI SDK
- [FastAPI](https://fastapi.tiangolo.com/) - Web框架
- [WebSocket](https://websockets.readthedocs.io/) - 实时通信

---

*诺玛Agent - 让AI更智能，让交互更自然*