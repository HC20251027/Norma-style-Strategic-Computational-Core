# 贡献指南

感谢您对诺玛式战略计算中枢项目的兴趣！我们欢迎所有形式的贡献。

## 如何贡献

### 报告问题
如果您发现了bug或有功能建议，请创建一个Issue：
- 使用清晰简洁的标题
- 详细描述问题或建议
- 提供重现步骤（如果适用）
- 包含环境信息（Python版本、操作系统等）

### 代码贡献

#### 开发环境设置
1. Fork本仓库
2. 克隆您的fork到本地
3. 创建虚拟环境
4. 安装开发依赖
5. 运行测试确保环境正常

```bash
# 克隆仓库
git clone https://github.com/HC20251027/Norma-style-Strategic-Computational-Core.git
cd Norma-style-Strategic-Computational-Core

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
pip install -r requirements-dev.txt  # 如果有开发依赖
```

#### 代码规范
- 遵循PEP 8代码风格
- 使用有意义的变量和函数名
- 添加适当的文档字符串
- 编写单元测试
- 确保所有测试通过

#### 提交规范
使用清晰的提交信息：
- `feat:` 新功能
- `fix:` bug修复
- `docs:` 文档更新
- `style:` 代码格式调整
- `refactor:` 代码重构
- `test:` 测试相关
- `chore:` 构建过程或辅助工具的变动

示例：
```
feat: 添加新的音频处理模块
fix: 修复聊天界面的内存泄漏问题
docs: 更新API文档
```

#### Pull Request流程
1. 从主分支创建功能分支
2. 进行代码修改并添加测试
3. 确保所有测试通过
4. 更新相关文档
5. 提交Pull Request

PR描述应包含：
- 修改内容概述
- 测试结果
- 相关的Issue编号

## 项目结构
```
诺玛式战略计算中枢/
├── ai_core/           # AI核心模块
│   ├── agents/        # 智能体组件
│   ├── knowledge_base/ # 知识库
│   ├── learning_engine/ # 学习引擎
│   └── reasoning_system/ # 推理系统
├── core/              # 核心计算引擎
│   ├── computational_engine/ # 计算引擎
│   ├── decision_engine/ # 决策引擎
│   ├── optimization_core/ # 优化核心
│   └── strategic_analyzer/ # 战略分析器
├── data_hub/          # 数据中心
│   ├── analytics/     # 分析模块
│   ├── pipelines/     # 数据管道
│   ├── processors/    # 数据处理器
│   └── storage/       # 存储模块
└── docs/              # 文档
```

## 测试
运行测试套件：
```bash
pytest tests/
```

运行特定测试：
```bash
pytest tests/test_specific_module.py
```

## 许可证
通过贡献代码，您同意您的贡献将在MIT许可证下发布。

## 联系我们
如有问题，请创建Issue或联系项目维护者。

感谢您的贡献！