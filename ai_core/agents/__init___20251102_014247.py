"""
Toolcall抽象层实现

实现职责分离：
- Agent: 负责调度（调用工具、管理上下文）
- MCP: 负责执行（运行工具函数）
- LLM: 负责规划（工具选择、参数推断）

主要组件：
- ToolcallManager: 工具调用管理器
- ToolSelector: 智能工具选择器
- ParameterInferencer: 参数推断器
- ContextManager: 上下文管理器
- PerformanceOptimizer: 性能优化器
- ErrorHandler: 错误处理器
"""

from .core import (
    ToolcallManager,
    ToolSelector,
    ParameterInferencer,
    ContextManager,
    PerformanceOptimizer,
    ErrorHandler
)

from .llm_interface import LLMInterface
from .models import (
    ToolcallRequest,
    ToolcallResponse,
    ToolcallContext,
    ToolSelectionResult,
    ParameterInferenceResult,
    ExecutionPlan
)

from .registry import ToolcallRegistry

__all__ = [
    # 核心组件
    "ToolcallManager",
    "ToolSelector", 
    "ParameterInferencer",
    "ContextManager",
    "PerformanceOptimizer",
    "ErrorHandler",
    
    # 接口
    "LLMInterface",
    
    # 数据模型
    "ToolcallRequest",
    "ToolcallResponse", 
    "ToolcallContext",
    "ToolSelectionResult",
    "ParameterInferenceResult",
    "ExecutionPlan",
    
    # 注册中心
    "ToolcallRegistry"
]

__version__ = "1.0.0"