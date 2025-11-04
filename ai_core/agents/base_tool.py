"""
基础工具类
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..core.models import ToolDefinition, ToolParameter, ToolCategory, SecurityLevel


logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """基础工具类"""
    
    def __init__(self):
        self._definition: Optional[ToolDefinition] = None
        self._execution_count = 0
        self._last_execution: Optional[datetime] = None
    
    @abstractmethod
    def get_tool_definition(self) -> ToolDefinition:
        """获取工具定义"""
        pass
    
    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """执行工具"""
        pass
    
    def get_definition(self) -> ToolDefinition:
        """获取工具定义（缓存）"""
        if self._definition is None:
            self._definition = self.get_tool_definition()
        return self._definition
    
    def validate_parameters(self, **kwargs) -> tuple[bool, Optional[str]]:
        """验证参数"""
        definition = self.get_definition()
        
        # 检查必需参数
        for param in definition.parameters:
            if param.required and param.name not in kwargs:
                return False, f"缺少必需参数: {param.name}"
        
        # 检查参数类型和值
        for param_name, value in kwargs.items():
            param_def = next((p for p in definition.parameters if p.name == param_name), None)
            if param_def:
                # 类型检查
                if param_def.type != "any" and not isinstance(value, self._get_python_type(param_def.type)):
                    return False, f"参数 {param_name} 类型错误，期望: {param_def.type}"
                
                # 选项检查
                if param_def.options and value not in param_def.options:
                    return False, f"参数 {param_name} 值不在允许范围内: {param_def.options}"
                
                # 自定义验证
                if param_def.validation and not param_def.validation(value):
                    return False, f"参数 {param_name} 验证失败"
        
        return True, None
    
    def _get_python_type(self, type_str: str):
        """获取Python类型"""
        type_mapping = {
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict
        }
        return type_mapping.get(type_str, str)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取工具统计"""
        return {
            "execution_count": self._execution_count,
            "last_execution": self._last_execution.isoformat() if self._last_execution else None,
            "definition": self.get_definition().to_dict()
        }
    
    def _update_execution_stats(self):
        """更新执行统计"""
        self._execution_count += 1
        self._last_execution = datetime.now()


class SystemMonitorTool(BaseTool):
    """系统监控工具基类"""
    
    def get_tool_definition(self) -> ToolDefinition:
        return ToolDefinition(
            id="system_monitor_base",
            name="系统监控工具",
            description="系统资源监控和管理",
            category=ToolCategory.SYSTEM_MONITOR,
            security_level=SecurityLevel.MEDIUM,
            parameters=[
                ToolParameter("metric", "str", "监控指标", True, options=["cpu", "memory", "disk", "network"]),
                ToolParameter("duration", "int", "监控时长（秒）", False, 60),
                ToolParameter("threshold", "float", "告警阈值", False, 80.0)
            ]
        )
    
    async def execute(self, metric: str, duration: int = 60, threshold: float = 80.0) -> Dict[str, Any]:
        """执行系统监控"""
        self._update_execution_stats()
        
        # 这里应该实现具体的监控逻辑
        return {
            "metric": metric,
            "current_value": 0.0,
            "threshold": threshold,
            "status": "normal",
            "duration": duration
        }


class NetworkSecurityTool(BaseTool):
    """网络安全工具基类"""
    
    def get_tool_definition(self) -> ToolDefinition:
        return ToolDefinition(
            id="network_security_base",
            name="网络安全工具",
            description="网络安全检测和分析",
            category=ToolCategory.NETWORK_SECURITY,
            security_level=SecurityLevel.HIGH,
            parameters=[
                ToolParameter("target", "str", "目标地址", True),
                ToolParameter("scan_type", "str", "扫描类型", True, options=["port", "vulnerability", "malware"]),
                ToolParameter("timeout", "int", "超时时间", False, 30)
            ]
        )
    
    async def execute(self, target: str, scan_type: str, timeout: int = 30) -> Dict[str, Any]:
        """执行网络安全检测"""
        self._update_execution_stats()
        
        # 这里应该实现具体的安全检测逻辑
        return {
            "target": target,
            "scan_type": scan_type,
            "status": "completed",
            "findings": []
        }


class BloodlineAnalysisTool(BaseTool):
    """血统分析工具基类"""
    
    def get_tool_definition(self) -> ToolDefinition:
        return ToolDefinition(
            id="bloodline_analysis_base",
            name="血统分析工具",
            description="龙族血统检测和分析",
            category=ToolCategory.BLOODLINE_ANALYSIS,
            security_level=SecurityLevel.MEDIUM,
            parameters=[
                ToolParameter("sample_data", "str", "样本数据", True),
                ToolParameter("analysis_type", "str", "分析类型", True, options=["purity", "heritage", "traits"]),
                ToolParameter("depth", "int", "分析深度", False, 3)
            ]
        )
    
    async def execute(self, sample_data: str, analysis_type: str, depth: int = 3) -> Dict[str, Any]:
        """执行血统分析"""
        self._update_execution_stats()
        
        # 这里应该实现具体的血统分析逻辑
        return {
            "sample_data": sample_data,
            "analysis_type": analysis_type,
            "depth": depth,
            "results": {}
        }


class DataProcessingTool(BaseTool):
    """数据处理工具基类"""
    
    def get_tool_definition(self) -> ToolDefinition:
        return ToolDefinition(
            id="data_processing_base",
            name="数据处理工具",
            description="数据处理和转换",
            category=ToolCategory.DATA_PROCESSING,
            security_level=SecurityLevel.LOW,
            parameters=[
                ToolParameter("data", "any", "输入数据", True),
                ToolParameter("operation", "str", "操作类型", True, options=["transform", "filter", "aggregate"]),
                ToolParameter("parameters", "dict", "操作参数", False, {})
            ]
        )
    
    async def execute(self, data: Any, operation: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """执行数据处理"""
        self._update_execution_stats()
        parameters = parameters or {}
        
        # 这里应该实现具体的数据处理逻辑
        return {
            "operation": operation,
            "input_size": len(str(data)),
            "output_data": data,
            "processing_time": 0.0
        }


class FileManagementTool(BaseTool):
    """文件管理工具基类"""
    
    def get_tool_definition(self) -> ToolDefinition:
        return ToolDefinition(
            id="file_management_base",
            name="文件管理工具",
            description="文件操作和管理",
            category=ToolCategory.FILE_MANAGEMENT,
            security_level=SecurityLevel.MEDIUM,
            parameters=[
                ToolParameter("file_path", "str", "文件路径", True),
                ToolParameter("operation", "str", "操作类型", True, options=["read", "write", "delete", "copy"]),
                ToolParameter("content", "str", "文件内容", False)
            ]
        )
    
    async def execute(self, file_path: str, operation: str, content: str = None) -> Dict[str, Any]:
        """执行文件管理"""
        self._update_execution_stats()
        
        # 这里应该实现具体的文件操作逻辑
        return {
            "file_path": file_path,
            "operation": operation,
            "success": True,
            "message": "操作完成"
        }