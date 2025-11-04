"""
工具注册和管理中心
"""

import logging
from typing import Dict, List, Optional, Any, Callable, Type
from collections import defaultdict
import asyncio
import threading
from datetime import datetime

from .models import ToolDefinition, ToolCategory, ToolStatus, SecurityLevel
from ..tools.base_tool import BaseTool


logger = logging.getLogger(__name__)


class ToolRegistry:
    """工具注册中心"""
    
    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._tool_instances: Dict[str, BaseTool] = {}
        self._tool_classes: Dict[str, Type[BaseTool]] = {}
        self._categories: Dict[ToolCategory, List[str]] = defaultdict(list)
        self._lock = threading.RLock()
        self._subscribers: List[Callable] = []
        
    def register_tool(self, tool_class: Type[BaseTool], definition: Optional[ToolDefinition] = None) -> bool:
        """注册工具"""
        try:
            with self._lock:
                # 创建工具实例
                tool_instance = tool_class()
                
                # 如果没有提供定义，从工具类获取
                if definition is None:
                    definition = tool_instance.get_tool_definition()
                
                # 验证工具定义
                if not self._validate_tool_definition(definition):
                    logger.error(f"工具定义验证失败: {definition.id}")
                    return False
                
                # 检查是否已存在
                if definition.id in self._tools:
                    logger.warning(f"工具已存在，将覆盖: {definition.id}")
                
                # 注册工具
                self._tools[definition.id] = definition
                self._tool_instances[definition.id] = tool_instance
                self._tool_classes[definition.id] = tool_class
                self._categories[definition.category].append(definition.id)
                
                # 通知订阅者
                self._notify_subscribers("register", definition)
                
                logger.info(f"工具注册成功: {definition.name} ({definition.id})")
                return True
                
        except Exception as e:
            logger.error(f"工具注册失败: {e}")
            return False
    
    def unregister_tool(self, tool_id: str) -> bool:
        """注销工具"""
        try:
            with self._lock:
                if tool_id not in self._tools:
                    logger.warning(f"工具不存在: {tool_id}")
                    return False
                
                definition = self._tools[tool_id]
                
                # 从各个集合中移除
                del self._tools[tool_id]
                if tool_id in self._tool_instances:
                    del self._tool_instances[tool_id]
                if tool_id in self._tool_classes:
                    del self._tool_classes[tool_id]
                
                # 从分类中移除
                if tool_id in self._categories[definition.category]:
                    self._categories[definition.category].remove(tool_id)
                
                # 通知订阅者
                self._notify_subscribers("unregister", definition)
                
                logger.info(f"工具注销成功: {definition.name} ({tool_id})")
                return True
                
        except Exception as e:
            logger.error(f"工具注销失败: {e}")
            return False
    
    def get_tool(self, tool_id: str) -> Optional[ToolDefinition]:
        """获取工具定义"""
        return self._tools.get(tool_id)
    
    def get_tool_instance(self, tool_id: str) -> Optional[BaseTool]:
        """获取工具实例"""
        return self._tool_instances.get(tool_id)
    
    def list_tools(self, category: Optional[ToolCategory] = None, 
                   status: Optional[ToolStatus] = None) -> List[ToolDefinition]:
        """列出工具"""
        tools = list(self._tools.values())
        
        if category:
            tools = [tool for tool in tools if tool.category == category]
        
        if status:
            tools = [tool for tool in tools if tool.status == status]
        
        return sorted(tools, key=lambda x: x.name)
    
    def list_tools_by_category(self) -> Dict[ToolCategory, List[ToolDefinition]]:
        """按分类列出工具"""
        result = {}
        for category in ToolCategory:
            result[category] = self.list_tools(category=category)
        return result
    
    def search_tools(self, query: str) -> List[ToolDefinition]:
        """搜索工具"""
        query = query.lower()
        results = []
        
        for tool in self._tools.values():
            # 搜索名称、描述、标签
            if (query in tool.name.lower() or 
                query in tool.description.lower() or 
                any(query in tag.lower() for tag in tool.tags)):
                results.append(tool)
        
        return sorted(results, key=lambda x: x.name)
    
    def update_tool_status(self, tool_id: str, status: ToolStatus) -> bool:
        """更新工具状态"""
        try:
            with self._lock:
                if tool_id not in self._tools:
                    return False
                
                self._tools[tool_id].status = status
                self._tools[tool_id].updated_at = datetime.now()
                
                # 通知订阅者
                self._notify_subscribers("status_update", self._tools[tool_id])
                
                logger.info(f"工具状态更新成功: {tool_id} -> {status.value}")
                return True
                
        except Exception as e:
            logger.error(f"工具状态更新失败: {e}")
            return False
    
    def validate_tool_execution(self, tool_id: str, parameters: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """验证工具执行参数"""
        try:
            tool = self.get_tool(tool_id)
            if not tool:
                return False, "工具不存在"
            
            # 检查工具状态
            if tool.status != ToolStatus.ACTIVE:
                return False, f"工具状态不正确: {tool.status.value}"
            
            # 验证必需参数
            for param in tool.parameters:
                if param.required and param.name not in parameters:
                    return False, f"缺少必需参数: {param.name}"
                
                # 验证参数值
                if param.name in parameters and param.validation:
                    if not param.validation(parameters[param.name]):
                        return False, f"参数验证失败: {param.name}"
            
            # 验证参数类型
            for param_name, value in parameters.items():
                param_def = next((p for p in tool.parameters if p.name == param_name), None)
                if param_def and not self._validate_parameter_type(value, param_def.type):
                    return False, f"参数类型错误: {param_name}"
            
            return True, None
            
        except Exception as e:
            logger.error(f"工具执行验证失败: {e}")
            return False, f"验证过程出错: {str(e)}"
    
    def get_tool_statistics(self) -> Dict[str, Any]:
        """获取工具统计信息"""
        with self._lock:
            total_tools = len(self._tools)
            active_tools = len([t for t in self._tools.values() if t.status == ToolStatus.ACTIVE])
            
            category_stats = {}
            for category in ToolCategory:
                category_stats[category.value] = len(self._categories[category])
            
            security_stats = {}
            for level in SecurityLevel:
                security_stats[level.value] = len([
                    t for t in self._tools.values() if t.security_level == level
                ])
            
            return {
                "total_tools": total_tools,
                "active_tools": active_tools,
                "inactive_tools": total_tools - active_tools,
                "categories": category_stats,
                "security_levels": security_stats,
                "last_updated": datetime.now().isoformat()
            }
    
    def subscribe(self, callback: Callable) -> None:
        """订阅工具注册事件"""
        self._subscribers.append(callback)
    
    def unsubscribe(self, callback: Callable) -> None:
        """取消订阅"""
        if callback in self._subscribers:
            self._subscribers.remove(callback)
    
    def _notify_subscribers(self, event: str, data: Any) -> None:
        """通知订阅者"""
        for callback in self._subscribers:
            try:
                callback(event, data)
            except Exception as e:
                logger.error(f"通知订阅者失败: {e}")
    
    def _validate_tool_definition(self, definition: ToolDefinition) -> bool:
        """验证工具定义"""
        try:
            # 基本字段验证
            if not all([definition.id, definition.name, definition.description]):
                return False
            
            # 参数验证
            param_names = set()
            for param in definition.parameters:
                if param.name in param_names:
                    return False  # 重复参数名
                param_names.add(param.name)
            
            return True
            
        except Exception:
            return False
    
    def _validate_parameter_type(self, value: Any, expected_type: str) -> bool:
        """验证参数类型"""
        type_mapping = {
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "any": object
        }
        
        expected_python_type = type_mapping.get(expected_type.lower())
        if expected_python_type is None:
            return True  # 未知类型，默认通过
        
        return isinstance(value, expected_python_type)


# 全局工具注册中心实例
global_registry = ToolRegistry()