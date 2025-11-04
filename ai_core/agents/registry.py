"""
Toolcall注册中心

负责注册和管理ToolcallManager实例，提供全局访问接口
"""

import logging
from typing import Dict, Optional, Any, List
from datetime import datetime

from .core.toolcall_manager import ToolcallManager
from .models import ToolcallRequest, ToolcallResponse, OptimizationConfig
from .llm_interface import LLMInterface


logger = logging.getLogger(__name__)


class ToolcallRegistry:
    """Toolcall注册中心
    
    全局注册表，管理所有ToolcallManager实例
    """
    
    _instance: Optional['ToolcallRegistry'] = None
    _managers: Dict[str, ToolcallManager] = {}
    _configs: Dict[str, OptimizationConfig] = {}
    _llm_interfaces: Dict[str, LLMInterface] = {}
    
    def __new__(cls) -> 'ToolcallRegistry':
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def register_manager(
        self, 
        manager_id: str, 
        manager: ToolcallManager, 
        config: Optional[OptimizationConfig] = None
    ) -> bool:
        """注册ToolcallManager"""
        try:
            if manager_id in self._managers:
                logger.warning(f"管理器已存在，将覆盖: {manager_id}")
            
            self._managers[manager_id] = manager
            if config:
                self._configs[manager_id] = config
            
            logger.info(f"注册ToolcallManager: {manager_id}")
            return True
            
        except Exception as e:
            logger.error(f"注册ToolcallManager失败: {e}")
            return False
    
    def unregister_manager(self, manager_id: str) -> bool:
        """注销ToolcallManager"""
        try:
            if manager_id not in self._managers:
                logger.warning(f"管理器不存在: {manager_id}")
                return False
            
            # 关闭管理器
            manager = self._managers[manager_id]
            if hasattr(manager, 'shutdown'):
                import asyncio
                asyncio.create_task(manager.shutdown())
            
            # 清理资源
            del self._managers[manager_id]
            if manager_id in self._configs:
                del self._configs[manager_id]
            if manager_id in self._llm_interfaces:
                del self._llm_interfaces[manager_id]
            
            logger.info(f"注销ToolcallManager: {manager_id}")
            return True
            
        except Exception as e:
            logger.error(f"注销ToolcallManager失败: {e}")
            return False
    
    def get_manager(self, manager_id: str) -> Optional[ToolcallManager]:
        """获取ToolcallManager"""
        return self._managers.get(manager_id)
    
    def list_managers(self) -> List[str]:
        """列出所有管理器ID"""
        return list(self._managers.keys())
    
    async def process_request(
        self, 
        manager_id: str, 
        request: ToolcallRequest
    ) -> Optional[ToolcallResponse]:
        """通过指定管理器处理请求"""
        manager = self.get_manager(manager_id)
        if not manager:
            logger.error(f"管理器不存在: {manager_id}")
            return None
        
        return await manager.process_request(request)
    
    async def batch_process_requests(
        self, 
        manager_id: str, 
        requests: List[ToolcallRequest]
    ) -> List[Optional[ToolcallResponse]]:
        """批量处理请求"""
        manager = self.get_manager(manager_id)
        if not manager:
            logger.error(f"管理器不存在: {manager_id}")
            return [None] * len(requests)
        
        return await manager.batch_process_requests(requests)
    
    async def cancel_request(self, manager_id: str, request_id: str) -> bool:
        """取消请求"""
        manager = self.get_manager(manager_id)
        if not manager:
            logger.error(f"管理器不存在: {manager_id}")
            return False
        
        return await manager.cancel_request(request_id)
    
    async def get_request_status(self, manager_id: str, request_id: str) -> Optional[Dict[str, Any]]:
        """获取请求状态"""
        manager = self.get_manager(manager_id)
        if not manager:
            logger.error(f"管理器不存在: {manager_id}")
            return None
        
        return await manager.get_request_status(request_id)
    
    async def get_system_statistics(self, manager_id: str) -> Optional[Dict[str, Any]]:
        """获取系统统计信息"""
        manager = self.get_manager(manager_id)
        if not manager:
            logger.error(f"管理器不存在: {manager_id}")
            return None
        
        return await manager.get_system_statistics()
    
    async def health_check(self, manager_id: str) -> Optional[Dict[str, Any]]:
        """健康检查"""
        manager = self.get_manager(manager_id)
        if not manager:
            logger.error(f"管理器不存在: {manager_id}")
            return None
        
        return await manager.health_check()
    
    async def optimize_configuration(self, manager_id: str) -> Optional[OptimizationConfig]:
        """优化配置"""
        manager = self.get_manager(manager_id)
        if not manager:
            logger.error(f"管理器不存在: {manager_id}")
            return None
        
        return await manager.optimize_configuration()
    
    def get_registry_statistics(self) -> Dict[str, Any]:
        """获取注册中心统计信息"""
        return {
            "total_managers": len(self._managers),
            "manager_ids": list(self._managers.keys()),
            "configs": {mid: config.to_dict() for mid, config in self._configs.items()},
            "timestamp": datetime.now().isoformat()
        }
    
    async def shutdown_all(self):
        """关闭所有管理器"""
        logger.info("正在关闭所有ToolcallManager...")
        
        for manager_id, manager in list(self._managers.items()):
            try:
                if hasattr(manager, 'shutdown'):
                    await manager.shutdown()
                logger.info(f"已关闭管理器: {manager_id}")
            except Exception as e:
                logger.error(f"关闭管理器失败: {manager_id}, {e}")
        
        # 清理注册表
        self._managers.clear()
        self._configs.clear()
        self._llm_interfaces.clear()
        
        logger.info("所有ToolcallManager已关闭")


# 全局注册中心实例
global_registry = ToolcallRegistry()


# 便捷函数
def register_toolcall_manager(
    manager_id: str,
    llm_interface: LLMInterface,
    tool_registry: Any,  # 这里应该是MCP的工具注册中心
    config: Optional[OptimizationConfig] = None
) -> bool:
    """便捷函数：注册ToolcallManager"""
    try:
        # 创建ToolcallManager
        manager = ToolcallManager(
            llm_interface=llm_interface,
            tool_registry=tool_registry,
            config=config
        )
        
        # 注册到全局注册中心
        return global_registry.register_manager(manager_id, manager, config)
        
    except Exception as e:
        logger.error(f"注册ToolcallManager失败: {e}")
        return False


async def process_toolcall_request(
    manager_id: str,
    user_query: str,
    session_id: str,
    user_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Optional[ToolcallResponse]:
    """便捷函数：处理工具调用请求"""
    try:
        # 创建请求对象
        request = ToolcallRequest(
            user_query=user_query,
            session_id=session_id,
            user_id=user_id,
            context=context or {},
            **kwargs
        )
        
        # 处理请求
        return await global_registry.process_request(manager_id, request)
        
    except Exception as e:
        logger.error(f"处理工具调用请求失败: {e}")
        return None


async def get_toolcall_status(
    manager_id: str,
    request_id: str
) -> Optional[Dict[str, Any]]:
    """便捷函数：获取工具调用状态"""
    return await global_registry.get_request_status(manager_id, request_id)


async def get_toolcall_statistics(manager_id: str) -> Optional[Dict[str, Any]]:
    """便捷函数：获取工具调用统计信息"""
    return await global_registry.get_system_statistics(manager_id)


async def health_check_toolcall_system(manager_id: str) -> Optional[Dict[str, Any]]:
    """便捷函数：健康检查"""
    return await global_registry.health_check(manager_id)


# 装饰器：自动注册工具调用管理器
def toolcall_manager(manager_id: str, config: Optional[OptimizationConfig] = None):
    """装饰器：自动注册工具调用管理器"""
    def decorator(cls):
        original_init = cls.__init__
        
        def new_init(self, llm_interface: LLMInterface, tool_registry: Any):
            # 调用原始初始化
            original_init(self, llm_interface, tool_registry, config)
            
            # 注册到全局注册中心
            global_registry.register_manager(manager_id, self, config)
        
        cls.__init__ = new_init
        return cls
    
    return decorator