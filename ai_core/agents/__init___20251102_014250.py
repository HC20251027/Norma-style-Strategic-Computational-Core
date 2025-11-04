"""
多智能体协作系统测试包
包含各种功能测试和集成测试
"""

import asyncio
import unittest
import logging
from typing import Dict, List, Any

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 测试工具函数
def create_test_agent(agent_id: str, capabilities: List[str] = None) -> Dict[str, Any]:
    """创建测试智能体"""
    if capabilities is None:
        capabilities = ["task_execution", "data_processing"]
    
    return {
        "id": agent_id,
        "name": f"TestAgent_{agent_id}",
        "capabilities": capabilities,
        "status": "active",
        "load": 0.0,
        "performance_score": 1.0
    }

def create_test_task(task_id: str, priority: int = 1, dependencies: List[str] = None) -> Dict[str, Any]:
    """创建测试任务"""
    if dependencies is None:
        dependencies = []
    
    return {
        "id": task_id,
        "name": f"TestTask_{task_id}",
        "description": f"测试任务 {task_id}",
        "priority": priority,
        "dependencies": dependencies,
        "estimated_duration": 10,
        "required_capabilities": ["task_execution"],
        "status": "pending"
    }

class AsyncTestCase(unittest.TestCase):
    """异步测试基类"""
    
    def setUp(self):
        """测试前设置"""
        self.loop = asyncio.get_event_loop()
    
    def tearDown(self):
        """测试后清理"""
        pass
    
    async def asyncSetUp(self):
        """异步设置"""
        pass
    
    async def asyncTearDown(self):
        """异步清理"""
        pass

# 导出测试工具
__all__ = [
    'AsyncTestCase',
    'create_test_agent',
    'create_test_task',
    'logger'
]