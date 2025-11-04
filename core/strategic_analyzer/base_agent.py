"""
智能体基类 - 基于Agno框架的核心抽象
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from datetime import datetime
import asyncio
import json
import uuid
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import logging


class AgentStatus(Enum):
    """智能体状态枚举"""
    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class TaskPriority(Enum):
    """任务优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Task:
    """任务数据类"""
    id: str
    agent_type: str
    priority: TaskPriority
    payload: Dict[str, Any]
    created_at: datetime
    timeout: Optional[int] = None
    callback: Optional[Callable] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AgentMessage:
    """智能体消息数据类"""
    id: str
    sender_id: str
    receiver_id: str
    message_type: str
    payload: Dict[str, Any]
    timestamp: datetime
    priority: TaskPriority = TaskPriority.NORMAL
    correlation_id: Optional[str] = None


class BaseAgent(ABC):
    """智能体基类"""
    
    def __init__(self, agent_id: str, agent_type: str, config: Dict[str, Any] = None):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config = config or {}
        self.status = AgentStatus.IDLE
        self.capabilities = []
        self.load = 0.0
        self.created_at = datetime.now()
        self.last_heartbeat = datetime.now()
        self.logger = logging.getLogger(f"agent.{agent_type}.{agent_id}")
        self.message_handlers = {}
        self.task_queue = asyncio.Queue()
        self.is_running = False
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    @abstractmethod
    async def initialize(self) -> bool:
        """初始化智能体"""
        pass
    
    @abstractmethod
    async def process_task(self, task: Task) -> Dict[str, Any]:
        """处理任务"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> bool:
        """关闭智能体"""
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """获取智能体状态"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "status": self.status.value,
            "load": self.load,
            "capabilities": self.capabilities,
            "created_at": self.created_at.isoformat(),
            "last_heartbeat": self.last_heartbeat.isoformat()
        }
    
    async def send_message(self, message: AgentMessage) -> bool:
        """发送消息到其他智能体"""
        try:
            # 消息路由逻辑将在协调器中实现
            self.logger.info(f"发送消息 {message.id} 到 {message.receiver_id}")
            return True
        except Exception as e:
            self.logger.error(f"发送消息失败: {e}")
            return False
    
    async def receive_message(self, message: AgentMessage) -> Dict[str, Any]:
        """接收来自其他智能体的消息"""
        try:
            self.logger.info(f"接收消息 {message.id} 从 {message.sender_id}")
            handler = self.message_handlers.get(message.message_type)
            if handler:
                return await handler(message)
            else:
                self.logger.warning(f"未找到消息处理器: {message.message_type}")
                return {"status": "handled", "message": "No handler found"}
        except Exception as e:
            self.logger.error(f"处理消息失败: {e}")
            return {"status": "error", "error": str(e)}
    
    def register_message_handler(self, message_type: str, handler: Callable):
        """注册消息处理器"""
        self.message_handlers[message_type] = handler
    
    async def update_heartbeat(self):
        """更新心跳"""
        self.last_heartbeat = datetime.now()
    
    def can_handle_task(self, task: Task) -> bool:
        """判断是否可以处理任务"""
        return (self.status == AgentStatus.IDLE and 
                self.agent_type == task.agent_type and
                self.load < 0.8)  # 负载阈值
    
    async def start(self):
        """启动智能体"""
        if self.is_running:
            return
        
        self.is_running = True
        self.status = AgentStatus.IDLE
        
        # 启动任务处理循环
        asyncio.create_task(self._task_processor())
        
        # 启动心跳更新
        asyncio.create_task(self._heartbeat_updater())
        
        self.logger.info(f"智能体 {self.agent_id} 已启动")
    
    async def stop(self):
        """停止智能体"""
        self.is_running = False
        self.status = AgentStatus.OFFLINE
        self.executor.shutdown(wait=True)
        self.logger.info(f"智能体 {self.agent_id} 已停止")
    
    async def _task_processor(self):
        """任务处理器"""
        while self.is_running:
            try:
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                await self._execute_task(task)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"任务处理错误: {e}")
    
    async def _execute_task(self, task: Task):
        """执行任务"""
        try:
            self.status = AgentStatus.BUSY
            self.load = min(self.load + 0.1, 1.0)
            
            result = await self.process_task(task)
            
            if task.callback:
                await task.callback(result)
            
            self.load = max(self.load - 0.1, 0.0)
            self.status = AgentStatus.IDLE
            
        except Exception as e:
            self.logger.error(f"任务执行失败: {e}")
            self.status = AgentStatus.ERROR
    
    async def _heartbeat_updater(self):
        """心跳更新器"""
        while self.is_running:
            await self.update_heartbeat()
            await asyncio.sleep(30)  # 30秒心跳间隔