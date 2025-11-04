"""
诺玛指挥中枢主Agent - 系统核心协调器
"""
import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
from .base_agent import BaseAgent, AgentMessage, Task, TaskPriority, AgentStatus


class NormaCommandCenter(BaseAgent):
    """诺玛指挥中枢主智能体"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            agent_id="norma-command-center",
            agent_type="command_center",
            config=config
        )
        self.specialist_agents = {}  # 专业智能体映射
        self.agent_pools = {}  # 智能体池
        self.active_tasks = {}  # 活跃任务
        self.load_balancer = None
        self.event_bus = None
        self.message_queue = asyncio.Queue()
        self.logger = logging.getLogger("norma.command.center")
        
    async def initialize(self) -> bool:
        """初始化诺玛指挥中枢"""
        try:
            self.logger.info("初始化诺玛指挥中枢...")
            
            # 注册消息处理器
            self._register_message_handlers()
            
            # 启动消息处理循环
            asyncio.create_task(self._message_processor())
            
            # 启动任务监控
            asyncio.create_task(self._task_monitor())
            
            # 启动负载监控
            asyncio.create_task(self._load_monitor())
            
            self.logger.info("诺玛指挥中枢初始化完成")
            return True
            
        except Exception as e:
            self.logger.error(f"初始化失败: {e}")
            return False
    
    async def process_task(self, task: Task) -> Dict[str, Any]:
        """处理任务"""
        try:
            self.logger.info(f"处理任务: {task.id}")
            
            # 任务路由到合适的专业智能体
            specialist = await self._route_task(task)
            
            if not specialist:
                return {
                    "status": "error",
                    "message": f"没有可用的专业智能体处理任务类型: {task.agent_type}"
                }
            
            # 执行任务
            result = await specialist.process_task(task)
            
            # 记录任务完成
            self.active_tasks.pop(task.id, None)
            
            return {
                "status": "success",
                "task_id": task.id,
                "result": result,
                "processed_by": specialist.agent_id
            }
            
        except Exception as e:
            self.logger.error(f"任务处理失败: {e}")
            return {
                "status": "error",
                "task_id": task.id,
                "error": str(e)
            }
    
    async def shutdown(self) -> bool:
        """关闭诺玛指挥中枢"""
        try:
            self.logger.info("关闭诺玛指挥中枢...")
            
            # 停止所有专业智能体
            for agent in self.specialist_agents.values():
                if hasattr(agent, 'stop'):
                    await agent.stop()
            
            # 停止内部任务
            self.is_running = False
            
            self.logger.info("诺玛指挥中枢已关闭")
            return True
            
        except Exception as e:
            self.logger.error(f"关闭失败: {e}")
            return False
    
    def register_specialist_agent(self, agent_type: str, agent: BaseAgent):
        """注册专业智能体"""
        self.specialist_agents[agent_type] = agent
        self.logger.info(f"注册专业智能体: {agent_type} -> {agent.agent_id}")
    
    def create_agent_pool(self, agent_type: str, pool_size: int = 3):
        """创建智能体池"""
        if agent_type not in self.agent_pools:
            self.agent_pools[agent_type] = []
        
        self.logger.info(f"创建智能体池: {agent_type} (大小: {pool_size})")
    
    async def broadcast_message(self, message: AgentMessage, target_types: List[str] = None):
        """广播消息"""
        targets = target_types or list(self.specialist_agents.keys())
        
        for agent_type in targets:
            if agent_type in self.specialist_agents:
                agent = self.specialist_agents[agent_type]
                message.receiver_id = agent.agent_id
                await agent.receive_message(message)
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        specialist_status = {}
        for agent_type, agent in self.specialist_agents.items():
            specialist_status[agent_type] = agent.get_status()
        
        return {
            "command_center": self.get_status(),
            "specialist_agents": specialist_status,
            "agent_pools": {
                pool_type: len(agents) 
                for pool_type, agents in self.agent_pools.items()
            },
            "active_tasks": len(self.active_tasks),
            "timestamp": datetime.now().isoformat()
        }
    
    def _register_message_handlers(self):
        """注册消息处理器"""
        self.register_message_handler("task_request", self._handle_task_request)
        self.register_message_handler("status_update", self._handle_status_update)
        self.register_message_handler("agent_registration", self._handle_agent_registration)
        self.register_message_handler("load_report", self._handle_load_report)
    
    async def _handle_task_request(self, message: AgentMessage) -> Dict[str, Any]:
        """处理任务请求"""
        task_data = message.payload.get("task")
        if not task_data:
            return {"status": "error", "message": "任务数据缺失"}
        
        task = Task(
            id=task_data.get("id", str(uuid.uuid4())),
            agent_type=task_data.get("agent_type"),
            priority=TaskPriority(task_data.get("priority", 2)),
            payload=task_data.get("payload", {}),
            created_at=datetime.now(),
            timeout=task_data.get("timeout")
        )
        
        self.active_tasks[task.id] = task
        
        return {"status": "accepted", "task_id": task.id}
    
    async def _handle_status_update(self, message: AgentMessage) -> Dict[str, Any]:
        """处理状态更新"""
        agent_id = message.sender_id
        status_data = message.payload.get("status", {})
        
        self.logger.info(f"收到智能体状态更新: {agent_id}")
        
        return {"status": "acknowledged"}
    
    async def _handle_agent_registration(self, message: AgentMessage) -> Dict[str, Any]:
        """处理智能体注册"""
        agent_info = message.payload.get("agent_info", {})
        agent_type = agent_info.get("agent_type")
        
        if agent_type:
            self.register_specialist_agent(agent_type, message.sender_id)
        
        return {"status": "registered"}
    
    async def _handle_load_report(self, message: AgentMessage) -> Dict[str, Any]:
        """处理负载报告"""
        load_data = message.payload.get("load", {})
        
        self.logger.debug(f"收到负载报告: {message.sender_id} -> {load_data}")
        
        return {"status": "acknowledged"}
    
    async def _route_task(self, task: Task) -> Optional[BaseAgent]:
        """路由任务到合适的智能体"""
        # 负载均衡逻辑
        available_agents = []
        
        for agent_type, agent in self.specialist_agents.items():
            if agent_type == task.agent_type and agent.can_handle_task(task):
                available_agents.append(agent)
        
        if not available_agents:
            return None
        
        # 选择负载最低的智能体
        best_agent = min(available_agents, key=lambda a: a.load)
        
        self.logger.info(f"任务 {task.id} 路由到智能体 {best_agent.agent_id}")
        return best_agent
    
    async def _message_processor(self):
        """消息处理器"""
        while self.is_running:
            try:
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                await self._process_message(message)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"消息处理错误: {e}")
    
    async def _process_message(self, message: AgentMessage):
        """处理消息"""
        handler = self.message_handlers.get(message.message_type)
        if handler:
            await handler(message)
    
    async def _task_monitor(self):
        """任务监控器"""
        while self.is_running:
            try:
                # 检查超时任务
                current_time = datetime.now()
                timeout_tasks = []
                
                for task_id, task in self.active_tasks.items():
                    if task.timeout:
                        elapsed = (current_time - task.created_at).total_seconds()
                        if elapsed > task.timeout:
                            timeout_tasks.append(task_id)
                
                # 处理超时任务
                for task_id in timeout_tasks:
                    self.logger.warning(f"任务超时: {task_id}")
                    self.active_tasks.pop(task_id, None)
                
                await asyncio.sleep(10)  # 10秒检查间隔
                
            except Exception as e:
                self.logger.error(f"任务监控错误: {e}")
    
    async def _load_monitor(self):
        """负载监控器"""
        while self.is_running:
            try:
                # 收集所有智能体的负载信息
                load_data = {}
                for agent_type, agent in self.specialist_agents.items():
                    load_data[agent_type] = agent.load
                
                self.logger.debug(f"系统负载: {load_data}")
                
                await asyncio.sleep(30)  # 30秒监控间隔
                
            except Exception as e:
                self.logger.error(f"负载监控错误: {e}")