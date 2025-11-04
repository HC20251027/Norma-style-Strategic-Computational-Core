#!/usr/bin/env python3
"""
智能体协作管理器
负责管理多个智能体之间的协作关系、任务分配和协调

作者: 皇
创建时间: 2025-10-31
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Callable
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path

from ..utils.logger import MultiAgentLogger
from ..communication.message_bus import MessageBus
from ..core.types import AgentStatus, AgentCapability, CollaborationType
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"
    ERROR = "error"


class CollaborationType(Enum):
    """协作类型枚举"""
    SEQUENTIAL = "sequential"  # 顺序协作
    PARALLEL = "parallel"      # 并行协作
    PIPELINE = "pipeline"      # 流水线协作
    SUPERVISOR = "supervisor"  # 监督者模式
    PEER_TO_PEER = "peer_to_peer"  # 点对点协作


@dataclass
class AgentCapability:
    """智能体能力定义"""
    name: str
    description: str
    complexity_level: int  # 1-10，复杂度级别
    estimated_duration: float  # 预估执行时间（秒）
    resource_requirements: Dict[str, Any]  # 资源需求
    success_rate: float  # 成功率 0-1
    quality_score: float  # 质量评分 0-1


@dataclass
class CollaborationSession:
    """协作会话"""
    session_id: str
    created_at: datetime
    participants: List[str]  # 参与者智能体ID列表
    collaboration_type: CollaborationType
    task_id: Optional[str] = None
    status: str = "active"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class AgentCollaborationManager:
    """智能体协作管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化智能体协作管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.manager_id = str(uuid.uuid4())
        self.start_time = datetime.now()
        
        # 初始化日志
        self.logger = MultiAgentLogger("collaboration_manager")
        
        # 核心组件
        self.message_bus: Optional[MessageBus] = None
        self.task_distributor: Optional[TaskDistributor] = None
        self.agent_pool: Optional[AgentPoolManager] = None
        self.collaboration_patterns: Optional[CollaborationPatterns] = None
        self.performance_evaluator: Optional[PerformanceEvaluator] = None
        
        # 智能体管理
        self.registered_agents: Dict[str, Dict[str, Any]] = {}
        self.agent_capabilities: Dict[str, List[AgentCapability]] = {}
        self.agent_status: Dict[str, AgentStatus] = {}
        self.active_collaborations: Dict[str, CollaborationSession] = {}
        
        # 协作历史
        self.collaboration_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Any] = {}
        
        # 配置
        self.config = self._load_config(config_path)
        
        # 事件回调
        self.event_callbacks: Dict[str, List[Callable]] = {}
        
        self.logger.info(f"智能体协作管理器 {self.manager_id} 已初始化")
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """加载配置"""
        default_config = {
            "collaboration": {
                "max_concurrent_collaborations": 50,
                "default_collaboration_timeout": 3600,  # 1小时
                "enable_auto_scaling": True,
                "load_balancing_strategy": "round_robin",
                "task_timeout": 1800,  # 30分钟
                "retry_attempts": 3
            },
            "agent_management": {
                "heartbeat_interval": 30,
                "status_check_interval": 60,
                "max_inactive_time": 300,  # 5分钟
                "auto_recovery": True
            },
            "performance": {
                "evaluation_interval": 300,  # 5分钟
                "metrics_retention_days": 30,
                "efficiency_threshold": 0.8,
                "quality_threshold": 0.85
            },
            "communication": {
                "message_buffer_size": 1000,
                "broadcast_interval": 5,
                "sync_timeout": 30
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    self._deep_merge(default_config, user_config)
            except Exception as e:
                self.logger.warning(f"配置文件加载失败，使用默认配置: {e}")
        
        return default_config
    
    def _deep_merge(self, base: Dict, update: Dict) -> None:
        """深度合并配置字典"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    async def initialize(self) -> bool:
        """初始化协作管理器"""
        try:
            self.logger.info("初始化智能体协作管理器...")
            
            # 初始化消息总线
            self.message_bus = MessageBus(
                buffer_size=self.config["communication"]["message_buffer_size"]
            )
            await self.message_bus.initialize()
            self.logger.info("✓ 消息总线初始化完成")
            
            # 初始化任务分配器
            self.task_distributor = TaskDistributor(
                message_bus=self.message_bus,
                timeout=self.config["collaboration"]["task_timeout"],
                max_retries=self.config["collaboration"]["retry_attempts"]
            )
            await self.task_distributor.initialize()
            self.logger.info("✓ 任务分配器初始化完成")
            
            # 初始化智能体池管理器
            self.agent_pool = AgentPoolManager(
                message_bus=self.message_bus,
                max_agents=self.config["collaboration"]["max_concurrent_collaborations"]
            )
            await self.agent_pool.initialize()
            self.logger.info("✓ 智能体池管理器初始化完成")
            
            # 初始化协作模式
            self.collaboration_patterns = CollaborationPatterns(
                message_bus=self.message_bus
            )
            await self.collaboration_patterns.initialize()
            self.logger.info("✓ 协作模式初始化完成")
            
            # 初始化性能评估器
            self.performance_evaluator = PerformanceEvaluator()
            await self.performance_evaluator.initialize()
            self.logger.info("✓ 性能评估器初始化完成")
            
            # 启动后台任务
            asyncio.create_task(self._status_monitor())
            asyncio.create_task(self._performance_evaluator())
            
            self.logger.info("智能体协作管理器初始化完成")
            return True
            
        except Exception as e:
            self.logger.error(f"智能体协作管理器初始化失败: {e}")
            return False
    
    async def register_agent(
        self, 
        agent_id: str, 
        agent_info: Dict[str, Any],
        capabilities: List[AgentCapability]
    ) -> bool:
        """注册智能体
        
        Args:
            agent_id: 智能体ID
            agent_info: 智能体信息
            capabilities: 智能体能力列表
            
        Returns:
            注册是否成功
        """
        try:
            if agent_id in self.registered_agents:
                self.logger.warning(f"智能体 {agent_id} 已存在，将更新信息")
            
            self.registered_agents[agent_id] = {
                **agent_info,
                "registered_at": datetime.now().isoformat(),
                "last_heartbeat": datetime.now().isoformat()
            }
            
            self.agent_capabilities[agent_id] = capabilities
            self.agent_status[agent_id] = AgentStatus.IDLE
            
            # 注册到智能体池
            if self.agent_pool:
                await self.agent_pool.register_agent(agent_id, agent_info, capabilities)
            
            # 发送注册事件
            await self._emit_event("agent.registered", {
                "agent_id": agent_id,
                "capabilities": [asdict(cap) for cap in capabilities]
            })
            
            self.logger.info(f"智能体 {agent_id} 注册成功")
            return True
            
        except Exception as e:
            self.logger.error(f"智能体 {agent_id} 注册失败: {e}")
            return False
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """注销智能体"""
        try:
            if agent_id not in self.registered_agents:
                self.logger.warning(f"智能体 {agent_id} 不存在")
                return False
            
            # 从智能体池移除
            if self.agent_pool:
                await self.agent_pool.unregister_agent(agent_id)
            
            # 清理相关数据
            del self.registered_agents[agent_id]
            if agent_id in self.agent_capabilities:
                del self.agent_capabilities[agent_id]
            if agent_id in self.agent_status:
                del self.agent_status[agent_id]
            
            # 发送注销事件
            await self._emit_event("agent.unregistered", {
                "agent_id": agent_id
            })
            
            self.logger.info(f"智能体 {agent_id} 注销成功")
            return True
            
        except Exception as e:
            self.logger.error(f"智能体 {agent_id} 注销失败: {e}")
            return False
    
    async def create_collaboration(
        self,
        collaboration_type: CollaborationType,
        participant_agents: List[str],
        task_description: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """创建协作会话
        
        Args:
            collaboration_type: 协作类型
            participant_agents: 参与者智能体ID列表
            task_description: 任务描述
            metadata: 附加元数据
            
        Returns:
            协作会话ID
        """
        try:
            # 验证参与者
            valid_agents = []
            for agent_id in participant_agents:
                if agent_id in self.registered_agents:
                    valid_agents.append(agent_id)
                else:
                    self.logger.warning(f"智能体 {agent_id} 不存在")
            
            if not valid_agents:
                raise ValueError("没有有效的参与者智能体")
            
            # 创建协作会话
            session_id = str(uuid.uuid4())
            session = CollaborationSession(
                session_id=session_id,
                created_at=datetime.now(),
                participants=valid_agents,
                collaboration_type=collaboration_type,
                metadata=metadata or {}
            )
            
            self.active_collaborations[session_id] = session
            
            # 初始化协作模式
            if self.collaboration_patterns:
                await self.collaboration_patterns.initialize_collaboration(
                    session_id, collaboration_type, valid_agents, task_description
                )
            
            # 发送协作创建事件
            await self._emit_event("collaboration.created", {
                "session_id": session_id,
                "type": collaboration_type.value,
                "participants": valid_agents,
                "task_description": task_description
            })
            
            self.logger.info(f"创建协作会话 {session_id}，类型: {collaboration_type.value}")
            return session_id
            
        except Exception as e:
            self.logger.error(f"创建协作会话失败: {e}")
            raise
    
    async def execute_collaboration(
        self,
        session_id: str,
        task_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行协作任务
        
        Args:
            session_id: 协作会话ID
            task_data: 任务数据
            
        Returns:
            执行结果
        """
        try:
            if session_id not in self.active_collaborations:
                raise ValueError(f"协作会话 {session_id} 不存在")
            
            session = self.active_collaborations[session_id]
            
            # 根据协作类型执行不同的协作模式
            if self.collaboration_patterns:
                result = await self.collaboration_patterns.execute_collaboration(
                    session_id, session.collaboration_type, session.participants, task_data
                )
                
                # 更新协作历史
                self.collaboration_history.append({
                    "session_id": session_id,
                    "executed_at": datetime.now().isoformat(),
                    "participants": session.participants,
                    "type": session.collaboration_type.value,
                    "result": result,
                    "duration": result.get("duration", 0)
                })
                
                # 发送执行完成事件
                await self._emit_event("collaboration.completed", {
                    "session_id": session_id,
                    "result": result
                })
                
                return result
            else:
                raise RuntimeError("协作模式未初始化")
                
        except Exception as e:
            self.logger.error(f"执行协作任务失败 {session_id}: {e}")
            
            # 发送执行失败事件
            await self._emit_event("collaboration.failed", {
                "session_id": session_id,
                "error": str(e)
            })
            
            raise
    
    async def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """获取智能体状态"""
        if agent_id not in self.registered_agents:
            return {"error": "智能体不存在"}
        
        return {
            "agent_id": agent_id,
            "status": self.agent_status.get(agent_id, AgentStatus.OFFLINE).value,
            "info": self.registered_agents[agent_id],
            "capabilities": [asdict(cap) for cap in self.agent_capabilities.get(agent_id, [])],
            "last_heartbeat": self.registered_agents[agent_id].get("last_heartbeat")
        }
    
    async def get_collaboration_status(self, session_id: str) -> Dict[str, Any]:
        """获取协作状态"""
        if session_id not in self.active_collaborations:
            return {"error": "协作会话不存在"}
        
        session = self.active_collaborations[session_id]
        
        return {
            "session_id": session_id,
            "status": session.status,
            "type": session.collaboration_type.value,
            "participants": session.participants,
            "created_at": session.created_at.isoformat(),
            "task_id": session.task_id,
            "metadata": session.metadata
        }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "manager_id": self.manager_id,
            "status": "running",
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "registered_agents": len(self.registered_agents),
            "active_collaborations": len(self.active_collaborations),
            "agent_status_distribution": {
                status.value: count 
                for status, count in self._get_agent_status_distribution().items()
            },
            "collaboration_types_distribution": self._get_collaboration_type_distribution(),
            "performance_metrics": self.performance_metrics
        }
    
    def _get_agent_status_distribution(self) -> Dict[AgentStatus, int]:
        """获取智能体状态分布"""
        distribution = {}
        for status in AgentStatus:
            distribution[status] = 0
        
        for status in self.agent_status.values():
            distribution[status] += 1
        
        return distribution
    
    def _get_collaboration_type_distribution(self) -> Dict[str, int]:
        """获取协作类型分布"""
        distribution = {}
        for session in self.active_collaborations.values():
            collab_type = session.collaboration_type.value
            distribution[collab_type] = distribution.get(collab_type, 0) + 1
        
        return distribution
    
    async def _status_monitor(self):
        """状态监控后台任务"""
        while True:
            try:
                await asyncio.sleep(self.config["agent_management"]["status_check_interval"])
                
                current_time = datetime.now()
                inactive_agents = []
                
                for agent_id, agent_info in self.registered_agents.items():
                    last_heartbeat = datetime.fromisoformat(
                        agent_info["last_heartbeat"]
                    )
                    
                    # 检查是否超时
                    if (current_time - last_heartbeat).total_seconds() > \
                       self.config["agent_management"]["max_inactive_time"]:
                        inactive_agents.append(agent_id)
                
                # 处理不活跃的智能体
                for agent_id in inactive_agents:
                    self.logger.warning(f"智能体 {agent_id} 长时间未活跃，标记为离线")
                    self.agent_status[agent_id] = AgentStatus.OFFLINE
                    
                    await self._emit_event("agent.inactive", {
                        "agent_id": agent_id,
                        "inactive_duration": (
                            current_time - datetime.fromisoformat(
                                self.registered_agents[agent_id]["last_heartbeat"]
                            )
                        ).total_seconds()
                    })
                
            except Exception as e:
                self.logger.error(f"状态监控任务出错: {e}")
    
    async def _performance_evaluator(self):
        """性能评估后台任务"""
        while True:
            try:
                await asyncio.sleep(self.config["performance"]["evaluation_interval"])
                
                if self.performance_evaluator:
                    metrics = await self.performance_evaluator.evaluate_system_performance(
                        self.registered_agents,
                        self.collaboration_history,
                        self.active_collaborations
                    )
                    
                    self.performance_metrics = metrics
                    
                    await self._emit_event("performance.evaluated", metrics)
                
            except Exception as e:
                self.logger.error(f"性能评估任务出错: {e}")
    
    def on_event(self, event_type: str, callback: Callable):
        """注册事件回调"""
        if event_type not in self.event_callbacks:
            self.event_callbacks[event_type] = []
        self.event_callbacks[event_type].append(callback)
    
    async def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """发送事件"""
        if event_type in self.event_callbacks:
            for callback in self.event_callbacks[event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                except Exception as e:
                    self.logger.error(f"事件回调执行失败 {event_type}: {e}")
        
        # 通过消息总线发送事件
        if self.message_bus:
            await self.message_bus.publish_event(event_type, data)
    
    async def shutdown(self):
        """关闭协作管理器"""
        try:
            self.logger.info("关闭智能体协作管理器...")
            
            # 取消所有活跃协作
            for session_id in list(self.active_collaborations.keys()):
                await self._terminate_collaboration(session_id)
            
            # 关闭各个组件
            if self.performance_evaluator:
                await self.performance_evaluator.shutdown()
            
            if self.collaboration_patterns:
                await self.collaboration_patterns.shutdown()
            
            if self.agent_pool:
                await self.agent_pool.shutdown()
            
            if self.task_distributor:
                await self.task_distributor.shutdown()
            
            if self.message_bus:
                await self.message_bus.shutdown()
            
            self.logger.info("智能体协作管理器已关闭")
            
        except Exception as e:
            self.logger.error(f"关闭智能体协作管理器时出错: {e}")
    
    async def _terminate_collaboration(self, session_id: str):
        """终止协作会话"""
        try:
            if session_id in self.active_collaborations:
                session = self.active_collaborations[session_id]
                session.status = "terminated"
                
                await self._emit_event("collaboration.terminated", {
                    "session_id": session_id,
                    "reason": "manager_shutdown"
                })
                
                del self.active_collaborations[session_id]
                
        except Exception as e:
            self.logger.error(f"终止协作会话失败 {session_id}: {e}")