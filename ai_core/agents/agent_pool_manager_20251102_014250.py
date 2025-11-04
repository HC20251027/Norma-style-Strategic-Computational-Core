#!/usr/bin/env python3
"""
智能体池管理器
负责智能体的注册、注销、健康检查和生命周期管理

作者: 皇
创建时间: 2025-10-31
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Callable, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import psutil

from ..utils.logger import MultiAgentLogger
from ..communication.message_bus import MessageBus
from .agent_collaboration_manager import AgentCapability, AgentStatus


class PoolStrategy(Enum):
    """池策略枚举"""
    FIXED_SIZE = "fixed_size"          # 固定大小
    DYNAMIC_SCALING = "dynamic_scaling" # 动态伸缩
    AUTO_SCALING = "auto_scaling"      # 自动伸缩
    PREDICTIVE_SCALING = "predictive_scaling"  # 预测性伸缩


class HealthStatus(Enum):
    """健康状态枚举"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class AgentInfo:
    """智能体信息"""
    agent_id: str
    name: str
    agent_type: str
    capabilities: List[AgentCapability]
    status: AgentStatus
    registered_at: datetime
    last_heartbeat: datetime
    health_status: HealthStatus = HealthStatus.UNKNOWN
    resource_usage: Dict[str, float] = None
    performance_metrics: Dict[str, float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.resource_usage is None:
            self.resource_usage = {}
        if self.performance_metrics is None:
            self.performance_metrics = {}
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def is_healthy(self) -> bool:
        """检查智能体是否健康"""
        return self.health_status == HealthStatus.HEALTHY
    
    @property
    def is_available(self) -> bool:
        """检查智能体是否可用"""
        return (self.status == AgentStatus.IDLE and 
                self.health_status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED])
    
    @property
    def uptime_seconds(self) -> float:
        """运行时间（秒）"""
        return (datetime.now() - self.registered_at).total_seconds()


@dataclass
class PoolConfiguration:
    """池配置"""
    strategy: PoolStrategy
    min_agents: int = 1
    max_agents: int = 100
    target_utilization: float = 0.7  # 目标利用率
    scale_up_threshold: float = 0.8   # 扩容阈值
    scale_down_threshold: float = 0.3 # 缩容阈值
    health_check_interval: int = 30   # 健康检查间隔（秒）
    heartbeat_timeout: int = 120      # 心跳超时（秒）
    auto_recovery: bool = True
    resource_limits: Dict[str, float] = None
    
    def __post_init__(self):
        if self.resource_limits is None:
            self.resource_limits = {
                "cpu_percent": 80.0,
                "memory_percent": 80.0,
                "disk_percent": 90.0
            }


class AgentPoolManager:
    """智能体池管理器"""
    
    def __init__(self, message_bus: MessageBus, max_agents: int = 100):
        """初始化智能体池管理器
        
        Args:
            message_bus: 消息总线
            max_agents: 最大智能体数量
        """
        self.pool_id = str(uuid.uuid4())
        self.message_bus = message_bus
        self.max_agents = max_agents
        
        # 初始化日志
        self.logger = MultiAgentLogger("agent_pool_manager")
        
        # 池管理
        self.agents: Dict[str, AgentInfo] = {}
        self.pool_config = PoolConfiguration(
            strategy=PoolStrategy.DYNAMIC_SCALING,
            max_agents=max_agents
        )
        
        # 能力索引
        self.capability_index: Dict[str, Set[str]] = defaultdict(set)
        self.type_index: Dict[str, Set[str]] = defaultdict(set)
        
        # 统计信息
        self.metrics = {
            "total_registrations": 0,
            "total_unregistrations": 0,
            "average_uptime": 0.0,
            "health_distribution": defaultdict(int),
            "utilization_history": deque(maxlen=100),
            "resource_usage_avg": defaultdict(float)
        }
        
        # 事件回调
        self.event_callbacks: Dict[str, List[Callable]] = {}
        
        self.logger.info(f"智能体池管理器 {self.pool_id} 已初始化")
    
    async def initialize(self) -> bool:
        """初始化智能体池管理器"""
        try:
            self.logger.info("初始化智能体池管理器...")
            
            # 订阅消息总线事件
            await self.message_bus.subscribe("agent.heartbeat", self._handle_agent_heartbeat)
            await self.message_bus.subscribe("agent.status_update", self._handle_agent_status_update)
            
            # 启动后台任务
            asyncio.create_task(self._health_monitor())
            asyncio.create_task(self._resource_monitor())
            asyncio.create_task(self._auto_scaler())
            asyncio.create_task(self._metrics_collector())
            
            self.logger.info("智能体池管理器初始化完成")
            return True
            
        except Exception as e:
            self.logger.error(f"智能体池管理器初始化失败: {e}")
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
            # 检查是否超过最大数量
            if len(self.agents) >= self.max_agents:
                self.logger.warning(f"智能体池已满，无法注册智能体 {agent_id}")
                return False
            
            # 检查智能体是否已存在
            if agent_id in self.agents:
                self.logger.warning(f"智能体 {agent_id} 已存在，将更新信息")
                await self.unregister_agent(agent_id)
            
            # 创建智能体信息
            agent = AgentInfo(
                agent_id=agent_id,
                name=agent_info.get("name", agent_id),
                agent_type=agent_info.get("type", "general"),
                capabilities=capabilities,
                status=AgentStatus.IDLE,
                registered_at=datetime.now(),
                last_heartbeat=datetime.now(),
                health_status=HealthStatus.UNKNOWN,
                metadata=agent_info.get("metadata", {})
            )
            
            # 注册智能体
            self.agents[agent_id] = agent
            
            # 更新索引
            await self._update_agent_index(agent_id, capabilities)
            
            # 更新统计
            self.metrics["total_registrations"] += 1
            
            # 发送注册事件
            await self._emit_event("agent.registered", {
                "agent_id": agent_id,
                "agent_type": agent.agent_type,
                "capabilities": [cap.name for cap in capabilities],
                "pool_size": len(self.agents)
            })
            
            self.logger.info(f"智能体 {agent_id} 已注册到池中")
            return True
            
        except Exception as e:
            self.logger.error(f"注册智能体失败 {agent_id}: {e}")
            return False
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """注销智能体"""
        try:
            if agent_id not in self.agents:
                self.logger.warning(f"智能体 {agent_id} 不存在")
                return False
            
            agent = self.agents[agent_id]
            
            # 从索引中移除
            await self._remove_agent_index(agent_id, agent.capabilities)
            
            # 移除智能体
            del self.agents[agent_id]
            
            # 更新统计
            self.metrics["total_unregistrations"] += 1
            
            # 发送注销事件
            await self._emit_event("agent.unregistered", {
                "agent_id": agent_id,
                "uptime_seconds": agent.uptime_seconds,
                "pool_size": len(self.agents)
            })
            
            self.logger.info(f"智能体 {agent_id} 已从池中注销")
            return True
            
        except Exception as e:
            self.logger.error(f"注销智能体失败 {agent_id}: {e}")
            return False
    
    async def update_agent_heartbeat(self, agent_id: str, resource_usage: Optional[Dict[str, float]] = None) -> bool:
        """更新智能体心跳"""
        try:
            if agent_id not in self.agents:
                return False
            
            agent = self.agents[agent_id]
            agent.last_heartbeat = datetime.now()
            
            # 更新资源使用情况
            if resource_usage:
                agent.resource_usage.update(resource_usage)
            
            # 发送心跳事件
            await self._emit_event("agent.heartbeat_received", {
                "agent_id": agent_id,
                "timestamp": agent.last_heartbeat.isoformat(),
                "resource_usage": agent.resource_usage
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"更新智能体心跳失败 {agent_id}: {e}")
            return False
    
    async def update_agent_status(self, agent_id: str, status: AgentStatus) -> bool:
        """更新智能体状态"""
        try:
            if agent_id not in self.agents:
                return False
            
            old_status = self.agents[agent_id].status
            self.agents[agent_id].status = status
            
            # 发送状态更新事件
            await self._emit_event("agent.status_updated", {
                "agent_id": agent_id,
                "old_status": old_status.value,
                "new_status": status.value
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"更新智能体状态失败 {agent_id}: {e}")
            return False
    
    async def find_available_agents(
        self,
        required_capabilities: Optional[List[str]] = None,
        agent_type: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[AgentInfo]:
        """查找可用的智能体
        
        Args:
            required_capabilities: 必需的能力列表
            agent_type: 智能体类型
            limit: 返回数量限制
            
        Returns:
            可用智能体列表
        """
        available_agents = []
        
        for agent in self.agents.values():
            # 检查是否可用
            if not agent.is_available:
                continue
            
            # 检查类型匹配
            if agent_type and agent.agent_type != agent_type:
                continue
            
            # 检查能力匹配
            if required_capabilities:
                agent_capabilities = {cap.name for cap in agent.capabilities}
                if not all(cap in agent_capabilities for cap in required_capabilities):
                    continue
            
            available_agents.append(agent)
        
        # 按性能和健康状态排序
        available_agents.sort(key=lambda a: (
            a.health_status.value,
            a.performance_metrics.get("success_rate", 0.0),
            -len(a.current_tasks) if hasattr(a, 'current_tasks') else 0
        ), reverse=True)
        
        # 应用限制
        if limit:
            available_agents = available_agents[:limit]
        
        return available_agents
    
    async def get_agent_by_id(self, agent_id: str) -> Optional[AgentInfo]:
        """根据ID获取智能体信息"""
        return self.agents.get(agent_id)
    
    async def get_agents_by_capability(self, capability: str) -> List[AgentInfo]:
        """根据能力获取智能体列表"""
        agent_ids = self.capability_index.get(capability, set())
        return [self.agents[agent_id] for agent_id in agent_ids if agent_id in self.agents]
    
    async def get_agents_by_type(self, agent_type: str) -> List[AgentInfo]:
        """根据类型获取智能体列表"""
        agent_ids = self.type_index.get(agent_type, set())
        return [self.agents[agent_id] for agent_id in agent_ids if agent_id in self.agents]
    
    async def _update_agent_index(self, agent_id: str, capabilities: List[AgentCapability]):
        """更新智能体索引"""
        # 能力索引
        for capability in capabilities:
            self.capability_index[capability.name].add(agent_id)
        
        # 类型索引（从能力中推断或从元数据获取）
        agent = self.agents[agent_id]
        self.type_index[agent.agent_type].add(agent_id)
    
    async def _remove_agent_index(self, agent_id: str, capabilities: List[AgentCapability]):
        """移除智能体索引"""
        # 能力索引
        for capability in capabilities:
            if agent_id in self.capability_index[capability.name]:
                self.capability_index[capability.name].remove(agent_id)
            
            # 如果集合为空，删除键
            if not self.capability_index[capability.name]:
                del self.capability_index[capability.name]
        
        # 类型索引
        agent = self.agents.get(agent_id)
        if agent:
            if agent_id in self.type_index[agent.agent_type]:
                self.type_index[agent.agent_type].remove(agent_id)
            
            if not self.type_index[agent.agent_type]:
                del self.type_index[agent.agent_type]
    
    async def _health_monitor(self):
        """健康监控后台任务"""
        while True:
            try:
                await asyncio.sleep(self.pool_config.health_check_interval)
                
                current_time = datetime.now()
                unhealthy_agents = []
                
                for agent_id, agent in self.agents.items():
                    # 检查心跳超时
                    time_since_heartbeat = (current_time - agent.last_heartbeat).total_seconds()
                    if time_since_heartbeat > self.pool_config.heartbeat_timeout:
                        agent.health_status = HealthStatus.UNHEALTHY
                        unhealthy_agents.append(agent_id)
                        continue
                    
                    # 检查资源使用情况
                    if await self._check_resource_health(agent):
                        agent.health_status = HealthStatus.HEALTHY
                    else:
                        agent.health_status = HealthStatus.DEGRADED
                
                # 处理不健康的智能体
                for agent_id in unhealthy_agents:
                    self.logger.warning(f"智能体 {agent_id} 心跳超时，标记为不健康")
                    
                    await self._emit_event("agent.unhealthy", {
                        "agent_id": agent_id,
                        "reason": "heartbeat_timeout",
                        "last_heartbeat": self.agents[agent_id].last_heartbeat.isoformat()
                    })
                    
                    # 如果启用了自动恢复，尝试重新连接
                    if self.pool_config.auto_recovery:
                        asyncio.create_task(self._attempt_agent_recovery(agent_id))
                
            except Exception as e:
                self.logger.error(f"健康监控任务出错: {e}")
    
    async def _check_resource_health(self, agent: AgentInfo) -> bool:
        """检查智能体资源健康状态"""
        resource_usage = agent.resource_usage
        
        # 检查CPU使用率
        cpu_percent = resource_usage.get("cpu_percent", 0)
        if cpu_percent > self.pool_config.resource_limits["cpu_percent"]:
            return False
        
        # 检查内存使用率
        memory_percent = resource_usage.get("memory_percent", 0)
        if memory_percent > self.pool_config.resource_limits["memory_percent"]:
            return False
        
        # 检查磁盘使用率
        disk_percent = resource_usage.get("disk_percent", 0)
        if disk_percent > self.pool_config.resource_limits["disk_percent"]:
            return False
        
        return True
    
    async def _attempt_agent_recovery(self, agent_id: str):
        """尝试恢复智能体"""
        try:
            # 等待一段时间后再次检查
            await asyncio.sleep(60)
            
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                current_time = datetime.now()
                
                # 如果智能体恢复心跳，则标记为健康
                if (current_time - agent.last_heartbeat).total_seconds() <= self.pool_config.heartbeat_timeout:
                    agent.health_status = HealthStatus.HEALTHY
                    
                    await self._emit_event("agent.recovered", {
                        "agent_id": agent_id,
                        "recovery_time": current_time.isoformat()
                    })
                    
                    self.logger.info(f"智能体 {agent_id} 已恢复")
                else:
                    # 如果仍然没有心跳，则注销
                    self.logger.warning(f"智能体 {agent_id} 恢复失败，将被注销")
                    await self.unregister_agent(agent_id)
            
        except Exception as e:
            self.logger.error(f"尝试恢复智能体失败 {agent_id}: {e}")
    
    async def _resource_monitor(self):
        """资源监控后台任务"""
        while True:
            try:
                await asyncio.sleep(60)  # 每分钟监控一次
                
                # 收集系统资源使用情况
                system_resources = await self._collect_system_resources()
                
                # 更新智能体资源使用情况
                for agent_id, agent in self.agents.items():
                    # 模拟智能体资源使用情况（实际实现中需要从智能体获取）
                    if not agent.resource_usage:
                        agent.resource_usage = {
                            "cpu_percent": system_resources["cpu_percent"] * 0.1,
                            "memory_percent": system_resources["memory_percent"] * 0.1,
                            "disk_percent": system_resources["disk_percent"] * 0.1
                        }
                
                # 发送资源监控事件
                await self._emit_event("resource.monitored", {
                    "timestamp": datetime.now().isoformat(),
                    "system_resources": system_resources,
                    "agent_count": len(self.agents)
                })
                
            except Exception as e:
                self.logger.error(f"资源监控任务出错: {e}")
    
    async def _collect_system_resources(self) -> Dict[str, float]:
        """收集系统资源使用情况"""
        try:
            return {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent,
                "network_io": psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {}
            }
        except Exception as e:
            self.logger.error(f"收集系统资源失败: {e}")
            return {"cpu_percent": 0, "memory_percent": 0, "disk_percent": 0}
    
    async def _auto_scaler(self):
        """自动伸缩后台任务"""
        while True:
            try:
                await asyncio.sleep(120)  # 每2分钟检查一次
                
                if self.pool_config.strategy in [PoolStrategy.DYNAMIC_SCALING, PoolStrategy.AUTO_SCALING]:
                    await self._perform_auto_scaling()
                
            except Exception as e:
                self.logger.error(f"自动伸缩任务出错: {e}")
    
    async def _perform_auto_scaling(self):
        """执行自动伸缩"""
        current_utilization = await self._calculate_pool_utilization()
        
        # 记录利用率历史
        self.metrics["utilization_history"].append(current_utilization)
        
        # 检查是否需要扩容
        if (current_utilization > self.pool_config.scale_up_threshold and 
            len(self.agents) < self.pool_config.max_agents):
            
            await self._scale_up()
        
        # 检查是否需要缩容
        elif (current_utilization < self.pool_config.scale_down_threshold and 
              len(self.agents) > self.pool_config.min_agents):
            
            await self._scale_down()
    
    async def _calculate_pool_utilization(self) -> float:
        """计算池利用率"""
        if not self.agents:
            return 0.0
        
        available_agents = sum(1 for agent in self.agents.values() if agent.is_available)
        return 1.0 - (available_agents / len(self.agents))
    
    async def _scale_up(self):
        """扩容"""
        self.logger.info("执行池扩容")
        
        await self._emit_event("pool.scaling_up", {
            "current_size": len(self.agents),
            "target_size": min(len(self.agents) + 5, self.pool_config.max_agents),
            "reason": "high_utilization"
        })
        
        # 实际扩容逻辑需要根据具体部署环境实现
        # 这里只是发送事件通知
    
    async def _scale_down(self):
        """缩容"""
        self.logger.info("执行池缩容")
        
        # 选择最不活跃的智能体进行缩容
        candidates = []
        for agent in self.agents.values():
            if agent.is_available:
                candidates.append((agent.agent_id, agent.uptime_seconds))
        
        # 按运行时间排序，运行时间长的优先缩容
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        agents_to_remove = min(3, len(candidates))
        for i in range(agents_to_remove):
            agent_id = candidates[i][0]
            await self.unregister_agent(agent_id)
        
        await self._emit_event("pool.scaling_down", {
            "current_size": len(self.agents),
            "removed_agents": agents_to_remove,
            "reason": "low_utilization"
        })
    
    async def _metrics_collector(self):
        """指标收集后台任务"""
        while True:
            try:
                await asyncio.sleep(300)  # 每5分钟收集一次
                
                # 更新健康状态分布
                health_dist = defaultdict(int)
                for agent in self.agents.values():
                    health_dist[agent.health_status.value] += 1
                
                self.metrics["health_distribution"] = dict(health_dist)
                
                # 计算平均运行时间
                if self.agents:
                    total_uptime = sum(agent.uptime_seconds for agent in self.agents.values())
                    self.metrics["average_uptime"] = total_uptime / len(self.agents)
                
                # 计算平均资源使用情况
                if self.agents:
                    resource_totals = defaultdict(float)
                    for agent in self.agents.values():
                        for resource, usage in agent.resource_usage.items():
                            resource_totals[resource] += usage
                    
                    for resource, total in resource_totals.items():
                        self.metrics["resource_usage_avg"][resource] = total / len(self.agents)
                
                # 发送指标收集事件
                await self._emit_event("pool.metrics_collected", {
                    "timestamp": datetime.now().isoformat(),
                    "metrics": dict(self.metrics)
                })
                
            except Exception as e:
                self.logger.error(f"指标收集任务出错: {e}")
    
    async def _handle_agent_heartbeat(self, data: Dict[str, Any]):
        """处理智能体心跳事件"""
        agent_id = data.get("agent_id")
        resource_usage = data.get("resource_usage", {})
        
        if agent_id:
            await self.update_agent_heartbeat(agent_id, resource_usage)
    
    async def _handle_agent_status_update(self, data: Dict[str, Any]):
        """处理智能体状态更新事件"""
        agent_id = data.get("agent_id")
        status = data.get("status")
        
        if agent_id and status:
            try:
                agent_status = AgentStatus(status)
                await self.update_agent_status(agent_id, agent_status)
            except ValueError:
                self.logger.warning(f"未知的智能体状态: {status}")
    
    def configure_pool(self, config: PoolConfiguration):
        """配置池参数"""
        self.pool_config = config
        self.max_agents = config.max_agents
        
        self.logger.info(f"池配置已更新: {config.strategy.value}")
    
    def get_pool_status(self) -> Dict[str, Any]:
        """获取池状态"""
        available_agents = sum(1 for agent in self.agents.values() if agent.is_available)
        utilization = 1.0 - (available_agents / len(self.agents)) if self.agents else 0.0
        
        return {
            "pool_id": self.pool_id,
            "current_size": len(self.agents),
            "max_size": self.max_agents,
            "available_agents": available_agents,
            "utilization": utilization,
            "strategy": self.pool_config.strategy.value,
            "health_distribution": dict(self.metrics["health_distribution"]),
            "capability_distribution": {
                capability: len(agents) 
                for capability, agents in self.capability_index.items()
            },
            "type_distribution": {
                agent_type: len(agents)
                for agent_type, agents in self.type_index.items()
            },
            "metrics": dict(self.metrics)
        }
    
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
        """关闭智能体池管理器"""
        try:
            self.logger.info("关闭智能体池管理器...")
            
            # 注销所有智能体
            agent_ids = list(self.agents.keys())
            for agent_id in agent_ids:
                await self.unregister_agent(agent_id)
            
            self.logger.info("智能体池管理器已关闭")
            
        except Exception as e:
            self.logger.error(f"关闭智能体池管理器时出错: {e}")