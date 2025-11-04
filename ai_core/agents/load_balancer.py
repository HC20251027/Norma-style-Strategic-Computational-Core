#!/usr/bin/env python3
"""
负载均衡器
负责在多个智能体之间分配工作负载，实现负载均衡

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
import statistics

from ..utils.logger import MultiAgentLogger
from ..communication.message_bus import MessageBus
from .agent_pool_manager import AgentInfo, HealthStatus


class LoadBalancingStrategy(Enum):
    """负载均衡策略枚举"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_RESPONSE_TIME = "least_response_time"
    RESOURCE_BASED = "resource_based"
    PREDICTIVE = "predictive"
    ADAPTIVE = "adaptive"


class LoadMetric(Enum):
    """负载指标枚举"""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    ACTIVE_TASKS = "active_tasks"
    RESPONSE_TIME = "response_time"
    SUCCESS_RATE = "success_rate"
    QUEUE_LENGTH = "queue_length"


@dataclass
class LoadBalanceRequest:
    """负载均衡请求"""
    request_id: str
    task_type: str
    required_capabilities: List[str]
    priority: int
    resource_requirements: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: datetime
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AgentLoadInfo:
    """智能体负载信息"""
    agent_id: str
    current_load: float
    capacity: float
    utilization: float
    active_tasks: int
    queue_length: int
    response_time_avg: float
    success_rate: float
    last_updated: datetime
    resource_usage: Dict[str, float]
    
    @property
    def available_capacity(self) -> float:
        """可用容量"""
        return max(0, self.capacity - self.current_load)
    
    @property
    def load_score(self) -> float:
        """负载评分（越低越好）"""
        return self.utilization * 0.4 + (self.active_tasks / max(self.capacity, 1)) * 0.3 + self.response_time_avg * 0.3


class LoadBalancer:
    """负载均衡器"""
    
    def __init__(self, message_bus: MessageBus):
        """初始化负载均衡器
        
        Args:
            message_bus: 消息总线
        """
        self.balancer_id = str(uuid.uuid4())
        self.message_bus = message_bus
        
        # 初始化日志
        self.logger = MultiAgentLogger("load_balancer")
        
        # 负载均衡策略
        self.strategy = LoadBalancingStrategy.ADAPTIVE
        self.strategy_weights = {
            LoadBalancingStrategy.ROUND_ROBIN: 0.1,
            LoadBalancingStrategy.LEAST_CONNECTIONS: 0.2,
            LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN: 0.15,
            LoadBalancingStrategy.LEAST_RESPONSE_TIME: 0.25,
            LoadBalancingStrategy.RESOURCE_BASED: 0.2,
            LoadBalancingStrategy.PREDICTIVE: 0.05,
            LoadBalancingStrategy.ADAPTIVE: 0.05
        }
        
        # 智能体负载信息
        self.agent_loads: Dict[str, AgentLoadInfo] = {}
        self.agent_weights: Dict[str, float] = {}  # 智能体权重
        
        # 负载历史
        self.load_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        
        # 统计信息
        self.metrics = {
            "total_requests": 0,
            "successful_balances": 0,
            "failed_balances": 0,
            "average_response_time": 0.0,
            "load_distribution_variance": 0.0,
            "strategy_effectiveness": defaultdict(float)
        }
        
        # 请求队列
        self.pending_requests: deque = deque()
        
        # 事件回调
        self.event_callbacks: Dict[str, List[Callable]] = {}
        
        # 轮询计数器
        self.round_robin_counter = 0
        
        self.logger.info(f"负载均衡器 {self.balancer_id} 已初始化")
    
    async def initialize(self) -> bool:
        """初始化负载均衡器"""
        try:
            self.logger.info("初始化负载均衡器...")
            
            # 订阅消息总线事件
            await self.message_bus.subscribe("agent.load_update", self._handle_agent_load_update)
            await self.message_bus.subscribe("agent.performance_update", self._handle_agent_performance_update)
            await self.message_bus.subscribe("task.completed", self._handle_task_completed)
            await self.message_bus.subscribe("task.failed", self._handle_task_failed)
            
            # 启动后台任务
            asyncio.create_task(self._load_monitor())
            asyncio.create_task(self._request_processor())
            asyncio.create_task(self._performance_analyzer())
            asyncio.create_task(self._adaptive_optimizer())
            
            self.logger.info("负载均衡器初始化完成")
            return True
            
        except Exception as e:
            self.logger.error(f"负载均衡器初始化失败: {e}")
            return False
    
    async def register_agent(self, agent_info: AgentInfo):
        """注册智能体"""
        try:
            agent_id = agent_info.agent_id
            
            # 创建负载信息
            load_info = AgentLoadInfo(
                agent_id=agent_id,
                current_load=0.0,
                capacity=self._calculate_agent_capacity(agent_info),
                utilization=0.0,
                active_tasks=0,
                queue_length=0,
                response_time_avg=1.0,
                success_rate=0.8,
                last_updated=datetime.now(),
                resource_usage=agent_info.resource_usage.copy()
            )
            
            self.agent_loads[agent_id] = load_info
            
            # 设置默认权重
            self.agent_weights[agent_id] = 1.0
            
            # 发送注册事件
            await self._emit_event("agent.registered_for_lb", {
                "agent_id": agent_id,
                "capacity": load_info.capacity,
                "weight": self.agent_weights[agent_id]
            })
            
            self.logger.info(f"智能体 {agent_id} 已注册到负载均衡器")
            
        except Exception as e:
            self.logger.error(f"注册智能体到负载均衡器失败: {e}")
    
    async def unregister_agent(self, agent_id: str):
        """注销智能体"""
        try:
            if agent_id in self.agent_loads:
                del self.agent_loads[agent_id]
            
            if agent_id in self.agent_weights:
                del self.agent_weights[agent_id]
            
            # 清理历史数据
            if agent_id in self.load_history:
                del self.load_history[agent_id]
            
            if agent_id in self.performance_history:
                del self.performance_history[agent_id]
            
            # 发送注销事件
            await self._emit_event("agent.unregistered_from_lb", {
                "agent_id": agent_id
            })
            
            self.logger.info(f"智能体 {agent_id} 已从负载均衡器注销")
            
        except Exception as e:
            self.logger.error(f"从负载均衡器注销智能体失败: {e}")
    
    async def balance_load(
        self,
        request: LoadBalanceRequest,
        available_agents: List[AgentInfo]
    ) -> Optional[str]:
        """执行负载均衡
        
        Args:
            request: 负载均衡请求
            available_agents: 可用智能体列表
            
        Returns:
            选中的智能体ID
        """
        try:
            self.metrics["total_requests"] += 1
            
            if not available_agents:
                self.logger.warning("没有可用的智能体进行负载均衡")
                self.metrics["failed_balances"] += 1
                return None
            
            # 根据策略选择智能体
            selected_agent_id = await self._select_agent_with_strategy(
                request, available_agents, self.strategy
            )
            
            if selected_agent_id:
                # 更新智能体负载
                await self._update_agent_load(selected_agent_id, request)
                
                # 记录负载历史
                self.load_history[selected_agent_id].append(datetime.now())
                
                self.metrics["successful_balances"] += 1
                
                # 发送负载均衡事件
                await self._emit_event("load.balanced", {
                    "request_id": request.request_id,
                    "selected_agent": selected_agent_id,
                    "strategy": self.strategy.value,
                    "available_agents": len(available_agents)
                })
                
                self.logger.info(f"负载均衡完成，选择智能体 {selected_agent_id}")
                return selected_agent_id
            else:
                self.metrics["failed_balances"] += 1
                self.logger.warning("负载均衡失败，没有找到合适的智能体")
                return None
                
        except Exception as e:
            self.logger.error(f"负载均衡失败: {e}")
            self.metrics["failed_balances"] += 1
            return None
    
    async def _select_agent_with_strategy(
        self,
        request: LoadBalanceRequest,
        available_agents: List[AgentInfo],
        strategy: LoadBalancingStrategy
    ) -> Optional[str]:
        """使用指定策略选择智能体"""
        if not available_agents:
            return None
        
        if len(available_agents) == 1:
            return available_agents[0].agent_id
        
        agent_scores = {}
        
        for agent in available_agents:
            agent_id = agent.agent_id
            
            if strategy == LoadBalancingStrategy.ROUND_ROBIN:
                score = self._calculate_round_robin_score(agent_id)
            
            elif strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                score = self._calculate_least_connections_score(agent)
            
            elif strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
                score = self._calculate_weighted_round_robin_score(agent_id)
            
            elif strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
                score = self._calculate_least_response_time_score(agent)
            
            elif strategy == LoadBalancingStrategy.RESOURCE_BASED:
                score = self._calculate_resource_based_score(agent, request)
            
            elif strategy == LoadBalancingStrategy.PREDICTIVE:
                score = await self._calculate_predictive_score(agent, request)
            
            elif strategy == LoadBalancingStrategy.ADAPTIVE:
                score = await self._calculate_adaptive_score(agent, request)
            
            else:
                score = 0.0
            
            agent_scores[agent_id] = score
        
        # 选择评分最高的智能体
        if agent_scores:
            selected_agent_id = max(agent_scores.items(), key=lambda x: x[1])[0]
            return selected_agent_id
        
        return None
    
    def _calculate_round_robin_score(self, agent_id: str) -> float:
        """计算轮询评分"""
        # 简单的轮询逻辑
        return -self.round_robin_counter  # 负号使得轮询计数器增加时评分降低
    
    def _calculate_least_connections_score(self, agent: AgentInfo) -> float:
        """计算最少连接评分"""
        load_info = self.agent_loads.get(agent.agent_id)
        if not load_info:
            return 0.0
        
        # 连接数越少评分越高
        return 1.0 / (load_info.active_tasks + 1)
    
    def _calculate_weighted_round_robin_score(self, agent_id: str) -> float:
        """计算加权轮询评分"""
        weight = self.agent_weights.get(agent_id, 1.0)
        return weight * (1.0 / (self.round_robin_counter + 1))
    
    def _calculate_least_response_time_score(self, agent: AgentInfo) -> float:
        """计算最少响应时间评分"""
        load_info = self.agent_loads.get(agent.agent_id)
        if not load_info:
            return 0.0
        
        # 响应时间越短评分越高
        return 1.0 / max(load_info.response_time_avg, 0.1)
    
    def _calculate_resource_based_score(self, agent: AgentInfo, request: LoadBalanceRequest) -> float:
        """计算基于资源的评分"""
        load_info = self.agent_loads.get(agent.agent_id)
        if not load_info:
            return 0.0
        
        score = 0.0
        
        # CPU使用率评分
        cpu_usage = load_info.resource_usage.get("cpu_percent", 0)
        cpu_score = max(0, 1.0 - cpu_usage / 100.0)
        score += cpu_score * 0.4
        
        # 内存使用率评分
        memory_usage = load_info.resource_usage.get("memory_percent", 0)
        memory_score = max(0, 1.0 - memory_usage / 100.0)
        score += memory_score * 0.3
        
        # 可用容量评分
        capacity_score = load_info.available_capacity / load_info.capacity
        score += capacity_score * 0.3
        
        return score
    
    async def _calculate_predictive_score(self, agent: AgentInfo, request: LoadBalanceRequest) -> float:
        """计算预测性评分"""
        load_info = self.agent_loads.get(agent.agent_id)
        if not load_info:
            return 0.0
        
        score = 0.0
        
        # 基于历史负载趋势预测
        if agent.agent_id in self.load_history:
            recent_loads = list(self.load_history[agent.agent_id])
            if len(recent_loads) >= 2:
                # 计算负载趋势
                time_diff = (recent_loads[-1] - recent_loads[-2]).total_seconds()
                if time_diff > 0:
                    load_trend = 1.0 / time_diff  # 负载增长趋势
                    score += max(0, 1.0 - load_trend) * 0.3
        
        # 基于任务类型历史表现
        task_type_performance = await self._get_task_type_performance(agent.agent_id, request.task_type)
        score += task_type_performance * 0.4
        
        # 基于当前负载
        current_load_score = 1.0 - load_info.utilization
        score += current_load_score * 0.3
        
        return score
    
    async def _calculate_adaptive_score(self, agent: AgentInfo, request: LoadBalanceRequest) -> float:
        """计算自适应评分"""
        load_info = self.agent_loads.get(agent.agent_id)
        if not load_info:
            return 0.0
        
        score = 0.0
        
        # 综合多种策略的评分
        strategies = [
            (LoadBalancingStrategy.LEAST_CONNECTIONS, 0.25),
            (LoadBalancingStrategy.LEAST_RESPONSE_TIME, 0.25),
            (LoadBalancingStrategy.RESOURCE_BASED, 0.25),
            (LoadBalancingStrategy.PREDICTIVE, 0.25)
        ]
        
        for strategy, weight in strategies:
            strategy_score = await self._get_strategy_score(agent, request, strategy)
            score += strategy_score * weight
        
        # 考虑智能体健康状态
        if agent.health_status == HealthStatus.HEALTHY:
            score *= 1.0
        elif agent.health_status == HealthStatus.DEGRADED:
            score *= 0.7
        else:
            score *= 0.3
        
        return score
    
    async def _get_strategy_score(
        self,
        agent: AgentInfo,
        request: LoadBalanceRequest,
        strategy: LoadBalancingStrategy
    ) -> float:
        """获取指定策略的评分"""
        if strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._calculate_least_connections_score(agent)
        elif strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return self._calculate_least_response_time_score(agent)
        elif strategy == LoadBalancingStrategy.RESOURCE_BASED:
            return self._calculate_resource_based_score(agent, request)
        elif strategy == LoadBalancingStrategy.PREDICTIVE:
            return await self._calculate_predictive_score(agent, request)
        else:
            return 0.0
    
    async def _get_task_type_performance(self, agent_id: str, task_type: str) -> float:
        """获取智能体对特定任务类型的历史表现"""
        # 简化的性能评估
        # 实际实现中需要维护更详细的历史数据
        performance_data = self.performance_history.get(f"{agent_id}_{task_type}", deque())
        
        if performance_data:
            recent_performance = list(performance_data)[-10:]  # 最近10次
            return statistics.mean(recent_performance) if recent_performance else 0.5
        
        return 0.5  # 默认性能
    
    def _calculate_agent_capacity(self, agent_info: AgentInfo) -> float:
        """计算智能体容量"""
        # 基于智能体能力和资源配置计算容量
        base_capacity = 10.0  # 基础容量
        
        # 根据能力数量调整
        capability_factor = len(agent_info.capabilities) * 0.5
        
        # 根据资源使用情况调整
        resource_factor = 1.0
        if agent_info.resource_usage:
            cpu_cores = agent_info.resource_usage.get("cpu_cores", 4)
            memory_gb = agent_info.resource_usage.get("memory_gb", 8)
            resource_factor = min(2.0, (cpu_cores / 4.0 + memory_gb / 8.0) / 2.0)
        
        return base_capacity * (1 + capability_factor) * resource_factor
    
    async def _update_agent_load(self, agent_id: str, request: LoadBalanceRequest):
        """更新智能体负载"""
        if agent_id in self.agent_loads:
            load_info = self.agent_loads[agent_id]
            
            # 增加当前负载
            load_increment = self._calculate_load_increment(request)
            load_info.current_load += load_increment
            load_info.active_tasks += 1
            
            # 更新利用率
            load_info.utilization = load_info.current_load / load_info.capacity
            
            # 更新最后更新时间
            load_info.last_updated = datetime.now()
            
            # 发送负载更新事件
            await self._emit_event("agent.load_updated", {
                "agent_id": agent_id,
                "load_increment": load_increment,
                "current_load": load_info.current_load,
                "utilization": load_info.utilization
            })
    
    def _calculate_load_increment(self, request: LoadBalanceRequest) -> float:
        """计算负载增量"""
        base_load = 1.0
        
        # 根据任务优先级调整
        priority_factor = request.priority / 10.0
        
        # 根据资源需求调整
        resource_factor = 1.0
        if "cpu_intensive" in request.resource_requirements:
            resource_factor *= 1.5
        if "memory_intensive" in request.resource_requirements:
            resource_factor *= 1.3
        
        return base_load * priority_factor * resource_factor
    
    async def _load_monitor(self):
        """负载监控后台任务"""
        while True:
            try:
                await asyncio.sleep(30)  # 每30秒监控一次
                
                # 检查负载过高的智能体
                overloaded_agents = []
                for agent_id, load_info in self.agent_loads.items():
                    if load_info.utilization > 0.9:  # 负载超过90%
                        overloaded_agents.append(agent_id)
                
                # 处理过载的智能体
                for agent_id in overloaded_agents:
                    self.logger.warning(f"智能体 {agent_id} 负载过载: {self.agent_loads[agent_id].utilization:.2%}")
                    
                    await self._emit_event("agent.overloaded", {
                        "agent_id": agent_id,
                        "utilization": self.agent_loads[agent_id].utilization,
                        "current_load": self.agent_loads[agent_id].current_load
                    })
                
                # 计算负载分布方差
                if self.agent_loads:
                    utilizations = [load.utilization for load in self.agent_loads.values()]
                    self.metrics["load_distribution_variance"] = statistics.variance(utilizations)
                
            except Exception as e:
                self.logger.error(f"负载监控任务出错: {e}")
    
    async def _request_processor(self):
        """请求处理后台任务"""
        while True:
            try:
                await asyncio.sleep(1)  # 每秒处理一次
                
                # 处理待处理的请求
                while self.pending_requests:
                    request = self.pending_requests.popleft()
                    
                    # 这里应该调用具体的负载均衡逻辑
                    # 实际实现中需要与智能体池管理器协作
                    
            except Exception as e:
                self.logger.error(f"请求处理任务出错: {e}")
    
    async def _performance_analyzer(self):
        """性能分析后台任务"""
        while True:
            try:
                await asyncio.sleep(60)  # 每分钟分析一次
                
                # 分析各策略的效果
                await self._analyze_strategy_effectiveness()
                
                # 更新智能体权重
                await self._update_agent_weights()
                
            except Exception as e:
                self.logger.error(f"性能分析任务出错: {e}")
    
    async def _analyze_strategy_effectiveness(self):
        """分析策略效果"""
        total_requests = self.metrics["total_requests"]
        if total_requests == 0:
            return
        
        successful_requests = self.metrics["successful_balances"]
        success_rate = successful_requests / total_requests
        
        # 分析各种策略的历史效果
        for strategy in LoadBalancingStrategy:
            # 这里需要更详细的历史数据来计算策略效果
            # 简化实现
            effectiveness = success_rate * (1.0 - self.metrics["load_distribution_variance"])
            self.metrics["strategy_effectiveness"][strategy.value] = effectiveness
    
    async def _update_agent_weights(self):
        """更新智能体权重"""
        for agent_id, load_info in self.agent_loads.items():
            # 基于性能和负载调整权重
            performance_score = load_info.success_rate
            load_score = 1.0 - load_info.utilization
            
            # 综合评分
            new_weight = (performance_score * 0.6 + load_score * 0.4)
            
            # 平滑更新权重
            current_weight = self.agent_weights.get(agent_id, 1.0)
            updated_weight = current_weight * 0.8 + new_weight * 0.2
            
            self.agent_weights[agent_id] = max(0.1, min(5.0, updated_weight))
    
    async def _adaptive_optimizer(self):
        """自适应优化后台任务"""
        while True:
            try:
                await asyncio.sleep(300)  # 每5分钟优化一次
                
                # 根据历史表现调整策略权重
                await self._adapt_strategy_weights()
                
                # 检测是否需要切换策略
                await self._check_strategy_switch()
                
            except Exception as e:
                self.logger.error(f"自适应优化任务出错: {e}")
    
    async def _adapt_strategy_weights(self):
        """调整策略权重"""
        effectiveness = self.metrics["strategy_effectiveness"]
        
        if effectiveness:
            # 归一化效果评分
            total_effectiveness = sum(effectiveness.values())
            if total_effectiveness > 0:
                for strategy, eff in effectiveness.items():
                    normalized_eff = eff / total_effectiveness
                    # 平滑更新权重
                    current_weight = self.strategy_weights.get(LoadBalancingStrategy(strategy), 0.1)
                    self.strategy_weights[LoadBalancingStrategy(strategy)] = current_weight * 0.7 + normalized_eff * 0.3
    
    async def _check_strategy_switch(self):
        """检查是否需要切换策略"""
        # 简化的策略切换逻辑
        # 如果当前策略效果不佳，考虑切换到其他策略
        
        current_effectiveness = self.metrics["strategy_effectiveness"].get(self.strategy.value, 0)
        best_strategy = max(
            self.metrics["strategy_effectiveness"].items(),
            key=lambda x: x[1],
            default=(self.strategy.value, current_effectiveness)
        )
        
        # 如果最佳策略与当前策略不同，且效果明显更好，则切换
        if (best_strategy[0] != self.strategy.value and 
            best_strategy[1] > current_effectiveness * 1.2):
            
            old_strategy = self.strategy
            self.strategy = LoadBalancingStrategy(best_strategy[0])
            
            await self._emit_event("load_balancer.strategy_switched", {
                "old_strategy": old_strategy.value,
                "new_strategy": self.strategy.value,
                "reason": "performance_optimization"
            })
            
            self.logger.info(f"负载均衡策略已切换: {old_strategy.value} -> {self.strategy.value}")
    
    async def _handle_agent_load_update(self, data: Dict[str, Any]):
        """处理智能体负载更新事件"""
        agent_id = data.get("agent_id")
        load_data = data.get("load_data", {})
        
        if agent_id in self.agent_loads:
            load_info = self.agent_loads[agent_id]
            
            # 更新负载信息
            for key, value in load_data.items():
                if hasattr(load_info, key):
                    setattr(load_info, key, value)
            
            load_info.last_updated = datetime.now()
    
    async def _handle_agent_performance_update(self, data: Dict[str, Any]):
        """处理智能体性能更新事件"""
        agent_id = data.get("agent_id")
        performance_data = data.get("performance", {})
        
        if agent_id in self.agent_loads:
            load_info = self.agent_loads[agent_id]
            
            # 更新性能指标
            if "success_rate" in performance_data:
                load_info.success_rate = performance_data["success_rate"]
            
            if "response_time" in performance_data:
                load_info.response_time_avg = performance_data["response_time"]
            
            # 记录性能历史
            task_type = performance_data.get("task_type", "general")
            history_key = f"{agent_id}_{task_type}"
            self.performance_history[history_key].append(load_info.success_rate)
    
    async def _handle_task_completed(self, data: Dict[str, Any]):
        """处理任务完成事件"""
        agent_id = data.get("agent_id")
        task_id = data.get("task_id")
        
        if agent_id in self.agent_loads:
            load_info = self.agent_loads[agent_id]
            
            # 减少负载
            load_info.active_tasks -= 1
            load_info.current_load = max(0, load_info.current_load - 1.0)
            load_info.utilization = load_info.current_load / load_info.capacity
            
            # 更新成功率
            if hasattr(load_info, 'success_rate'):
                load_info.success_rate = load_info.success_rate * 0.9 + 1.0 * 0.1
    
    async def _handle_task_failed(self, data: Dict[str, Any]):
        """处理任务失败事件"""
        agent_id = data.get("agent_id")
        
        if agent_id in self.agent_loads:
            load_info = self.agent_loads[agent_id]
            
            # 减少负载（失败任务也算完成）
            load_info.active_tasks -= 1
            load_info.current_load = max(0, load_info.current_load - 1.0)
            load_info.utilization = load_info.current_load / load_info.capacity
            
            # 更新成功率
            if hasattr(load_info, 'success_rate'):
                load_info.success_rate = load_info.success_rate * 0.9 + 0.0 * 0.1
    
    def get_load_balance_status(self) -> Dict[str, Any]:
        """获取负载均衡状态"""
        avg_utilization = 0.0
        if self.agent_loads:
            avg_utilization = statistics.mean(load.utilization for load in self.agent_loads.values())
        
        return {
            "balancer_id": self.balancer_id,
            "current_strategy": self.strategy.value,
            "registered_agents": len(self.agent_loads),
            "average_utilization": avg_utilization,
            "strategy_weights": {k.value: v for k, v in self.strategy_weights.items()},
            "metrics": dict(self.metrics),
            "agent_loads": {
                agent_id: {
                    "utilization": load.utilization,
                    "active_tasks": load.active_tasks,
                    "response_time": load.response_time_avg,
                    "success_rate": load.success_rate
                }
                for agent_id, load in self.agent_loads.items()
            }
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
        """关闭负载均衡器"""
        try:
            self.logger.info("关闭负载均衡器...")
            
            # 注销所有智能体
            agent_ids = list(self.agent_loads.keys())
            for agent_id in agent_ids:
                await self.unregister_agent(agent_id)
            
            self.logger.info("负载均衡器已关闭")
            
        except Exception as e:
            self.logger.error(f"关闭负载均衡器时出错: {e}")