"""
智能体池管理和负载均衡
"""
import asyncio
import time
import random
from typing import Dict, Any, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from collections import defaultdict, deque
import heapq


class LoadBalancingStrategy(Enum):
    """负载均衡策略"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_RESPONSE_TIME = "least_response_time"
    RESOURCE_BASED = "resource_based"
    RANDOM = "random"


@dataclass
class AgentMetrics:
    """智能体指标"""
    agent_id: str
    current_load: float  # 0.0 - 1.0
    active_connections: int
    response_time: float  # 平均响应时间(毫秒)
    success_rate: float  # 成功率 0.0 - 1.0
    cpu_usage: float
    memory_usage: float
    last_heartbeat: datetime
    total_requests: int
    failed_requests: int


@dataclass
class PoolConfig:
    """智能体池配置"""
    pool_name: str
    agent_type: str
    min_size: int
    max_size: int
    scaling_policy: str  # auto, manual
    health_check_interval: int  # 秒
    load_threshold: float  # 负载阈值
    response_time_threshold: float  # 响应时间阈值
    scaling_up_cooldown: int  # 扩容冷却时间(秒)
    scaling_down_cooldown: int  # 缩容冷却时间(秒)


class AgentPool:
    """智能体池"""
    
    def __init__(self, config: PoolConfig):
        self.config = config
        self.agents = {}  # agent_id -> AgentMetrics
        self.available_agents = deque()  # 可用智能体队列
        self.agent_creation_func = None  # 创建智能体的函数
        self.logger = logging.getLogger(f"agent.pool.{config.pool_name}")
        
    def set_agent_creation_func(self, creation_func: Callable):
        """设置智能体创建函数"""
        self.agent_creation_func = creation_func
    
    def add_agent(self, agent_id: str, metrics: AgentMetrics):
        """添加智能体到池中"""
        self.agents[agent_id] = metrics
        self.available_agents.append(agent_id)
        self.logger.info(f"添加智能体到池: {agent_id}")
    
    def remove_agent(self, agent_id: str):
        """从池中移除智能体"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            # 从可用队列中移除
            self.available_agents = deque([aid for aid in self.available_agents if aid != agent_id])
            self.logger.info(f"从池中移除智能体: {agent_id}")
    
    def get_available_agents(self) -> List[str]:
        """获取可用智能体列表"""
        available = []
        for agent_id in self.available_agents:
            if agent_id in self.agents:
                metrics = self.agents[agent_id]
                if self._is_agent_healthy(metrics):
                    available.append(agent_id)
        return available
    
    def update_metrics(self, agent_id: str, metrics: AgentMetrics):
        """更新智能体指标"""
        if agent_id in self.agents:
            self.agents[agent_id] = metrics
            
            # 更新可用性
            if self._is_agent_healthy(metrics):
                if agent_id not in self.available_agents:
                    self.available_agents.append(agent_id)
            else:
                # 智能体不健康，从可用队列中移除
                self.available_agents = deque([aid for aid in self.available_agents if aid != agent_id])
    
    def _is_agent_healthy(self, metrics: AgentMetrics) -> bool:
        """检查智能体是否健康"""
        # 检查负载
        if metrics.current_load > self.config.load_threshold:
            return False
        
        # 检查响应时间
        if metrics.response_time > self.config.response_time_threshold:
            return False
        
        # 检查心跳
        if (datetime.now() - metrics.last_heartbeat).total_seconds() > 60:
            return False
        
        # 检查成功率
        if metrics.total_requests > 0:
            success_rate = (metrics.total_requests - metrics.failed_requests) / metrics.total_requests
            if success_rate < 0.8:  # 成功率低于80%
                return False
        
        return True
    
    def get_pool_status(self) -> Dict[str, Any]:
        """获取池状态"""
        total_agents = len(self.agents)
        available_agents = len(self.get_available_agents())
        
        if total_agents > 0:
            avg_load = sum(m.current_load for m in self.agents.values()) / total_agents
            avg_response_time = sum(m.response_time for m in self.agents.values()) / total_agents
            avg_success_rate = sum(m.success_rate for m in self.agents.values()) / total_agents
        else:
            avg_load = 0.0
            avg_response_time = 0.0
            avg_success_rate = 0.0
        
        return {
            "pool_name": self.config.pool_name,
            "agent_type": self.config.agent_type,
            "total_agents": total_agents,
            "available_agents": available_agents,
            "min_size": self.config.min_size,
            "max_size": self.config.max_size,
            "avg_load": avg_load,
            "avg_response_time": avg_response_time,
            "avg_success_rate": avg_success_rate,
            "scaling_policy": self.config.scaling_policy
        }


class LoadBalancer:
    """负载均衡器"""
    
    def __init__(self):
        self.pools = {}  # pool_name -> AgentPool
        self.strategies = {
            LoadBalancingStrategy.ROUND_ROBIN: self._round_robin,
            LoadBalancingStrategy.LEAST_CONNECTIONS: self._least_connections,
            LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN: self._weighted_round_robin,
            LoadBalancingStrategy.LEAST_RESPONSE_TIME: self._least_response_time,
            LoadBalancingStrategy.RESOURCE_BASED: self._resource_based,
            LoadBalancingStrategy.RANDOM: self._random
        }
        self.round_robin_counters = defaultdict(int)  # pool_name -> counter
        self.logger = logging.getLogger("load.balancer")
    
    def create_pool(self, config: PoolConfig):
        """创建智能体池"""
        pool = AgentPool(config)
        self.pools[config.pool_name] = pool
        self.logger.info(f"创建智能体池: {config.pool_name}")
        return pool
    
    def get_pool(self, pool_name: str) -> Optional[AgentPool]:
        """获取智能体池"""
        return self.pools.get(pool_name)
    
    def select_agent(self, pool_name: str, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN) -> Optional[str]:
        """选择智能体"""
        pool = self.get_pool(pool_name)
        if not pool:
            return None
        
        available_agents = pool.get_available_agents()
        if not available_agents:
            return None
        
        selector = self.strategies.get(strategy)
        if selector:
            return selector(pool, available_agents)
        else:
            self.logger.warning(f"未知的负载均衡策略: {strategy}")
            return available_agents[0] if available_agents else None
    
    def _round_robin(self, pool: AgentPool, available_agents: List[str]) -> str:
        """轮询策略"""
        counter = self.round_robin_counters[pool.config.pool_name]
        selected_agent = available_agents[counter % len(available_agents)]
        self.round_robin_counters[pool.config.pool_name] += 1
        return selected_agent
    
    def _least_connections(self, pool: AgentPool, available_agents: List[str]) -> str:
        """最少连接策略"""
        min_connections = float('inf')
        selected_agent = available_agents[0]
        
        for agent_id in available_agents:
            metrics = pool.agents[agent_id]
            if metrics.active_connections < min_connections:
                min_connections = metrics.active_connections
                selected_agent = agent_id
        
        return selected_agent
    
    def _weighted_round_robin(self, pool: AgentPool, available_agents: List[str]) -> str:
        """加权轮询策略"""
        weights = []
        for agent_id in available_agents:
            metrics = pool.agents[agent_id]
            # 计算权重：基于成功率、负载、响应时间
            weight = (
                metrics.success_rate * 0.4 +
                (1 - metrics.current_load) * 0.3 +
                (1 - min(metrics.response_time / 1000, 1.0)) * 0.3
            )
            weights.append(weight)
        
        # 加权随机选择
        total_weight = sum(weights)
        if total_weight == 0:
            return available_agents[0]
        
        random_weight = random.uniform(0, total_weight)
        current_weight = 0
        
        for i, weight in enumerate(weights):
            current_weight += weight
            if random_weight <= current_weight:
                return available_agents[i]
        
        return available_agents[-1]
    
    def _least_response_time(self, pool: AgentPool, available_agents: List[str]) -> str:
        """最少响应时间策略"""
        min_response_time = float('inf')
        selected_agent = available_agents[0]
        
        for agent_id in available_agents:
            metrics = pool.agents[agent_id]
            if metrics.response_time < min_response_time:
                min_response_time = metrics.response_time
                selected_agent = agent_id
        
        return selected_agent
    
    def _resource_based(self, pool: AgentPool, available_agents: List[str]) -> str:
        """基于资源策略"""
        best_score = -1
        selected_agent = available_agents[0]
        
        for agent_id in available_agents:
            metrics = pool.agents[agent_id]
            # 计算资源分数：CPU、内存、负载的综合评估
            cpu_score = 1 - metrics.cpu_usage
            memory_score = 1 - metrics.memory_usage
            load_score = 1 - metrics.current_load
            
            total_score = (cpu_score + memory_score + load_score) / 3
            
            if total_score > best_score:
                best_score = total_score
                selected_agent = agent_id
        
        return selected_agent
    
    def _random(self, pool: AgentPool, available_agents: List[str]) -> str:
        """随机策略"""
        return random.choice(available_agents)
    
    def get_load_balancer_status(self) -> Dict[str, Any]:
        """获取负载均衡器状态"""
        pool_statuses = {}
        for pool_name, pool in self.pools.items():
            pool_statuses[pool_name] = pool.get_pool_status()
        
        return {
            "total_pools": len(self.pools),
            "pools": pool_statuses,
            "strategies": [strategy.value for strategy in LoadBalancingStrategy]
        }


class AutoScaler:
    """自动扩缩容器"""
    
    def __init__(self, load_balancer: LoadBalancer):
        self.load_balancer = load_balancer
        self.scaling_history = []
        self.last_scaling_time = {}
        self.logger = logging.getLogger("auto.scaler")
    
    async def start_auto_scaling(self):
        """启动自动扩缩容"""
        while True:
            try:
                await self._check_scaling_conditions()
                await asyncio.sleep(30)  # 30秒检查一次
            except Exception as e:
                self.logger.error(f"自动扩缩容错误: {e}")
                await asyncio.sleep(30)
    
    async def _check_scaling_conditions(self):
        """检查扩缩容条件"""
        for pool_name, pool in self.load_balancer.pools.items():
            if pool.config.scaling_policy != "auto":
                continue
            
            await self._evaluate_pool_scaling(pool)
    
    async def _evaluate_pool_scaling(self, pool: AgentPool):
        """评估池的扩缩容"""
        current_size = len(pool.agents)
        available_agents = len(pool.get_available_agents())
        
        # 计算平均负载
        if current_size > 0:
            avg_load = sum(m.current_load for m in pool.agents.values()) / current_size
        else:
            avg_load = 0.0
        
        # 扩容条件
        should_scale_up = (
            available_agents == 0 or  # 没有可用智能体
            avg_load > 0.8 or  # 平均负载过高
            (current_size < pool.config.min_size)  # 低于最小规模
        )
        
        # 缩容条件
        should_scale_down = (
            current_size > pool.config.min_size and  # 大于最小规模
            avg_load < 0.3 and  # 平均负载较低
            available_agents > pool.config.min_size  # 可用智能体超过最小规模
        )
        
        if should_scale_up:
            await self._scale_up(pool)
        elif should_scale_down:
            await self._scale_down(pool)
    
    async def _scale_up(self, pool: AgentPool):
        """扩容"""
        current_size = len(pool.agents)
        if current_size >= pool.config.max_size:
            return
        
        # 检查冷却时间
        last_scale_time = self.last_scaling_time.get(pool.config.pool_name)
        if last_scale_time and (datetime.now() - last_scale_time).total_seconds() < pool.config.scaling_up_cooldown:
            return
        
        # 创建新智能体
        if pool.agent_creation_func:
            try:
                new_agent_id = await pool.agent_creation_func(pool.config.agent_type)
                
                # 添加到池中
                metrics = AgentMetrics(
                    agent_id=new_agent_id,
                    current_load=0.0,
                    active_connections=0,
                    response_time=0.0,
                    success_rate=1.0,
                    cpu_usage=0.0,
                    memory_usage=0.0,
                    last_heartbeat=datetime.now(),
                    total_requests=0,
                    failed_requests=0
                )
                
                pool.add_agent(new_agent_id, metrics)
                
                # 记录扩容
                scaling_event = {
                    "pool_name": pool.config.pool_name,
                    "action": "scale_up",
                    "agent_id": new_agent_id,
                    "timestamp": datetime.now().isoformat(),
                    "pool_size": len(pool.agents)
                }
                self.scaling_history.append(scaling_event)
                self.last_scaling_time[pool.config.pool_name] = datetime.now()
                
                self.logger.info(f"扩容成功: {pool.config.pool_name} -> {new_agent_id}")
                
            except Exception as e:
                self.logger.error(f"扩容失败: {e}")
    
    async def _scale_down(self, pool: AgentPool):
        """缩容"""
        available_agents = pool.get_available_agents()
        
        if len(available_agents) <= pool.config.min_size:
            return
        
        # 检查冷却时间
        last_scale_time = self.last_scaling_time.get(pool.config.pool_name)
        if last_scale_time and (datetime.now() - last_scale_time).total_seconds() < pool.config.scaling_down_cooldown:
            return
        
        # 选择负载最低的智能体进行缩容
        agent_to_remove = None
        min_load = float('inf')
        
        for agent_id in available_agents:
            metrics = pool.agents[agent_id]
            if metrics.current_load < min_load:
                min_load = metrics.current_load
                agent_to_remove = agent_id
        
        if agent_to_remove:
            # 这里应该实现优雅关闭智能体的逻辑
            pool.remove_agent(agent_to_remove)
            
            # 记录缩容
            scaling_event = {
                "pool_name": pool.config.pool_name,
                "action": "scale_down",
                "agent_id": agent_to_remove,
                "timestamp": datetime.now().isoformat(),
                "pool_size": len(pool.agents)
            }
            self.scaling_history.append(scaling_event)
            self.last_scaling_time[pool.config.pool_name] = datetime.now()
            
            self.logger.info(f"缩容成功: {pool.config.pool_name} <- {agent_to_remove}")
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """获取扩缩容状态"""
        return {
            "total_scaling_events": len(self.scaling_history),
            "recent_events": self.scaling_history[-10:],  # 最近10个事件
            "last_scaling_times": {
                pool_name: timestamp.isoformat()
                for pool_name, timestamp in self.last_scaling_time.items()
            }
        }