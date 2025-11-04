"""
智能Agent调度器

支持复杂任务的智能调度，包括任务分析、资源评估和调度决策
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import json
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class TaskComplexity(Enum):
    """任务复杂度等级"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    CRITICAL = "critical"


class TaskType(Enum):
    """任务类型"""
    COMPUTATION = "computation"
    IO_INTENSIVE = "io_intensive"
    MEMORY_INTENSIVE = "memory_intensive"
    NETWORK_INTENSIVE = "network_intensive"
    MULTIMODAL = "multimodal"
    CONVERSATION = "conversation"
    ANALYSIS = "analysis"


@dataclass
class Task:
    """任务定义"""
    id: str
    type: TaskType
    complexity: TaskComplexity
    priority: int
    payload: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    estimated_duration: float = 0.0
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    deadline: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentCapability:
    """Agent能力定义"""
    agent_id: str
    supported_task_types: List[TaskType]
    max_concurrent_tasks: int
    current_load: int = 0
    performance_score: float = 1.0
    resource_capacity: Dict[str, float] = field(default_factory=dict)
    specialization_score: Dict[TaskType, float] = field(default_factory=dict)


@dataclass
class SchedulingDecision:
    """调度决策"""
    task_id: str
    assigned_agent_id: str
    estimated_start_time: float
    estimated_completion_time: float
    confidence_score: float
    reasoning: str


class AgentScheduler:
    """智能Agent调度器"""
    
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self.agents: Dict[str, AgentCapability] = {}
        self.pending_tasks: List[Task] = []
        self.running_tasks: Dict[str, SchedulingDecision] = {}
        self.completed_tasks: List[str] = []
        self.scheduling_history: List[SchedulingDecision] = []
        
        # 调度策略配置
        self.scheduling_strategies = {
            'load_balancing': self._load_balancing_strategy,
            'capability_matching': self._capability_matching_strategy,
            'performance_optimization': self._performance_optimization_strategy,
            'deadline_aware': self._deadline_aware_strategy
        }
        
        self.current_strategy = 'capability_matching'
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = asyncio.Lock()
        
        # 性能统计
        self.scheduling_stats = {
            'total_scheduled': 0,
            'successful_assignments': 0,
            'failed_assignments': 0,
            'average_scheduling_time': 0.0,
            'strategy_performance': {}
        }
    
    async def register_agent(self, agent: AgentCapability) -> bool:
        """注册Agent"""
        try:
            self.agents[agent.agent_id] = agent
            logger.info(f"Agent {agent.agent_id} 注册成功")
            return True
        except Exception as e:
            logger.error(f"Agent注册失败: {e}")
            return False
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """注销Agent"""
        try:
            if agent_id in self.agents:
                del self.agents[agent_id]
                logger.info(f"Agent {agent_id} 注销成功")
                return True
            return False
        except Exception as e:
            logger.error(f"Agent注销失败: {e}")
            return False
    
    async def submit_task(self, task: Task) -> str:
        """提交任务"""
        async with self._lock:
            self.pending_tasks.append(task)
            task_id = task.id
            logger.info(f"任务 {task_id} 已提交，类型: {task.type.value}, 复杂度: {task.complexity.value}")
            return task_id
    
    async def schedule_task(self, task_id: str) -> Optional[SchedulingDecision]:
        """调度单个任务"""
        try:
            # 查找任务
            task = None
            for t in self.pending_tasks:
                if t.id == task_id:
                    task = t
                    break
            
            if not task:
                logger.warning(f"任务 {task_id} 不存在")
                return None
            
            # 选择调度策略
            strategy = self.scheduling_strategies.get(self.current_strategy, 
                                                     self._capability_matching_strategy)
            
            # 执行调度
            start_time = time.time()
            decision = await strategy(task)
            scheduling_time = time.time() - start_time
            
            if decision:
                # 更新统计
                self.scheduling_stats['total_scheduled'] += 1
                self.scheduling_stats['successful_assignments'] += 1
                self.scheduling_stats['average_scheduling_time'] = (
                    (self.scheduling_stats['average_scheduling_time'] * 
                     (self.scheduling_stats['successful_assignments'] - 1) + scheduling_time) /
                    self.scheduling_stats['successful_assignments']
                )
                
                # 记录调度历史
                self.scheduling_history.append(decision)
                
                # 更新Agent负载
                if decision.assigned_agent_id in self.agents:
                    self.agents[decision.assigned_agent_id].current_load += 1
                
                # 移动到运行中
                self.pending_tasks.remove(task)
                self.running_tasks[task_id] = decision
                
                logger.info(f"任务 {task_id} 调度成功，分配给Agent {decision.assigned_agent_id}")
                return decision
            else:
                self.scheduling_stats['failed_assignments'] += 1
                logger.warning(f"任务 {task_id} 调度失败")
                return None
                
        except Exception as e:
            logger.error(f"任务调度异常: {e}")
            self.scheduling_stats['failed_assignments'] += 1
            return None
    
    async def schedule_batch(self, max_concurrent: int = 5) -> List[SchedulingDecision]:
        """批量调度任务"""
        decisions = []
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def schedule_with_semaphore(task):
            async with semaphore:
                return await self.schedule_task(task.id)
        
        # 创建调度任务
        schedule_tasks = [schedule_with_semaphore(task) for task in self.pending_tasks]
        
        # 并发执行调度
        results = await asyncio.gather(*schedule_tasks, return_exceptions=True)
        
        # 收集有效结果
        for result in results:
            if isinstance(result, SchedulingDecision):
                decisions.append(result)
            elif isinstance(result, Exception):
                logger.error(f"批量调度异常: {result}")
        
        return decisions
    
    async def _load_balancing_strategy(self, task: Task) -> Optional[SchedulingDecision]:
        """负载均衡策略"""
        if not self.agents:
            return None
        
        # 选择负载最轻的Agent
        best_agent = min(self.agents.values(), 
                        key=lambda a: a.current_load / a.max_concurrent_tasks)
        
        if best_agent.current_load >= best_agent.max_concurrent_tasks:
            return None
        
        estimated_start = time.time()
        estimated_completion = estimated_start + task.estimated_duration
        
        return SchedulingDecision(
            task_id=task.id,
            assigned_agent_id=best_agent.agent_id,
            estimated_start_time=estimated_start,
            estimated_completion_time=estimated_completion,
            confidence_score=0.8,
            reasoning="基于负载均衡策略"
        )
    
    async def _capability_matching_strategy(self, task: Task) -> Optional[SchedulingDecision]:
        """能力匹配策略"""
        if not self.agents:
            return None
        
        # 筛选支持该任务类型的Agent
        capable_agents = [
            agent for agent in self.agents.values()
            if task.type in agent.supported_task_types
        ]
        
        if not capable_agents:
            # 如果没有专门支持的Agent，选择负载最轻的
            return await self._load_balancing_strategy(task)
        
        # 计算匹配分数
        scored_agents = []
        for agent in capable_agents:
            if agent.current_load >= agent.max_concurrent_tasks:
                continue
            
            # 基础分数 = 专业化分数 * 性能分数 * 负载因子
            specialization = agent.specialization_score.get(task.type, 0.5)
            performance = agent.performance_score
            load_factor = 1.0 - (agent.current_load / agent.max_concurrent_tasks)
            
            total_score = specialization * performance * load_factor
            scored_agents.append((agent, total_score))
        
        if not scored_agents:
            return None
        
        # 选择分数最高的Agent
        best_agent, score = max(scored_agents, key=lambda x: x[1])
        
        estimated_start = time.time()
        estimated_completion = estimated_start + task.estimated_duration
        
        return SchedulingDecision(
            task_id=task.id,
            assigned_agent_id=best_agent.agent_id,
            estimated_start_time=estimated_start,
            estimated_completion_time=estimated_completion,
            confidence_score=min(score, 1.0),
            reasoning=f"能力匹配分数: {score:.2f}"
        )
    
    async def _performance_optimization_strategy(self, task: Task) -> Optional[SchedulingDecision]:
        """性能优化策略"""
        if not self.agents:
            return None
        
        # 筛选支持该任务类型的Agent
        capable_agents = [
            agent for agent in self.agents.values()
            if task.type in agent.supported_task_types and 
               agent.current_load < agent.max_concurrent_tasks
        ]
        
        if not capable_agents:
            return None
        
        # 基于历史性能选择最佳Agent
        best_agent = max(capable_agents, key=lambda a: a.performance_score)
        
        estimated_start = time.time()
        estimated_completion = estimated_start + task.estimated_duration
        
        return SchedulingDecision(
            task_id=task.id,
            assigned_agent_id=best_agent.agent_id,
            estimated_start_time=estimated_start,
            estimated_completion_time=estimated_completion,
            confidence_score=best_agent.performance_score,
            reasoning="基于历史性能优化"
        )
    
    async def _deadline_aware_strategy(self, task: Task) -> Optional[SchedulingDecision]:
        """截止时间感知策略"""
        if not self.agents:
            return None
        
        if not task.deadline:
            # 如果没有截止时间，使用能力匹配策略
            return await self._capability_matching_strategy(task)
        
        time_remaining = task.deadline - time.time()
        
        if time_remaining <= 0:
            logger.warning(f"任务 {task.id} 已超时")
            return None
        
        # 筛选能在截止时间前完成的Agent
        capable_agents = []
        for agent in self.agents.values():
            if (task.type in agent.supported_task_types and 
                agent.current_load < agent.max_concurrent_tasks):
                
                # 估算完成时间
                estimated_duration = task.estimated_duration / agent.performance_score
                if time.time() + estimated_duration <= task.deadline:
                    capable_agents.append(agent)
        
        if not capable_agents:
            return None
        
        # 选择最快能完成的Agent
        best_agent = min(capable_agents, key=lambda a: a.performance_score)
        
        estimated_start = time.time()
        estimated_completion = estimated_start + task.estimated_duration / best_agent.performance_score
        
        return SchedulingDecision(
            task_id=task.id,
            assigned_agent_id=best_agent.agent_id,
            estimated_start_time=estimated_start,
            estimated_completion_time=estimated_completion,
            confidence_score=0.9,
            reasoning="基于截止时间优化"
        )
    
    async def update_agent_performance(self, agent_id: str, success: bool, 
                                     execution_time: float):
        """更新Agent性能指标"""
        if agent_id not in self.agents:
            return
        
        agent = self.agents[agent_id]
        
        # 更新性能分数 (指数移动平均)
        alpha = 0.1  # 学习率
        if success:
            performance_delta = min(execution_time / 1000.0, 1.0)  # 归一化
        else:
            performance_delta = -0.5
        
        agent.performance_score = max(0.1, min(2.0, 
            agent.performance_score + alpha * performance_delta))
    
    async def complete_task(self, task_id: str, success: bool, 
                          execution_time: float):
        """任务完成处理"""
        async with self._lock:
            if task_id in self.running_tasks:
                decision = self.running_tasks[task_id]
                
                # 更新Agent负载
                if decision.assigned_agent_id in self.agents:
                    agent = self.agents[decision.assigned_agent_id]
                    agent.current_load = max(0, agent.current_load - 1)
                    
                    # 更新性能
                    await self.update_agent_performance(
                        decision.assigned_agent_id, success, execution_time)
                
                # 移动到完成列表
                self.completed_tasks.append(task_id)
                del self.running_tasks[task_id]
                
                logger.info(f"任务 {task_id} 已完成，成功: {success}")
    
    def get_scheduling_stats(self) -> Dict[str, Any]:
        """获取调度统计信息"""
        return {
            'total_agents': len(self.agents),
            'pending_tasks': len(self.pending_tasks),
            'running_tasks': len(self.running_tasks),
            'completed_tasks': len(self.completed_tasks),
            'scheduling_stats': self.scheduling_stats.copy(),
            'agent_loads': {
                agent_id: {
                    'current_load': agent.current_load,
                    'max_load': agent.max_concurrent_tasks,
                    'performance_score': agent.performance_score
                }
                for agent_id, agent in self.agents.items()
            }
        }
    
    async def set_scheduling_strategy(self, strategy: str):
        """设置调度策略"""
        if strategy in self.scheduling_strategies:
            self.current_strategy = strategy
            logger.info(f"调度策略已切换为: {strategy}")
        else:
            logger.warning(f"未知调度策略: {strategy}")
    
    async def cleanup(self):
        """清理资源"""
        self.executor.shutdown(wait=True)
        logger.info("Agent调度器已清理")