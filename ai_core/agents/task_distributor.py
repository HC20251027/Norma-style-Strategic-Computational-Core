#!/usr/bin/env python3
"""
任务分配器
负责将任务智能分配给合适的智能体

作者: 皇
创建时间: 2025-10-31
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from enum import Enum
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import heapq

from ..utils.logger import MultiAgentLogger
from ..communication.message_bus import MessageBus
from ..core.types import AgentCapability, AgentStatus


class TaskType(Enum):
    """任务类型枚举"""
    COMPUTATION = "computation"
    ANALYSIS = "analysis"
    GENERATION = "generation"
    VALIDATION = "validation"
    COORDINATION = "coordination"
    MONITORING = "monitoring"
    COMMUNICATION = "communication"
    LEARNING = "learning"


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class AssignmentStrategy(Enum):
    """分配策略枚举"""
    ROUND_ROBIN = "round_robin"
    LOAD_BALANCED = "load_balanced"
    CAPABILITY_BASED = "capability_based"
    PERFORMANCE_BASED = "performance_based"
    HYBRID = "hybrid"


@dataclass
class Task:
    """任务定义"""
    task_id: str
    task_type: TaskType
    description: str
    priority: int  # 1-10，10为最高优先级
    created_at: datetime
    deadline: Optional[datetime] = None
    required_capabilities: List[str] = None
    estimated_duration: float = 0.0
    resource_requirements: Dict[str, Any] = None
    input_data: Dict[str, Any] = None
    metadata: Dict[str, Any] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def __post_init__(self):
        if self.required_capabilities is None:
            self.required_capabilities = []
        if self.resource_requirements is None:
            self.resource_requirements = {}
        if self.input_data is None:
            self.input_data = {}
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def is_expired(self) -> bool:
        """检查任务是否过期"""
        if self.deadline:
            return datetime.now() > self.deadline
        return False
    
    @property
    def urgency_score(self) -> float:
        """计算紧急程度评分"""
        if self.is_expired:
            return float('inf')
        
        # 基于优先级和剩余时间计算紧急度
        time_factor = 1.0
        if self.deadline:
            time_remaining = (self.deadline - datetime.now()).total_seconds()
            time_factor = max(0.1, min(1.0, time_remaining / 3600))  # 1小时内权重较高
        
        return self.priority * time_factor
    
    @property
    def can_retry(self) -> bool:
        """检查是否可以重试"""
        return self.retry_count < self.max_retries


@dataclass
class AgentAssignment:
    """智能体分配信息"""
    assignment_id: str
    task_id: str
    agent_id: str
    assigned_at: datetime
    estimated_completion: datetime
    assignment_strategy: AssignmentStrategy
    confidence_score: float  # 分配置信度 0-1
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TaskDistributor:
    """任务分配器"""
    
    def __init__(self, message_bus: MessageBus, timeout: int = 1800, max_retries: int = 3):
        """初始化任务分配器
        
        Args:
            message_bus: 消息总线
            timeout: 任务超时时间（秒）
            max_retries: 最大重试次数
        """
        self.distributor_id = str(uuid.uuid4())
        self.message_bus = message_bus
        self.timeout = timeout
        self.max_retries = max_retries
        
        # 初始化日志
        self.logger = MultiAgentLogger("task_distributor")
        
        # 任务管理
        self.pending_tasks: List[Task] = []
        self.active_assignments: Dict[str, AgentAssignment] = {}
        self.completed_tasks: List[Task] = []
        self.failed_tasks: List[Task] = []
        
        # 智能体信息
        self.agent_capabilities: Dict[str, List[AgentCapability]] = {}
        self.agent_status: Dict[str, AgentStatus] = {}
        self.agent_load: Dict[str, int] = defaultdict(int)
        self.agent_performance: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # 分配策略
        self.assignment_strategy = AssignmentStrategy.HYBRID
        self.strategy_weights = {
            AssignmentStrategy.ROUND_ROBIN: 0.1,
            AssignmentStrategy.LOAD_BALANCED: 0.3,
            AssignmentStrategy.CAPABILITY_BASED: 0.4,
            AssignmentStrategy.PERFORMANCE_BASED: 0.2
        }
        
        # 统计信息
        self.metrics = {
            "tasks_submitted": 0,
            "tasks_assigned": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "average_assignment_time": 0.0,
            "assignment_accuracy": 0.0,
            "agent_utilization": defaultdict(float)
        }
        
        # 轮询计数器
        self.round_robin_counter = 0
        
        # 事件回调
        self.event_callbacks: Dict[str, List[Callable]] = {}
        
        self.logger.info(f"任务分配器 {self.distributor_id} 已初始化")
    
    async def initialize(self) -> bool:
        """初始化任务分配器"""
        try:
            self.logger.info("初始化任务分配器...")
            
            # 订阅消息总线事件
            await self.message_bus.subscribe("agent.status_update", self._handle_agent_status_update)
            await self.message_bus.subscribe("agent.performance_update", self._handle_agent_performance_update)
            
            # 启动后台任务
            asyncio.create_task(self._task_scheduler())
            asyncio.create_task(self._assignment_monitor())
            asyncio.create_task(self._performance_tracker())
            
            self.logger.info("任务分配器初始化完成")
            return True
            
        except Exception as e:
            self.logger.error(f"任务分配器初始化失败: {e}")
            return False
    
    async def submit_task(self, task: Task) -> str:
        """提交任务
        
        Args:
            task: 任务对象
            
        Returns:
            任务ID
        """
        try:
            # 验证任务
            if task.is_expired:
                raise ValueError(f"任务 {task.task_id} 已过期")
            
            # 添加到待处理任务队列
            self.pending_tasks.append(task)
            self.metrics["tasks_submitted"] += 1
            
            # 按紧急程度排序
            self.pending_tasks.sort(key=lambda t: t.urgency_score, reverse=True)
            
            # 发送任务提交事件
            await self._emit_event("task.submitted", {
                "task_id": task.task_id,
                "task_type": task.task_type.value,
                "priority": task.priority,
                "urgency_score": task.urgency_score
            })
            
            self.logger.info(f"任务 {task.task_id} 已提交，类型: {task.task_type.value}, 优先级: {task.priority}")
            return task.task_id
            
        except Exception as e:
            self.logger.error(f"提交任务失败: {e}")
            raise
    
    async def assign_task(self, task_id: str, strategy: Optional[AssignmentStrategy] = None) -> Optional[AgentAssignment]:
        """分配任务给智能体
        
        Args:
            task_id: 任务ID
            strategy: 分配策略，如果为None则使用默认策略
            
        Returns:
            分配信息
        """
        try:
            # 查找任务
            task = self._find_task_by_id(task_id)
            if not task:
                raise ValueError(f"任务 {task_id} 不存在")
            
            # 选择分配策略
            if strategy is None:
                strategy = self.assignment_strategy
            
            # 找到合适的智能体
            suitable_agents = await self._find_suitable_agents(task)
            if not suitable_agents:
                self.logger.warning(f"没有找到适合执行任务 {task_id} 的智能体")
                return None
            
            # 使用指定策略选择最佳智能体
            selected_agent = await self._select_agent_with_strategy(task, suitable_agents, strategy)
            if not selected_agent:
                self.logger.warning(f"无法为任务 {task_id} 选择合适的智能体")
                return None
            
            # 创建分配
            assignment = await self._create_assignment(task, selected_agent, strategy)
            
            # 更新智能体负载
            self.agent_load[selected_agent] += 1
            
            # 移动任务状态
            self.pending_tasks.remove(task)
            self.active_assignments[assignment.assignment_id] = assignment
            
            self.metrics["tasks_assigned"] += 1
            
            # 发送任务分配事件
            await self._emit_event("task.assigned", {
                "task_id": task_id,
                "agent_id": selected_agent,
                "assignment_id": assignment.assignment_id,
                "strategy": strategy.value,
                "estimated_completion": assignment.estimated_completion.isoformat()
            })
            
            self.logger.info(f"任务 {task_id} 已分配给智能体 {selected_agent}")
            return assignment
            
        except Exception as e:
            self.logger.error(f"分配任务失败 {task_id}: {e}")
            return None
    
    async def _find_suitable_agents(self, task: Task) -> List[str]:
        """找到适合执行任务的智能体"""
        suitable_agents = []
        
        # 检查每个智能体是否满足任务要求
        for agent_id, capabilities in self.agent_capabilities.items():
            # 检查智能体状态
            if self.agent_status.get(agent_id) != AgentStatus.IDLE:
                continue
            
            # 检查能力匹配
            has_required_capabilities = all(
                any(cap.name == required_cap for cap in capabilities)
                for required_cap in task.required_capabilities
            )
            
            if not has_required_capabilities:
                continue
            
            # 检查资源需求
            if not self._check_resource_requirements(agent_id, task.resource_requirements):
                continue
            
            suitable_agents.append(agent_id)
        
        return suitable_agents
    
    def _check_resource_requirements(self, agent_id: str, requirements: Dict[str, Any]) -> bool:
        """检查智能体是否满足资源需求"""
        # 简化的资源检查逻辑
        # 实际实现中需要根据具体资源类型进行检查
        
        agent_performance = self.agent_performance.get(agent_id, {})
        
        # 检查CPU需求
        if "cpu_cores" in requirements:
            available_cores = agent_performance.get("cpu_cores", 4)
            if available_cores < requirements["cpu_cores"]:
                return False
        
        # 检查内存需求
        if "memory_gb" in requirements:
            available_memory = agent_performance.get("memory_gb", 8)
            if available_memory < requirements["memory_gb"]:
                return False
        
        return True
    
    async def _select_agent_with_strategy(
        self, 
        task: Task, 
        suitable_agents: List[str], 
        strategy: AssignmentStrategy
    ) -> Optional[str]:
        """使用指定策略选择智能体"""
        if not suitable_agents:
            return None
        
        if len(suitable_agents) == 1:
            return suitable_agents[0]
        
        if strategy == AssignmentStrategy.ROUND_ROBIN:
            return self._select_round_robin(suitable_agents)
        
        elif strategy == AssignmentStrategy.LOAD_BALANCED:
            return self._select_load_balanced(suitable_agents)
        
        elif strategy == AssignmentStrategy.CAPABILITY_BASED:
            return self._select_capability_based(task, suitable_agents)
        
        elif strategy == AssignmentStrategy.PERFORMANCE_BASED:
            return self._select_performance_based(task, suitable_agents)
        
        elif strategy == AssignmentStrategy.HYBRID:
            return await self._select_hybrid(task, suitable_agents)
        
        else:
            return suitable_agents[0]
    
    def _select_round_robin(self, agents: List[str]) -> str:
        """轮询选择"""
        agent = agents[self.round_robin_counter % len(agents)]
        self.round_robin_counter += 1
        return agent
    
    def _select_load_balanced(self, agents: List[str]) -> str:
        """负载均衡选择"""
        return min(agents, key=lambda a: self.agent_load.get(a, 0))
    
    def _select_capability_based(self, task: Task, agents: List[str]) -> str:
        """基于能力选择"""
        agent_scores = {}
        
        for agent_id in agents:
            capabilities = self.agent_capabilities.get(agent_id, [])
            score = 0.0
            
            # 计算能力匹配分数
            for required_cap in task.required_capabilities:
                for cap in capabilities:
                    if cap.name == required_cap:
                        # 考虑复杂度和成功率
                        score += cap.success_rate * (1.0 / max(cap.complexity_level, 1))
                        break
            
            agent_scores[agent_id] = score
        
        return max(agent_scores.items(), key=lambda x: x[1])[0]
    
    def _select_performance_based(self, task: Task, agents: List[str]) -> str:
        """基于性能选择"""
        agent_scores = {}
        
        for agent_id in agents:
            performance = self.agent_performance.get(agent_id, {})
            score = performance.get("success_rate", 0.5)
            
            # 考虑任务类型匹配度
            task_type_performance = performance.get(f"{task.task_type.value}_success_rate", 0.5)
            score = (score + task_type_performance) / 2
            
            agent_scores[agent_id] = score
        
        return max(agent_scores.items(), key=lambda x: x[1])[0]
    
    async def _select_hybrid(self, task: Task, agents: List[str]) -> str:
        """混合策略选择"""
        agent_scores = {}
        
        for agent_id in agents:
            score = 0.0
            
            # 负载均衡分数
            load_score = 1.0 / (self.agent_load.get(agent_id, 0) + 1)
            score += load_score * self.strategy_weights[AssignmentStrategy.LOAD_BALANCED]
            
            # 能力匹配分数
            capability_score = self._calculate_capability_score(agent_id, task)
            score += capability_score * self.strategy_weights[AssignmentStrategy.CAPABILITY_BASED]
            
            # 性能分数
            performance_score = self.agent_performance.get(agent_id, {}).get("success_rate", 0.5)
            score += performance_score * self.strategy_weights[AssignmentStrategy.PERFORMANCE_BASED]
            
            # 轮询分数（避免总是选择同一个智能体）
            round_robin_score = 1.0 if agent_id == agents[self.round_robin_counter % len(agents)] else 0.5
            score += round_robin_score * self.strategy_weights[AssignmentStrategy.ROUND_ROBIN]
            
            agent_scores[agent_id] = score
        
        selected_agent = max(agent_scores.items(), key=lambda x: x[1])[0]
        self.round_robin_counter += 1
        
        return selected_agent
    
    def _calculate_capability_score(self, agent_id: str, task: Task) -> float:
        """计算智能体能力匹配分数"""
        capabilities = self.agent_capabilities.get(agent_id, [])
        score = 0.0
        
        for required_cap in task.required_capabilities:
            for cap in capabilities:
                if cap.name == required_cap:
                    # 考虑复杂度和质量
                    complexity_factor = 1.0 / max(cap.complexity_level, 1)
                    quality_factor = cap.quality_score
                    score += complexity_factor * quality_factor
                    break
        
        return score / max(len(task.required_capabilities), 1)
    
    async def _create_assignment(
        self, 
        task: Task, 
        agent_id: str, 
        strategy: AssignmentStrategy
    ) -> AgentAssignment:
        """创建分配"""
        assignment_id = str(uuid.uuid4())
        assigned_at = datetime.now()
        
        # 估算完成时间
        estimated_duration = task.estimated_duration
        estimated_completion = assigned_at + timedelta(seconds=estimated_duration)
        
        # 计算置信度
        confidence_score = await self._calculate_assignment_confidence(task, agent_id)
        
        assignment = AgentAssignment(
            assignment_id=assignment_id,
            task_id=task.task_id,
            agent_id=agent_id,
            assigned_at=assigned_at,
            estimated_completion=estimated_completion,
            assignment_strategy=strategy,
            confidence_score=confidence_score,
            metadata={
                "task_priority": task.priority,
                "required_capabilities": task.required_capabilities,
                "resource_requirements": task.resource_requirements
            }
        )
        
        return assignment
    
    async def _calculate_assignment_confidence(self, task: Task, agent_id: str) -> float:
        """计算分配置信度"""
        confidence = 0.0
        
        # 能力匹配置信度
        capability_confidence = self._calculate_capability_score(agent_id, task)
        confidence += capability_confidence * 0.4
        
        # 性能置信度
        performance = self.agent_performance.get(agent_id, {})
        performance_confidence = performance.get("success_rate", 0.5)
        confidence += performance_confidence * 0.3
        
        # 负载置信度
        load = self.agent_load.get(agent_id, 0)
        load_confidence = max(0.0, 1.0 - load / 10.0)
        confidence += load_confidence * 0.2
        
        # 历史成功率
        task_type_success = performance.get(f"{task.task_type.value}_success_rate", 0.5)
        confidence += task_type_success * 0.1
        
        return min(1.0, confidence)
    
    def _find_task_by_id(self, task_id: str) -> Optional[Task]:
        """根据ID查找任务"""
        for task in self.pending_tasks:
            if task.task_id == task_id:
                return task
        
        for assignment in self.active_assignments.values():
            if assignment.task_id == task_id:
                # 从已完成任务中查找
                for task in self.completed_tasks:
                    if task.task_id == task_id:
                        return task
                for task in self.failed_tasks:
                    if task.task_id == task_id:
                        return task
        
        return None
    
    async def complete_task(self, task_id: str, result: Dict[str, Any]) -> bool:
        """完成任务"""
        try:
            # 查找分配
            assignment = None
            for a in self.active_assignments.values():
                if a.task_id == task_id:
                    assignment = a
                    break
            
            if not assignment:
                self.logger.warning(f"找不到任务 {task_id} 的分配信息")
                return False
            
            # 更新智能体负载
            self.agent_load[assignment.agent_id] = max(0, self.agent_load[assignment.agent_id] - 1)
            
            # 移动到已完成任务
            task = self._find_task_by_id(task_id)
            if task:
                self.completed_tasks.append(task)
            
            # 移除分配
            del self.active_assignments[assignment.assignment_id]
            
            self.metrics["tasks_completed"] += 1
            
            # 发送任务完成事件
            await self._emit_event("task.completed", {
                "task_id": task_id,
                "agent_id": assignment.agent_id,
                "result": result,
                "completion_time": datetime.now().isoformat()
            })
            
            self.logger.info(f"任务 {task_id} 已完成")
            return True
            
        except Exception as e:
            self.logger.error(f"完成任务失败 {task_id}: {e}")
            return False
    
    async def fail_task(self, task_id: str, error: str) -> bool:
        """标记任务失败"""
        try:
            # 查找分配
            assignment = None
            for a in self.active_assignments.values():
                if a.task_id == task_id:
                    assignment = a
                    break
            
            if not assignment:
                self.logger.warning(f"找不到任务 {task_id} 的分配信息")
                return False
            
            # 更新智能体负载
            self.agent_load[assignment.agent_id] = max(0, self.agent_load[assignment.agent_id] - 1)
            
            # 更新任务重试次数
            task = self._find_task_by_id(task_id)
            if task:
                task.retry_count += 1
                
                # 检查是否需要重试
                if task.can_retry:
                    self.logger.info(f"任务 {task_id} 失败，准备重试 ({task.retry_count}/{task.max_retries})")
                    self.pending_tasks.append(task)
                    return True
            
            # 移动到失败任务
            if task:
                self.failed_tasks.append(task)
            
            # 移除分配
            del self.active_assignments[assignment.assignment_id]
            
            self.metrics["tasks_failed"] += 1
            
            # 发送任务失败事件
            await self._emit_event("task.failed", {
                "task_id": task_id,
                "agent_id": assignment.agent_id,
                "error": error,
                "retry_count": task.retry_count if task else 0
            })
            
            self.logger.warning(f"任务 {task_id} 失败: {error}")
            return True
            
        except Exception as e:
            self.logger.error(f"标记任务失败时出错 {task_id}: {e}")
            return False
    
    async def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        try:
            # 从待处理任务中移除
            task = None
            for t in self.pending_tasks:
                if t.task_id == task_id:
                    task = t
                    break
            
            if task:
                self.pending_tasks.remove(task)
                
                # 发送任务取消事件
                await self._emit_event("task.cancelled", {
                    "task_id": task_id,
                    "reason": "user_cancelled"
                })
                
                self.logger.info(f"任务 {task_id} 已取消")
                return True
            
            # 检查是否正在执行
            for assignment in self.active_assignments.values():
                if assignment.task_id == task_id:
                    # 标记为取消
                    del self.active_assignments[assignment.assignment_id]
                    
                    await self._emit_event("task.cancelled", {
                        "task_id": task_id,
                        "reason": "execution_cancelled"
                    })
                    
                    self.logger.info(f"正在执行的任务 {task_id} 已取消")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"取消任务失败 {task_id}: {e}")
            return False
    
    def register_agent(
        self, 
        agent_id: str, 
        capabilities: List[AgentCapability]
    ):
        """注册智能体"""
        self.agent_capabilities[agent_id] = capabilities
        self.agent_status[agent_id] = AgentStatus.IDLE
        self.agent_performance[agent_id] = {
            "success_rate": 0.5,
            "average_response_time": 1.0,
            "cpu_cores": 4,
            "memory_gb": 8
        }
    
    def unregister_agent(self, agent_id: str):
        """注销智能体"""
        if agent_id in self.agent_capabilities:
            del self.agent_capabilities[agent_id]
        
        if agent_id in self.agent_status:
            del self.agent_status[agent_id]
        
        if agent_id in self.agent_performance:
            del self.agent_performance[agent_id]
        
        if agent_id in self.agent_load:
            del self.agent_load[agent_id]
    
    async def _task_scheduler(self):
        """任务调度器后台任务"""
        while True:
            try:
                await asyncio.sleep(5)  # 每5秒调度一次
                
                # 处理待处理任务
                while self.pending_tasks:
                    task = self.pending_tasks[0]  # 取最紧急的任务
                    
                    # 尝试分配任务
                    assignment = await self.assign_task(task.task_id)
                    if assignment:
                        self.pending_tasks.remove(task)
                    else:
                        # 暂时无法分配，等待下次调度
                        break
                
            except Exception as e:
                self.logger.error(f"任务调度器出错: {e}")
    
    async def _assignment_monitor(self):
        """分配监控后台任务"""
        while True:
            try:
                await asyncio.sleep(30)  # 每30秒检查一次
                
                current_time = datetime.now()
                timeout_assignments = []
                
                for assignment_id, assignment in self.active_assignments.items():
                    if current_time > assignment.estimated_completion + timedelta(seconds=self.timeout):
                        timeout_assignments.append(assignment_id)
                
                # 处理超时的分配
                for assignment_id in timeout_assignments:
                    assignment = self.active_assignments[assignment_id]
                    self.logger.warning(f"任务 {assignment.task_id} 超时")
                    
                    # 标记任务失败
                    await self.fail_task(assignment.task_id, "Task timeout")
                
            except Exception as e:
                self.logger.error(f"分配监控任务出错: {e}")
    
    async def _performance_tracker(self):
        """性能跟踪后台任务"""
        while True:
            try:
                await asyncio.sleep(60)  # 每分钟更新一次
                
                # 更新智能体利用率
                total_tasks = self.metrics["tasks_completed"] + self.metrics["tasks_failed"]
                if total_tasks > 0:
                    for agent_id in self.agent_capabilities:
                        completed_tasks = sum(1 for task in self.completed_tasks 
                                            if self._get_assignment_agent(task.task_id) == agent_id)
                        utilization = completed_tasks / total_tasks
                        self.metrics["agent_utilization"][agent_id] = utilization
                
            except Exception as e:
                self.logger.error(f"性能跟踪任务出错: {e}")
    
    def _get_assignment_agent(self, task_id: str) -> Optional[str]:
        """获取任务的分配智能体"""
        for assignment in self.active_assignments.values():
            if assignment.task_id == task_id:
                return assignment.agent_id
        return None
    
    async def _handle_agent_status_update(self, data: Dict[str, Any]):
        """处理智能体状态更新"""
        agent_id = data.get("agent_id")
        status = data.get("status")
        
        if agent_id and status:
            try:
                self.agent_status[agent_id] = AgentStatus(status)
            except ValueError:
                self.logger.warning(f"未知的智能体状态: {status}")
    
    async def _handle_agent_performance_update(self, data: Dict[str, Any]):
        """处理智能体性能更新"""
        agent_id = data.get("agent_id")
        performance_data = data.get("performance", {})
        
        if agent_id and performance_data:
            self.agent_performance[agent_id].update(performance_data)
    
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
    
    def get_distribution_status(self) -> Dict[str, Any]:
        """获取分配状态"""
        return {
            "distributor_id": self.distributor_id,
            "pending_tasks": len(self.pending_tasks),
            "active_assignments": len(self.active_assignments),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "registered_agents": len(self.agent_capabilities),
            "assignment_strategy": self.assignment_strategy.value,
            "metrics": dict(self.metrics),
            "agent_load": dict(self.agent_load),
            "agent_utilization": dict(self.metrics["agent_utilization"])
        }
    
    async def shutdown(self):
        """关闭任务分配器"""
        try:
            self.logger.info("关闭任务分配器...")
            
            # 等待所有活跃任务完成
            while self.active_assignments:
                await asyncio.sleep(1)
            
            self.logger.info("任务分配器已关闭")
            
        except Exception as e:
            self.logger.error(f"关闭任务分配器时出错: {e}")