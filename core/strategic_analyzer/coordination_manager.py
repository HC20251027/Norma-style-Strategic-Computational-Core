#!/usr/bin/env python3
"""
协调管理器
负责多智能体系统的高级协调和调度

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
import heapq

from ..utils.logger import MultiAgentLogger
from .agent_collaboration_manager import AgentCollaborationManager, CollaborationType, AgentCapability


class CoordinationStrategy(Enum):
    """协调策略枚举"""
    CENTRALIZED = "centralized"      # 集中式协调
    DISTRIBUTED = "distributed"      # 分布式协调
    HYBRID = "hybrid"                # 混合式协调
    FEDERATED = "federated"          # 联邦式协调


class TaskPriority(Enum):
    """任务优先级"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


@dataclass
class CoordinationTask:
    """协调任务"""
    task_id: str
    task_type: str
    description: str
    priority: TaskPriority
    created_at: datetime
    deadline: Optional[datetime] = None
    required_capabilities: List[str] = None
    estimated_duration: float = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.required_capabilities is None:
            self.required_capabilities = []
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
        if not self.deadline:
            return 0.0
        
        time_remaining = (self.deadline - datetime.now()).total_seconds()
        if time_remaining <= 0:
            return float('inf')  # 已过期，紧急度最高
        
        # 基于剩余时间和优先级计算紧急度
        priority_weights = {
            TaskPriority.CRITICAL: 4.0,
            TaskPriority.HIGH: 3.0,
            TaskPriority.MEDIUM: 2.0,
            TaskPriority.LOW: 1.0
        }
        
        return priority_weights[self.priority] * (1.0 / max(time_remaining, 1))


@dataclass
class CoordinationPlan:
    """协调计划"""
    plan_id: str
    created_at: datetime
    tasks: List[CoordinationTask]
    execution_order: List[str]  # 任务执行顺序
    agent_assignments: Dict[str, List[str]]  # 智能体ID -> 任务ID列表
    dependencies: Dict[str, List[str]]  # 任务ID -> 依赖的任务ID列表
    estimated_completion: datetime
    status: str = "pending"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class CoordinationManager:
    """协调管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化协调管理器"""
        self.coordinator_id = str(uuid.uuid4())
        self.start_time = datetime.now()
        
        # 初始化日志
        self.logger = MultiAgentLogger("coordination_manager")
        
        # 核心组件
        self.collaboration_manager: Optional[AgentCollaborationManager] = None
        
        # 协调策略
        self.coordination_strategy = CoordinationStrategy.HYBRID
        self.coordination_rules: Dict[str, Any] = {}
        
        # 任务管理
        self.pending_tasks: List[CoordinationTask] = []
        self.active_plans: Dict[str, CoordinationPlan] = {}
        self.completed_plans: List[CoordinationPlan] = []
        
        # 智能体能力映射
        self.capability_agent_map: Dict[str, Set[str]] = defaultdict(set)
        self.agent_load: Dict[str, int] = defaultdict(int)
        self.agent_performance: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # 调度队列（优先级队列）
        self.task_queue: List[Tuple[float, int, CoordinationTask]] = []
        self.task_counter = 0
        
        # 统计信息
        self.metrics = {
            "tasks_processed": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "average_completion_time": 0.0,
            "agent_utilization": defaultdict(float),
            "coordination_overhead": 0.0
        }
        
        # 事件回调
        self.event_callbacks: Dict[str, List[Callable]] = {}
        
        # 配置
        self.config = self._load_config(config_path)
        
        self.logger.info(f"协调管理器 {self.coordinator_id} 已初始化")
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """加载配置"""
        default_config = {
            "coordination": {
                "strategy": "hybrid",
                "max_concurrent_plans": 20,
                "task_timeout": 3600,
                "plan_timeout": 7200,
                "max_retry_attempts": 3,
                "enable_predictive_scheduling": True,
                "load_balancing_threshold": 0.8
            },
            "scheduling": {
                "priority_weights": {
                    "critical": 4.0,
                    "high": 3.0,
                    "medium": 2.0,
                    "low": 1.0
                },
                "time_decay_factor": 0.1,
                "performance_weight": 0.3,
                "load_weight": 0.4,
                "capability_weight": 0.3
            },
            "optimization": {
                "enable_auto_scaling": True,
                "efficiency_threshold": 0.75,
                "quality_threshold": 0.85,
                "response_time_target": 30.0
            }
        }
        
        if config_path:
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
    
    async def initialize(self, collaboration_manager: AgentCollaborationManager):
        """初始化协调管理器"""
        try:
            self.logger.info("初始化协调管理器...")
            
            self.collaboration_manager = collaboration_manager
            
            # 设置协调策略
            strategy_name = self.config["coordination"]["strategy"]
            self.coordination_strategy = CoordinationStrategy(strategy_name)
            
            # 初始化协调规则
            await self._initialize_coordination_rules()
            
            # 启动后台任务
            asyncio.create_task(self._task_scheduler())
            asyncio.create_task(self._plan_monitor())
            asyncio.create_task(self._performance_optimizer())
            
            self.logger.info("协调管理器初始化完成")
            return True
            
        except Exception as e:
            self.logger.error(f"协调管理器初始化失败: {e}")
            return False
    
    async def _initialize_coordination_rules(self):
        """初始化协调规则"""
        self.coordination_rules = {
            "capability_based_assignment": True,
            "load_balancing": True,
            "performance_optimization": True,
            "fault_tolerance": True,
            "resource_sharing": True,
            "conflict_resolution": "priority_based"
        }
    
    async def submit_task(self, task: CoordinationTask) -> str:
        """提交协调任务"""
        try:
            # 验证任务
            if task.is_expired:
                raise ValueError(f"任务 {task.task_id} 已过期")
            
            # 添加到待处理任务队列
            heapq.heappush(
                self.task_queue,
                (task.urgency_score, self.task_counter, task)
            )
            self.task_counter += 1
            
            self.pending_tasks.append(task)
            
            # 发送任务提交事件
            await self._emit_event("task.submitted", {
                "task_id": task.task_id,
                "priority": task.priority.value,
                "urgency_score": task.urgency_score
            })
            
            self.logger.info(f"任务 {task.task_id} 已提交，优先级: {task.priority.name}")
            return task.task_id
            
        except Exception as e:
            self.logger.error(f"提交任务失败: {e}")
            raise
    
    async def create_coordination_plan(
        self,
        tasks: List[CoordinationTask],
        constraints: Optional[Dict[str, Any]] = None
    ) -> str:
        """创建协调计划"""
        try:
            plan_id = str(uuid.uuid4())
            
            # 分析任务依赖关系
            dependencies = await self._analyze_task_dependencies(tasks)
            
            # 确定执行顺序
            execution_order = await self._determine_execution_order(tasks, dependencies)
            
            # 分配智能体
            agent_assignments = await self._assign_agents_to_tasks(tasks, execution_order)
            
            # 估算完成时间
            estimated_completion = await self._estimate_completion_time(
                tasks, execution_order, agent_assignments
            )
            
            # 创建协调计划
            plan = CoordinationPlan(
                plan_id=plan_id,
                created_at=datetime.now(),
                tasks=tasks,
                execution_order=execution_order,
                agent_assignments=agent_assignments,
                dependencies=dependencies,
                estimated_completion=estimated_completion,
                metadata=constraints or {}
            )
            
            self.active_plans[plan_id] = plan
            
            # 发送计划创建事件
            await self._emit_event("plan.created", {
                "plan_id": plan_id,
                "task_count": len(tasks),
                "estimated_completion": estimated_completion.isoformat()
            })
            
            self.logger.info(f"协调计划 {plan_id} 已创建，包含 {len(tasks)} 个任务")
            return plan_id
            
        except Exception as e:
            self.logger.error(f"创建协调计划失败: {e}")
            raise
    
    async def execute_plan(self, plan_id: str) -> Dict[str, Any]:
        """执行协调计划"""
        try:
            if plan_id not in self.active_plans:
                raise ValueError(f"协调计划 {plan_id} 不存在")
            
            plan = self.active_plans[plan_id]
            plan.status = "executing"
            
            execution_results = []
            start_time = datetime.now()
            
            # 按顺序执行任务
            for task_id in plan.execution_order:
                task = next(t for t in plan.tasks if t.task_id == task_id)
                assigned_agents = plan.agent_assignments.get(task_id, [])
                
                try:
                    # 执行任务
                    result = await self._execute_task_with_agents(task, assigned_agents)
                    execution_results.append({
                        "task_id": task_id,
                        "status": "completed",
                        "result": result,
                        "execution_time": result.get("duration", 0)
                    })
                    
                    # 更新智能体负载
                    for agent_id in assigned_agents:
                        self.agent_load[agent_id] -= 1
                    
                except Exception as e:
                    execution_results.append({
                        "task_id": task_id,
                        "status": "failed",
                        "error": str(e)
                    })
                    
                    self.logger.error(f"任务 {task_id} 执行失败: {e}")
            
            # 计算执行统计
            end_time = datetime.now()
            total_duration = (end_time - start_time).total_seconds()
            success_rate = sum(1 for r in execution_results if r["status"] == "completed") / len(execution_results)
            
            # 更新计划状态
            plan.status = "completed" if success_rate > 0.5 else "failed"
            
            # 移动到已完成计划列表
            self.completed_plans.append(plan)
            del self.active_plans[plan_id]
            
            # 更新统计信息
            self.metrics["tasks_processed"] += len(tasks)
            self.metrics["tasks_completed"] += sum(1 for r in execution_results if r["status"] == "completed")
            self.metrics["tasks_failed"] += sum(1 for r in execution_results if r["status"] == "failed")
            
            result = {
                "plan_id": plan_id,
                "status": plan.status,
                "execution_results": execution_results,
                "total_duration": total_duration,
                "success_rate": success_rate,
                "completed_at": end_time.isoformat()
            }
            
            # 发送计划执行完成事件
            await self._emit_event("plan.completed", result)
            
            self.logger.info(f"协调计划 {plan_id} 执行完成，成功率: {success_rate:.2%}")
            return result
            
        except Exception as e:
            self.logger.error(f"执行协调计划失败 {plan_id}: {e}")
            
            if plan_id in self.active_plans:
                self.active_plans[plan_id].status = "failed"
            
            raise
    
    async def _analyze_task_dependencies(self, tasks: List[CoordinationTask]) -> Dict[str, List[str]]:
        """分析任务依赖关系"""
        dependencies = defaultdict(list)
        
        # 简单的依赖分析：基于任务类型和描述
        for i, task1 in enumerate(tasks):
            for j, task2 in enumerate(tasks):
                if i != j:
                    # 检查是否存在依赖关系
                    if await self._check_task_dependency(task1, task2):
                        dependencies[task1.task_id].append(task2.task_id)
        
        return dict(dependencies)
    
    async def _check_task_dependency(self, task1: CoordinationTask, task2: CoordinationTask) -> bool:
        """检查两个任务是否存在依赖关系"""
        # 简化的依赖检查逻辑
        # 实际实现中可以使用更复杂的依赖分析算法
        
        # 基于任务类型
        if task1.task_type == "preprocessing" and task2.task_type == "processing":
            return True
        if task1.task_type == "processing" and task2.task_type == "postprocessing":
            return True
        
        # 基于能力需求
        for cap in task1.required_capabilities:
            if cap in [c for c in task2.required_capabilities]:
                # 检查是否存在资源冲突
                if self._has_resource_conflict(task1, task2):
                    return True
        
        return False
    
    def _has_resource_conflict(self, task1: CoordinationTask, task2: CoordinationTask) -> bool:
        """检查任务间是否存在资源冲突"""
        # 简化的资源冲突检查
        for resource in task1.metadata.get("resources", []):
            if resource in task2.metadata.get("resources", []):
                return True
        return False
    
    async def _determine_execution_order(
        self, 
        tasks: List[CoordinationTask], 
        dependencies: Dict[str, List[str]]
    ) -> List[str]:
        """确定任务执行顺序（拓扑排序）"""
        # 构建有向图
        graph = defaultdict(list)
        in_degree = defaultdict(int)
        
        # 初始化
        for task in tasks:
            in_degree[task.task_id] = 0
        
        # 添加边
        for task_id, deps in dependencies.items():
            for dep in deps:
                graph[dep].append(task_id)
                in_degree[task_id] += 1
        
        # 拓扑排序
        queue = [task_id for task_id in in_degree if in_degree[task_id] == 0]
        result = []
        
        while queue:
            # 按优先级排序
            queue.sort(key=lambda x: next(t for t in tasks if t.task_id == x).priority.value)
            current = queue.pop(0)
            result.append(current)
            
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return result
    
    async def _assign_agents_to_tasks(
        self, 
        tasks: List[CoordinationTask], 
        execution_order: List[str]
    ) -> Dict[str, List[str]]:
        """为任务分配智能体"""
        assignments = {}
        
        for task_id in execution_order:
            task = next(t for t in tasks if t.task_id == task_id)
            
            # 找到适合的智能体
            suitable_agents = await self._find_suitable_agents(task)
            
            # 选择最佳智能体
            selected_agents = await self._select_optimal_agents(task, suitable_agents)
            
            assignments[task_id] = selected_agents
            
            # 更新智能体负载
            for agent_id in selected_agents:
                self.agent_load[agent_id] += 1
        
        return assignments
    
    async def _find_suitable_agents(self, task: CoordinationTask) -> List[str]:
        """找到适合执行任务的智能体"""
        suitable_agents = []
        
        for capability in task.required_capabilities:
            agents_with_capability = self.capability_agent_map.get(capability, set())
            suitable_agents.extend(agents_with_capability)
        
        # 去重并过滤掉忙碌的智能体
        suitable_agents = list(set(suitable_agents))
        suitable_agents = [
            agent_id for agent_id in suitable_agents 
            if self.agent_load.get(agent_id, 0) < self.config["coordination"]["load_balancing_threshold"] * 10
        ]
        
        return suitable_agents
    
    async def _select_optimal_agents(
        self, 
        task: CoordinationTask, 
        suitable_agents: List[str]
    ) -> List[str]:
        """选择最优的智能体"""
        if not suitable_agents:
            return []
        
        # 评估每个智能体
        agent_scores = []
        for agent_id in suitable_agents:
            score = await self._evaluate_agent_for_task(agent_id, task)
            agent_scores.append((agent_id, score))
        
        # 按评分排序，选择前几个
        agent_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 需要的智能体数量（基于任务复杂度）
        needed_agents = min(len(suitable_agents), max(1, task.estimated_duration // 60))
        
        return [agent_id for agent_id, _ in agent_scores[:needed_agents]]
    
    async def _evaluate_agent_for_task(self, agent_id: str, task: CoordinationTask) -> float:
        """评估智能体对任务的适合度"""
        score = 0.0
        
        # 性能评分
        performance = self.agent_performance.get(agent_id, {})
        score += performance.get("success_rate", 0.5) * self.config["scheduling"]["performance_weight"]
        
        # 负载评分（负载越低评分越高）
        load = self.agent_load.get(agent_id, 0)
        max_load = self.config["coordination"]["load_balancing_threshold"] * 10
        load_score = max(0, 1 - load / max_load)
        score += load_score * self.config["scheduling"]["load_weight"]
        
        # 能力匹配评分
        agent_capabilities = self.capability_agent_map.get(task.required_capabilities[0], set())
        if agent_id in agent_capabilities:
            score += self.config["scheduling"]["capability_weight"]
        
        return score
    
    async def _estimate_completion_time(
        self,
        tasks: List[CoordinationTask],
        execution_order: List[str],
        agent_assignments: Dict[str, List[str]]
    ) -> datetime:
        """估算完成时间"""
        current_time = datetime.now()
        total_time = 0.0
        
        for task_id in execution_order:
            task = next(t for t in tasks if t.task_id == task_id)
            assigned_agents = agent_assignments.get(task_id, [])
            
            # 考虑并行度
            parallel_factor = min(len(assigned_agents), 4)  # 最多4个并行
            task_time = task.estimated_duration / parallel_factor
            
            total_time += task_time
        
        return current_time + timedelta(seconds=total_time)
    
    async def _execute_task_with_agents(
        self, 
        task: CoordinationTask, 
        agent_ids: List[str]
    ) -> Dict[str, Any]:
        """使用指定智能体执行任务"""
        if not self.collaboration_manager:
            raise RuntimeError("协作管理器未初始化")
        
        # 创建协作会话
        collaboration_type = CollaborationType.PARALLEL if len(agent_ids) > 1 else CollaborationType.SEQUENTIAL
        
        session_id = await self.collaboration_manager.create_collaboration(
            collaboration_type=collaboration_type,
            participant_agents=agent_ids,
            task_description=task.description
        )
        
        # 执行任务
        result = await self.collaboration_manager.execute_collaboration(
            session_id=session_id,
            task_data={
                "task_id": task.task_id,
                "task_type": task.task_type,
                "metadata": task.metadata
            }
        )
        
        return result
    
    async def _task_scheduler(self):
        """任务调度器后台任务"""
        while True:
            try:
                await asyncio.sleep(5)  # 每5秒调度一次
                
                # 处理待处理任务
                while self.task_queue:
                    urgency_score, _, task = heapq.heappop(self.task_queue)
                    
                    # 检查任务是否仍然有效
                    if task.is_expired:
                        self.logger.warning(f"任务 {task.task_id} 已过期，跳过")
                        continue
                    
                    # 创建协调计划
                    plan_id = await self.create_coordination_plan([task])
                    
                    # 异步执行计划
                    asyncio.create_task(self.execute_plan(plan_id))
                
            except Exception as e:
                self.logger.error(f"任务调度器出错: {e}")
    
    async def _plan_monitor(self):
        """计划监控后台任务"""
        while True:
            try:
                await asyncio.sleep(30)  # 每30秒检查一次
                
                current_time = datetime.now()
                expired_plans = []
                
                for plan_id, plan in self.active_plans.items():
                    if current_time > plan.estimated_completion + timedelta(seconds=self.config["coordination"]["plan_timeout"]):
                        expired_plans.append(plan_id)
                
                # 处理过期的计划
                for plan_id in expired_plans:
                    self.logger.warning(f"计划 {plan_id} 已过期，标记为失败")
                    self.active_plans[plan_id].status = "expired"
                
            except Exception as e:
                self.logger.error(f"计划监控任务出错: {e}")
    
    async def _performance_optimizer(self):
        """性能优化后台任务"""
        while True:
            try:
                await asyncio.sleep(60)  # 每分钟优化一次
                
                # 分析性能指标
                await self._analyze_performance_metrics()
                
                # 优化智能体分配策略
                await self._optimize_agent_assignment_strategy()
                
            except Exception as e:
                self.logger.error(f"性能优化任务出错: {e}")
    
    async def _analyze_performance_metrics(self):
        """分析性能指标"""
        if not self.metrics["tasks_processed"]:
            return
        
        # 计算平均完成时间
        if self.completed_plans:
            total_time = sum(
                (plan.estimated_completion - plan.created_at).total_seconds() 
                for plan in self.completed_plans
            )
            self.metrics["average_completion_time"] = total_time / len(self.completed_plans)
        
        # 计算智能体利用率
        for agent_id in self.agent_load:
            total_tasks = sum(1 for plan in self.completed_plans 
                            for task_id, agents in plan.agent_assignments.items() 
                            if agent_id in agents)
            utilization = total_tasks / max(self.metrics["tasks_processed"], 1)
            self.metrics["agent_utilization"][agent_id] = utilization
    
    async def _optimize_agent_assignment_strategy(self):
        """优化智能体分配策略"""
        # 基于历史性能调整权重
        for agent_id, performance in self.agent_performance.items():
            if performance.get("success_rate", 0) < 0.7:
                # 降低低性能智能体的优先级
                self.agent_load[agent_id] *= 1.2
    
    def register_agent_capabilities(self, agent_id: str, capabilities: List[AgentCapability]):
        """注册智能体能力"""
        for capability in capabilities:
            self.capability_agent_map[capability.name].add(agent_id)
    
    def update_agent_performance(self, agent_id: str, performance_data: Dict[str, float]):
        """更新智能体性能数据"""
        self.agent_performance[agent_id].update(performance_data)
    
    async def get_coordination_status(self) -> Dict[str, Any]:
        """获取协调状态"""
        return {
            "coordinator_id": self.coordinator_id,
            "status": "running",
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "pending_tasks": len(self.pending_tasks),
            "active_plans": len(self.active_plans),
            "completed_plans": len(self.completed_plans),
            "metrics": self.metrics,
            "coordination_strategy": self.coordination_strategy.value,
            "agent_load_distribution": dict(self.agent_load),
            "capability_mapping": {
                cap: list(agents) for cap, agents in self.capability_agent_map.items()
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
    
    async def shutdown(self):
        """关闭协调管理器"""
        try:
            self.logger.info("关闭协调管理器...")
            
            # 等待所有活跃计划完成
            while self.active_plans:
                await asyncio.sleep(1)
            
            self.logger.info("协调管理器已关闭")
            
        except Exception as e:
            self.logger.error(f"关闭协调管理器时出错: {e}")