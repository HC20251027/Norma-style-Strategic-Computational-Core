#!/usr/bin/env python3
"""
任务协调器
负责协调多个任务的执行顺序、依赖关系和资源分配

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
from .task_distributor import Task, TaskType, TaskStatus, AssignmentStrategy
from ..communication.message_bus import MessageBus


class CoordinationMode(Enum):
    """协调模式枚举"""
    SEQUENTIAL = "sequential"      # 顺序执行
    PARALLEL = "parallel"          # 并行执行
    PIPELINE = "pipeline"          # 流水线执行
    CONDITIONAL = "conditional"    # 条件执行
    ADAPTIVE = "adaptive"          # 自适应执行


class DependencyType(Enum):
    """依赖类型枚举"""
    FINISH_TO_START = "finish_to_start"      # 完成到开始
    START_TO_START = "start_to_start"        # 开始到开始
    FINISH_TO_FINISH = "finish_to_finish"    # 完成到完成
    START_TO_FINISH = "start_to_finish"      # 开始到完成


@dataclass
class TaskDependency:
    """任务依赖关系"""
    dependency_id: str
    predecessor_task_id: str
    successor_task_id: str
    dependency_type: DependencyType
    lag_time: float = 0.0  # 滞后时间（秒）
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class CoordinationWorkflow:
    """协调工作流"""
    workflow_id: str
    name: str
    created_at: datetime
    tasks: List[Task]
    dependencies: List[TaskDependency]
    coordination_mode: CoordinationMode
    max_parallel_tasks: int = 10
    timeout: float = 3600.0
    metadata: Dict[str, Any] = None
    status: str = "created"
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def task_count(self) -> int:
        """任务数量"""
        return len(self.tasks)
    
    @property
    def dependency_count(self) -> int:
        """依赖关系数量"""
        return len(self.dependencies)
    
    def get_task_by_id(self, task_id: str) -> Optional[Task]:
        """根据ID获取任务"""
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None
    
    def get_dependencies_for_task(self, task_id: str) -> List[TaskDependency]:
        """获取任务的所有依赖关系"""
        return [dep for dep in self.dependencies if dep.successor_task_id == task_id]
    
    def get_successors(self, task_id: str) -> List[str]:
        """获取任务的后继任务"""
        successors = []
        for dep in self.dependencies:
            if dep.predecessor_task_id == task_id:
                successors.append(dep.successor_task_id)
        return successors
    
    def get_predecessors(self, task_id: str) -> List[str]:
        """获取任务的前置任务"""
        predecessors = []
        for dep in self.dependencies:
            if dep.successor_task_id == task_id:
                predecessors.append(dep.predecessor_task_id)
        return predecessors


@dataclass
class WorkflowExecution:
    """工作流执行状态"""
    execution_id: str
    workflow_id: str
    started_at: datetime
    status: str = "running"
    completed_tasks: List[str] = None
    failed_tasks: List[str] = None
    current_tasks: Set[str] = None
    results: Dict[str, Any] = None
    error_info: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.completed_tasks is None:
            self.completed_tasks = []
        if self.failed_tasks is None:
            self.failed_tasks = []
        if self.current_tasks is None:
            self.current_tasks = set()
        if self.results is None:
            self.results = {}
        if self.error_info is None:
            self.error_info = {}
    
    @property
    def is_completed(self) -> bool:
        """检查是否完成"""
        return self.status == "completed"
    
    @property
    def is_failed(self) -> bool:
        """检查是否失败"""
        return self.status == "failed"
    
    @property
    def progress(self) -> float:
        """执行进度"""
        total_tasks = len(self.completed_tasks) + len(self.failed_tasks)
        if total_tasks == 0:
            return 0.0
        return len(self.completed_tasks) / total_tasks


class TaskCoordinator:
    """任务协调器"""
    
    def __init__(self, message_bus: MessageBus):
        """初始化任务协调器
        
        Args:
            message_bus: 消息总线
        """
        self.coordinator_id = str(uuid.uuid4())
        self.message_bus = message_bus
        
        # 初始化日志
        self.logger = MultiAgentLogger("task_coordinator")
        
        # 工作流管理
        self.workflows: Dict[str, CoordinationWorkflow] = {}
        self.active_executions: Dict[str, WorkflowExecution] = {}
        self.completed_executions: List[WorkflowExecution] = []
        
        # 执行跟踪
        self.task_executions: Dict[str, Dict[str, Any]] = {}  # task_id -> execution_info
        self.execution_queue: deque = deque()
        
        # 协调策略
        self.default_coordination_mode = CoordinationMode.PARALLEL
        self.max_concurrent_workflows = 20
        
        # 统计信息
        self.metrics = {
            "workflows_created": 0,
            "workflows_completed": 0,
            "workflows_failed": 0,
            "tasks_coordinated": 0,
            "average_execution_time": 0.0,
            "coordination_efficiency": 0.0
        }
        
        # 事件回调
        self.event_callbacks: Dict[str, List[Callable]] = {}
        
        self.logger.info(f"任务协调器 {self.coordinator_id} 已初始化")
    
    async def initialize(self) -> bool:
        """初始化任务协调器"""
        try:
            self.logger.info("初始化任务协调器...")
            
            # 订阅消息总线事件
            await self.message_bus.subscribe("task.completed", self._handle_task_completed)
            await self.message_bus.subscribe("task.failed", self._handle_task_failed)
            
            # 启动后台任务
            asyncio.create_task(self._execution_scheduler())
            asyncio.create_task(self._workflow_monitor())
            asyncio.create_task(self._dependency_resolver())
            
            self.logger.info("任务协调器初始化完成")
            return True
            
        except Exception as e:
            self.logger.error(f"任务协调器初始化失败: {e}")
            return False
    
    async def create_workflow(
        self,
        name: str,
        tasks: List[Task],
        dependencies: Optional[List[TaskDependency]] = None,
        coordination_mode: Optional[CoordinationMode] = None,
        max_parallel_tasks: int = 10,
        timeout: float = 3600.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """创建协调工作流
        
        Args:
            name: 工作流名称
            tasks: 任务列表
            dependencies: 依赖关系列表
            coordination_mode: 协调模式
            max_parallel_tasks: 最大并行任务数
            timeout: 超时时间
            metadata: 附加元数据
            
        Returns:
            工作流ID
        """
        try:
            workflow_id = str(uuid.uuid4())
            
            # 设置默认协调模式
            if coordination_mode is None:
                coordination_mode = self.default_coordination_mode
            
            # 创建工作流
            workflow = CoordinationWorkflow(
                workflow_id=workflow_id,
                name=name,
                created_at=datetime.now(),
                tasks=tasks,
                dependencies=dependencies or [],
                coordination_mode=coordination_mode,
                max_parallel_tasks=max_parallel_tasks,
                timeout=timeout,
                metadata=metadata or {}
            )
            
            # 验证工作流
            await self._validate_workflow(workflow)
            
            self.workflows[workflow_id] = workflow
            self.metrics["workflows_created"] += 1
            
            # 发送工作流创建事件
            await self._emit_event("workflow.created", {
                "workflow_id": workflow_id,
                "name": name,
                "task_count": len(tasks),
                "coordination_mode": coordination_mode.value
            })
            
            self.logger.info(f"工作流 {workflow_id} ({name}) 已创建，包含 {len(tasks)} 个任务")
            return workflow_id
            
        except Exception as e:
            self.logger.error(f"创建工作流失败: {e}")
            raise
    
    async def execute_workflow(self, workflow_id: str) -> str:
        """执行工作流
        
        Args:
            workflow_id: 工作流ID
            
        Returns:
            执行ID
        """
        try:
            if workflow_id not in self.workflows:
                raise ValueError(f"工作流 {workflow_id} 不存在")
            
            workflow = self.workflows[workflow_id]
            
            # 检查是否超过最大并发数
            if len(self.active_executions) >= self.max_concurrent_workflows:
                raise RuntimeError("已达到最大并发工作流数量")
            
            # 创建执行实例
            execution_id = str(uuid.uuid4())
            execution = WorkflowExecution(
                execution_id=execution_id,
                workflow_id=workflow_id,
                started_at=datetime.now()
            )
            
            self.active_executions[execution_id] = execution
            
            # 添加到执行队列
            self.execution_queue.append(execution_id)
            
            # 发送工作流开始执行事件
            await self._emit_event("workflow.started", {
                "workflow_id": workflow_id,
                "execution_id": execution_id,
                "name": workflow.name
            })
            
            self.logger.info(f"工作流 {workflow_id} 开始执行，执行ID: {execution_id}")
            return execution_id
            
        except Exception as e:
            self.logger.error(f"执行工作流失败 {workflow_id}: {e}")
            raise
    
    async def _validate_workflow(self, workflow: CoordinationWorkflow):
        """验证工作流的正确性"""
        # 检查任务ID是否唯一
        task_ids = [task.task_id for task in workflow.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("工作流中存在重复的任务ID")
        
        # 检查依赖关系是否有效
        task_id_set = set(task_ids)
        for dependency in workflow.dependencies:
            if dependency.predecessor_task_id not in task_id_set:
                raise ValueError(f"依赖关系中的前置任务 {dependency.predecessor_task_id} 不存在")
            if dependency.successor_task_id not in task_id_set:
                raise ValueError(f"依赖关系中的后继任务 {dependency.successor_task_id} 不存在")
        
        # 检查是否存在循环依赖
        if await self._has_circular_dependency(workflow):
            raise ValueError("工作流中存在循环依赖")
    
    async def _has_circular_dependency(self, workflow: CoordinationWorkflow) -> bool:
        """检查是否存在循环依赖"""
        # 使用DFS检测循环
        visited = set()
        rec_stack = set()
        
        def has_cycle(task_id: str) -> bool:
            if task_id in rec_stack:
                return True
            if task_id in visited:
                return False
            
            visited.add(task_id)
            rec_stack.add(task_id)
            
            for successor in workflow.get_successors(task_id):
                if has_cycle(successor):
                    return True
            
            rec_stack.remove(task_id)
            return False
        
        for task in workflow.tasks:
            if task.task_id not in visited:
                if has_cycle(task.task_id):
                    return True
        
        return False
    
    async def _execution_scheduler(self):
        """执行调度器后台任务"""
        while True:
            try:
                await asyncio.sleep(1)  # 每秒调度一次
                
                # 处理执行队列
                while self.execution_queue:
                    execution_id = self.execution_queue.popleft()
                    if execution_id in self.active_executions:
                        await self._schedule_workflow_execution(execution_id)
                
            except Exception as e:
                self.logger.error(f"执行调度器出错: {e}")
    
    async def _schedule_workflow_execution(self, execution_id: str):
        """调度工作流执行"""
        execution = self.active_executions[execution_id]
        workflow = self.workflows[execution.workflow_id]
        
        try:
            # 根据协调模式调度任务
            if workflow.coordination_mode == CoordinationMode.SEQUENTIAL:
                await self._schedule_sequential_execution(execution, workflow)
            elif workflow.coordination_mode == CoordinationMode.PARALLEL:
                await self._schedule_parallel_execution(execution, workflow)
            elif workflow.coordination_mode == CoordinationMode.PIPELINE:
                await self._schedule_pipeline_execution(execution, workflow)
            elif workflow.coordination_mode == CoordinationMode.CONDITIONAL:
                await self._schedule_conditional_execution(execution, workflow)
            elif workflow.coordination_mode == CoordinationMode.ADAPTIVE:
                await self._schedule_adaptive_execution(execution, workflow)
            
        except Exception as e:
            self.logger.error(f"调度工作流执行失败 {execution_id}: {e}")
            execution.status = "failed"
            execution.error_info["scheduling_error"] = str(e)
    
    async def _schedule_sequential_execution(self, execution: WorkflowExecution, workflow: CoordinationWorkflow):
        """顺序执行调度"""
        # 按拓扑排序顺序执行任务
        sorted_tasks = await self._topological_sort(workflow)
        
        for task in sorted_tasks:
            if task.task_id not in execution.completed_tasks and task.task_id not in execution.failed_tasks:
                execution.current_tasks.add(task.task_id)
                
                # 启动任务执行
                await self._execute_task(task, execution)
                
                # 等待任务完成
                while task.task_id in execution.current_tasks:
                    await asyncio.sleep(0.1)
    
    async def _schedule_parallel_execution(self, execution: WorkflowExecution, workflow: CoordinationWorkflow):
        """并行执行调度"""
        # 找到所有可执行的任务（没有未完成的前置任务）
        ready_tasks = await self._get_ready_tasks(execution, workflow)
        
        # 限制并发数量
        tasks_to_execute = ready_tasks[:workflow.max_parallel_tasks]
        
        for task in tasks_to_execute:
            execution.current_tasks.add(task.task_id)
            
            # 异步启动任务执行
            asyncio.create_task(self._execute_task(task, execution))
    
    async def _schedule_pipeline_execution(self, execution: WorkflowExecution, workflow: CoordinationWorkflow):
        """流水线执行调度"""
        # 将任务按流水线阶段分组
        pipeline_stages = await self._create_pipeline_stages(workflow)
        
        for stage in pipeline_stages:
            # 等待前一阶段完成
            while execution.current_tasks:
                await asyncio.sleep(0.1)
            
            # 执行当前阶段的所有任务
            for task in stage:
                execution.current_tasks.add(task.task_id)
                asyncio.create_task(self._execute_task(task, execution))
    
    async def _schedule_conditional_execution(self, execution: WorkflowExecution, workflow: CoordinationWorkflow):
        """条件执行调度"""
        # 简化的条件执行：基于任务结果决定后续执行
        ready_tasks = await self._get_ready_tasks(execution, workflow)
        
        for task in ready_tasks:
            execution.current_tasks.add(task.task_id)
            asyncio.create_task(self._execute_task_with_condition(task, execution))
    
    async def _schedule_adaptive_execution(self, execution: WorkflowExecution, workflow: CoordinationWorkflow):
        """自适应执行调度"""
        # 根据系统负载和任务特性动态调整执行策略
        system_load = await self._get_system_load()
        
        if system_load < 0.3:
            # 低负载：使用并行执行
            await self._schedule_parallel_execution(execution, workflow)
        elif system_load < 0.7:
            # 中等负载：使用流水线执行
            await self._schedule_pipeline_execution(execution, workflow)
        else:
            # 高负载：使用顺序执行
            await self._schedule_sequential_execution(execution, workflow)
    
    async def _topological_sort(self, workflow: CoordinationWorkflow) -> List[Task]:
        """拓扑排序"""
        # 构建图
        graph = defaultdict(list)
        in_degree = defaultdict(int)
        
        # 初始化
        for task in workflow.tasks:
            in_degree[task.task_id] = 0
        
        # 添加边
        for dependency in workflow.dependencies:
            graph[dependency.predecessor_task_id].append(dependency.successor_task_id)
            in_degree[dependency.successor_task_id] += 1
        
        # 拓扑排序
        queue = deque([task_id for task_id in in_degree if in_degree[task_id] == 0])
        result = []
        
        while queue:
            current = queue.popleft()
            task = workflow.get_task_by_id(current)
            if task:
                result.append(task)
            
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return result
    
    async def _get_ready_tasks(self, execution: WorkflowExecution, workflow: CoordinationWorkflow) -> List[Task]:
        """获取可执行的任务"""
        ready_tasks = []
        
        for task in workflow.tasks:
            if task.task_id in execution.completed_tasks or task.task_id in execution.failed_tasks:
                continue
            
            # 检查前置任务是否完成
            predecessors = workflow.get_predecessors(task.task_id)
            all_predecessors_complete = all(
                pred_id in execution.completed_tasks for pred_id in predecessors
            )
            
            if all_predecessors_complete:
                ready_tasks.append(task)
        
        return ready_tasks
    
    async def _create_pipeline_stages(self, workflow: CoordinationWorkflow) -> List[List[Task]]:
        """创建流水线阶段"""
        # 简化的流水线阶段划分
        # 实际实现中可以根据任务类型和依赖关系更智能地划分
        
        stages = []
        remaining_tasks = workflow.tasks.copy()
        
        while remaining_tasks:
            current_stage = []
            tasks_to_remove = []
            
            for task in remaining_tasks:
                # 检查是否可以添加到当前阶段
                predecessors = workflow.get_predecessors(task.task_id)
                can_add = all(
                    pred not in [t.task_id for t in remaining_tasks] 
                    for pred in predecessors
                )
                
                if can_add:
                    current_stage.append(task)
                    tasks_to_remove.append(task)
            
            if not current_stage:
                # 如果没有任务可以添加，添加剩余的第一个任务
                current_stage.append(remaining_tasks[0])
                tasks_to_remove.append(remaining_tasks[0])
            
            stages.append(current_stage)
            
            # 移除已分配的任务
            for task in tasks_to_remove:
                remaining_tasks.remove(task)
        
        return stages
    
    async def _get_system_load(self) -> float:
        """获取系统负载"""
        # 简化的负载计算
        total_agents = len(self.active_executions) * 2  # 假设每个执行需要2个智能体
        if total_agents == 0:
            return 0.0
        
        active_tasks = sum(len(exec.current_tasks) for exec in self.active_executions.values())
        return min(1.0, active_tasks / total_agents)
    
    async def _execute_task(self, task: Task, execution: WorkflowExecution):
        """执行任务"""
        try:
            # 记录任务执行信息
            self.task_executions[task.task_id] = {
                "execution_id": execution.execution_id,
                "started_at": datetime.now(),
                "status": "running"
            }
            
            # 发送任务执行事件
            await self._emit_event("task.execution_started", {
                "task_id": task.task_id,
                "execution_id": execution.execution_id,
                "workflow_id": execution.workflow_id
            })
            
            # 模拟任务执行（实际实现中需要调用具体的任务执行器）
            await asyncio.sleep(task.estimated_duration)
            
            # 标记任务完成
            execution.current_tasks.discard(task.task_id)
            execution.completed_tasks.append(task.task_id)
            execution.results[task.task_id] = {"status": "completed", "result": "success"}
            
            # 更新任务执行信息
            self.task_executions[task.task_id]["status"] = "completed"
            self.task_executions[task.task_id]["completed_at"] = datetime.now()
            
            # 检查工作流是否完成
            await self._check_workflow_completion(execution)
            
        except Exception as e:
            # 任务执行失败
            execution.current_tasks.discard(task.task_id)
            execution.failed_tasks.append(task.task_id)
            execution.error_info[task.task_id] = str(e)
            
            # 更新任务执行信息
            self.task_executions[task.task_id]["status"] = "failed"
            self.task_executions[task.task_id]["error"] = str(e)
            
            self.logger.error(f"任务 {task.task_id} 执行失败: {e}")
            
            # 检查工作流是否需要停止
            await self._check_workflow_failure(execution)
    
    async def _execute_task_with_condition(self, task: Task, execution: WorkflowExecution):
        """执行带条件的任务"""
        # 先执行任务
        await self._execute_task(task, execution)
        
        # 检查是否需要根据结果调整后续执行
        result = execution.results.get(task.task_id)
        if result and result.get("condition_failed"):
            # 如果条件失败，可能需要跳过某些后续任务
            await self._handle_conditional_skip(task, execution)
    
    async def _handle_conditional_skip(self, task: Task, execution: WorkflowExecution):
        """处理条件跳过"""
        workflow = self.workflows[execution.workflow_id]
        
        # 找到依赖当前任务的后续任务
        successors = workflow.get_successors(task.task_id)
        
        for successor_id in successors:
            if successor_id not in execution.completed_tasks and successor_id not in execution.failed_tasks:
                # 标记为跳过
                execution.failed_tasks.append(successor_id)
                execution.error_info[successor_id] = "skipped_due_to_condition"
    
    async def _check_workflow_completion(self, execution: WorkflowExecution):
        """检查工作流是否完成"""
        workflow = self.workflows[execution.workflow_id]
        
        # 检查所有任务是否完成或失败
        all_tasks_finished = True
        for task in workflow.tasks:
            if task.task_id not in execution.completed_tasks and task.task_id not in execution.failed_tasks:
                all_tasks_finished = False
                break
        
        if all_tasks_finished:
            execution.status = "completed"
            self.metrics["workflows_completed"] += 1
            
            # 移动到已完成执行
            self.completed_executions.append(execution)
            del self.active_executions[execution.execution_id]
            
            # 发送工作流完成事件
            await self._emit_event("workflow.completed", {
                "execution_id": execution.execution_id,
                "workflow_id": execution.workflow_id,
                "completed_tasks": execution.completed_tasks,
                "failed_tasks": execution.failed_tasks,
                "progress": execution.progress
            })
            
            self.logger.info(f"工作流 {execution.workflow_id} 执行完成")
    
    async def _check_workflow_failure(self, execution: WorkflowExecution):
        """检查工作流是否失败"""
        workflow = self.workflows[execution.workflow_id]
        
        # 如果失败任务过多，标记工作流失败
        failure_rate = len(execution.failed_tasks) / len(workflow.tasks)
        if failure_rate > 0.5:  # 失败率超过50%
            execution.status = "failed"
            self.metrics["workflows_failed"] += 1
            
            # 移动到已完成执行
            self.completed_executions.append(execution)
            del self.active_executions[execution.execution_id]
            
            # 发送工作流失败事件
            await self._emit_event("workflow.failed", {
                "execution_id": execution.execution_id,
                "workflow_id": execution.workflow_id,
                "failure_rate": failure_rate,
                "error_info": execution.error_info
            })
            
            self.logger.warning(f"工作流 {execution.workflow_id} 执行失败")
    
    async def _workflow_monitor(self):
        """工作流监控后台任务"""
        while True:
            try:
                await asyncio.sleep(30)  # 每30秒检查一次
                
                current_time = datetime.now()
                timeout_executions = []
                
                for execution_id, execution in self.active_executions.items():
                    workflow = self.workflows[execution.workflow_id]
                    
                    # 检查是否超时
                    if (current_time - execution.started_at).total_seconds() > workflow.timeout:
                        timeout_executions.append(execution_id)
                
                # 处理超时的执行
                for execution_id in timeout_executions:
                    execution = self.active_executions[execution_id]
                    execution.status = "failed"
                    execution.error_info["timeout"] = f"Execution exceeded {workflow.timeout} seconds"
                    
                    self.logger.warning(f"工作流执行 {execution_id} 超时")
                    
                    # 移动到已完成执行
                    self.completed_executions.append(execution)
                    del self.active_executions[execution_id]
                
            except Exception as e:
                self.logger.error(f"工作流监控任务出错: {e}")
    
    async def _dependency_resolver(self):
        """依赖关系解析器后台任务"""
        while True:
            try:
                await asyncio.sleep(10)  # 每10秒解析一次依赖关系
                
                # 检查是否有新的依赖关系需要处理
                for execution_id, execution in self.active_executions.items():
                    workflow = self.workflows[execution.workflow_id]
                    
                    # 检查是否有新的任务可以执行
                    if workflow.coordination_mode == CoordinationMode.PARALLEL:
                        await self._schedule_parallel_execution(execution, workflow)
                    elif workflow.coordination_mode == CoordinationMode.PIPELINE:
                        await self._schedule_pipeline_execution(execution, workflow)
                
            except Exception as e:
                self.logger.error(f"依赖关系解析任务出错: {e}")
    
    async def _handle_task_completed(self, data: Dict[str, Any]):
        """处理任务完成事件"""
        task_id = data.get("task_id")
        result = data.get("result", {})
        
        if task_id in self.task_executions:
            execution_id = self.task_executions[task_id]["execution_id"]
            if execution_id in self.active_executions:
                execution = self.active_executions[execution_id]
                
                # 更新执行状态
                if task_id in execution.current_tasks:
                    execution.current_tasks.discard(task_id)
                    execution.completed_tasks.append(task_id)
                    execution.results[task_id] = result
                
                # 检查工作流是否完成
                await self._check_workflow_completion(execution)
    
    async def _handle_task_failed(self, data: Dict[str, Any]):
        """处理任务失败事件"""
        task_id = data.get("task_id")
        error = data.get("error", "")
        
        if task_id in self.task_executions:
            execution_id = self.task_executions[task_id]["execution_id"]
            if execution_id in self.active_executions:
                execution = self.active_executions[execution_id]
                
                # 更新执行状态
                if task_id in execution.current_tasks:
                    execution.current_tasks.discard(task_id)
                    execution.failed_tasks.append(task_id)
                    execution.error_info[task_id] = error
                
                # 检查工作流是否需要停止
                await self._check_workflow_failure(execution)
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """获取工作流状态"""
        if workflow_id not in self.workflows:
            return {"error": "工作流不存在"}
        
        workflow = self.workflows[workflow_id]
        
        # 查找活跃的执行
        active_execution = None
        for execution in self.active_executions.values():
            if execution.workflow_id == workflow_id:
                active_execution = execution
                break
        
        return {
            "workflow_id": workflow_id,
            "name": workflow.name,
            "status": workflow.status,
            "coordination_mode": workflow.coordination_mode.value,
            "task_count": workflow.task_count,
            "dependency_count": workflow.dependency_count,
            "active_execution": {
                "execution_id": active_execution.execution_id,
                "status": active_execution.status,
                "progress": active_execution.progress,
                "current_tasks": list(active_execution.current_tasks),
                "completed_tasks": active_execution.completed_tasks,
                "failed_tasks": active_execution.failed_tasks
            } if active_execution else None,
            "created_at": workflow.created_at.isoformat()
        }
    
    def get_coordination_status(self) -> Dict[str, Any]:
        """获取协调状态"""
        return {
            "coordinator_id": self.coordinator_id,
            "total_workflows": len(self.workflows),
            "active_executions": len(self.active_executions),
            "completed_executions": len(self.completed_executions),
            "default_coordination_mode": self.default_coordination_mode.value,
            "max_concurrent_workflows": self.max_concurrent_workflows,
            "metrics": self.metrics,
            "execution_queue_size": len(self.execution_queue)
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
        """关闭任务协调器"""
        try:
            self.logger.info("关闭任务协调器...")
            
            # 等待所有活跃执行完成
            while self.active_executions:
                await asyncio.sleep(1)
            
            self.logger.info("任务协调器已关闭")
            
        except Exception as e:
            self.logger.error(f"关闭任务协调器时出错: {e}")