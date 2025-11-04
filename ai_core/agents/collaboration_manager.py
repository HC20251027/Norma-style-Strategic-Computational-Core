"""
Agent间协作和任务分配机制
"""
import asyncio
import uuid
import time
from typing import Dict, Any, List, Optional, Set, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from collections import defaultdict, deque
import json


class CollaborationMode(Enum):
    """协作模式"""
    SEQUENTIAL = "sequential"      # 顺序执行
    PARALLEL = "parallel"          # 并行执行
    PIPELINE = "pipeline"          # 流水线模式
    SUPERVISOR = "supervisor"      # 监督者模式
    PEER_TO_PEER = "peer_to_peer"  # 点对点模式


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class TaskDependency(Enum):
    """任务依赖类型"""
    FINISH_TO_START = "finish_to_start"  # 完成到开始
    START_TO_START = "start_to_start"    # 开始到开始
    FINISH_TO_FINISH = "finish_to_finish"  # 完成到完成
    START_TO_FINISH = "start_to_finish"    # 开始到完成


@dataclass
class TaskNode:
    """任务节点"""
    task_id: str
    agent_id: str
    task_type: str
    priority: int
    payload: Dict[str, Any]
    dependencies: List[str]  # 依赖的任务ID列表
    created_at: datetime
    estimated_duration: Optional[int] = None  # 预估执行时间(秒)
    timeout: Optional[int] = None


@dataclass
class CollaborationTask:
    """协作任务"""
    collaboration_id: str
    mode: CollaborationMode
    tasks: List[TaskNode]
    participants: List[str]
    created_at: datetime
    status: TaskStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class TaskAssignment:
    """任务分配"""
    assignment_id: str
    task_id: str
    agent_id: str
    assigned_at: datetime
    estimated_completion: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING


class TaskScheduler:
    """任务调度器"""
    
    def __init__(self):
        self.task_queue = deque()  # 任务队列
        self.running_tasks = {}  # 运行中的任务
        self.completed_tasks = {}  # 已完成的任务
        self.failed_tasks = {}  # 失败的任务
        self.task_dependencies = {}  # 任务依赖关系
        self.agent_workload = defaultdict(float)  # 智能体工作负载
        self.logger = logging.getLogger("task.scheduler")
    
    def add_task(self, task: TaskNode) -> str:
        """添加任务到调度队列"""
        self.task_queue.append(task)
        self.task_dependencies[task.task_id] = task.dependencies
        self.logger.info(f"添加任务到队列: {task.task_id}")
        return task.task_id
    
    def add_tasks(self, tasks: List[TaskNode]):
        """批量添加任务"""
        for task in tasks:
            self.add_task(task)
    
    def get_ready_tasks(self) -> List[TaskNode]:
        """获取可执行的任务"""
        ready_tasks = []
        
        for task in list(self.task_queue):
            # 检查依赖是否满足
            if self._are_dependencies_satisfied(task.task_id):
                ready_tasks.append(task)
        
        return ready_tasks
    
    def _are_dependencies_satisfied(self, task_id: str) -> bool:
        """检查任务依赖是否满足"""
        dependencies = self.task_dependencies.get(task_id, [])
        
        for dep_id in dependencies:
            if dep_id not in self.completed_tasks:
                return False
        
        return True
    
    def assign_task(self, task: TaskNode, agent_id: str) -> TaskAssignment:
        """分配任务给智能体"""
        assignment = TaskAssignment(
            assignment_id=str(uuid.uuid4()),
            task_id=task.task_id,
            agent_id=agent_id,
            assigned_at=datetime.now()
        )
        
        # 计算预估完成时间
        if task.estimated_duration:
            assignment.estimated_completion = datetime.now() + timedelta(seconds=task.estimated_duration)
        
        # 从队列中移除任务
        try:
            self.task_queue.remove(task)
        except ValueError:
            pass  # 任务可能已经在运行中
        
        # 添加到运行中任务
        self.running_tasks[task.task_id] = {
            "task": task,
            "assignment": assignment,
            "start_time": datetime.now()
        }
        
        # 更新智能体负载
        self.agent_workload[agent_id] += 1
        
        self.logger.info(f"分配任务: {task.task_id} -> {agent_id}")
        return assignment
    
    def complete_task(self, task_id: str, result: Dict[str, Any]):
        """完成任务"""
        if task_id in self.running_tasks:
            task_info = self.running_tasks.pop(task_id)
            task = task_info["task"]
            assignment = task_info["assignment"]
            
            # 移动到已完成任务
            self.completed_tasks[task_id] = {
                "task": task,
                "assignment": assignment,
                "result": result,
                "completed_at": datetime.now()
            }
            
            # 更新智能体负载
            self.agent_workload[assignment.agent_id] = max(0, self.agent_workload[assignment.agent_id] - 1)
            
            self.logger.info(f"任务完成: {task_id}")
    
    def fail_task(self, task_id: str, error: str):
        """任务失败"""
        if task_id in self.running_tasks:
            task_info = self.running_tasks.pop(task_id)
            task = task_info["task"]
            assignment = task_info["assignment"]
            
            # 移动到失败任务
            self.failed_tasks[task_id] = {
                "task": task,
                "assignment": assignment,
                "error": error,
                "failed_at": datetime.now()
            }
            
            # 更新智能体负载
            self.agent_workload[assignment.agent_id] = max(0, self.agent_workload[assignment.agent_id] - 1)
            
            self.logger.error(f"任务失败: {task_id} - {error}")
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """获取调度器状态"""
        return {
            "pending_tasks": len(self.task_queue),
            "running_tasks": len(self.running_tasks),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "agent_workload": dict(self.agent_workload)
        }


class CollaborationManager:
    """协作管理器"""
    
    def __init__(self, task_scheduler: TaskScheduler):
        self.task_scheduler = task_scheduler
        self.active_collaborations = {}  # 活跃协作
        self.collaboration_history = []  # 协作历史
        self.collaboration_templates = {}  # 协作模板
        self.logger = logging.getLogger("collaboration.manager")
    
    async def create_collaboration(self, mode: CollaborationMode, 
                                  tasks: List[TaskNode], 
                                  participants: List[str]) -> str:
        """创建协作"""
        collaboration_id = str(uuid.uuid4())
        
        collaboration = CollaborationTask(
            collaboration_id=collaboration_id,
            mode=mode,
            tasks=tasks,
            participants=participants,
            created_at=datetime.now(),
            status=TaskStatus.PENDING
        )
        
        self.active_collaborations[collaboration_id] = collaboration
        
        # 根据协作模式执行
        if mode == CollaborationMode.SEQUENTIAL:
            await self._execute_sequential(collaboration)
        elif mode == CollaborationMode.PARALLEL:
            await self._execute_parallel(collaboration)
        elif mode == CollaborationMode.PIPELINE:
            await self._execute_pipeline(collaboration)
        elif mode == CollaborationMode.SUPERVISOR:
            await self._execute_supervisor(collaboration)
        elif mode == CollaborationMode.PEER_TO_PEER:
            await self._execute_peer_to_peer(collaboration)
        
        return collaboration_id
    
    async def _execute_sequential(self, collaboration: CollaborationTask):
        """顺序执行模式"""
        self.logger.info(f"开始顺序协作: {collaboration.collaboration_id}")
        
        try:
            collaboration.status = TaskStatus.RUNNING
            
            # 按依赖关系排序任务
            sorted_tasks = self._topological_sort(collaboration.tasks)
            
            results = {}
            for task in sorted_tasks:
                # 分配任务
                assignment = self.task_scheduler.assign_task(task, task.agent_id)
                
                # 等待任务完成（简化实现）
                await asyncio.sleep(0.1)  # 模拟执行时间
                
                # 模拟任务完成
                self.task_scheduler.complete_task(task.task_id, {"result": "success"})
                results[task.task_id] = {"result": "success"}
            
            collaboration.status = TaskStatus.COMPLETED
            collaboration.result = results
            
        except Exception as e:
            collaboration.status = TaskStatus.FAILED
            collaboration.error = str(e)
            self.logger.error(f"顺序协作失败: {e}")
    
    async def _execute_parallel(self, collaboration: CollaborationTask):
        """并行执行模式"""
        self.logger.info(f"开始并行协作: {collaboration.collaboration_id}")
        
        try:
            collaboration.status = TaskStatus.RUNNING
            
            # 并行执行所有无依赖的任务
            ready_tasks = [task for task in collaboration.tasks 
                          if not task.dependencies]
            
            # 分配所有就绪任务
            assignments = []
            for task in ready_tasks:
                assignment = self.task_scheduler.assign_task(task, task.agent_id)
                assignments.append(assignment)
            
            # 等待所有任务完成（简化实现）
            await asyncio.sleep(0.1)
            
            # 模拟任务完成
            for task in ready_tasks:
                self.task_scheduler.complete_task(task.task_id, {"result": "success"})
            
            collaboration.status = TaskStatus.COMPLETED
            collaboration.result = {"parallel_results": "success"}
            
        except Exception as e:
            collaboration.status = TaskStatus.FAILED
            collaboration.error = str(e)
            self.logger.error(f"并行协作失败: {e}")
    
    async def _execute_pipeline(self, collaboration: CollaborationTask):
        """流水线执行模式"""
        self.logger.info(f"开始流水线协作: {collaboration.collaboration_id}")
        
        try:
            collaboration.status = TaskStatus.RUNNING
            
            # 流水线执行：每个阶段处理一个任务
            pipeline_stages = len(collaboration.tasks)
            
            for i, task in enumerate(collaboration.tasks):
                # 分配任务到对应阶段的智能体
                stage_agent = collaboration.participants[i % len(collaboration.participants)]
                assignment = self.task_scheduler.assign_task(task, stage_agent)
                
                # 模拟流水线处理时间
                await asyncio.sleep(0.05)
                
                # 完成任务
                self.task_scheduler.complete_task(task.task_id, {"stage": i, "result": "success"})
            
            collaboration.status = TaskStatus.COMPLETED
            collaboration.result = {"pipeline_result": "success"}
            
        except Exception as e:
            collaboration.status = TaskStatus.FAILED
            collaboration.error = str(e)
            self.logger.error(f"流水线协作失败: {e}")
    
    async def _execute_supervisor(self, collaboration: CollaborationTask):
        """监督者执行模式"""
        self.logger.info(f"开始监督者协作: {collaboration.collaboration_id}")
        
        try:
            collaboration.status = TaskStatus.RUNNING
            
            # 选择监督者（第一个参与者）
            supervisor = collaboration.participants[0]
            workers = collaboration.participants[1:]
            
            # 监督者分配任务给工作者
            for i, task in enumerate(collaboration.tasks):
                worker = workers[i % len(workers)]
                assignment = self.task_scheduler.assign_task(task, worker)
                
                # 监督者监控任务执行
                await asyncio.sleep(0.1)
                
                # 完成任务
                self.task_scheduler.complete_task(task.task_id, {"supervised": True, "result": "success"})
            
            collaboration.status = TaskStatus.COMPLETED
            collaboration.result = {"supervised_result": "success"}
            
        except Exception as e:
            collaboration.status = TaskStatus.FAILED
            collaboration.error = str(e)
            self.logger.error(f"监督者协作失败: {e}")
    
    async def _execute_peer_to_peer(self, collaboration: CollaborationTask):
        """点对点执行模式"""
        self.logger.info(f"开始点对点协作: {collaboration.collaboration_id}")
        
        try:
            collaboration.status = TaskStatus.RUNNING
            
            # 点对点协作：智能体之间直接通信和协调
            for i, task in enumerate(collaboration.tasks):
                # 任务可以在任意参与者上执行
                agent = collaboration.participants[i % len(collaboration.participants)]
                assignment = self.task_scheduler.assign_task(task, agent)
                
                # 模拟点对点协调
                await asyncio.sleep(0.1)
                
                # 完成任务
                self.task_scheduler.complete_task(task.task_id, {"peer_coordination": True, "result": "success"})
            
            collaboration.status = TaskStatus.COMPLETED
            collaboration.result = {"peer_result": "success"}
            
        except Exception as e:
            collaboration.status = TaskStatus.FAILED
            collaboration.error = str(e)
            self.logger.error(f"点对点协作失败: {e}")
    
    def _topological_sort(self, tasks: List[TaskNode]) -> List[TaskNode]:
        """拓扑排序"""
        # 简化的拓扑排序实现
        sorted_tasks = []
        remaining_tasks = tasks.copy()
        
        while remaining_tasks:
            # 找到没有依赖或依赖已满足的任务
            ready_task = None
            for task in remaining_tasks:
                if all(dep_id in [t.task_id for t in sorted_tasks] for dep_id in task.dependencies):
                    ready_task = task
                    break
            
            if ready_task:
                sorted_tasks.append(ready_task)
                remaining_tasks.remove(ready_task)
            else:
                # 存在循环依赖，选择任意任务
                sorted_tasks.append(remaining_tasks.pop(0))
        
        return sorted_tasks
    
    def get_collaboration_status(self, collaboration_id: str) -> Optional[Dict[str, Any]]:
        """获取协作状态"""
        collaboration = self.active_collaborations.get(collaboration_id)
        if not collaboration:
            return None
        
        return {
            "collaboration_id": collaboration.collaboration_id,
            "mode": collaboration.mode.value,
            "status": collaboration.status.value,
            "participants": collaboration.participants,
            "task_count": len(collaboration.tasks),
            "created_at": collaboration.created_at.isoformat(),
            "result": collaboration.result,
            "error": collaboration.error
        }
    
    def get_all_collaborations_status(self) -> Dict[str, Any]:
        """获取所有协作状态"""
        active_status = {}
        for collab_id, collaboration in self.active_collaborations.items():
            active_status[collab_id] = self.get_collaboration_status(collab_id)
        
        return {
            "active_collaborations": active_status,
            "total_active": len(self.active_collaborations),
            "total_history": len(self.collaboration_history)
        }


class TaskDistributor:
    """任务分发器"""
    
    def __init__(self, collaboration_manager: CollaborationManager):
        self.collaboration_manager = collaboration_manager
        self.distribution_rules = {}  # 分发规则
        self.agent_capabilities = {}  # 智能体能力映射
        self.logger = logging.getLogger("task.distributor")
    
    def register_agent_capability(self, agent_id: str, capabilities: List[str]):
        """注册智能体能力"""
        self.agent_capabilities[agent_id] = capabilities
        self.logger.info(f"注册智能体能力: {agent_id} -> {capabilities}")
    
    def set_distribution_rule(self, task_type: str, rule: Dict[str, Any]):
        """设置分发规则"""
        self.distribution_rules[task_type] = rule
        self.logger.info(f"设置分发规则: {task_type}")
    
    def distribute_task(self, task: TaskNode) -> Optional[str]:
        """分发任务"""
        # 根据任务类型选择分发策略
        rule = self.distribution_rules.get(task.task_type, {})
        strategy = rule.get("strategy", "capability_based")
        
        if strategy == "capability_based":
            return self._distribute_by_capability(task)
        elif strategy == "round_robin":
            return self._distribute_round_robin(task)
        elif strategy == "load_balanced":
            return self._distribute_load_balanced(task)
        elif strategy == "collaboration":
            return self._distribute_collaboration(task)
        else:
            self.logger.warning(f"未知的分发策略: {strategy}")
            return None
    
    def _distribute_by_capability(self, task: TaskNode) -> str:
        """基于能力分发"""
        # 查找具有所需能力的智能体
        suitable_agents = []
        for agent_id, capabilities in self.agent_capabilities.items():
            if task.task_type in capabilities:
                suitable_agents.append(agent_id)
        
        if not suitable_agents:
            # 如果没有找到合适的智能体，返回默认智能体
            return "default_agent"
        
        # 选择负载最低的智能体
        scheduler_status = self.collaboration_manager.task_scheduler.get_scheduler_status()
        agent_workload = scheduler_status["agent_workload"]
        
        best_agent = min(suitable_agents, key=lambda a: agent_workload.get(a, 0))
        return best_agent
    
    def _distribute_round_robin(self, task: TaskNode) -> str:
        """轮询分发"""
        # 简化实现：随机选择
        suitable_agents = list(self.agent_capabilities.keys())
        return suitable_agents[hash(task.task_id) % len(suitable_agents)]
    
    def _distribute_load_balanced(self, task: TaskNode) -> str:
        """负载均衡分发"""
        scheduler_status = self.collaboration_manager.task_scheduler.get_scheduler_status()
        agent_workload = scheduler_status["agent_workload"]
        
        # 选择负载最低的智能体
        all_agents = list(self.agent_capabilities.keys())
        if not all_agents:
            return "default_agent"
        
        best_agent = min(all_agents, key=lambda a: agent_workload.get(a, 0))
        return best_agent
    
    def _distribute_collaboration(self, task: TaskNode) -> str:
        """协作分发"""
        # 创建一个简单的协作任务
        collaboration_task = TaskNode(
            task_id=f"collab_{task.task_id}",
            agent_id="collaboration_manager",
            task_type="collaboration",
            priority=task.priority,
            payload={"original_task": asdict(task)},
            dependencies=[],
            created_at=datetime.now()
        )
        
        # 这里应该创建实际的协作任务
        # 简化实现：返回协调器
        return "collaboration_coordinator"
    
    def get_distribution_status(self) -> Dict[str, Any]:
        """获取分发状态"""
        return {
            "registered_agents": len(self.agent_capabilities),
            "distribution_rules": self.distribution_rules,
            "agent_capabilities": self.agent_capabilities
        }