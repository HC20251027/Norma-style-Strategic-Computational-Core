"""
任务模型定义

定义任务相关的数据模型和枚举类型
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import uuid


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"           # 等待中
    PLANNING = "planning"         # 规划中
    READY = "ready"              # 准备就绪
    RUNNING = "running"          # 执行中
    COMPLETED = "completed"      # 已完成
    FAILED = "failed"           # 执行失败
    CANCELLED = "cancelled"     # 已取消
    RETRYING = "retrying"       # 重试中
    SKIPPED = "skipped"         # 已跳过


class TaskPriority(Enum):
    """任务优先级枚举"""
    LOW = 1      # 低优先级
    NORMAL = 2   # 普通优先级
    HIGH = 3     # 高优先级
    CRITICAL = 4 # 关键优先级


class TaskDependency:
    """任务依赖关系"""
    
    def __init__(
        self,
        source_task_id: str,
        target_task_id: str,
        dependency_type: str = "finish_to_start",
        delay: int = 0,
        condition: Optional[str] = None
    ):
        self.id = str(uuid.uuid4())
        self.source_task_id = source_task_id
        self.target_task_id = target_task_id
        self.dependency_type = dependency_type  # finish_to_start, start_to_start, finish_to_finish, start_to_finish
        self.delay = delay  # 延迟时间（分钟）
        self.condition = condition  # 依赖条件（可选）
        self.created_at = datetime.now()
    
    def __repr__(self):
        return f"TaskDependency({self.source_task_id} -> {self.target_task_id})"


@dataclass
class Task:
    """任务数据模型"""
    
    # 基本信息
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    
    # 状态信息
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.NORMAL
    progress: float = 0.0  # 0.0 - 1.0
    
    # 时间信息
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_duration: Optional[int] = None  # 预计耗时（分钟）
    actual_duration: Optional[int] = None     # 实际耗时（分钟）
    
    # 任务内容
    task_type: str = "generic"  # 任务类型
    parameters: Dict[str, Any] = field(default_factory=dict)
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    
    # 执行信息
    assigned_to: Optional[str] = None
    executor_type: str = "llm"  # llm, human, system, agent
    retry_count: int = 0
    max_retries: int = 3
    
    # 依赖关系
    dependencies: List[TaskDependency] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)  # 依赖此任务的任务ID列表
    
    # 错误信息
    error_message: Optional[str] = None
    error_details: Dict[str, Any] = field(default_factory=dict)
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def update_status(self, new_status: TaskStatus, error_message: Optional[str] = None):
        """更新任务状态"""
        old_status = self.status
        self.status = new_status
        self.updated_at = datetime.now()
        
        if new_status == TaskStatus.RUNNING and self.started_at is None:
            self.started_at = datetime.now()
        elif new_status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            self.completed_at = datetime.now()
            if self.started_at:
                self.actual_duration = int((self.completed_at - self.started_at).total_seconds() / 60)
        
        if error_message:
            self.error_message = error_message
    
    def update_progress(self, progress: float):
        """更新任务进度"""
        self.progress = max(0.0, min(1.0, progress))
        self.updated_at = datetime.now()
    
    def add_dependency(self, dependency: TaskDependency):
        """添加依赖关系"""
        self.dependencies.append(dependency)
    
    def add_dependent(self, task_id: str):
        """添加依赖此任务的任务"""
        if task_id not in self.dependents:
            self.dependents.append(task_id)
    
    def can_execute(self) -> bool:
        """检查任务是否可以执行"""
        if self.status != TaskStatus.READY:
            return False
        
        # 检查所有依赖是否完成
        for dep in self.dependencies:
            # 这里需要通过外部检查依赖任务的状态
            # 简化实现，实际需要传入任务管理器
            pass
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority.value,
            "progress": self.progress,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "estimated_duration": self.estimated_duration,
            "actual_duration": self.actual_duration,
            "task_type": self.task_type,
            "parameters": self.parameters,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "assigned_to": self.assigned_to,
            "executor_type": self.executor_type,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "dependencies": [
                {
                    "id": dep.id,
                    "source_task_id": dep.source_task_id,
                    "target_task_id": dep.target_task_id,
                    "dependency_type": dep.dependency_type,
                    "delay": dep.delay,
                    "condition": dep.condition
                }
                for dep in self.dependencies
            ],
            "dependents": self.dependents,
            "error_message": self.error_message,
            "error_details": self.error_details,
            "metadata": self.metadata,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """从字典创建任务实例"""
        task = cls()
        task.id = data["id"]
        task.name = data["name"]
        task.description = data["description"]
        task.status = TaskStatus(data["status"])
        task.priority = TaskPriority(data["priority"])
        task.progress = data["progress"]
        
        # 处理时间字段
        if data["created_at"]:
            task.created_at = datetime.fromisoformat(data["created_at"])
        if data["updated_at"]:
            task.updated_at = datetime.fromisoformat(data["updated_at"])
        if data["started_at"]:
            task.started_at = datetime.fromisoformat(data["started_at"])
        if data["completed_at"]:
            task.completed_at = datetime.fromisoformat(data["completed_at"])
        
        task.estimated_duration = data.get("estimated_duration")
        task.actual_duration = data.get("actual_duration")
        task.task_type = data.get("task_type", "generic")
        task.parameters = data.get("parameters", {})
        task.inputs = data.get("inputs", {})
        task.outputs = data.get("outputs", {})
        task.assigned_to = data.get("assigned_to")
        task.executor_type = data.get("executor_type", "llm")
        task.retry_count = data.get("retry_count", 0)
        task.max_retries = data.get("max_retries", 3)
        task.error_message = data.get("error_message")
        task.error_details = data.get("error_details", {})
        task.metadata = data.get("metadata", {})
        task.tags = data.get("tags", [])
        
        # 处理依赖关系
        task.dependents = data.get("dependents", [])
        
        return task


@dataclass
class TaskExecutionPlan:
    """任务执行计划"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    
    # 任务列表
    tasks: List[Task] = field(default_factory=list)
    
    # 依赖关系
    dependencies: List[TaskDependency] = field(default_factory=list)
    
    # 执行参数
    max_parallel_tasks: int = 5
    execution_mode: str = "sequential"  # sequential, parallel, hybrid
    
    # 状态
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    
    def add_task(self, task: Task):
        """添加任务"""
        self.tasks.append(task)
    
    def add_dependency(self, dependency: TaskDependency):
        """添加依赖关系"""
        self.dependencies.append(dependency)
    
    def get_task_by_id(self, task_id: str) -> Optional[Task]:
        """根据ID获取任务"""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None
    
    def update_progress(self):
        """更新整体进度"""
        if not self.tasks:
            self.progress = 0.0
            return
        
        total_progress = sum(task.progress for task in self.tasks)
        self.progress = total_progress / len(self.tasks)