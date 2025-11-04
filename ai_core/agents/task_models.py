"""
任务模型定义

定义任务状态、优先级、任务对象和结果等核心模型
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Optional, List, Union
from dataclasses import dataclass, field
import json
import logging


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"          # 等待中
    SCHEDULED = "scheduled"      # 已调度
    RUNNING = "running"          # 运行中
    PAUSED = "paused"            # 已暂停
    CANCELLED = "cancelled"      # 已取消
    COMPLETED = "completed"      # 已完成
    FAILED = "failed"           # 失败
    TIMEOUT = "timeout"         # 超时


class TaskPriority(Enum):
    """任务优先级枚举"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


@dataclass
class TaskConfig:
    """任务配置"""
    timeout: Optional[int] = None           # 超时时间（秒）
    retry_count: int = 3                    # 重试次数
    retry_delay: float = 1.0               # 重试延迟（秒）
    max_concurrent: int = 10               # 最大并发数
    priority: TaskPriority = TaskPriority.NORMAL  # 优先级
    dependencies: List[str] = field(default_factory=list)  # 依赖任务ID
    tags: List[str] = field(default_factory=list)         # 标签
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据


@dataclass
class TaskMetrics:
    """任务指标"""
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration: Optional[float] = None       # 执行时长（秒）
    retry_count: int = 0                   # 重试次数
    progress: float = 0.0                  # 进度百分比
    memory_usage: Optional[float] = None   # 内存使用量
    cpu_usage: Optional[float] = None      # CPU使用率


@dataclass
class TaskResult:
    """任务结果"""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    traceback: Optional[str] = None
    metrics: TaskMetrics = field(default_factory=TaskMetrics)
    cached: bool = False                   # 是否来自缓存
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "traceback": self.traceback,
            "metrics": {
                "created_at": self.metrics.created_at.isoformat(),
                "started_at": self.metrics.started_at.isoformat() if self.metrics.started_at else None,
                "completed_at": self.metrics.completed_at.isoformat() if self.metrics.completed_at else None,
                "duration": self.metrics.duration,
                "retry_count": self.metrics.retry_count,
                "progress": self.metrics.progress,
                "memory_usage": self.metrics.memory_usage,
                "cpu_usage": self.metrics.cpu_usage
            },
            "cached": self.cached
        }


@dataclass
class Task:
    """异步任务"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    func: Optional[Callable] = None
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    config: TaskConfig = field(default_factory=TaskConfig)
    metrics: TaskMetrics = field(default_factory=TaskMetrics)
    result: Optional[Any] = None
    error: Optional[str] = None
    traceback: Optional[str] = None
    dependencies_resolved: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.name:
            self.name = f"Task_{self.id[:8]}"
    
    def update_status(self, status: TaskStatus):
        """更新任务状态"""
        self.status = status
        self.updated_at = datetime.now()
        
        # 更新指标
        if status == TaskStatus.RUNNING and not self.metrics.started_at:
            self.metrics.started_at = datetime.now()
        elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.TIMEOUT]:
            self.metrics.completed_at = datetime.now()
            if self.metrics.started_at:
                self.metrics.duration = (self.metrics.completed_at - self.metrics.started_at).total_seconds()
    
    def update_progress(self, progress: float):
        """更新进度"""
        self.metrics.progress = max(0, min(100, progress))
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "config": {
                "timeout": self.config.timeout,
                "retry_count": self.config.retry_count,
                "retry_delay": self.config.retry_delay,
                "max_concurrent": self.config.max_concurrent,
                "priority": self.config.priority.value,
                "dependencies": self.config.dependencies,
                "tags": self.config.tags,
                "metadata": self.config.metadata
            },
            "metrics": {
                "created_at": self.metrics.created_at.isoformat(),
                "started_at": self.metrics.started_at.isoformat() if self.metrics.started_at else None,
                "completed_at": self.metrics.completed_at.isoformat() if self.metrics.completed_at else None,
                "duration": self.metrics.duration,
                "retry_count": self.metrics.retry_count,
                "progress": self.metrics.progress,
                "memory_usage": self.metrics.memory_usage,
                "cpu_usage": self.metrics.cpu_usage
            },
            "dependencies_resolved": self.dependencies_resolved,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    def is_finished(self) -> bool:
        """检查任务是否已完成"""
        return self.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.TIMEOUT]
    
    def can_run(self) -> bool:
        """检查任务是否可以运行"""
        return (
            self.status in [TaskStatus.PENDING, TaskStatus.SCHEDULED] and
            self.dependencies_resolved and
            not self.is_finished()
        )
    
    def get_timeout_time(self) -> Optional[datetime]:
        """获取超时时间"""
        if self.config.timeout and self.metrics.started_at:
            return self.metrics.started_at + timedelta(seconds=self.config.timeout)
        return None


class TaskRegistry:
    """任务注册表"""
    
    def __init__(self):
        self._tasks: Dict[str, Task] = {}
        self._lock = asyncio.Lock()
    
    async def add_task(self, task: Task):
        """添加任务"""
        async with self._lock:
            self._tasks[task.id] = task
    
    async def get_task(self, task_id: str) -> Optional[Task]:
        """获取任务"""
        async with self._lock:
            return self._tasks.get(task_id)
    
    async def remove_task(self, task_id: str):
        """移除任务"""
        async with self._lock:
            self._tasks.pop(task_id, None)
    
    async def list_tasks(self, status: Optional[TaskStatus] = None) -> List[Task]:
        """列出任务"""
        async with self._lock:
            tasks = list(self._tasks.values())
            if status:
                tasks = [t for t in tasks if t.status == status]
            return sorted(tasks, key=lambda t: t.created_at)
    
    async def update_task(self, task: Task):
        """更新任务"""
        async with self._lock:
            self._tasks[task.id] = task
    
    async def get_running_tasks(self) -> List[Task]:
        """获取运行中的任务"""
        async with self._lock:
            return [t for t in self._tasks.values() if t.status == TaskStatus.RUNNING]
    
    async def get_pending_tasks(self) -> List[Task]:
        """获取等待中的任务"""
        async with self._lock:
            return [t for t in self._tasks.values() if t.status == TaskStatus.PENDING]