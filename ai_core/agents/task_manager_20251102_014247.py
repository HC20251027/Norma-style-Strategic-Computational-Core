"""
多任务管理器
实现任务的并行处理、调度、切换和管理功能
"""

import asyncio
import uuid
import json
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """任务优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


class Task:
    """任务类"""
    
    def __init__(self, 
                 task_id: str,
                 session_id: str,
                 task_type: str,
                 description: str,
                 priority: str = 'normal',
                 dependencies: Optional[List[str]] = None,
                 estimated_duration: Optional[int] = None):
        self.task_id = task_id
        self.session_id = session_id
        self.task_type = task_type
        self.description = description
        self.priority = TaskPriority[priority.upper()]
        self.dependencies = dependencies or []
        self.estimated_duration = estimated_duration
        
        self.status = TaskStatus.PENDING
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.progress = 0
        self.result: Optional[Any] = None
        self.error: Optional[str] = None
        
        # 任务执行器
        self.executor: Optional[Callable] = None
        self.context: Dict[str, Any] = {}
        
        # 资源占用
        self.cpu_usage = 0
        self.memory_usage = 0
        self.io_operations = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'task_id': self.task_id,
            'session_id': self.session_id,
            'task_type': self.task_type,
            'description': self.description,
            'priority': self.priority.name,
            'status': self.status.value,
            'progress': self.progress,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'estimated_duration': self.estimated_duration,
            'dependencies': self.dependencies,
            'context': self.context
        }


class MultiTaskManager:
    """多任务管理器"""
    
    def __init__(self, max_concurrent_tasks: int = 10):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.tasks: Dict[str, Task] = {}
        self.task_queue: List[str] = []
        self.running_tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, Task] = {}
        
        # 任务执行器
        self.executor_pool = ThreadPoolExecutor(max_workers=max_concurrent_tasks)
        self.task_semaphore = asyncio.Semaphore(max_concurrent_tasks)
        
        # 任务调度器
        self.scheduler_running = False
        self.scheduler_task: Optional[asyncio.Task] = None
        
        # 回调函数
        self.on_task_start: Optional[Callable] = None
        self.on_task_progress: Optional[Callable] = None
        self.on_task_complete: Optional[Callable] = None
        self.on_task_fail: Optional[Callable] = None
        
        self.logger = logging.getLogger(__name__)
    
    async def create_task(self,
                         session_id: str,
                         task_type: str,
                         description: str,
                         priority: str = 'normal',
                         dependencies: Optional[List[str]] = None,
                         executor: Optional[Callable] = None,
                         estimated_duration: Optional[int] = None) -> str:
        """创建任务"""
        task_id = str(uuid.uuid4())
        
        task = Task(
            task_id=task_id,
            session_id=session_id,
            task_type=task_type,
            description=description,
            priority=priority,
            dependencies=dependencies,
            estimated_duration=estimated_duration
        )
        
        if executor:
            task.executor = executor
        
        self.tasks[task_id] = task
        self.task_queue.append(task_id)
        
        # 按优先级排序队列
        self._sort_queue_by_priority()
        
        self.logger.info(f"Created task {task_id} for session {session_id}")
        return task_id
    
    async def start_task(self, task_id: str) -> bool:
        """启动任务"""
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        
        # 检查依赖
        if not await self._check_dependencies(task):
            return False
        
        # 检查并发限制
        if len(self.running_tasks) >= self.max_concurrent_tasks:
            return False
        
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        
        self.running_tasks[task_id] = task
        
        # 异步执行任务
        asyncio.create_task(self._execute_task(task))
        
        if self.on_task_start:
            await self.on_task_start(task)
        
        self.logger.info(f"Started task {task_id}")
        return True
    
    async def _execute_task(self, task: Task):
        """执行任务"""
        try:
            async with self.task_semaphore:
                if task.executor:
                    # 如果有自定义执行器
                    if asyncio.iscoroutinefunction(task.executor):
                        result = await task.executor(task)
                    else:
                        result = task.executor(task)
                else:
                    # 默认任务执行逻辑
                    result = await self._default_task_execution(task)
                
                task.result = result
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now()
                task.progress = 100
                
        except Exception as e:
            task.error = str(e)
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            
            self.logger.error(f"Task {task.task_id} failed: {e}")
            
            if self.on_task_fail:
                await self.on_task_fail(task)
        finally:
            # 移动到已完成任务
            self.completed_tasks[task.task_id] = task
            if task.task_id in self.running_tasks:
                del self.running_tasks[task.task_id]
            
            if self.on_task_complete:
                await self.on_task_complete(task)
    
    async def _default_task_execution(self, task: Task) -> Any:
        """默认任务执行逻辑"""
        # 模拟任务执行过程
        steps = 5
        for i in range(steps):
            await asyncio.sleep(0.1)  # 模拟处理时间
            task.progress = int((i + 1) / steps * 100)
            
            if self.on_task_progress:
                await self.on_task_progress(task)
            
            # 检查是否被中断
            if task.status == TaskStatus.CANCELLED:
                raise asyncio.CancelledError("Task was cancelled")
        
        return f"Task {task.task_id} completed successfully"
    
    async def _check_dependencies(self, task: Task) -> bool:
        """检查任务依赖"""
        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                return False
            if self.completed_tasks[dep_id].status != TaskStatus.COMPLETED:
                return False
        return True
    
    def _sort_queue_by_priority(self):
        """按优先级排序任务队列"""
        def get_priority_value(task_id: str) -> int:
            task = self.tasks[task_id]
            return task.priority.value
        
        self.task_queue.sort(key=get_priority_value, reverse=True)
    
    async def pause_task(self, task_id: str) -> bool:
        """暂停任务"""
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        if task.status == TaskStatus.RUNNING:
            task.status = TaskStatus.PAUSED
            self.logger.info(f"Paused task {task_id}")
            return True
        
        return False
    
    async def resume_task(self, task_id: str) -> bool:
        """恢复任务"""
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        if task.status == TaskStatus.PAUSED:
            task.status = TaskStatus.RUNNING
            if task_id not in self.running_tasks:
                self.running_tasks[task_id] = task
                asyncio.create_task(self._execute_task(task))
            self.logger.info(f"Resumed task {task_id}")
            return True
        
        return False
    
    async def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        task.status = TaskStatus.CANCELLED
        task.completed_at = datetime.now()
        
        if task_id in self.running_tasks:
            del self.running_tasks[task_id]
        
        self.completed_tasks[task_id] = task
        self.logger.info(f"Cancelled task {task_id}")
        return True
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        if task_id not in self.tasks:
            return None
        
        task = self.tasks[task_id]
        return {
            'task_id': task_id,
            'status': task.status.value,
            'progress': task.progress,
            'started_at': task.started_at.isoformat() if task.started_at else None,
            'completed_at': task.completed_at.isoformat() if task.completed_at else None,
            'estimated_duration': task.estimated_duration,
            'priority': task.priority.name,
            'error': task.error
        }
    
    async def get_active_tasks(self, session_id: str) -> List[Dict[str, Any]]:
        """获取会话的活跃任务"""
        active_tasks = []
        
        for task_id, task in self.running_tasks.items():
            if task.session_id == session_id:
                active_tasks.append(task.to_dict())
        
        return active_tasks
    
    async def get_session_tasks(self, session_id: str) -> List[Dict[str, Any]]:
        """获取会话的所有任务"""
        session_tasks = []
        
        for task in self.tasks.values():
            if task.session_id == session_id:
                session_tasks.append(task.to_dict())
        
        return session_tasks
    
    async def get_estimated_time(self, task_id: str) -> Optional[int]:
        """获取任务预计完成时间"""
        if task_id not in self.tasks:
            return None
        
        task = self.tasks[task_id]
        if task.estimated_duration:
            if task.started_at:
                elapsed = (datetime.now() - task.started_at).total_seconds()
                remaining = task.estimated_duration - elapsed
                return max(0, int(remaining))
            return task.estimated_duration
        
        # 根据进度估算
        if task.progress > 0:
            elapsed = (datetime.now() - task.started_at).total_seconds() if task.started_at else 0
            estimated_total = elapsed / (task.progress / 100)
            remaining = estimated_total - elapsed
            return max(0, int(remaining))
        
        return None
    
    async def switch_task_priority(self, task_id: str, new_priority: str) -> bool:
        """切换任务优先级"""
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        task.priority = TaskPriority[new_priority.upper()]
        
        # 重新排序队列
        self._sort_queue_by_priority()
        
        self.logger.info(f"Switched task {task_id} priority to {new_priority}")
        return True
    
    async def get_system_load(self) -> Dict[str, Any]:
        """获取系统负载"""
        return {
            'running_tasks': len(self.running_tasks),
            'queued_tasks': len(self.task_queue),
            'completed_tasks': len(self.completed_tasks),
            'total_tasks': len(self.tasks),
            'max_concurrent': self.max_concurrent_tasks,
            'cpu_usage': sum(task.cpu_usage for task in self.running_tasks.values()),
            'memory_usage': sum(task.memory_usage for task in self.running_tasks.values())
        }
    
    async def get_total_tasks(self) -> int:
        """获取总任务数"""
        return len(self.tasks)
    
    async def start_scheduler(self):
        """启动任务调度器"""
        if self.scheduler_running:
            return
        
        self.scheduler_running = True
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        self.logger.info("Task scheduler started")
    
    async def stop_scheduler(self):
        """停止任务调度器"""
        self.scheduler_running = False
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Task scheduler stopped")
    
    async def _scheduler_loop(self):
        """调度器主循环"""
        while self.scheduler_running:
            try:
                # 检查队列中的任务
                while self.task_queue and len(self.running_tasks) < self.max_concurrent_tasks:
                    task_id = self.task_queue.pop(0)
                    if task_id in self.tasks:
                        await self.start_task(task_id)
                
                await asyncio.sleep(0.1)  # 避免过度消耗CPU
                
            except Exception as e:
                self.logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(1)
    
    def register_callbacks(self,
                          on_task_start: Optional[Callable] = None,
                          on_task_progress: Optional[Callable] = None,
                          on_task_complete: Optional[Callable] = None,
                          on_task_fail: Optional[Callable] = None):
        """注册回调函数"""
        self.on_task_start = on_task_start
        self.on_task_progress = on_task_progress
        self.on_task_complete = on_task_complete
        self.on_task_fail = on_task_fail