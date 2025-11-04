"""
并发执行器

支持任务并发执行、资源控制和负载均衡
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
import threading
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class ResourceType(Enum):
    """资源类型"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    GPU = "gpu"
    CUSTOM = "custom"


@dataclass
class Task:
    """并发任务"""
    id: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: int = 0
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    resource_requirements: Dict[ResourceType, float] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    callback: Optional[Callable] = None
    error_callback: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    def __lt__(self, other):
        return self.priority > other.priority


@dataclass
class ExecutionResult:
    """执行结果"""
    task_id: str
    status: TaskStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    retry_count: int = 0
    resource_usage: Dict[ResourceType, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceConstraint:
    """资源约束"""
    resource_type: ResourceType
    limit: float
    current_usage: float = 0.0
    allocation_strategy: str = "fair"  # fair, priority, fifo


class ConcurrentExecutor:
    """并发执行器"""
    
    def __init__(self, max_workers: int = 10, max_memory_mb: int = 1024):
        self.max_workers = max_workers
        self.max_memory_mb = max_memory_mb
        
        # 任务管理
        self.pending_tasks: List[Task] = []
        self.running_tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, ExecutionResult] = {}
        self.task_futures: Dict[str, Future] = {}
        
        # 线程池
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.asyncio_pool = []
        
        # 资源管理
        self.resource_constraints: Dict[ResourceType, ResourceConstraint] = {}
        self._initialize_default_constraints()
        
        # 统计信息
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'cancelled_tasks': 0,
            'average_execution_time': 0.0,
            'throughput': 0.0,
            'resource_efficiency': 0.0
        }
        
        # 队列和锁
        self._lock = threading.Lock()
        self._task_queue = asyncio.Queue()
        self._running = False
        
        # 监控和回调
        self.task_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self.global_callbacks: List[Callable] = []
    
    def _initialize_default_constraints(self):
        """初始化默认资源约束"""
        self.resource_constraints = {
            ResourceType.CPU: ResourceConstraint(ResourceType.CPU, limit=max_workers),
            ResourceType.MEMORY: ResourceConstraint(ResourceType.MEMORY, limit=self.max_memory_mb),
            ResourceType.DISK: ResourceConstraint(ResourceType.DISK, limit=100.0),  # 100% 磁盘使用率
            ResourceType.NETWORK: ResourceConstraint(ResourceType.NETWORK, limit=1000.0),  # 1000 Mbps
        }
    
    async def start(self):
        """启动并发执行器"""
        self._running = True
        
        # 启动异步工作线程
        for i in range(min(self.max_workers, 5)):
            task = asyncio.create_task(self._async_worker(f"async-worker-{i}"))
            self.asyncio_pool.append(task)
        
        # 启动资源监控
        monitor_task = asyncio.create_task(self._resource_monitor())
        self.asyncio_pool.append(monitor_task)
        
        logger.info(f"并发执行器已启动，最大工作线程: {self.max_workers}")
    
    async def stop(self):
        """停止并发执行器"""
        self._running = False
        
        # 取消所有异步任务
        for task in self.asyncio_pool:
            task.cancel()
        
        # 关闭线程池
        self.thread_pool.shutdown(wait=True)
        
        logger.info("并发执行器已停止")
    
    async def submit(self, task: Task) -> str:
        """提交任务"""
        async with asyncio.Lock():
            try:
                # 检查资源约束
                if not await self._check_resource_constraints(task):
                    logger.warning(f"任务 {task.id} 资源不足，被拒绝")
                    return ""
                
                # 检查依赖
                if not await self._check_dependencies(task):
                    logger.warning(f"任务 {task.id} 依赖未满足，被拒绝")
                    return ""
                
                # 添加到待执行队列
                self.pending_tasks.append(task)
                heapq.heappush(self.pending_tasks, task)
                
                # 统计
                self.stats['total_tasks'] += 1
                
                logger.info(f"任务 {task.id} 已提交，优先级: {task.priority}")
                return task.id
                
            except Exception as e:
                logger.error(f"提交任务失败: {e}")
                return ""
    
    async def submit_batch(self, tasks: List[Task]) -> List[str]:
        """批量提交任务"""
        task_ids = []
        
        for task in tasks:
            task_id = await self.submit(task)
            if task_id:
                task_ids.append(task_id)
        
        logger.info(f"批量提交完成，成功: {len(task_ids)}/{len(tasks)}")
        return task_ids
    
    async def execute_task(self, task_id: str) -> Optional[ExecutionResult]:
        """执行单个任务"""
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
            
            # 移动到运行队列
            self.pending_tasks.remove(task)
            self.running_tasks[task_id] = task
            task.started_at = time.time()
            
            # 分配资源
            await self._allocate_resources(task)
            
            # 执行任务
            result = await self._run_task(task)
            
            # 释放资源
            await self._release_resources(task)
            
            # 记录结果
            self.completed_tasks[task_id] = result
            
            # 触发回调
            await self._trigger_callbacks(task_id, result)
            
            # 从运行队列移除
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
            
            return result
            
        except Exception as e:
            logger.error(f"执行任务异常: {e}")
            return ExecutionResult(
                task_id=task_id,
                status=TaskStatus.FAILED,
                error=str(e)
            )
    
    async def _run_task(self, task: Task) -> ExecutionResult:
        """运行任务"""
        start_time = time.time()
        
        try:
            # 根据任务类型选择执行方式
            if asyncio.iscoroutinefunction(task.func):
                # 异步函数
                if task.timeout:
                    result = await asyncio.wait_for(
                        task.func(*task.args, **task.kwargs),
                        timeout=task.timeout
                    )
                else:
                    result = await task.func(*task.args, **task.kwargs)
            else:
                # 同步函数，使用线程池
                if task.timeout:
                    future = self.thread_pool.submit(task.func, *task.args, **task.kwargs)
                    result = future.result(timeout=task.timeout)
                else:
                    result = task.func(*task.args, **task.kwargs)
            
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                task_id=task.id,
                status=TaskStatus.COMPLETED,
                result=result,
                execution_time=execution_time,
                retry_count=task.retry_count
            )
            
        except asyncio.TimeoutError:
            return ExecutionResult(
                task_id=task.id,
                status=TaskStatus.TIMEOUT,
                error="任务执行超时",
                execution_time=time.time() - start_time,
                retry_count=task.retry_count
            )
        except Exception as e:
            # 处理重试
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                logger.warning(f"任务 {task.id} 执行失败，准备重试 ({task.retry_count}/{task.max_retries})")
                
                # 等待后重试
                await asyncio.sleep(2 ** task.retry_count)
                return await self._run_task(task)
            else:
                return ExecutionResult(
                    task_id=task.id,
                    status=TaskStatus.FAILED,
                    error=str(e),
                    execution_time=time.time() - start_time,
                    retry_count=task.retry_count
                )
    
    async def _check_resource_constraints(self, task: Task) -> bool:
        """检查资源约束"""
        for resource_type, required in task.resource_requirements.items():
            if resource_type in self.resource_constraints:
                constraint = self.resource_constraints[resource_type]
                if constraint.current_usage + required > constraint.limit:
                    return False
        return True
    
    async def _check_dependencies(self, task: Task) -> bool:
        """检查依赖"""
        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                dep_result = self.completed_tasks.get(dep_id)
                if not dep_result or dep_result.status != TaskStatus.COMPLETED:
                    return False
        return True
    
    async def _allocate_resources(self, task: Task):
        """分配资源"""
        for resource_type, required in task.resource_requirements.items():
            if resource_type in self.resource_constraints:
                self.resource_constraints[resource_type].current_usage += required
    
    async def _release_resources(self, task: Task):
        """释放资源"""
        for resource_type, required in task.resource_requirements.items():
            if resource_type in self.resource_constraints:
                constraint = self.resource_constraints[resource_type]
                constraint.current_usage = max(0, constraint.current_usage - required)
    
    async def _async_worker(self, worker_id: str):
        """异步工作线程"""
        logger.debug(f"异步工作线程 {worker_id} 已启动")
        
        while self._running:
            try:
                # 获取下一个任务
                if self.pending_tasks:
                    task = heapq.heappop(self.pending_tasks)
                    
                    # 执行任务
                    await self.execute_task(task.id)
                else:
                    await asyncio.sleep(0.1)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"异步工作线程 {worker_id} 异常: {e}")
    
    async def _resource_monitor(self):
        """资源监控"""
        while self._running:
            try:
                await asyncio.sleep(10)  # 每10秒监控一次
                
                # 记录资源使用情况
                for resource_type, constraint in self.resource_constraints.items():
                    usage_percent = (constraint.current_usage / constraint.limit) * 100
                    
                    if usage_percent > 90:
                        logger.warning(f"资源使用率过高: {resource_type.value} {usage_percent:.1f}%")
                
                # 更新统计
                await self._update_stats()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"资源监控异常: {e}")
    
    async def _update_stats(self):
        """更新统计信息"""
        try:
            total = self.stats['total_tasks']
            if total > 0:
                self.stats['failed_tasks'] = len([
                    r for r in self.completed_tasks.values()
                    if r.status == TaskStatus.FAILED
                ])
                
                # 计算资源效率
                total_usage = sum(
                    c.current_usage / c.limit 
                    for c in self.resource_constraints.values()
                )
                self.stats['resource_efficiency'] = total_usage / len(self.resource_constraints)
                
        except Exception as e:
            logger.error(f"更新统计失败: {e}")
    
    async def _trigger_callbacks(self, task_id: str, result: ExecutionResult):
        """触发回调函数"""
        try:
            # 任务特定回调
            callbacks = self.task_callbacks.get(task_id, [])
            for callback in callbacks:
                if asyncio.iscoroutinefunction(callback):
                    await callback(result)
                else:
                    callback(result)
            
            # 全局回调
            for callback in self.global_callbacks:
                if asyncio.iscoroutinefunction(callback):
                    await callback(task_id, result)
                else:
                    callback(task_id, result)
            
        except Exception as e:
            logger.error(f"触发回调失败: {e}")
    
    async def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        try:
            # 从待执行队列移除
            for i, task in enumerate(self.pending_tasks):
                if task.id == task_id:
                    del self.pending_tasks[i]
                    heapq.heapify(self.pending_tasks)
                    
                    # 记录取消结果
                    self.completed_tasks[task_id] = ExecutionResult(
                        task_id=task_id,
                        status=TaskStatus.CANCELLED
                    )
                    
                    self.stats['cancelled_tasks'] += 1
                    logger.info(f"任务 {task_id} 已取消")
                    return True
            
            # 从运行队列移除
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
                
                # 记录取消结果
                self.completed_tasks[task_id] = ExecutionResult(
                    task_id=task_id,
                    status=TaskStatus.CANCELLED
                )
                
                self.stats['cancelled_tasks'] += 1
                logger.info(f"运行中的任务 {task_id} 已取消")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"取消任务失败: {e}")
            return False
    
    def add_task_callback(self, task_id: str, callback: Callable):
        """添加任务回调"""
        self.task_callbacks[task_id].append(callback)
    
    def add_global_callback(self, callback: Callable):
        """添加全局回调"""
        self.global_callbacks.append(callback)
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """获取任务状态"""
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id].status
        elif task_id in self.running_tasks:
            return TaskStatus.RUNNING
        elif any(task.id == task_id for task in self.pending_tasks):
            return TaskStatus.PENDING
        else:
            return None
    
    def get_execution_result(self, task_id: str) -> Optional[ExecutionResult]:
        """获取执行结果"""
        return self.completed_tasks.get(task_id)
    
    def get_resource_usage(self) -> Dict[str, float]:
        """获取资源使用情况"""
        return {
            resource_type.value: {
                'current': constraint.current_usage,
                'limit': constraint.limit,
                'usage_percent': (constraint.current_usage / constraint.limit) * 100
            }
            for resource_type, constraint in self.resource_constraints.items()
        }
    
    def get_executor_stats(self) -> Dict[str, Any]:
        """获取执行器统计"""
        return {
            'pending_tasks': len(self.pending_tasks),
            'running_tasks': len(self.running_tasks),
            'completed_tasks': len(self.completed_tasks),
            'max_workers': self.max_workers,
            'stats': self.stats.copy(),
            'resource_usage': self.get_resource_usage(),
            'queue_size': len(self.pending_tasks)
        }