"""
高级任务调度器
支持复杂任务依赖关系、优先级调度、动态资源分配
"""

import asyncio
import logging
from typing import Dict, List, Set, Optional, Any, Callable, Union
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
import uuid
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """任务优先级枚举"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class TaskDependency:
    """任务依赖关系"""
    task_id: str
    depends_on: List[str] = field(default_factory=list)
    blocking_tasks: Set[str] = field(default_factory=set)
    
    def add_dependency(self, dependency_task_id: str):
        """添加依赖任务"""
        if dependency_task_id not in self.depends_on:
            self.depends_on.append(dependency_task_id)
    
    def remove_dependency(self, dependency_task_id: str):
        """移除依赖任务"""
        if dependency_task_id in self.depends_on:
            self.depends_on.remove(dependency_task_id)


@dataclass
class Task:
    """任务定义"""
    id: str
    name: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    dependencies: TaskDependency = field(default_factory=lambda: TaskDependency(""))
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[Exception] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    execution_time: float = 0.0
    metadata: dict = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.dependencies.task_id:
            self.dependencies.task_id = self.id


class TaskScheduler:
    """高级任务调度器"""
    
    def __init__(self, max_workers: int = 10, enable_monitoring: bool = True):
        self.max_workers = max_workers
        self.enable_monitoring = enable_monitoring
        
        # 任务存储
        self.tasks: Dict[str, Task] = {}
        self.task_queue = deque()
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.completed_tasks: Dict[str, Task] = {}
        self.failed_tasks: Dict[str, Task] = {}
        
        # 依赖管理
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self.completed_dependencies: Dict[str, Set[str]] = defaultdict(set)
        
        # 优先级队列
        self.priority_queues: Dict[TaskPriority, deque] = {
            priority: deque() for priority in TaskPriority
        }
        
        # 执行控制
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.running = False
        self._lock = threading.RLock()
        self._shutdown_event = asyncio.Event()
        
        # 监控统计
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'running_tasks': 0,
            'avg_execution_time': 0.0,
            'total_execution_time': 0.0
        }
        
        logger.info(f"任务调度器初始化完成，最大工作线程数: {max_workers}")
    
    async def submit_task(self, task: Task) -> str:
        """提交任务"""
        with self._lock:
            self.tasks[task.id] = task
            self.stats['total_tasks'] += 1
            
            # 构建依赖图
            for dep_id in task.dependencies.depends_on:
                self.dependency_graph[task.id].add(dep_id)
                self.reverse_dependency_graph[dep_id].add(task.id)
            
            logger.info(f"任务已提交: {task.name} (ID: {task.id})")
            
            # 如果任务可以立即执行，加入队列
            if self._can_execute_task(task):
                await self._add_to_queue(task)
            
            return task.id
    
    async def submit_batch(self, tasks: List[Task]) -> List[str]:
        """批量提交任务"""
        task_ids = []
        for task in tasks:
            task_id = await self.submit_task(task)
            task_ids.append(task_id)
        
        logger.info(f"批量提交任务完成，共 {len(tasks)} 个任务")
        return task_ids
    
    def _can_execute_task(self, task: Task) -> bool:
        """检查任务是否可以执行"""
        # 检查依赖是否完成
        for dep_id in task.dependencies.depends_on:
            if dep_id not in self.completed_dependencies:
                return False
            if dep_id not in self.completed_tasks:
                return False
        
        # 检查任务状态
        return task.status == TaskStatus.PENDING
    
    async def _add_to_queue(self, task: Task):
        """将任务加入优先级队列"""
        self.priority_queues[task.priority].append(task)
        logger.debug(f"任务已加入队列: {task.name} (优先级: {task.priority.name})")
    
    async def start(self):
        """启动调度器"""
        if self.running:
            logger.warning("调度器已在运行中")
            return
        
        self.running = True
        self._shutdown_event.clear()
        
        # 启动调度循环
        asyncio.create_task(self._scheduler_loop())
        logger.info("任务调度器已启动")
    
    async def stop(self):
        """停止调度器"""
        if not self.running:
            return
        
        self.running = False
        self._shutdown_event.set()
        
        # 取消所有运行中的任务
        for task in self.running_tasks.values():
            task.cancel()
        
        # 等待任务完成
        if self.running_tasks:
            await asyncio.gather(*self.running_tasks.values(), return_exceptions=True)
        
        # 关闭线程池
        self.executor.shutdown(wait=True)
        
        logger.info("任务调度器已停止")
    
    async def _scheduler_loop(self):
        """调度器主循环"""
        while self.running:
            try:
                # 检查是否有任务可以执行
                await self._process_ready_tasks()
                
                # 更新统计信息
                self._update_stats()
                
                # 等待一段时间
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"调度器循环错误: {e}")
                await asyncio.sleep(1)
    
    async def _process_ready_tasks(self):
        """处理就绪任务"""
        # 按优先级检查队列
        for priority in TaskPriority:
            queue = self.priority_queues[priority]
            
            while queue:
                task = queue.popleft()
                
                # 再次检查依赖和状态
                if not self._can_execute_task(task):
                    # 如果不能执行，重新放回队列
                    queue.appendleft(task)
                    break
                
                # 执行任务
                await self._execute_task(task)
    
    async def _execute_task(self, task: Task):
        """执行任务"""
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        self.stats['running_tasks'] += 1
        
        # 创建异步任务
        async_task = asyncio.create_task(self._run_task(task))
        self.running_tasks[task.id] = async_task
        
        logger.info(f"开始执行任务: {task.name} (ID: {task.id})")
    
    async def _run_task(self, task: Task):
        """运行单个任务"""
        try:
            # 设置超时
            if task.timeout:
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        self.executor, task.func, *task.args, **task.kwargs
                    ),
                    timeout=task.timeout
                )
            else:
                result = await asyncio.get_event_loop().run_in_executor(
                    self.executor, task.func, *task.args, **task.kwargs
                )
            
            # 任务成功完成
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
            task.execution_time = task.completed_at - task.started_at
            
            # 更新统计
            self.completed_tasks[task.id] = task
            self.stats['completed_tasks'] += 1
            self.stats['running_tasks'] -= 1
            self.stats['total_execution_time'] += task.execution_time
            
            # 通知依赖此任务的其它任务
            await self._notify_dependent_tasks(task.id)
            
            logger.info(f"任务完成: {task.name} (耗时: {task.execution_time:.2f}s)")
            
        except asyncio.TimeoutError:
            await self._handle_task_timeout(task)
        except Exception as e:
            await self._handle_task_error(task, e)
        finally:
            # 清理运行中的任务记录
            if task.id in self.running_tasks:
                del self.running_tasks[task.id]
    
    async def _handle_task_timeout(self, task: Task):
        """处理任务超时"""
        task.status = TaskStatus.TIMEOUT
        task.error = TimeoutError(f"任务执行超时: {task.timeout}s")
        task.completed_at = time.time()
        
        self.failed_tasks[task.id] = task
        self.stats['failed_tasks'] += 1
        self.stats['running_tasks'] -= 1
        
        logger.error(f"任务超时: {task.name}")
        
        # 尝试重试
        await self._retry_task(task)
    
    async def _handle_task_error(self, task: Task, error: Exception):
        """处理任务错误"""
        task.status = TaskStatus.FAILED
        task.error = error
        task.completed_at = time.time()
        
        self.failed_tasks[task.id] = task
        self.stats['failed_tasks'] += 1
        self.stats['running_tasks'] -= 1
        
        logger.error(f"任务执行失败: {task.name}, 错误: {error}")
        
        # 尝试重试
        await self._retry_task(task)
    
    async def _retry_task(self, task: Task):
        """重试任务"""
        if task.retry_count < task.max_retries:
            task.retry_count += 1
            task.status = TaskStatus.PENDING
            task.started_at = None
            task.completed_at = None
            task.error = None
            
            # 重新加入队列
            await self._add_to_queue(task)
            
            logger.info(f"任务重试: {task.name} (第 {task.retry_count} 次)")
        else:
            logger.error(f"任务重试次数已达上限: {task.name}")
    
    async def _notify_dependent_tasks(self, completed_task_id: str):
        """通知依赖任务"""
        # 记录完成的依赖
        for task_id in self.reverse_dependency_graph[completed_task_id]:
            self.completed_dependencies[task_id].add(completed_task_id)
            
            # 检查是否可以执行
            if task_id in self.tasks:
                task = self.tasks[task_id]
                if self._can_execute_task(task):
                    await self._add_to_queue(task)
    
    def _update_stats(self):
        """更新统计信息"""
        if self.stats['completed_tasks'] > 0:
            self.stats['avg_execution_time'] = (
                self.stats['total_execution_time'] / self.stats['completed_tasks']
            )
    
    async def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        with self._lock:
            if task_id not in self.tasks:
                return False
            
            task = self.tasks[task_id]
            
            if task.status == TaskStatus.RUNNING:
                # 取消运行中的任务
                if task_id in self.running_tasks:
                    self.running_tasks[task_id].cancel()
                    del self.running_tasks[task_id]
                
                task.status = TaskStatus.CANCELLED
                self.stats['running_tasks'] -= 1
            
            elif task.status == TaskStatus.PENDING:
                # 从队列中移除
                for queue in self.priority_queues.values():
                    try:
                        queue.remove(task)
                        break
                    except ValueError:
                        continue
                
                task.status = TaskStatus.CANCELLED
            
            logger.info(f"任务已取消: {task.name}")
            return True
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """获取任务状态"""
        if task_id in self.tasks:
            return self.tasks[task_id].status
        return None
    
    def get_task_result(self, task_id: str) -> Optional[Any]:
        """获取任务结果"""
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id].result
        return None
    
    def get_task_error(self, task_id: str) -> Optional[Exception]:
        """获取任务错误"""
        if task_id in self.failed_tasks:
            return self.failed_tasks[task_id].error
        return None
    
    def get_statistics(self) -> dict:
        """获取调度器统计信息"""
        return self.stats.copy()
    
    def get_running_tasks(self) -> List[Task]:
        """获取运行中的任务"""
        return [
            task for task in self.tasks.values() 
            if task.status == TaskStatus.RUNNING
        ]
    
    def get_pending_tasks(self) -> List[Task]:
        """获取等待中的任务"""
        return [
            task for task in self.tasks.values() 
            if task.status == TaskStatus.PENDING
        ]
    
    def get_completed_tasks(self) -> List[Task]:
        """获取已完成的任务"""
        return list(self.completed_tasks.values())
    
    def get_failed_tasks(self) -> List[Task]:
        """获取失败的任务"""
        return list(self.failed_tasks.values())
    
    async def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> bool:
        """等待任务完成"""
        if task_id not in self.tasks:
            return False
        
        start_time = time.time()
        while True:
            task = self.tasks[task_id]
            
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                return True
            
            if timeout and (time.time() - start_time) >= timeout:
                return False
            
            await asyncio.sleep(0.1)
    
    async def wait_for_all_tasks(self, timeout: Optional[float] = None) -> bool:
        """等待所有任务完成"""
        start_time = time.time()
        
        while True:
            with self._lock:
                if not self.running_tasks:
                    return True
            
            if timeout and (time.time() - start_time) >= timeout:
                return False
            
            await asyncio.sleep(0.1)


# 辅助函数
def create_task(
    name: str,
    func: Callable,
    args: tuple = (),
    kwargs: dict = None,
    priority: TaskPriority = TaskPriority.NORMAL,
    timeout: Optional[float] = None,
    max_retries: int = 3,
    dependencies: List[str] = None
) -> Task:
    """创建任务"""
    task_id = str(uuid.uuid4())
    kwargs = kwargs or {}
    dependencies = dependencies or []
    
    task = Task(
        id=task_id,
        name=name,
        func=func,
        args=args,
        kwargs=kwargs,
        priority=priority,
        timeout=timeout,
        max_retries=max_retries
    )
    
    task.dependencies.depends_on = dependencies
    
    return task


# 示例用法
async def example_usage():
    """示例用法"""
    scheduler = TaskScheduler(max_workers=5)
    
    # 定义任务函数
    def task1():
        time.sleep(2)
        return "任务1完成"
    
    def task2():
        time.sleep(1)
        return "任务2完成"
    
    def task3():
        time.sleep(1)
        return "任务3完成"
    
    # 创建任务
    task_1 = create_task("任务1", task1, priority=TaskPriority.HIGH)
    task_2 = create_task("任务2", task2, dependencies=[task_1.id])
    task_3 = create_task("任务3", task3, priority=TaskPriority.LOW)
    
    # 提交任务
    await scheduler.submit_batch([task_1, task_2, task_3])
    
    # 启动调度器
    await scheduler.start()
    
    # 等待所有任务完成
    await scheduler.wait_for_all_tasks()
    
    # 获取结果
    print("任务1结果:", scheduler.get_task_result(task_1.id))
    print("任务2结果:", scheduler.get_task_result(task_2.id))
    print("任务3结果:", scheduler.get_task_result(task_3.id))
    
    # 停止调度器
    await scheduler.stop()


if __name__ == "__main__":
    asyncio.run(example_usage())