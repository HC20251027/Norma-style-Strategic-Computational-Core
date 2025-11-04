"""
任务调度器

负责任务的调度、优先级管理和依赖关系处理
"""

import asyncio
import heapq
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Callable, Any
from collections import defaultdict, deque
from enum import Enum

from .task_models import Task, TaskStatus, TaskPriority, TaskRegistry


class SchedulingStrategy(Enum):
    """调度策略"""
    FIFO = "fifo"                    # 先进先出
    PRIORITY = "priority"            # 优先级调度
    FAIR = "fair"                    # 公平调度
    LOAD_BALANCED = "load_balanced"  # 负载均衡


class TaskScheduler:
    """任务调度器"""
    
    def __init__(self, max_concurrent: int = 10, strategy: SchedulingStrategy = SchedulingStrategy.PRIORITY):
        self.max_concurrent = max_concurrent
        self.strategy = strategy
        self.logger = logging.getLogger(__name__)
        
        # 任务队列
        self._pending_queue: List[Task] = []
        self._priority_queue: List[tuple] = []  # (priority, timestamp, task)
        self._scheduled_tasks: Dict[str, Task] = {}
        self._running_tasks: Set[str] = set()
        
        # 依赖管理
        self._dependencies: Dict[str, Set[str]] = defaultdict(set)  # task_id -> dependencies
        self._dependents: Dict[str, Set[str]] = defaultdict(set)   # task_id -> dependents
        
        # 负载均衡
        self._worker_loads: Dict[str, int] = defaultdict(int)
        
        # 回调函数
        self._task_scheduled_callbacks: List[Callable] = []
        self._task_completed_callbacks: List[Callable] = []
        
        # 控制任务
        self._scheduler_task: Optional[asyncio.Task] = None
        self._is_running = False
    
    async def start(self):
        """启动调度器"""
        if self._is_running:
            return
        
        self._is_running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        self.logger.info("任务调度器已启动")
    
    async def stop(self):
        """停止调度器"""
        if not self._is_running:
            return
        
        self._is_running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("任务调度器已停止")
    
    async def submit_task(self, task: Task) -> str:
        """提交任务"""
        # 检查依赖关系
        await self._resolve_dependencies(task)
        
        # 添加到队列
        task.update_status(TaskStatus.PENDING)
        
        if self.strategy == SchedulingStrategy.PRIORITY:
            # 按优先级调度
            priority_value = task.config.priority.value
            heapq.heappush(self._priority_queue, (priority_value, task.created_at, task))
        else:
            # 其他调度策略
            self._pending_queue.append(task)
        
        self.logger.info(f"任务已提交: {task.id} ({task.name})")
        return task.id
    
    async def schedule_task(self, task_id: str, delay: float = 0) -> bool:
        """调度任务（带延迟）"""
        task = self._get_task_by_id(task_id)
        if not task:
            return False
        
        async def _schedule_delayed():
            if delay > 0:
                await asyncio.sleep(delay)
            
            if task.can_run():
                task.update_status(TaskStatus.SCHEDULED)
                self._scheduled_tasks[task_id] = task
                await self._notify_scheduled_callbacks(task)
        
        asyncio.create_task(_schedule_delayed())
        return True
    
    async def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        # 从队列中移除
        removed = False
        
        # 从优先级队列中移除
        self._priority_queue = [
            (p, t, task) for p, t, task in self._priority_queue 
            if task.id != task_id
        ]
        heapq.heapify(self._priority_queue)
        
        # 从普通队列中移除
        self._pending_queue = [task for task in self._pending_queue if task.id != task_id]
        
        # 从计划任务中移除
        if task_id in self._scheduled_tasks:
            del self._scheduled_tasks[task_id]
            removed = True
        
        # 取消运行中的任务
        if task_id in self._running_tasks:
            self._running_tasks.remove(task_id)
            removed = True
        
        if removed:
            self.logger.info(f"任务已取消: {task_id}")
        
        return removed
    
    async def pause_task(self, task_id: str) -> bool:
        """暂停任务"""
        task = self._get_task_by_id(task_id)
        if not task:
            return False
        
        if task.status == TaskStatus.RUNNING:
            task.update_status(TaskStatus.PAUSED)
            self.logger.info(f"任务已暂停: {task_id}")
            return True
        
        return False
    
    async def resume_task(self, task_id: str) -> bool:
        """恢复任务"""
        task = self._get_task_by_id(task_id)
        if not task:
            return False
        
        if task.status == TaskStatus.PAUSED:
            task.update_status(TaskStatus.PENDING)
            await self.submit_task(task)
            self.logger.info(f"任务已恢复: {task_id}")
            return True
        
        return False
    
    def add_dependency(self, task_id: str, dependency_id: str):
        """添加任务依赖"""
        self._dependencies[task_id].add(dependency_id)
        self._dependents[dependency_id].add(task_id)
    
    def remove_dependency(self, task_id: str, dependency_id: str):
        """移除任务依赖"""
        self._dependencies[task_id].discard(dependency_id)
        self._dependents[dependency_id].discard(task_id)
    
    def get_dependencies(self, task_id: str) -> Set[str]:
        """获取任务依赖"""
        return self._dependencies.get(task_id, set())
    
    def get_dependents(self, task_id: str) -> Set[str]:
        """获取依赖此任务的其他任务"""
        return self._dependents.get(task_id, set())
    
    def add_task_scheduled_callback(self, callback: Callable):
        """添加任务调度回调"""
        self._task_scheduled_callbacks.append(callback)
    
    def add_task_completed_callback(self, callback: Callable):
        """添加任务完成回调"""
        self._task_completed_callbacks.append(callback)
    
    async def get_next_task(self) -> Optional[Task]:
        """获取下一个要执行的任务"""
        if len(self._running_tasks) >= self.max_concurrent:
            return None
        
        # 根据调度策略选择任务
        if self.strategy == SchedulingStrategy.PRIORITY:
            return await self._get_priority_task()
        elif self.strategy == SchedulingStrategy.FIFO:
            return await self._get_fifo_task()
        elif self.strategy == SchedulingStrategy.FAIR:
            return await self._get_fair_task()
        elif self.strategy == SchedulingStrategy.LOAD_BALANCED:
            return await self._get_load_balanced_task()
        
        return None
    
    async def mark_task_started(self, task_id: str):
        """标记任务开始执行"""
        self._running_tasks.add(task_id)
    
    async def mark_task_completed(self, task_id: str):
        """标记任务完成"""
        self._running_tasks.discard(task_id)
        
        # 通知依赖此任务的其他任务
        dependents = self.get_dependents(task_id)
        for dependent_id in dependents:
            await self._check_dependency_resolution(dependent_id)
        
        # 调用完成回调
        task = self._get_task_by_id(task_id)
        if task:
            await self._notify_completed_callbacks(task)
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """获取队列状态"""
        return {
            "pending_count": len(self._pending_queue) + len(self._priority_queue),
            "running_count": len(self._running_tasks),
            "scheduled_count": len(self._scheduled_tasks),
            "max_concurrent": self.max_concurrent,
            "strategy": self.strategy.value,
            "dependencies": {
                task_id: list(deps) for task_id, deps in self._dependencies.items()
            }
        }
    
    async def _scheduler_loop(self):
        """调度器主循环"""
        while self._is_running:
            try:
                # 获取下一个任务
                next_task = await self.get_next_task()
                
                if next_task:
                    # 标记任务开始执行
                    await self.mark_task_started(next_task.id)
                    
                    # 调用调度回调
                    await self._notify_scheduled_callbacks(next_task)
                
                # 短暂休眠，避免CPU占用过高
                await asyncio.sleep(0.01)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"调度器循环异常: {str(e)}")
                await asyncio.sleep(1)
    
    async def _resolve_dependencies(self, task: Task):
        """解析任务依赖"""
        if not task.config.dependencies:
            task.dependencies_resolved = True
            return
        
        # 检查所有依赖是否已完成
        resolved = True
        for dep_id in task.config.dependencies:
            dep_task = self._get_task_by_id(dep_id)
            if not dep_task or not dep_task.is_finished():
                resolved = False
                break
        
        task.dependencies_resolved = resolved
    
    async def _check_dependency_resolution(self, task_id: str):
        """检查依赖解析"""
        task = self._get_task_by_id(task_id)
        if not task:
            return
        
        await self._resolve_dependencies(task)
    
    async def _get_priority_task(self) -> Optional[Task]:
        """获取优先级任务"""
        while self._priority_queue:
            priority, timestamp, task = heapq.heappop(self._priority_queue)
            if task.can_run():
                return task
        return None
    
    async def _get_fifo_task(self) -> Optional[Task]:
        """获取FIFO任务"""
        while self._pending_queue:
            task = self._pending_queue.pop(0)
            if task.can_run():
                return task
        return None
    
    async def _get_fair_task(self) -> Optional[Task]:
        """获取公平调度任务"""
        # 简单的公平调度：轮询不同优先级的任务
        tasks_by_priority = defaultdict(list)
        
        for task in self._pending_queue:
            if task.can_run():
                tasks_by_priority[task.config.priority].append(task)
        
        # 按优先级顺序选择
        for priority in sorted(tasks_by_priority.keys(), reverse=True):
            if tasks_by_priority[priority]:
                task = tasks_by_priority[priority].pop(0)
                # 将剩余任务放回队列
                self._pending_queue.extend(tasks_by_priority[priority])
                return task
        
        return None
    
    async def _get_load_balanced_task(self) -> Optional[Task]:
        """获取负载均衡任务"""
        # 选择负载最轻的worker对应的任务
        # 这里简化处理，实际应该考虑worker的具体负载
        return await self._get_priority_task()
    
    def _get_task_by_id(self, task_id: str) -> Optional[Task]:
        """通过ID获取任务"""
        # 在各个队列中查找
        for queue in [self._pending_queue, self._scheduled_tasks.values()]:
            if isinstance(queue, dict):
                task = queue.get(task_id)
                if task:
                    return task
            else:
                for task in queue:
                    if task.id == task_id:
                        return task
        
        # 在优先级队列中查找
        for _, _, task in self._priority_queue:
            if task.id == task_id:
                return task
        
        return None
    
    async def _notify_scheduled_callbacks(self, task: Task):
        """通知任务调度回调"""
        for callback in self._task_scheduled_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(task)
                else:
                    callback(task)
            except Exception as e:
                self.logger.error(f"任务调度回调异常: {str(e)}")
    
    async def _notify_completed_callbacks(self, task: Task):
        """通知任务完成回调"""
        for callback in self._task_completed_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(task)
                else:
                    callback(task)
            except Exception as e:
                self.logger.error(f"任务完成回调异常: {str(e)}")


class DependencyResolver:
    """依赖解析器"""
    
    def __init__(self, scheduler: TaskScheduler):
        self.scheduler = scheduler
        self.logger = logging.getLogger(__name__)
    
    async def resolve_all_dependencies(self, tasks: List[Task]):
        """解析所有任务的依赖关系"""
        # 构建依赖图
        dependency_graph = self._build_dependency_graph(tasks)
        
        # 检测循环依赖
        cycles = self._detect_cycles(dependency_graph)
        if cycles:
            raise ValueError(f"检测到循环依赖: {cycles}")
        
        # 拓扑排序
        sorted_tasks = self._topological_sort(dependency_graph)
        
        # 按依赖顺序解析
        for task in sorted_tasks:
            await self.scheduler._resolve_dependencies(task)
    
    def _build_dependency_graph(self, tasks: List[Task]) -> Dict[str, Set[str]]:
        """构建依赖图"""
        graph = defaultdict(set)
        task_map = {task.id: task for task in tasks}
        
        for task in tasks:
            for dep_id in task.config.dependencies:
                if dep_id in task_map:
                    graph[task.id].add(dep_id)
        
        return graph
    
    def _detect_cycles(self, graph: Dict[str, Set[str]]) -> List[List[str]]:
        """检测循环依赖"""
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs(node: str, path: List[str]):
            if node in rec_stack:
                # 找到循环
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:] + [node])
                return
            
            if node in visited:
                return
            
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in graph.get(node, []):
                dfs(neighbor, path.copy())
            
            rec_stack.remove(node)
        
        for node in graph:
            if node not in visited:
                dfs(node, [])
        
        return cycles
    
    def _topological_sort(self, graph: Dict[str, Set[str]]) -> List[str]:
        """拓扑排序"""
        in_degree = defaultdict(int)
        
        # 计算入度
        for node in graph:
            for neighbor in graph[node]:
                in_degree[neighbor] += 1
        
        # 初始化队列
        queue = deque([node for node in graph if in_degree[node] == 0])
        result = []
        
        # Kahn算法
        while queue:
            node = queue.popleft()
            result.append(node)
            
            for neighbor in graph.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return result