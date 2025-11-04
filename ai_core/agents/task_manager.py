"""
异步任务管理器

整合所有组件，提供完整的异步任务管理功能
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, Union
from datetime import datetime

from .task_models import (
    Task, TaskResult, TaskStatus, TaskPriority, TaskConfig, TaskRegistry
)
from .task_executor import TaskExecutor
from .scheduler import TaskScheduler, SchedulingStrategy, DependencyResolver
from .cache_manager import CacheManager, CacheStrategy, TaskResultCache
from .storage import TaskStorage, JSONStorage
from .monitor import TaskMonitor, MonitorConfig


class AsyncTaskManager:
    """异步任务管理器
    
    提供完整的异步任务管理功能，包括：
    - 任务创建、提交、执行、取消
    - 任务调度和优先级管理
    - 任务状态跟踪和进度监控
    - 任务结果缓存和持久化
    - 依赖关系管理
    - 性能监控和告警
    """
    
    def __init__(
        self,
        max_concurrent: int = 10,
        cache_enabled: bool = True,
        cache_size: int = 1000,
        cache_strategy: CacheStrategy = CacheStrategy.LRU,
        storage_enabled: bool = True,
        storage_type: str = "sqlite",  # sqlite or json
        storage_path: str = "tasks.db",
        monitoring_enabled: bool = True,
        monitoring_config: MonitorConfig = None,
        scheduling_strategy: SchedulingStrategy = SchedulingStrategy.PRIORITY
    ):
        # 配置参数
        self.max_concurrent = max_concurrent
        self.cache_enabled = cache_enabled
        self.storage_enabled = storage_enabled
        self.monitoring_enabled = monitoring_enabled
        self.scheduling_strategy = scheduling_strategy
        
        # 日志
        self.logger = logging.getLogger(__name__)
        
        # 核心组件
        self.task_registry = TaskRegistry()
        self.task_executor = TaskExecutor()
        self.task_scheduler = TaskScheduler(
            max_concurrent=max_concurrent,
            strategy=scheduling_strategy
        )
        
        # 缓存组件
        if cache_enabled:
            self.cache_manager = CacheManager(
                max_size=cache_size,
                strategy=cache_strategy,
                enable_persistence=True
            )
            self.result_cache = TaskResultCache(self.cache_manager)
        else:
            self.cache_manager = None
            self.result_cache = None
        
        # 存储组件
        if storage_enabled:
            if storage_type.lower() == "sqlite":
                self.storage = TaskStorage(db_path=storage_path)
            else:
                self.storage = JSONStorage(storage_path=storage_path)
        else:
            self.storage = None
        
        # 监控组件
        if monitoring_enabled:
            self.monitor = TaskMonitor(monitoring_config or MonitorConfig())
            self.monitor.set_task_registry(self.task_registry)
        else:
            self.monitor = None
        
        # 依赖解析器
        self.dependency_resolver = DependencyResolver(self.task_scheduler)
        
        # 运行状态
        self.is_running = False
        self._main_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """启动任务管理器"""
        if self.is_running:
            return
        
        self.logger.info("启动异步任务管理器...")
        
        # 启动各个组件
        await self.task_scheduler.start()
        
        if self.cache_manager:
            await self.cache_manager.load_from_disk()
        
        if self.storage:
            await self.storage.start()
        
        if self.monitor:
            await self.monitor.start()
        
        # 设置调度器回调
        self.task_scheduler.add_task_scheduled_callback(self._on_task_scheduled)
        self.task_scheduler.add_task_completed_callback(self._on_task_completed)
        
        # 启动主循环
        self._main_task = asyncio.create_task(self._main_loop())
        
        self.is_running = True
        self.logger.info("异步任务管理器已启动")
    
    async def stop(self):
        """停止任务管理器"""
        if not self.is_running:
            return
        
        self.logger.info("停止异步任务管理器...")
        
        self.is_running = False
        
        # 停止主循环
        if self._main_task:
            self._main_task.cancel()
            try:
                await self._main_task
            except asyncio.CancelledError:
                pass
        
        # 停止各个组件
        await self.task_scheduler.stop()
        
        if self.cache_manager:
            await self.cache_manager.save_to_disk()
        
        if self.storage:
            await self.storage.stop()
        
        if self.monitor:
            await self.monitor.stop()
        
        self.logger.info("异步任务管理器已停止")
    
    async def submit_task(
        self,
        func: Callable,
        *args,
        name: str = "",
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: Optional[int] = None,
        retry_count: int = 3,
        retry_delay: float = 1.0,
        dependencies: List[str] = None,
        tags: List[str] = None,
        metadata: Dict[str, Any] = None,
        **kwargs
    ) -> str:
        """提交任务
        
        Args:
            func: 要执行的函数
            *args: 函数参数
            name: 任务名称
            priority: 任务优先级
            timeout: 超时时间（秒）
            retry_count: 重试次数
            retry_delay: 重试延迟（秒）
            dependencies: 依赖的任务ID列表
            tags: 任务标签
            metadata: 任务元数据
            **kwargs: 函数关键字参数
        
        Returns:
            任务ID
        """
        # 创建任务配置
        config = TaskConfig(
            timeout=timeout,
            retry_count=retry_count,
            retry_delay=retry_delay,
            priority=priority,
            dependencies=dependencies or [],
            tags=tags or [],
            metadata=metadata or {}
        )
        
        # 创建任务
        task = Task(
            name=name or f"Task_{func.__name__}",
            func=func,
            args=args,
            kwargs=kwargs,
            config=config
        )
        
        # 添加到注册表
        await self.task_registry.add_task(task)
        
        # 添加依赖关系
        for dep_id in config.dependencies:
            self.task_scheduler.add_dependency(task.id, dep_id)
        
        # 提交到调度器
        task_id = await self.task_scheduler.submit_task(task)
        
        self.logger.info(f"任务已提交: {task_id} ({task.name})")
        return task_id
    
    async def submit_batch_tasks(
        self,
        tasks: List[Dict[str, Any]]
    ) -> List[str]:
        """批量提交任务
        
        Args:
            tasks: 任务配置列表，每个元素包含：
                - func: 要执行的函数
                - args: 函数参数（可选）
                - kwargs: 函数关键字参数（可选）
                - name: 任务名称（可选）
                - priority: 任务优先级（可选）
                - timeout: 超时时间（可选）
                - retry_count: 重试次数（可选）
                - dependencies: 依赖任务ID列表（可选）
        
        Returns:
            任务ID列表
        """
        task_ids = []
        
        for task_config in tasks:
            task_id = await self.submit_task(**task_config)
            task_ids.append(task_id)
        
        self.logger.info(f"批量提交了 {len(task_ids)} 个任务")
        return task_ids
    
    async def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        # 从调度器取消
        cancelled = await self.task_scheduler.cancel_task(task_id)
        
        # 从执行器取消
        if cancelled:
            await self.task_executor.cancel_task(task_id)
            
            # 更新任务状态
            task = await self.task_registry.get_task(task_id)
            if task:
                old_status = task.status
                task.update_status(TaskStatus.CANCELLED)
                await self.task_registry.update_task(task)
                
                # 通知监控器
                if self.monitor:
                    await self.monitor.track_task_status(task, old_status, TaskStatus.CANCELLED)
                
                # 保存到存储
                if self.storage:
                    await self.storage.save_task(task)
        
        return cancelled
    
    async def pause_task(self, task_id: str) -> bool:
        """暂停任务"""
        return await self.task_scheduler.pause_task(task_id)
    
    async def resume_task(self, task_id: str) -> bool:
        """恢复任务"""
        return await self.task_scheduler.resume_task(task_id)
    
    async def get_task(self, task_id: str) -> Optional[Task]:
        """获取任务"""
        return await self.task_registry.get_task(task_id)
    
    async def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """获取任务结果"""
        # 首先检查缓存
        if self.result_cache:
            cached_result = await self.result_cache.get_cached_result(task_id)
            if cached_result:
                return cached_result
        
        # 从任务中获取结果
        task = await self.task_registry.get_task(task_id)
        if not task:
            return None
        
        # 检查是否需要执行
        if task.status == TaskStatus.COMPLETED and task.result is not None:
            result = TaskResult(
                task_id=task.id,
                status=task.status,
                result=task.result,
                metrics=task.metrics
            )
            
            # 缓存结果
            if self.result_cache:
                await self.result_cache.cache_task_result(task, result)
            
            return result
        
        return None
    
    async def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Optional[TaskResult]:
        """等待任务完成"""
        try:
            # 等待任务完成
            await asyncio.wait_for(self._wait_for_task_completion(task_id), timeout=timeout)
            return await self.get_task_result(task_id)
        except asyncio.TimeoutError:
            self.logger.warning(f"等待任务超时: {task_id}")
            return None
    
    async def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Task]:
        """列出任务"""
        return await self.task_registry.list_tasks(status=status)
    
    async def get_running_tasks(self) -> List[Task]:
        """获取运行中的任务"""
        return await self.task_registry.get_running_tasks()
    
    async def get_pending_tasks(self) -> List[Task]:
        """获取等待中的任务"""
        return await self.task_registry.get_pending_tasks()
    
    async def clear_completed_tasks(self) -> int:
        """清理已完成的任务"""
        completed_tasks = await self.list_tasks(status=TaskStatus.COMPLETED)
        cleared_count = 0
        
        for task in completed_tasks:
            if await self.task_registry.remove_task(task.id):
                cleared_count += 1
        
        self.logger.info(f"已清理 {cleared_count} 个已完成的任务")
        return cleared_count
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """获取队列状态"""
        scheduler_status = await self.task_scheduler.get_queue_status()
        
        # 添加执行器状态
        running_tasks = await self.task_executor.get_running_tasks()
        
        # 添加缓存状态
        cache_stats = {}
        if self.cache_manager:
            cache_stats = await self.cache_manager.get_stats()
        
        # 添加监控状态
        monitor_stats = {}
        if self.monitor:
            monitor_stats = await self.monitor.get_performance_stats()
        
        return {
            "scheduler": scheduler_status,
            "executor": {
                "running_tasks": len(running_tasks),
                "running_task_ids": running_tasks
            },
            "cache": cache_stats,
            "monitor": monitor_stats,
            "total_tasks": len(await self.task_registry.list_tasks()),
            "is_running": self.is_running
        }
    
    async def export_tasks(self, file_path: str, format: str = "json") -> bool:
        """导出任务"""
        if self.storage:
            return await self.storage.export_tasks(file_path, format)
        return False
    
    async def import_tasks(self, file_path: str, format: str = "json") -> int:
        """导入任务"""
        if self.storage:
            return await self.storage.import_tasks(file_path, format)
        return 0
    
    async def create_backup(self, backup_path: Optional[str] = None) -> str:
        """创建备份"""
        if self.storage and hasattr(self.storage, 'create_backup'):
            return await self.storage.create_backup(backup_path)
        raise NotImplementedError("当前存储类型不支持备份功能")
    
    def get_cache_manager(self) -> Optional[CacheManager]:
        """获取缓存管理器"""
        return self.cache_manager
    
    def get_storage(self) -> Optional[Union[TaskStorage, JSONStorage]]:
        """获取存储管理器"""
        return self.storage
    
    def get_monitor(self) -> Optional[TaskMonitor]:
        """获取监控器"""
        return self.monitor
    
    def get_scheduler(self) -> TaskScheduler:
        """获取调度器"""
        return self.task_scheduler
    
    def get_executor(self) -> TaskExecutor:
        """获取执行器"""
        return self.task_executor
    
    async def _main_loop(self):
        """主循环"""
        while self.is_running:
            try:
                # 获取下一个要执行的任务
                next_task = await self.task_scheduler.get_next_task()
                
                if next_task:
                    # 执行任务
                    asyncio.create_task(self._execute_task(next_task))
                
                # 短暂休眠
                await asyncio.sleep(0.01)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"主循环异常: {str(e)}")
                await asyncio.sleep(1)
    
    async def _execute_task(self, task: Task):
        """执行任务"""
        try:
            # 检查缓存
            if self.result_cache and await self.result_cache.is_result_cached(task.id):
                self.logger.info(f"任务结果来自缓存: {task.id}")
                result = await self.result_cache.get_cached_result(task.id)
                if result:
                    task.update_status(TaskStatus.COMPLETED)
                    task.result = result.result
                    await self.task_registry.update_task(task)
                    return
            
            # 执行任务
            result = await self.task_executor.execute_task(task)
            
            # 更新任务状态
            await self.task_registry.update_task(task)
            
            # 保存结果到缓存
            if self.result_cache and result.status == TaskStatus.COMPLETED:
                await self.result_cache.cache_task_result(task, result)
            
            # 保存到存储
            if self.storage:
                await self.storage.save_task(task)
                await self.storage.save_task_result(result)
            
            # 通知调度器任务完成
            await self.task_scheduler.mark_task_completed(task.id)
            
        except Exception as e:
            self.logger.error(f"执行任务异常: {task.id}, 错误: {str(e)}")
    
    async def _wait_for_task_completion(self, task_id: str):
        """等待任务完成"""
        while True:
            task = await self.task_registry.get_task(task_id)
            if not task:
                raise ValueError(f"任务不存在: {task_id}")
            
            if task.is_finished():
                break
            
            await asyncio.sleep(0.1)
    
    async def _on_task_scheduled(self, task: Task):
        """任务调度回调"""
        self.logger.debug(f"任务已调度: {task.id}")
    
    async def _on_task_completed(self, task: Task):
        """任务完成回调"""
        self.logger.info(f"任务已完成: {task.id} - {task.name}")
        
        # 通知监控器
        if self.monitor:
            old_status = TaskStatus.RUNNING  # 假设之前是运行状态
            await self.monitor.track_task_status(task, old_status, task.status)


# 便捷函数
def create_task_manager(
    max_concurrent: int = 10,
    cache_size: int = 1000,
    storage_path: str = "tasks.db",
    monitoring_enabled: bool = True
) -> AsyncTaskManager:
    """创建任务管理器的便捷函数"""
    return AsyncTaskManager(
        max_concurrent=max_concurrent,
        cache_enabled=True,
        cache_size=cache_size,
        storage_enabled=True,
        storage_path=storage_path,
        monitoring_enabled=monitoring_enabled
    )


async def quick_task(func: Callable, *args, **kwargs) -> Any:
    """快速执行单个任务的便捷函数"""
    if asyncio.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    else:
        # 同步函数在线程池中执行
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))