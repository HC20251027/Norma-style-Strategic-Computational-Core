"""
任务执行器

负责执行具体的异步任务，包括重试、超时处理、进度跟踪等功能
"""

import asyncio
import functools
import traceback
import psutil
import logging
from typing import Any, Callable, Optional
from datetime import datetime

from .task_models import Task, TaskStatus, TaskResult, TaskMetrics


class TaskExecutor:
    """任务执行器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._running_tasks: dict[str, asyncio.Task] = {}
        self._cancel_events: dict[str, asyncio.Event] = {}
    
    async def execute_task(self, task: Task) -> TaskResult:
        """执行任务"""
        try:
            # 检查任务是否可执行
            if not task.can_run():
                return TaskResult(
                    task_id=task.id,
                    status=task.status,
                    error=f"任务不可执行，状态: {task.status.value}"
                )
            
            # 更新任务状态为运行中
            task.update_status(TaskStatus.RUNNING)
            
            # 创建取消事件
            cancel_event = asyncio.Event()
            self._cancel_events[task.id] = cancel_event
            
            # 创建执行任务
            execution_task = asyncio.create_task(
                self._execute_with_monitoring(task, cancel_event),
                name=f"task_{task.id}"
            )
            self._running_tasks[task.id] = execution_task
            
            # 等待任务完成或取消
            try:
                result = await execution_task
                return result
            except asyncio.CancelledError:
                task.update_status(TaskStatus.CANCELLED)
                return TaskResult(
                    task_id=task.id,
                    status=TaskStatus.CANCELLED,
                    error="任务被取消"
                )
            finally:
                # 清理资源
                self._running_tasks.pop(task.id, None)
                self._cancel_events.pop(task.id, None)
                
        except Exception as e:
            task.update_status(TaskStatus.FAILED)
            self.logger.error(f"任务执行异常: {task.id}, 错误: {str(e)}")
            return TaskResult(
                task_id=task.id,
                status=TaskStatus.FAILED,
                error=str(e),
                traceback=traceback.format_exc()
            )
    
    async def _execute_with_monitoring(self, task: Task, cancel_event: asyncio.Event) -> TaskResult:
        """带监控的任务执行"""
        start_time = datetime.now()
        
        try:
            # 如果有超时设置，创建超时任务
            if task.config.timeout:
                timeout_task = asyncio.create_task(
                    asyncio.sleep(task.config.timeout)
                )
                
                # 等待任务完成或超时
                done, pending = await asyncio.wait(
                    [self._run_task_func(task, cancel_event), timeout_task],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # 取消未完成的任务
                for p in pending:
                    p.cancel()
                
                if timeout_task in done:
                    task.update_status(TaskStatus.TIMEOUT)
                    return TaskResult(
                        task_id=task.id,
                        status=TaskStatus.TIMEOUT,
                        error="任务执行超时"
                    )
                
                # 获取任务结果
                result = list(done)[0].result()
            else:
                result = await self._run_task_func(task, cancel_event)
            
            # 任务成功完成
            task.update_status(TaskStatus.COMPLETED)
            task.result = result
            
            return TaskResult(
                task_id=task.id,
                status=TaskStatus.COMPLETED,
                result=result,
                metrics=task.metrics
            )
            
        except asyncio.CancelledError:
            raise
        except Exception as e:
            task.update_status(TaskStatus.FAILED)
            return TaskResult(
                task_id=task.id,
                status=TaskStatus.FAILED,
                error=str(e),
                traceback=traceback.format_exc(),
                metrics=task.metrics
            )
    
    async def _run_task_func(self, task: Task, cancel_event: asyncio.Event) -> Any:
        """运行任务函数"""
        if not task.func:
            raise ValueError("任务函数未设置")
        
        # 包装函数以支持进度更新和取消检查
        wrapped_func = self._wrap_task_function(task.func, task, cancel_event)
        
        # 执行函数
        if asyncio.iscoroutinefunction(task.func):
            return await wrapped_func(*task.args, **task.kwargs)
        else:
            # 同步函数在线程池中执行
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, lambda: wrapped_func(*task.args, **task.kwargs)
            )
    
    def _wrap_task_function(self, func: Callable, task: Task, cancel_event: asyncio.Event) -> Callable:
        """包装任务函数以支持进度更新和取消检查"""
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 设置取消检查
            original_check = getattr(func, '_cancel_check', None)
            func._cancel_check = lambda: cancel_event.is_set()
            
            # 设置进度更新
            original_update = getattr(func, '_progress_update', None)
            func._progress_update = lambda progress: task.update_progress(progress)
            
            try:
                # 执行原始函数
                result = func(*args, **kwargs)
                
                # 如果是生成器或迭代器，需要特殊处理
                if hasattr(result, '__iter__') and not isinstance(result, (str, bytes)):
                    return self._handle_iterable_result(result, task, cancel_event)
                
                return result
                
            finally:
                # 恢复原始属性
                if original_check:
                    func._cancel_check = original_check
                if original_update:
                    func._progress_update = original_update
        
        return wrapper
    
    async def _handle_iterable_result(self, result, task: Task, cancel_event: asyncio.Event):
        """处理可迭代结果（如生成器）"""
        try:
            processed_results = []
            for i, item in enumerate(result):
                # 检查取消
                if cancel_event.is_set():
                    raise asyncio.CancelledError()
                
                # 更新进度
                progress = (i + 1) / len(result) * 100 if hasattr(result, '__len__') else 0
                task.update_progress(progress)
                
                processed_results.append(item)
                
                # 允许其他任务执行
                await asyncio.sleep(0)
            
            return processed_results
            
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.logger.error(f"处理可迭代结果时出错: {str(e)}")
            raise
    
    async def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        if task_id in self._running_tasks:
            execution_task = self._running_tasks[task_id]
            execution_task.cancel()
            
            # 设置取消事件
            if task_id in self._cancel_events:
                self._cancel_events[task_id].set()
            
            return True
        return False
    
    async def pause_task(self, task_id: str) -> bool:
        """暂停任务（通过取消实现）"""
        # 注意：真正的暂停需要更复杂的实现
        # 这里简化处理，实际应该实现真正的暂停/恢复机制
        return await self.cancel_task(task_id)
    
    async def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """获取任务状态"""
        if task_id in self._running_tasks:
            task = self._running_tasks[task_id]
            if task.done():
                if task.cancelled():
                    return TaskStatus.CANCELLED
                elif task.exception():
                    return TaskStatus.FAILED
                else:
                    return TaskStatus.COMPLETED
            else:
                return TaskStatus.RUNNING
        return None
    
    async def get_running_tasks(self) -> list[str]:
        """获取运行中的任务ID列表"""
        return list(self._running_tasks.keys())
    
    async def get_system_metrics(self) -> dict:
        """获取系统指标"""
        try:
            return {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent,
                "running_tasks": len(self._running_tasks)
            }
        except Exception as e:
            self.logger.error(f"获取系统指标失败: {str(e)}")
            return {}


class ProgressTracker:
    """进度跟踪器"""
    
    def __init__(self, task: Task):
        self.task = task
        self.logger = logging.getLogger(__name__)
    
    def update(self, progress: float, message: str = ""):
        """更新进度"""
        self.task.update_progress(progress)
        if message:
            self.logger.info(f"任务 {self.task.id} 进度: {progress:.1f}% - {message}")
    
    def complete(self, message: str = ""):
        """标记完成"""
        self.task.update_progress(100.0)
        if message:
            self.logger.info(f"任务 {self.task.id} 完成: {message}")
    
    def error(self, message: str = ""):
        """标记错误"""
        if message:
            self.logger.error(f"任务 {self.task.id} 错误: {message}")


def create_progress_tracker(task: Task) -> ProgressTracker:
    """创建进度跟踪器"""
    return ProgressTracker(task)


def progress_callback(task: Task):
    """进度回调装饰器"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 设置进度更新函数
            def update_progress(progress: float):
                task.update_progress(progress)
            
            # 临时设置进度更新函数
            original_update = getattr(func, '_progress_update', None)
            func._progress_update = update_progress
            
            try:
                result = func(*args, **kwargs)
                
                # 如果是生成器，包装以支持进度更新
                if hasattr(result, '__iter__') and not isinstance(result, (str, bytes)):
                    return _wrap_iterable_with_progress(result, task)
                
                return result
                
            finally:
                if original_update:
                    func._progress_update = original_update
        
        return wrapper
    return decorator


def _wrap_iterable_with_progress(iterable, task: Task):
    """包装可迭代对象以支持进度更新"""
    try:
        items = list(iterable)
        total = len(items)
        
        for i, item in enumerate(items):
            progress = (i + 1) / total * 100
            task.update_progress(progress)
            yield item
            
    except Exception as e:
        task.update_progress(0)
        raise