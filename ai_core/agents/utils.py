"""
工具函数

提供异步任务机制相关的工具函数和装饰器
"""

import asyncio
import functools
import time
import logging
from typing import Any, Callable, Dict, List, Optional, Union
from datetime import datetime

from .task_models import Task, TaskStatus, TaskPriority, TaskConfig
from .task_executor import ProgressTracker


def async_task(
    name: str = "",
    priority: TaskPriority = TaskPriority.NORMAL,
    timeout: Optional[int] = None,
    retry_count: int = 3,
    retry_delay: float = 1.0,
    dependencies: List[str] = None,
    tags: List[str] = None,
    metadata: Dict[str, Any] = None
):
    """异步任务装饰器
    
    用于将函数包装为可管理的异步任务
    
    Args:
        name: 任务名称
        priority: 任务优先级
        timeout: 超时时间（秒）
        retry_count: 重试次数
        retry_delay: 重试延迟（秒）
        dependencies: 依赖的任务ID列表
        tags: 任务标签
        metadata: 任务元数据
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # 这里返回包装后的函数，实际的任务提交逻辑在AsyncTaskManager中处理
            return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # 同步版本的包装
            return func(*args, **kwargs)
        
        # 设置任务属性
        wrapper = async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        wrapper._task_config = {
            "name": name or func.__name__,
            "priority": priority,
            "timeout": timeout,
            "retry_count": retry_count,
            "retry_delay": retry_delay,
            "dependencies": dependencies or [],
            "tags": tags or [],
            "metadata": metadata or {}
        }
        
        return wrapper
    
    return decorator


def retry_on_failure(
    max_retries: int = 3,
    delay: float = 1.0,
    exceptions: tuple = (Exception,),
    backoff_factor: float = 2.0
):
    """重试装饰器
    
    Args:
        max_retries: 最大重试次数
        delay: 初始延迟时间（秒）
        exceptions: 需要重试的异常类型
        backoff_factor: 退避因子
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        # 同步函数在线程池中执行
                        loop = asyncio.get_event_loop()
                        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        break
                    
                    # 计算延迟时间（指数退避）
                    wait_time = delay * (backoff_factor ** attempt)
                    await asyncio.sleep(wait_time)
            
            # 所有重试都失败了
            raise last_exception
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        break
                    
                    # 计算延迟时间（指数退避）
                    wait_time = delay * (backoff_factor ** attempt)
                    time.sleep(wait_time)
            
            # 所有重试都失败了
            raise last_exception
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


def timeout(seconds: float):
    """超时装饰器
    
    Args:
        seconds: 超时时间（秒）
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await asyncio.wait_for(
                func(*args, **kwargs) if asyncio.iscoroutinefunction(func) 
                else asyncio.get_event_loop().run_in_executor(None, lambda: func(*args, **kwargs)),
                timeout=seconds
            )
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # 同步函数的超时处理比较复杂，这里简化处理
            return func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


def progress_tracker(func):
    """进度跟踪装饰器
    
    用于自动跟踪函数执行进度
    """
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        # 尝试从参数中获取任务对象
        task = None
        for arg in args:
            if isinstance(arg, Task):
                task = arg
                break
        
        for key, value in kwargs.items():
            if isinstance(value, Task):
                task = value
                break
        
        if task:
            tracker = ProgressTracker(task)
            # 设置进度更新函数
            original_update = getattr(func, '_progress_update', None)
            func._progress_update = tracker.update
            
            try:
                result = await func(*args, **kwargs)
                tracker.complete()
                return result
            except Exception as e:
                tracker.error(str(e))
                raise
            finally:
                if original_update:
                    func._progress_update = original_update
        else:
            # 没有任务对象，正常执行
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


def measure_time(func):
    """性能测量装饰器
    
    用于测量函数执行时间
    """
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: func(*args, **kwargs))
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # 记录执行时间
            logger = logging.getLogger(__name__)
            logger.info(f"函数 {func.__name__} 执行时间: {execution_time:.3f} 秒")
            
            return result
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            
            logger = logging.getLogger(__name__)
            logger.error(f"函数 {func.__name__} 执行失败，耗时: {execution_time:.3f} 秒，错误: {str(e)}")
            raise
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            
            logger = logging.getLogger(__name__)
            logger.info(f"函数 {func.__name__} 执行时间: {execution_time:.3f} 秒")
            
            return result
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            
            logger = logging.getLogger(__name__)
            logger.error(f"函数 {func.__name__} 执行失败，耗时: {execution_time:.3f} 秒，错误: {str(e)}")
            raise
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


def batch_process(
    batch_size: int = 100,
    max_workers: int = 10,
    progress_callback: Optional[Callable] = None
):
    """批处理装饰器
    
    用于批量处理数据
    
    Args:
        batch_size: 批处理大小
        max_workers: 最大工作线程数
        progress_callback: 进度回调函数
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(data_list, *args, **kwargs):
            results = []
            total_items = len(data_list)
            
            # 分批处理
            for i in range(0, total_items, batch_size):
                batch = data_list[i:i + batch_size]
                
                # 并发处理当前批次
                if asyncio.iscoroutinefunction(func):
                    batch_results = await asyncio.gather(*[func(item, *args, **kwargs) for item in batch])
                else:
                    loop = asyncio.get_event_loop()
                    batch_results = await asyncio.gather(*[
                        loop.run_in_executor(None, lambda item=item: func(item, *args, **kwargs)) 
                        for item in batch
                    ])
                
                results.extend(batch_results)
                
                # 更新进度
                if progress_callback:
                    progress = min(100, (i + len(batch)) / total_items * 100)
                    await progress_callback(progress, f"处理进度: {i + len(batch)}/{total_items}")
            
            return results
        
        return async_wrapper
    return decorator


def rate_limit(calls_per_second: float):
    """速率限制装饰器
    
    Args:
        calls_per_second: 每秒最大调用次数
    """
    min_interval = 1.0 / calls_per_second
    
    def decorator(func):
        last_called = [0.0]  # 使用列表来避免闭包问题
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            
            if left_to_wait > 0:
                await asyncio.sleep(left_to_wait)
            
            last_called[0] = time.time()
            
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            
            last_called[0] = time.time()
            return func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


def memoize(ttl: Optional[float] = None):
    """记忆化装饰器
    
    Args:
        ttl: 缓存生存时间（秒）
    """
    def decorator(func):
        cache = {}
        cache_times = {}
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # 生成缓存键
            key = str(args) + str(sorted(kwargs.items()))
            
            # 检查缓存
            if key in cache:
                # 检查TTL
                if ttl is None or time.time() - cache_times[key] < ttl:
                    return cache[key]
            
            # 执行函数
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: func(*args, **kwargs))
            
            # 更新缓存
            cache[key] = result
            cache_times[key] = time.time()
            
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # 生成缓存键
            key = str(args) + str(sorted(kwargs.items()))
            
            # 检查缓存
            if key in cache:
                # 检查TTL
                if ttl is None or time.time() - cache_times[key] < ttl:
                    return cache[key]
            
            # 执行函数
            result = func(*args, **kwargs)
            
            # 更新缓存
            cache[key] = result
            cache_times[key] = time.time()
            
            return result
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


class TaskContext:
    """任务上下文管理器"""
    
    def __init__(self, task: Task):
        self.task = task
        self.logger = logging.getLogger(f"task.{task.id}")
    
    async def __aenter__(self):
        self.logger.info(f"开始执行任务: {self.task.name}")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.logger.error(f"任务执行异常: {exc_val}")
            self.task.update_status(TaskStatus.FAILED)
        else:
            self.logger.info(f"任务执行完成: {self.task.name}")
            self.task.update_status(TaskStatus.COMPLETED)
    
    def update_progress(self, progress: float, message: str = ""):
        """更新进度"""
        self.task.update_progress(progress)
        if message:
            self.logger.info(f"进度: {progress:.1f}% - {message}")
    
    def log_info(self, message: str):
        """记录信息日志"""
        self.logger.info(message)
    
    def log_warning(self, message: str):
        """记录警告日志"""
        self.logger.warning(message)
    
    def log_error(self, message: str):
        """记录错误日志"""
        self.logger.error(message)


def create_task_context(task: Task) -> TaskContext:
    """创建任务上下文"""
    return TaskContext(task)


# 工具函数
def format_duration(seconds: float) -> str:
    """格式化时长"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}分{secs:.1f}秒"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}小时{minutes}分{secs:.1f}秒"


def format_file_size(size_bytes: int) -> str:
    """格式化文件大小"""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"


def calculate_progress(completed: int, total: int) -> float:
    """计算进度百分比"""
    if total == 0:
        return 0.0
    return min(100.0, (completed / total) * 100.0)


def is_task_urgent(task: Task) -> bool:
    """判断任务是否紧急"""
    return task.config.priority in [TaskPriority.URGENT, TaskPriority.CRITICAL]


def get_task_urgency_score(task: Task) -> float:
    """获取任务紧急度评分"""
    base_score = task.config.priority.value
    
    # 根据等待时间增加评分
    wait_time = (datetime.now() - task.created_at).total_seconds()
    time_bonus = min(wait_time / 3600, 2)  # 最多增加2分
    
    return base_score + time_bonus


def validate_task_config(config: TaskConfig) -> List[str]:
    """验证任务配置"""
    errors = []
    
    if config.timeout is not None and config.timeout <= 0:
        errors.append("超时时间必须大于0")
    
    if config.retry_count < 0:
        errors.append("重试次数不能为负数")
    
    if config.retry_delay < 0:
        errors.append("重试延迟不能为负数")
    
    if config.max_concurrent <= 0:
        errors.append("最大并发数必须大于0")
    
    return errors