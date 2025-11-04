"""
非阻塞等待系统超时和失败处理器
"""

import asyncio
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass
from enum import Enum
import logging

from .config import NonBlockingConfig, TaskPriority
from .event_system import event_system, EventType
from .task_manager import TaskManager, Task, TaskStatus


class FailureType(Enum):
    """失败类型"""
    TIMEOUT = "timeout"
    EXCEPTION = "exception"
    CANCELLATION = "cancellation"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    DEPENDENCY_FAILURE = "dependency_failure"
    USER_CANCELLED = "user_cancelled"
    SYSTEM_ERROR = "system_error"


class RetryStrategy(Enum):
    """重试策略"""
    IMMEDIATE = "immediate"  # 立即重试
    LINEAR = "linear"        # 线性退避
    EXPONENTIAL = "exponential"  # 指数退避
    FIXED = "fixed"          # 固定间隔


@dataclass
class TimeoutConfig:
    """超时配置"""
    soft_timeout: Optional[float] = None  # 软超时（警告）
    hard_timeout: float = 300.0           # 硬超时（终止）
    timeout_action: str = "warn"          # 超时动作：warn, cancel, retry
    max_timeout_extensions: int = 3       # 最大超时扩展次数


@dataclass
class FailureInfo:
    """失败信息"""
    task_id: str
    failure_type: FailureType
    error_message: str
    timestamp: datetime
    retry_count: int = 0
    recovery_suggestions: List[str] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "failure_type": self.failure_type.value,
            "error_message": self.error_message,
            "timestamp": self.timestamp.isoformat(),
            "retry_count": self.retry_count,
            "recovery_suggestions": self.recovery_suggestions or [],
            "metadata": self.metadata or {}
        }


class RetryManager:
    """重试管理器"""
    
    def __init__(self, config: NonBlockingConfig):
        self.config = config
        self.retry_policies: Dict[str, Dict[str, Any]] = {}
        self.failure_history: Dict[str, List[FailureInfo]] = {}
        
    def set_retry_policy(self, task_id: str, strategy: RetryStrategy, max_retries: int = None, base_delay: float = None) -> None:
        """设置重试策略"""
        self.retry_policies[task_id] = {
            "strategy": strategy,
            "max_retries": max_retries or self.config.max_retries,
            "base_delay": base_delay or self.config.retry_delay_base,
            "multiplier": self.config.retry_delay_multiplier
        }
    
    def calculate_retry_delay(self, task_id: str, retry_count: int) -> float:
        """计算重试延迟"""
        policy = self.retry_policies.get(task_id, {})
        strategy = policy.get("strategy", RetryStrategy.EXPONENTIAL)
        base_delay = policy.get("base_delay", 1.0)
        multiplier = policy.get("multiplier", 2.0)
        
        if strategy == RetryStrategy.IMMEDIATE:
            return 0
        elif strategy == RetryStrategy.LINEAR:
            return base_delay * retry_count
        elif strategy == RetryStrategy.EXPONENTIAL:
            return base_delay * (multiplier ** retry_count)
        elif strategy == RetryStrategy.FIXED:
            return base_delay
        else:
            return base_delay
    
    def should_retry(self, task_id: str, failure_info: FailureInfo) -> bool:
        """判断是否应该重试"""
        policy = self.retry_policies.get(task_id, {})
        max_retries = policy.get("max_retries", self.config.max_retries)
        
        # 检查重试次数
        if failure_info.retry_count >= max_retries:
            return False
        
        # 根据失败类型决定是否重试
        non_retryable_failures = {
            FailureType.USER_CANCELLATION,
            FailureType.RESOURCE_EXHAUSTION,
            FailureType.SYSTEM_ERROR
        }
        
        if failure_info.failure_type in non_retryable_failures:
            return False
        
        return True
    
    def record_failure(self, failure_info: FailureInfo) -> None:
        """记录失败信息"""
        if failure_info.task_id not in self.failure_history:
            self.failure_history[failure_info.task_id] = []
        
        self.failure_history[failure_info.task_id].append(failure_info)
        
        # 保持历史记录在合理范围内
        if len(self.failure_history[failure_info.task_id]) > 50:
            self.failure_history[failure_info.task_id].pop(0)
    
    def get_failure_pattern(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取失败模式"""
        failures = self.failure_history.get(task_id, [])
        if not failures:
            return None
        
        # 分析失败模式
        failure_types = [f.failure_type for f in failures]
        error_messages = [f.error_message for f in failures]
        
        most_common_failure = max(set(failure_types), key=failure_types.count) if failure_types else None
        
        return {
            "total_failures": len(failures),
            "most_common_failure": most_common_failure.value if most_common_failure else None,
            "unique_error_messages": len(set(error_messages)),
            "recent_failures": len([f for f in failures if f.timestamp > datetime.now(timezone.utc) - timedelta(hours=1)])
        }


class TimeoutHandler:
    """超时处理器"""
    
    def __init__(self, task_manager: TaskManager, config: Optional[NonBlockingConfig] = None):
        self.task_manager = task_manager
        self.config = config or NonBlockingConfig()
        self.retry_manager = RetryManager(self.config)
        self.timeout_configs: Dict[str, TimeoutConfig] = {}
        self.timeout_tasks: Dict[str, asyncio.Task] = {}
        self.running = False
        
    async def start(self) -> None:
        """启动超时处理器"""
        self.running = True
        logging.info("TimeoutHandler started")
    
    async def stop(self) -> None:
        """停止超时处理器"""
        self.running = False
        
        # 取消所有超时任务
        for task in self.timeout_tasks.values():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self.timeout_tasks.clear()
        logging.info("TimeoutHandler stopped")
    
    def configure_task_timeout(self, task_id: str, timeout_config: TimeoutConfig) -> None:
        """配置任务超时"""
        self.timeout_configs[task_id] = timeout_config
    
    async def start_task_monitoring(self, task_id: str) -> None:
        """开始监控任务超时"""
        task = self.task_manager.get_task(task_id)
        if not task:
            return
        
        timeout_config = self.timeout_configs.get(task_id, TimeoutConfig())
        
        # 创建超时监控任务
        timeout_task = asyncio.create_task(self._monitor_task_timeout(task_id, timeout_config))
        self.timeout_tasks[task_id] = timeout_task
    
    async def _monitor_task_timeout(self, task_id: str, config: TimeoutConfig) -> None:
        """监控任务超时"""
        task = self.task_manager.get_task(task_id)
        if not task:
            return
        
        start_time = time.time()
        soft_timeout_triggered = False
        timeout_extensions = 0
        
        while self.running and task.status == TaskStatus.RUNNING:
            elapsed = time.time() - start_time
            
            # 检查软超时
            if config.soft_timeout and not soft_timeout_triggered and elapsed >= config.soft_timeout:
                soft_timeout_triggered = True
                await self._handle_soft_timeout(task_id, elapsed)
            
            # 检查硬超时
            if elapsed >= config.hard_timeout:
                await self._handle_hard_timeout(task_id, elapsed, config, timeout_extensions)
                break
            
            await asyncio.sleep(1.0)  # 每秒检查一次
        
        # 清理超时任务
        self.timeout_tasks.pop(task_id, None)
    
    async def _handle_soft_timeout(self, task_id: str, elapsed: float) -> None:
        """处理软超时"""
        await event_system.emit_ux_warning(
            "soft_timeout",
            f"任务执行时间较长 ({elapsed:.1f}秒)",
            {"task_id": task_id, "elapsed_time": elapsed}
        )
        
        # 发送警告给用户
        task = self.task_manager.get_task(task_id)
        if task and task.correlation_id:
            await event_system.emit_ux_feedback(
                "timeout_warning",
                "任务执行时间较长，请耐心等待...",
                {"task_id": task_id, "elapsed_time": elapsed},
                task.correlation_id
            )
    
    async def _handle_hard_timeout(self, task_id: str, elapsed: float, config: TimeoutConfig, timeout_extensions: int) -> None:
        """处理硬超时"""
        task = self.task_manager.get_task(task_id)
        if not task:
            return
        
        if config.timeout_action == "cancel" or timeout_extensions >= config.max_timeout_extensions:
            # 取消任务
            await self.task_manager.cancel_task(task_id)
            
            failure_info = FailureInfo(
                task_id=task_id,
                failure_type=FailureType.TIMEOUT,
                error_message=f"任务超时 ({elapsed:.1f}秒)",
                timestamp=datetime.now(timezone.utc),
                retry_count=task.retry_count
            )
            
            self.retry_manager.record_failure(failure_info)
            
            await event_system.emit_task_timeout(task_id, failure_info.error_message, task.correlation_id)
            
        elif config.timeout_action == "retry" and timeout_extensions < config.max_timeout_extensions:
            # 扩展超时并重试
            timeout_extensions += 1
            new_hard_timeout = config.hard_timeout * 1.5  # 增加50%超时时间
            
            updated_config = TimeoutConfig(
                soft_timeout=config.soft_timeout,
                hard_timeout=new_hard_timeout,
                timeout_action=config.timeout_action,
                max_timeout_extensions=config.max_timeout_extensions
            )
            
            self.timeout_configs[task_id] = updated_config
            
            await event_system.emit_ux_feedback(
                "timeout_extension",
                f"超时时间已延长，正在继续执行任务",
                {"task_id": task_id, "extension_count": timeout_extensions, "new_timeout": new_hard_timeout},
                task.correlation_id
            )
            
            # 重新开始监控
            await self.start_task_monitoring(task_id)
    
    async def handle_task_failure(self, task_id: str, error: Exception, failure_type: FailureType = FailureType.EXCEPTION) -> bool:
        """处理任务失败"""
        task = self.task_manager.get_task(task_id)
        if not task:
            return False
        
        failure_info = FailureInfo(
            task_id=task_id,
            failure_type=failure_type,
            error_message=str(error),
            timestamp=datetime.now(timezone.utc),
            retry_count=task.retry_count,
            metadata={
                "task_name": task.name,
                "task_priority": task.priority.value,
                "elapsed_time": task.duration
            }
        )
        
        # 添加恢复建议
        failure_info.recovery_suggestions = self._generate_recovery_suggestions(failure_info)
        
        # 记录失败
        self.retry_manager.record_failure(failure_info)
        
        # 判断是否应该重试
        if self.retry_manager.should_retry(task_id, failure_info):
            delay = self.retry_manager.calculate_retry_delay(task_id, task.retry_count)
            
            # 发送重试事件
            await event_system.emit_ux_feedback(
                "retry_scheduled",
                f"任务失败，{delay:.1f}秒后进行第{task.retry_count + 1}次重试",
                {
                    "task_id": task_id,
                    "error": str(error),
                    "retry_count": task.retry_count + 1,
                    "retry_delay": delay
                },
                task.correlation_id
            )
            
            # 安排重试
            asyncio.create_task(self._schedule_retry(task_id, delay))
            return True
        else:
            # 不重试，任务失败
            await event_system.emit_task_failed(task_id, str(error), task.correlation_id)
            return False
    
    def _generate_recovery_suggestions(self, failure_info: FailureInfo) -> List[str]:
        """生成恢复建议"""
        suggestions = []
        
        if failure_info.failure_type == FailureType.TIMEOUT:
            suggestions.extend([
                "考虑增加任务超时时间",
                "检查任务是否过于复杂，可以拆分为更小的子任务",
                "优化任务算法以提高执行效率"
            ])
        elif failure_info.failure_type == FailureType.EXCEPTION:
            suggestions.extend([
                "检查输入参数是否正确",
                "查看任务代码中的异常处理",
                "确认外部依赖服务是否可用"
            ])
        elif failure_info.failure_type == FailureType.RESOURCE_EXHAUSTION:
            suggestions.extend([
                "减少并发任务数量",
                "释放不必要的资源",
                "考虑升级系统资源"
            ])
        elif failure_info.failure_type == FailureType.DEPENDENCY_FAILURE:
            suggestions.extend([
                "检查依赖服务状态",
                "确认网络连接正常",
                "验证依赖配置正确性"
            ])
        
        return suggestions
    
    async def _schedule_retry(self, task_id: str, delay: float) -> None:
        """安排重试"""
        await asyncio.sleep(delay)
        
        task = self.task_manager.get_task(task_id)
        if task and task.status in [TaskStatus.FAILED, TaskStatus.TIMEOUT]:
            # 重置任务状态
            task.status = TaskStatus.PENDING
            task.error = None
            task.started_at = None
            task.completed_at = None
            
            # 重新加入队列
            await self.task_manager.submit_task(
                name=f"{task.name} (重试 {task.retry_count + 1})",
                func=task.func,
                *task.args,
                priority=task.priority,
                timeout=task.timeout,
                max_retries=task.max_retries,
                correlation_id=task.correlation_id,
                metadata=task.metadata,
                **task.kwargs
            )
    
    def get_failure_analysis(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取失败分析"""
        pattern = self.retry_manager.get_failure_pattern(task_id)
        if not pattern:
            return None
        
        task = self.task_manager.get_task(task_id)
        if not task:
            return None
        
        return {
            **pattern,
            "current_status": task.status.value,
            "suggested_actions": self._get_suggested_actions(pattern, task)
        }
    
    def _get_suggested_actions(self, pattern: Dict[str, Any], task: Task) -> List[str]:
        """获取建议操作"""
        actions = []
        
        if pattern["total_failures"] > 5:
            actions.append("任务失败次数较多，建议检查任务逻辑")
        
        if pattern["recent_failures"] > 3:
            actions.append("最近失败频繁，可能存在系统性问题")
        
        if task.priority == TaskPriority.URGENT:
            actions.append("紧急任务，建议手动干预")
        
        if task.timeout < 60:
            actions.append("超时时间较短，建议适当延长")
        
        return actions
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_failures = sum(len(failures) for failures in self.retry_manager.failure_history.values())
        retryable_failures = sum(
            1 for failures in self.retry_manager.failure_history.values()
            for failure in failures
            if self.retry_manager.should_retry(failure.task_id, failure)
        )
        
        return {
            "total_failures": total_failures,
            "retryable_failures": retryable_failures,
            "retry_rate": retryable_failures / total_failures if total_failures > 0 else 0,
            "active_timeouts": len(self.timeout_tasks),
            "configured_timeouts": len(self.timeout_configs)
        }