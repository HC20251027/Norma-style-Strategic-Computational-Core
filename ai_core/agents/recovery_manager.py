"""
恢复管理器

实现任务失败恢复和重试机制
"""

import asyncio
import time
import random
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import uuid

from .models import Task, TaskStatus, TaskExecutionPlan, TaskPriority
from .state_tracker import StateTracker


class RecoveryStrategy(Enum):
    """恢复策略"""
    IMMEDIATE_RETRY = "immediate_retry"      # 立即重试
    EXPONENTIAL_BACKOFF = "exponential_backoff"  # 指数退避
    FIXED_DELAY = "fixed_delay"             # 固定延迟
    ADAPTIVE = "adaptive"                   # 自适应策略
    MANUAL_REVIEW = "manual_review"         # 人工审核
    SKIP_AND_CONTINUE = "skip_and_continue" # 跳过继续


class FailureType(Enum):
    """失败类型"""
    TEMPORARY = "temporary"     # 临时性失败（网络、临时资源不可用等）
    PERMANENT = "permanent"     # 永久性失败（逻辑错误、配置错误等）
    RESOURCE = "resource"       # 资源不足
    DEPENDENCY = "dependency"   # 依赖失败
    TIMEOUT = "timeout"         # 超时失败
    UNKNOWN = "unknown"         # 未知失败


@dataclass
class RetryConfig:
    """重试配置"""
    max_retries: int = 3
    initial_delay: float = 1.0  # 初始延迟（秒）
    max_delay: float = 300.0    # 最大延迟（秒）
    backoff_factor: float = 2.0 # 退避因子
    jitter: bool = True         # 是否添加随机抖动
    strategy: RecoveryStrategy = RecoveryStrategy.EXPONENTIAL_BACKOFF
    
    # 特定失败类型的重试配置
    retry_configs: Dict[FailureType, Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self):
        # 设置不同失败类型的默认重试策略
        self.retry_configs = {
            FailureType.TEMPORARY: {
                "max_retries": 5,
                "strategy": RecoveryStrategy.EXPONENTIAL_BACKOFF,
                "initial_delay": 0.5
            },
            FailureType.RESOURCE: {
                "max_retries": 3,
                "strategy": RecoveryStrategy.FIXED_DELAY,
                "initial_delay": 10.0
            },
            FailureType.DEPENDENCY: {
                "max_retries": 2,
                "strategy": RecoveryStrategy.EXPONENTIAL_BACKOFF,
                "initial_delay": 5.0
            },
            FailureType.TIMEOUT: {
                "max_retries": 3,
                "strategy": RecoveryStrategy.FIXED_DELAY,
                "initial_delay": 2.0
            },
            FailureType.PERMANENT: {
                "max_retries": 0,
                "strategy": RecoveryStrategy.MANUAL_REVIEW
            }
        }


@dataclass
class RecoveryPlan:
    """恢复计划"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = ""
    original_error: str = ""
    failure_type: FailureType = FailureType.UNKNOWN
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.IMMEDIATE_RETRY
    retry_count: int = 0
    max_retries: int = 3
    next_retry_time: Optional[datetime] = None
    recovery_actions: List[str] = field(default_factory=list)
    status: str = "pending"  # pending, in_progress, completed, failed
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class RecoveryManager:
    """恢复管理器"""
    
    def __init__(
        self,
        state_tracker: StateTracker,
        default_retry_config: Optional[RetryConfig] = None
    ):
        self.state_tracker = state_tracker
        self.default_retry_config = default_retry_config or RetryConfig()
        
        # 恢复计划存储
        self.recovery_plans: Dict[str, RecoveryPlan] = {}
        self.active_recoveries: Dict[str, asyncio.Task] = {}
        
        # 恢复统计
        self.recovery_stats = {
            "total_recoveries": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0,
            "average_recovery_time": 0.0,
            "failure_type_distribution": {},
            "strategy_effectiveness": {}
        }
        
        # 回调函数
        self.on_recovery_start: Optional[Callable] = None
        self.on_recovery_complete: Optional[Callable] = None
        self.on_recovery_failed: Optional[Callable] = None
    
    async def handle_task_failure(
        self,
        task: Task,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        处理任务失败
        
        Returns:
            是否成功启动恢复流程
        """
        # 分析失败类型
        failure_type = self._analyze_failure_type(task, error, context)
        
        # 创建恢复计划
        recovery_plan = self._create_recovery_plan(task, error, failure_type, context)
        
        # 检查是否需要重试
        if not self._should_retry(recovery_plan):
            await self._mark_recovery_failed(recovery_plan, "达到最大重试次数")
            return False
        
        # 启动恢复流程
        return await self._start_recovery(recovery_plan)
    
    def _analyze_failure_type(
        self,
        task: Task,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> FailureType:
        """分析失败类型"""
        error_message = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # 基于错误类型和消息判断失败类型
        if any(keyword in error_message for keyword in [
            "timeout", "timed out", "connection timeout"
        ]):
            return FailureType.TIMEOUT
        
        elif any(keyword in error_message for keyword in [
            "connection", "network", "unreachable", "temporary",
            "service unavailable", "rate limit"
        ]):
            return FailureType.TEMPORARY
        
        elif any(keyword in error_message for keyword in [
            "resource", "memory", "disk", "quota", "limit exceeded"
        ]):
            return FailureType.RESOURCE
        
        elif any(keyword in error_message for keyword in [
            "dependency", " prerequisite", "required", "missing"
        ]):
            return FailureType.DEPENDENCY
        
        elif any(keyword in error_message for keyword in [
            "invalid", "error", "failed", "exception", "bug"
        ]):
            return FailureType.PERMANENT
        
        else:
            return FailureType.UNKNOWN
    
    def _create_recovery_plan(
        self,
        task: Task,
        error: Exception,
        failure_type: FailureType,
        context: Optional[Dict[str, Any]] = None
    ) -> RecoveryPlan:
        """创建恢复计划"""
        # 获取重试配置
        retry_config = self._get_retry_config(task, failure_type)
        
        # 确定恢复策略
        strategy = self._determine_recovery_strategy(task, failure_type, retry_config)
        
        # 生成恢复动作
        recovery_actions = self._generate_recovery_actions(task, failure_type, error, context)
        
        recovery_plan = RecoveryPlan(
            task_id=task.id,
            original_error=str(error),
            failure_type=failure_type,
            recovery_strategy=strategy,
            retry_count=task.retry_count,
            max_retries=retry_config.max_retries,
            recovery_actions=recovery_actions
        )
        
        # 计算下次重试时间
        if strategy in [RecoveryStrategy.EXPONENTIAL_BACKOFF, RecoveryStrategy.FIXED_DELAY]:
            recovery_plan.next_retry_time = self._calculate_next_retry_time(
                task.retry_count, retry_config, failure_type
            )
        
        self.recovery_plans[recovery_plan.id] = recovery_plan
        return recovery_plan
    
    def _get_retry_config(self, task: Task, failure_type: FailureType) -> RetryConfig:
        """获取重试配置"""
        # 任务特定的配置优先
        if hasattr(task, 'retry_config') and task.retry_config:
            return task.retry_config
        
        # 失败类型特定的配置
        if failure_type in self.default_retry_config.retry_configs:
            type_config = self.default_retry_config.retry_configs[failure_type]
            config = RetryConfig(
                max_retries=type_config.get("max_retries", self.default_retry_config.max_retries),
                strategy=RecoveryStrategy(type_config.get("strategy", self.default_retry_config.strategy)),
                initial_delay=type_config.get("initial_delay", self.default_retry_config.initial_delay)
            )
            return config
        
        # 默认配置
        return self.default_retry_config
    
    def _determine_recovery_strategy(
        self,
        task: Task,
        failure_type: FailureType,
        retry_config: RetryConfig
    ) -> RecoveryStrategy:
        """确定恢复策略"""
        # 基于失败类型选择策略
        if failure_type == FailureType.PERMANENT:
            return RecoveryStrategy.MANUAL_REVIEW
        elif failure_type == FailureType.RESOURCE:
            return RecoveryStrategy.FIXED_DELAY
        elif failure_type == FailureType.DEPENDENCY:
            return RecoveryStrategy.EXPONENTIAL_BACKOFF
        
        # 基于任务优先级调整策略
        if task.priority == TaskPriority.CRITICAL:
            if failure_type == FailureType.TEMPORARY:
                return RecoveryStrategy.IMMEDIATE_RETRY
            else:
                return RecoveryStrategy.ADAPTIVE
        elif task.priority == TaskPriority.LOW:
            return RecoveryStrategy.SKIP_AND_CONTINUE
        
        # 使用配置中的策略
        return retry_config.strategy
    
    def _generate_recovery_actions(
        self,
        task: Task,
        failure_type: FailureType,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """生成恢复动作"""
        actions = []
        
        # 基于失败类型的恢复动作
        if failure_type == FailureType.TEMPORARY:
            actions.extend([
                "检查网络连接",
                "验证外部服务状态",
                "增加超时时间",
                "重试执行"
            ])
        
        elif failure_type == FailureType.RESOURCE:
            actions.extend([
                "检查系统资源使用情况",
                "清理临时文件",
                "释放不必要的资源",
                "等待资源可用后重试"
            ])
        
        elif failure_type == FailureType.DEPENDENCY:
            actions.extend([
                "检查依赖任务状态",
                "验证输入数据完整性",
                "重新执行依赖任务",
                "更新任务依赖关系"
            ])
        
        elif failure_type == FailureType.TIMEOUT:
            actions.extend([
                "增加执行超时时间",
                "优化任务执行逻辑",
                "分解为更小的子任务",
                "并行执行优化"
            ])
        
        elif failure_type == FailureType.PERMANENT:
            actions.extend([
                "检查任务逻辑",
                "验证输入参数",
                "更新任务实现",
                "人工审核和修复"
            ])
        
        # 基于任务类型的特定动作
        if task.task_type == "llm":
            actions.extend([
                "检查LLM API状态",
                "验证提示词",
                "调整模型参数"
            ])
        elif task.task_type == "data_processing":
            actions.extend([
                "检查数据源",
                "验证数据格式",
                "清理损坏数据"
            ])
        
        return actions
    
    def _should_retry(self, recovery_plan: RecoveryPlan) -> bool:
        """判断是否应该重试"""
        return recovery_plan.retry_count < recovery_plan.max_retries
    
    def _calculate_next_retry_time(
        self,
        retry_count: int,
        retry_config: RetryConfig,
        failure_type: FailureType
    ) -> datetime:
        """计算下次重试时间"""
        base_delay = retry_config.initial_delay
        
        if retry_config.strategy == RecoveryStrategy.EXPONENTIAL_BACKOFF:
            delay = base_delay * (retry_config.backoff_factor ** retry_count)
        elif retry_config.strategy == RecoveryStrategy.FIXED_DELAY:
            delay = base_delay
        else:
            delay = base_delay
        
        # 限制最大延迟
        delay = min(delay, retry_config.max_delay)
        
        # 添加随机抖动
        if retry_config.jitter:
            jitter_range = delay * 0.1
            delay += random.uniform(-jitter_range, jitter_range)
        
        return datetime.now() + timedelta(seconds=delay)
    
    async def _start_recovery(self, recovery_plan: RecoveryPlan) -> bool:
        """启动恢复流程"""
        try:
            recovery_plan.status = "in_progress"
            recovery_plan.updated_at = datetime.now()
            
            # 记录恢复开始事件
            self.state_tracker._record_event(
                self.state_tracker.__class__.TaskEvent(
                    task_id=recovery_plan.task_id,
                    event_type="recovery_started",
                    data={
                        "recovery_plan_id": recovery_plan.id,
                        "failure_type": recovery_plan.failure_type.value,
                        "strategy": recovery_plan.recovery_strategy.value,
                        "retry_count": recovery_plan.retry_count
                    }
                )
            )
            
            # 触发恢复开始回调
            if self.on_recovery_start:
                await self._safe_callback(self.on_recovery_start, recovery_plan)
            
            # 根据策略执行恢复
            if recovery_plan.recovery_strategy == RecoveryStrategy.IMMEDIATE_RETRY:
                return await self._execute_immediate_retry(recovery_plan)
            elif recovery_plan.recovery_strategy == RecoveryStrategy.EXPONENTIAL_BACKOFF:
                return await self._execute_delayed_retry(recovery_plan)
            elif recovery_plan.recovery_strategy == RecoveryStrategy.FIXED_DELAY:
                return await self._execute_delayed_retry(recovery_plan)
            elif recovery_plan.recovery_strategy == RecoveryStrategy.ADAPTIVE:
                return await self._execute_adaptive_recovery(recovery_plan)
            elif recovery_plan.recovery_strategy == RecoveryStrategy.SKIP_AND_CONTINUE:
                return await self._execute_skip_and_continue(recovery_plan)
            else:
                return await self._execute_manual_review(recovery_plan)
        
        except Exception as e:
            await self._mark_recovery_failed(recovery_plan, str(e))
            return False
    
    async def _execute_immediate_retry(self, recovery_plan: RecoveryPlan) -> bool:
        """执行立即重试"""
        # 等待一小段时间确保系统稳定
        await asyncio.sleep(0.1)
        return await self._retry_task(recovery_plan)
    
    async def _execute_delayed_retry(self, recovery_plan: RecoveryPlan) -> bool:
        """执行延迟重试"""
        if not recovery_plan.next_retry_time:
            return False
        
        # 等待到重试时间
        wait_time = (recovery_plan.next_retry_time - datetime.now()).total_seconds()
        if wait_time > 0:
            await asyncio.sleep(wait_time)
        
        return await self._retry_task(recovery_plan)
    
    async def _execute_adaptive_recovery(self, recovery_plan: RecoveryPlan) -> bool:
        """执行自适应恢复"""
        task = self.state_tracker.tasks.get(recovery_plan.task_id)
        if not task:
            return False
        
        # 基于历史失败模式调整策略
        failure_history = self._get_failure_history(task.id)
        
        if len(failure_history) > 3:
            # 如果连续失败，考虑跳过或人工介入
            if all("timeout" in f["error"].lower() for f in failure_history[-3:]):
                # 连续超时，增加超时时间
                task.parameters["timeout"] = task.parameters.get("timeout", 30) * 2
                recovery_plan.recovery_actions.append("增加超时时间")
        
        return await self._retry_task(recovery_plan)
    
    async def _execute_skip_and_continue(self, recovery_plan: RecoveryPlan) -> bool:
        """执行跳过继续"""
        task = self.state_tracker.tasks.get(recovery_plan.task_id)
        if not task:
            return False
        
        # 标记任务为跳过
        task.update_status(TaskStatus.SKIPPED, f"跳过任务: {recovery_plan.original_error}")
        
        await self._mark_recovery_completed(recovery_plan, "任务已跳过")
        return True
    
    async def _execute_manual_review(self, recovery_plan: RecoveryPlan) -> bool:
        """执行人工审核"""
        # 标记任务需要人工审核
        task = self.state_tracker.tasks.get(recovery_plan.task_id)
        if task:
            task.metadata["requires_manual_review"] = True
            task.metadata["recovery_plan_id"] = recovery_plan.id
        
        # 记录需要人工审核的事件
        self.state_tracker._record_event(
            self.state_tracker.__class__.TaskEvent(
                task_id=recovery_plan.task_id,
                event_type="manual_review_required",
                data={
                    "recovery_plan_id": recovery_plan.id,
                    "error": recovery_plan.original_error,
                    "recovery_actions": recovery_plan.recovery_actions
                }
            )
        )
        
        return True
    
    async def _retry_task(self, recovery_plan: RecoveryPlan) -> bool:
        """重试任务"""
        task = self.state_tracker.tasks.get(recovery_plan.task_id)
        if not task:
            return False
        
        try:
            # 更新重试计数
            recovery_plan.retry_count += 1
            recovery_plan.updated_at = datetime.now()
            
            # 更新任务状态
            task.retry_count = recovery_plan.retry_count
            task.update_status(TaskStatus.RETRYING)
            
            # 重新执行任务（这里需要实际的执行逻辑）
            # 简化实现，模拟重试结果
            success = await self._simulate_task_retry(task, recovery_plan)
            
            if success:
                await self._mark_recovery_completed(recovery_plan, "任务重试成功")
                return True
            else:
                # 检查是否还可以重试
                if self._should_retry(recovery_plan):
                    # 计算下次重试时间
                    retry_config = self._get_retry_config(task, recovery_plan.failure_type)
                    recovery_plan.next_retry_time = self._calculate_next_retry_time(
                        recovery_plan.retry_count, retry_config, recovery_plan.failure_type
                    )
                    return False
                else:
                    await self._mark_recovery_failed(recovery_plan, "达到最大重试次数")
                    return False
        
        except Exception as e:
            await self._mark_recovery_failed(recovery_plan, str(e))
            return False
    
    async def _simulate_task_retry(self, task: Task, recovery_plan: RecoveryPlan) -> bool:
        """模拟任务重试（实际实现中需要替换为真实逻辑）"""
        # 模拟重试成功率随重试次数降低
        base_success_rate = 0.8
        retry_penalty = 0.1 * recovery_plan.retry_count
        success_rate = max(0.1, base_success_rate - retry_penalty)
        
        # 添加失败类型的影响
        if recovery_plan.failure_type == FailureType.PERMANENT:
            success_rate = 0.0
        elif recovery_plan.failure_type == FailureType.RESOURCE:
            success_rate *= 0.7
        
        return random.random() < success_rate
    
    async def _mark_recovery_completed(self, recovery_plan: RecoveryPlan, message: str):
        """标记恢复完成"""
        recovery_plan.status = "completed"
        recovery_plan.updated_at = datetime.now()
        
        self.recovery_stats["successful_recoveries"] += 1
        
        # 记录恢复完成事件
        self.state_tracker._record_event(
            self.state_tracker.__class__.TaskEvent(
                task_id=recovery_plan.task_id,
                event_type="recovery_completed",
                data={
                    "recovery_plan_id": recovery_plan.id,
                    "message": message,
                    "retry_count": recovery_plan.retry_count
                }
            )
        )
        
        if self.on_recovery_complete:
            await self._safe_callback(self.on_recovery_complete, recovery_plan, message)
    
    async def _mark_recovery_failed(self, recovery_plan: RecoveryPlan, message: str):
        """标记恢复失败"""
        recovery_plan.status = "failed"
        recovery_plan.updated_at = datetime.now()
        
        self.recovery_stats["failed_recoveries"] += 1
        
        # 更新任务状态为失败
        task = self.state_tracker.tasks.get(recovery_plan.task_id)
        if task:
            task.update_status(TaskStatus.FAILED, f"恢复失败: {message}")
        
        # 记录恢复失败事件
        self.state_tracker._record_event(
            self.state_tracker.__class__.TaskEvent(
                task_id=recovery_plan.task_id,
                event_type="recovery_failed",
                data={
                    "recovery_plan_id": recovery_plan.id,
                    "message": message,
                    "retry_count": recovery_plan.retry_count
                }
            )
        )
        
        if self.on_recovery_failed:
            await self._safe_callback(self.on_recovery_failed, recovery_plan, message)
    
    def _get_failure_history(self, task_id: str) -> List[Dict[str, Any]]:
        """获取失败历史"""
        events = self.state_tracker.get_task_events(task_id, "failed")
        return [{"timestamp": e["timestamp"], "error": e["data"].get("error_message", "")} for e in events]
    
    def _safe_callback(self, callback: Callable, *args, **kwargs):
        """安全执行回调函数"""
        try:
            if asyncio.iscoroutinefunction(callback):
                asyncio.create_task(callback(*args, **kwargs))
            else:
                callback(*args, **kwargs)
        except Exception as e:
            print(f"回调函数执行失败: {e}")
    
    def get_recovery_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取恢复状态"""
        for plan in self.recovery_plans.values():
            if plan.task_id == task_id:
                return {
                    "recovery_plan": plan.__dict__,
                    "recovery_stats": self.recovery_stats
                }
        return None
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """获取恢复统计信息"""
        return {
            "recovery_stats": self.recovery_stats,
            "active_recovery_plans": len([p for p in self.recovery_plans.values() if p.status == "in_progress"]),
            "pending_recovery_plans": len([p for p in self.recovery_plans.values() if p.status == "pending"]),
            "failure_type_distribution": self.recovery_stats["failure_type_distribution"]
        }