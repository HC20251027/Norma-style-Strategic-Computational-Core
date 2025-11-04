"""
任务执行的容错和恢复
负责处理任务失败、重试、故障转移和系统恢复
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Union, Set
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
import time
import threading
import uuid
import json
import pickle
from concurrent.futures import ThreadPoolExecutor
from abc import ABC, abstractmethod
import traceback

# 配置日志
logger = logging.getLogger(__name__)


class FailureType(Enum):
    """失败类型"""
    TIMEOUT = "timeout"
    EXCEPTION = "exception"
    RESOURCE_UNAVAILABLE = "resource_unavailable"
    NETWORK_ERROR = "network_error"
    SYSTEM_ERROR = "system_error"
    USER_CANCELLED = "user_cancelled"
    UNKNOWN = "unknown"


class RecoveryStrategy(Enum):
    """恢复策略"""
    RETRY = "retry"
    FAILOVER = "failover"
    CIRCUIT_BREAKER = "circuit_breaker"
    BULKHEAD = "bulkhead"
    TIMEOUT_EXTENSION = "timeout_extension"
    SKIP = "skip"
    MANUAL = "manual"


class CircuitBreakerState(Enum):
    """熔断器状态"""
    CLOSED = "closed"      # 正常状态
    OPEN = "open"          # 熔断状态
    HALF_OPEN = "half_open"  # 半开状态


@dataclass
class TaskFailure:
    """任务失败信息"""
    task_id: str
    failure_type: FailureType
    error: Exception
    timestamp: float
    retry_count: int = 0
    recovery_attempts: List[RecoveryStrategy] = field(default_factory=list)
    context: dict = field(default_factory=dict)
    resolved: bool = False
    resolution_time: Optional[float] = None
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'task_id': self.task_id,
            'failure_type': self.failure_type.value,
            'error': str(self.error),
            'timestamp': self.timestamp,
            'retry_count': self.retry_count,
            'recovery_attempts': [s.value for s in self.recovery_attempts],
            'context': self.context,
            'resolved': self.resolved,
            'resolution_time': self.resolution_time
        }


@dataclass
class RecoveryAction:
    """恢复动作"""
    id: str
    strategy: RecoveryStrategy
    task_id: str
    parameters: dict = field(default_factory=dict)
    status: str = "pending"  # pending, executing, completed, failed
    created_at: float = field(default_factory=time.time)
    executed_at: Optional[float] = None
    result: Any = None
    error: Optional[Exception] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


class CircuitBreaker:
    """熔断器"""
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.success_count = 0
        
        self._lock = threading.RLock()
        
        logger.info(f"熔断器 {name} 初始化完成")
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """调用函数，受熔断器保护"""
        with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    logger.info(f"熔断器 {self.name} 进入半开状态")
                else:
                    raise Exception(f"熔断器 {self.name} 开启，拒绝调用")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except self.expected_exception as e:
                self._on_failure()
                raise
    
    def _should_attempt_reset(self) -> bool:
        """检查是否应该尝试重置"""
        return (
            self.last_failure_time is not None and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )
    
    def _on_success(self):
        """成功回调"""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= 3:  # 连续3次成功则关闭熔断器
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                logger.info(f"熔断器 {self.name} 关闭")
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = max(0, self.failure_count - 1)
    
    def _on_failure(self):
        """失败回调"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.warning(f"熔断器 {self.name} 开启，失败次数: {self.failure_count}")
    
    def get_state(self) -> dict:
        """获取熔断器状态"""
        return {
            'name': self.name,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'last_failure_time': self.last_failure_time
        }


class TaskRecovery:
    """任务恢复管理器"""
    
    def __init__(
        self,
        max_retry_attempts: int = 3,
        retry_delay_base: float = 1.0,
        retry_delay_multiplier: float = 2.0,
        enable_circuit_breaker: bool = True
    ):
        self.max_retry_attempts = max_retry_attempts
        self.retry_delay_base = retry_delay_base
        self.retry_delay_multiplier = retry_delay_multiplier
        self.enable_circuit_breaker = enable_circuit_breaker
        
        # 失败管理
        self.task_failures: Dict[str, TaskFailure] = {}
        self.recovery_actions: Dict[str, RecoveryAction] = {}
        
        # 熔断器管理
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # 恢复统计
        self.recovery_stats = {
            'total_failures': 0,
            'recovered_tasks': 0,
            'failed_recoveries': 0,
            'avg_recovery_time': 0.0,
            'total_recovery_time': 0.0
        }
        
        # 回调函数
        self.failure_callbacks: List[Callable] = []
        self.recovery_callbacks: List[Callable] = []
        
        self._lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        logger.info("任务恢复管理器初始化完成")
    
    def register_circuit_breaker(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0
    ) -> CircuitBreaker:
        """注册熔断器"""
        breaker = CircuitBreaker(name, failure_threshold, recovery_timeout)
        self.circuit_breakers[name] = breaker
        logger.info(f"熔断器已注册: {name}")
        return breaker
    
    def add_failure_callback(self, callback: Callable):
        """添加失败回调"""
        self.failure_callbacks.append(callback)
    
    def add_recovery_callback(self, callback: Callable):
        """添加恢复回调"""
        self.recovery_callbacks.append(callback)
    
    async def handle_failure(
        self,
        task_id: str,
        error: Exception,
        failure_type: FailureType = FailureType.UNKNOWN,
        context: dict = None
    ) -> bool:
        """处理任务失败"""
        with self._lock:
            self.recovery_stats['total_failures'] += 1
            
            # 创建失败记录
            failure = TaskFailure(
                task_id=task_id,
                failure_type=failure_type,
                error=error,
                timestamp=time.time(),
                context=context or {}
            )
            
            self.task_failures[task_id] = failure
            
            logger.error(f"任务失败: {task_id}, 类型: {failure_type.value}, 错误: {error}")
            
            # 调用失败回调
            for callback in self.failure_callbacks:
                try:
                    await callback(failure)
                except Exception as e:
                    logger.error(f"失败回调错误: {e}")
            
            # 尝试恢复
            recovery_success = await self._attempt_recovery(failure)
            
            if recovery_success:
                failure.resolved = True
                failure.resolution_time = time.time()
                self.recovery_stats['recovered_tasks'] += 1
                logger.info(f"任务恢复成功: {task_id}")
            else:
                self.recovery_stats['failed_recoveries'] += 1
                logger.error(f"任务恢复失败: {task_id}")
            
            return recovery_success
    
    async def _attempt_recovery(self, failure: TaskFailure) -> bool:
        """尝试恢复任务"""
        start_time = time.time()
        
        # 根据失败类型选择恢复策略
        strategies = self._select_recovery_strategies(failure)
        
        for strategy in strategies:
            try:
                action = RecoveryAction(
                    id=str(uuid.uuid4()),
                    strategy=strategy,
                    task_id=failure.task_id
                )
                
                self.recovery_actions[action.id] = action
                
                # 执行恢复动作
                success = await self._execute_recovery_action(action, failure)
                
                if success:
                    # 记录恢复时间
                    recovery_time = time.time() - start_time
                    self.recovery_stats['total_recovery_time'] += recovery_time
                    self.recovery_stats['avg_recovery_time'] = (
                        self.recovery_stats['total_recovery_time'] / 
                        max(1, self.recovery_stats['recovered_tasks'])
                    )
                    
                    return True
                
            except Exception as e:
                logger.error(f"恢复动作执行失败: {strategy.value}, 错误: {e}")
                continue
        
        return False
    
    def _select_recovery_strategies(self, failure: TaskFailure) -> List[RecoveryStrategy]:
        """选择恢复策略"""
        strategies = []
        
        if failure.failure_type == FailureType.TIMEOUT:
            strategies = [
                RecoveryStrategy.TIMEOUT_EXTENSION,
                RecoveryStrategy.RETRY,
                RecoveryStrategy.FAILOVER
            ]
        elif failure.failure_type == FailureType.EXCEPTION:
            if failure.retry_count < self.max_retry_attempts:
                strategies = [
                    RecoveryStrategy.RETRY,
                    RecoveryStrategy.CIRCUIT_BREAKER
                ]
            else:
                strategies = [
                    RecoveryStrategy.FAILOVER,
                    RecoveryStrategy.SKIP
                ]
        elif failure.failure_type == FailureType.RESOURCE_UNAVAILABLE:
            strategies = [
                RecoveryStrategy.FAILOVER,
                RecoveryStrategy.BULKHEAD,
                RecoveryStrategy.RETRY
            ]
        elif failure.failure_type == FailureType.NETWORK_ERROR:
            strategies = [
                RecoveryStrategy.RETRY,
                RecoveryStrategy.CIRCUIT_BREAKER,
                RecoveryStrategy.FAILOVER
            ]
        else:
            strategies = [
                RecoveryStrategy.RETRY,
                RecoveryStrategy.SKIP
            ]
        
        return strategies
    
    async def _execute_recovery_action(
        self,
        action: RecoveryAction,
        failure: TaskFailure
    ) -> bool:
        """执行恢复动作"""
        action.status = "executing"
        action.executed_at = time.time()
        
        try:
            if action.strategy == RecoveryStrategy.RETRY:
                return await self._execute_retry(action, failure)
            elif action.strategy == RecoveryStrategy.FAILOVER:
                return await self._execute_failover(action, failure)
            elif action.strategy == RecoveryStrategy.CIRCUIT_BREAKER:
                return await self._execute_circuit_breaker(action, failure)
            elif action.strategy == RecoveryStrategy.BULKHEAD:
                return await self._execute_bulkhead(action, failure)
            elif action.strategy == RecoveryStrategy.TIMEOUT_EXTENSION:
                return await self._execute_timeout_extension(action, failure)
            elif action.strategy == RecoveryStrategy.SKIP:
                return await self._execute_skip(action, failure)
            else:
                logger.warning(f"未知的恢复策略: {action.strategy}")
                return False
                
        except Exception as e:
            action.status = "failed"
            action.error = e
            logger.error(f"恢复动作执行异常: {action.strategy.value}, 错误: {e}")
            return False
    
    async def _execute_retry(self, action: RecoveryAction, failure: TaskFailure) -> bool:
        """执行重试"""
        failure.retry_count += 1
        failure.recovery_attempts.append(RecoveryStrategy.RETRY)
        
        # 计算延迟时间（指数退避）
        delay = self.retry_delay_base * (self.retry_delay_multiplier ** failure.retry_count)
        
        logger.info(f"准备重试任务: {failure.task_id}, 延迟: {delay:.2f}s, 次数: {failure.retry_count}")
        
        await asyncio.sleep(delay)
        
        # 这里应该重新执行任务
        # 实际实现中需要调用任务调度器重新提交任务
        action.status = "completed"
        action.result = "retry_scheduled"
        
        return True
    
    async def _execute_failover(self, action: RecoveryAction, failure: TaskFailure) -> bool:
        """执行故障转移"""
        failure.recovery_attempts.append(RecoveryStrategy.FAILOVER)
        
        logger.info(f"执行故障转移: {failure.task_id}")
        
        # 这里应该将任务转移到备用资源或节点
        # 实际实现中需要与资源管理器配合
        
        action.status = "completed"
        action.result = "failover_initiated"
        
        return True
    
    async def _execute_circuit_breaker(self, action: RecoveryAction, failure: TaskFailure) -> bool:
        """执行熔断器保护"""
        failure.recovery_attempts.append(RecoveryStrategy.CIRCUIT_BREAKER)
        
        # 使用熔断器保护后续调用
        breaker_name = f"task_{failure.task_id}"
        if breaker_name not in self.circuit_breakers:
            self.register_circuit_breaker(breaker_name)
        
        breaker = self.circuit_breakers[breaker_name]
        
        # 熔断器会自动处理失败情况
        logger.info(f"启用熔断器保护: {breaker_name}")
        
        action.status = "completed"
        action.result = "circuit_breaker_enabled"
        
        return True
    
    async def _execute_bulkhead(self, action: RecoveryAction, failure: TaskFailure) -> bool:
        """执行隔板模式"""
        failure.recovery_attempts.append(RecoveryStrategy.BULKHEAD)
        
        logger.info(f"执行隔板隔离: {failure.task_id}")
        
        # 这里应该将任务隔离到独立的资源池
        # 实际实现中需要与资源管理器配合
        
        action.status = "completed"
        action.result = "bulkhead_isolation"
        
        return True
    
    async def _execute_timeout_extension(self, action: RecoveryAction, failure: TaskFailure) -> bool:
        """执行超时扩展"""
        failure.recovery_attempts.append(RecoveryStrategy.TIMEOUT_EXTENSION)
        
        logger.info(f"扩展任务超时: {failure.task_id}")
        
        # 这里应该增加任务超时时间
        # 实际实现中需要与任务调度器配合
        
        action.status = "completed"
        action.result = "timeout_extended"
        
        return True
    
    async def _execute_skip(self, action: RecoveryAction, failure: TaskFailure) -> bool:
        """执行跳过"""
        failure.recovery_attempts.append(RecoveryStrategy.SKIP)
        
        logger.info(f"跳过失败任务: {failure.task_id}")
        
        action.status = "completed"
        action.result = "task_skipped"
        
        return True
    
    def get_failure_info(self, task_id: str) -> Optional[TaskFailure]:
        """获取失败信息"""
        return self.task_failures.get(task_id)
    
    def get_all_failures(self, resolved: bool = None) -> List[TaskFailure]:
        """获取所有失败记录"""
        failures = list(self.task_failures.values())
        if resolved is not None:
            failures = [f for f in failures if f.resolved == resolved]
        return sorted(failures, key=lambda x: x.timestamp, reverse=True)
    
    def get_recovery_actions(self, task_id: str = None) -> List[RecoveryAction]:
        """获取恢复动作"""
        actions = list(self.recovery_actions.values())
        if task_id:
            actions = [a for a in actions if a.task_id == task_id]
        return sorted(actions, key=lambda x: x.created_at, reverse=True)
    
    def get_circuit_breaker_status(self, name: str = None) -> List[dict]:
        """获取熔断器状态"""
        if name:
            if name in self.circuit_breakers:
                return [self.circuit_breakers[name].get_state()]
            return []
        else:
            return [breaker.get_state() for breaker in self.circuit_breakers.values()]
    
    def get_recovery_statistics(self) -> dict:
        """获取恢复统计"""
        return self.recovery_stats.copy()
    
    def reset_circuit_breaker(self, name: str) -> bool:
        """重置熔断器"""
        if name in self.circuit_breakers:
            breaker = self.circuit_breakers[name]
            breaker.state = CircuitBreakerState.CLOSED
            breaker.failure_count = 0
            breaker.success_count = 0
            breaker.last_failure_time = None
            
            logger.info(f"熔断器已重置: {name}")
            return True
        return False
    
    def export_recovery_log(self, filepath: str):
        """导出恢复日志"""
        log_data = {
            'export_time': time.time(),
            'recovery_statistics': self.get_recovery_statistics(),
            'failures': [failure.to_dict() for failure in self.get_all_failures()],
            'recovery_actions': [
                {
                    'id': action.id,
                    'strategy': action.strategy.value,
                    'task_id': action.task_id,
                    'status': action.status,
                    'created_at': action.created_at,
                    'executed_at': action.executed_at,
                    'result': str(action.result) if action.result else None,
                    'error': str(action.error) if action.error else None
                }
                for action in self.get_recovery_actions()
            ],
            'circuit_breakers': self.get_circuit_breaker_status()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"恢复日志已导出到: {filepath}")


class FaultTolerance:
    """容错系统"""
    
    def __init__(self, recovery_manager: TaskRecovery):
        self.recovery_manager = recovery_manager
        self.health_checks: Dict[str, Callable] = {}
        self.health_status: Dict[str, bool] = {}
        
        # 容错策略配置
        self.fault_tolerance_config = {
            'enable_auto_recovery': True,
            'enable_health_monitoring': True,
            'health_check_interval': 30.0,
            'max_consecutive_failures': 10
        }
        
        self.consecutive_failures = 0
        self.last_health_check = 0
        
        logger.info("容错系统初始化完成")
    
    def register_health_check(self, name: str, check_func: Callable):
        """注册健康检查"""
        self.health_checks[name] = check_func
        logger.info(f"健康检查已注册: {name}")
    
    async def check_system_health(self) -> dict:
        """检查系统健康状态"""
        health_results = {}
        overall_healthy = True
        
        for name, check_func in self.health_checks.items():
            try:
                result = await check_func()
                self.health_status[name] = result
                health_results[name] = {
                    'healthy': result,
                    'timestamp': time.time()
                }
                
                if not result:
                    overall_healthy = False
                    
            except Exception as e:
                self.health_status[name] = False
                health_results[name] = {
                    'healthy': False,
                    'error': str(e),
                    'timestamp': time.time()
                }
                overall_healthy = False
        
        # 更新连续失败计数
        if overall_healthy:
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
        
        return {
            'overall_healthy': overall_healthy,
            'consecutive_failures': self.consecutive_failures,
            'checks': health_results,
            'timestamp': time.time()
        }
    
    async def handle_system_failure(self, failure_info: dict):
        """处理系统级失败"""
        logger.error(f"系统级失败: {failure_info}")
        
        # 根据失败情况采取相应措施
        if self.consecutive_failures >= self.fault_tolerance_config['max_consecutive_failures']:
            await self._initiate_emergency_recovery()
    
    async def _initiate_emergency_recovery(self):
        """启动紧急恢复"""
        logger.critical("启动紧急恢复程序")
        
        # 重置所有熔断器
        for name in self.recovery_manager.circuit_breakers:
            self.recovery_manager.reset_circuit_breaker(name)
        
        # 清理失败状态
        self.consecutive_failures = 0
        
        logger.info("紧急恢复完成")


# 示例健康检查函数
async def example_health_checks():
    """示例健康检查"""
    
    async def database_health_check():
        """数据库健康检查"""
        # 模拟数据库连接检查
        await asyncio.sleep(0.1)
        return True
    
    async def memory_health_check():
        """内存健康检查"""
        import psutil
        memory_percent = psutil.virtual_memory().percent
        return memory_percent < 90
    
    async def disk_health_check():
        """磁盘健康检查"""
        import psutil
        disk_percent = psutil.disk_usage('/').percent
        return disk_percent < 95
    
    return {
        'database': database_health_check,
        'memory': memory_health_check,
        'disk': disk_health_check
    }


# 示例用法
async def example_usage():
    """示例用法"""
    # 创建恢复管理器
    recovery_manager = TaskRecovery(max_retry_attempts=3)
    
    # 添加回调函数
    async def failure_handler(failure: TaskFailure):
        print(f"任务失败处理: {failure.task_id} - {failure.failure_type.value}")
    
    async def recovery_handler(failure: TaskFailure):
        print(f"任务恢复处理: {failure.task_id}")
    
    recovery_manager.add_failure_callback(failure_handler)
    recovery_manager.add_recovery_callback(recovery_handler)
    
    # 创建容错系统
    fault_tolerance = FaultTolerance(recovery_manager)
    
    # 注册健康检查
    health_checks = await example_health_checks()
    for name, check_func in health_checks.items():
        fault_tolerance.register_health_check(name, check_func)
    
    # 模拟任务失败
    test_errors = [
        (TimeoutError("任务超时"), FailureType.TIMEOUT),
        (ConnectionError("网络连接失败"), FailureType.NETWORK_ERROR),
        (ValueError("参数错误"), FailureType.EXCEPTION)
    ]
    
    for error, failure_type in test_errors:
        task_id = str(uuid.uuid4())
        success = await recovery_manager.handle_failure(
            task_id, error, failure_type, {'context': 'test'}
        )
        print(f"任务 {task_id} 恢复结果: {success}")
        
        await asyncio.sleep(1)
    
    # 检查系统健康
    health_result = await fault_tolerance.check_system_health()
    print(f"系统健康状态: {health_result}")
    
    # 获取统计信息
    stats = recovery_manager.get_recovery_statistics()
    print("恢复统计:", stats)
    
    # 导出恢复日志
    recovery_manager.export_recovery_log("recovery_log.json")
    
    print("容错和恢复示例完成")


if __name__ == "__main__":
    asyncio.run(example_usage())