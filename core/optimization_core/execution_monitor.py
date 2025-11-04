"""
执行监控器

提供任务执行监控、状态跟踪和异常检测功能
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import traceback
from collections import defaultdict, deque
import threading

logger = logging.getLogger(__name__)


class ExecutionState(Enum):
    """执行状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    RETRYING = "retrying"


class ExecutionType(Enum):
    """执行类型"""
    TASK = "task"
    PIPELINE = "pipeline"
    BATCH = "batch"
    SCHEDULED = "scheduled"
    EVENT_DRIVEN = "event_driven"


@dataclass
class ExecutionRecord:
    """执行记录"""
    execution_id: str
    execution_type: ExecutionType
    task_type: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    status: ExecutionState = ExecutionState.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    error_trace: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout: Optional[float] = None
    resource_usage: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'execution_id': self.execution_id,
            'execution_type': self.execution_type.value,
            'task_type': self.task_type,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'status': self.status.value,
            'result': self.result,
            'error': self.error,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'timeout': self.timeout,
            'resource_usage': self.resource_usage,
            'metadata': self.metadata,
            'tags': self.tags
        }


@dataclass
class ExecutionMetrics:
    """执行指标"""
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    cancelled_executions: int = 0
    timeout_executions: int = 0
    average_duration: float = 0.0
    min_duration: float = float('inf')
    max_duration: float = 0.0
    success_rate: float = 0.0
    error_rate: float = 0.0
    throughput: float = 0.0  # 每分钟执行数
    
    def update(self, execution: ExecutionRecord):
        """更新指标"""
        self.total_executions += 1
        
        if execution.status == ExecutionState.COMPLETED:
            self.successful_executions += 1
            if execution.duration:
                self.average_duration = (
                    (self.average_duration * (self.successful_executions - 1) + execution.duration) /
                    self.successful_executions
                )
                self.min_duration = min(self.min_duration, execution.duration)
                self.max_duration = max(self.max_duration, execution.duration)
        
        elif execution.status == ExecutionState.FAILED:
            self.failed_executions += 1
        elif execution.status == ExecutionState.CANCELLED:
            self.cancelled_executions += 1
        elif execution.status == ExecutionState.TIMEOUT:
            self.timeout_executions += 1
        
        # 计算成功率
        if self.total_executions > 0:
            self.success_rate = (self.successful_executions / self.total_executions) * 100
            self.error_rate = (self.failed_executions / self.total_executions) * 100


class ExecutionMonitor:
    """执行监控器"""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        
        # 执行记录管理
        self.active_executions: Dict[str, ExecutionRecord] = {}
        self.execution_history: deque = deque(maxlen=max_history)
        self.executions_by_type: Dict[ExecutionType, deque] = defaultdict(
            lambda: deque(maxlen=max_history // 4)
        )
        self.executions_by_task_type: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_history // 4)
        )
        
        # 指标管理
        self.overall_metrics = ExecutionMetrics()
        self.metrics_by_type: Dict[ExecutionType, ExecutionMetrics] = defaultdict(ExecutionMetrics)
        self.metrics_by_task_type: Dict[str, ExecutionMetrics] = defaultdict(ExecutionMetrics)
        
        # 监控配置
        self.monitoring_enabled = True
        self.alert_thresholds = {
            'error_rate': 10.0,  # 错误率超过10%触发告警
            'timeout_rate': 5.0,  # 超时率超过5%触发告警
            'avg_duration_increase': 50.0  # 平均执行时间增长超过50%触发告警
        }
        
        # 回调函数
        self.execution_callbacks: List[Callable] = []
        self.alert_callbacks: List[Callable] = []
        self.metrics_callbacks: List[Callable] = []
        
        # 统计和清理
        self._lock = threading.RLock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
        # 异常检测
        self.error_patterns: Dict[str, int] = defaultdict(int)
        self.slow_executions: List[str] = []  # 慢执行ID列表
    
    async def start(self):
        """启动执行监控"""
        if self._running:
            return
        
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("执行监控器已启动")
    
    async def stop(self):
        """停止执行监控"""
        if not self._running:
            return
        
        self._running = False
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("执行监控器已停止")
    
    def start_execution(self, execution_id: str, execution_type: ExecutionType,
                       task_type: str, timeout: Optional[float] = None,
                       metadata: Optional[Dict[str, Any]] = None,
                       tags: Optional[List[str]] = None) -> bool:
        """开始执行监控"""
        try:
            with self._lock:
                if execution_id in self.active_executions:
                    logger.warning(f"执行ID {execution_id} 已存在")
                    return False
                
                execution = ExecutionRecord(
                    execution_id=execution_id,
                    execution_type=execution_type,
                    task_type=task_type,
                    start_time=time.time(),
                    timeout=timeout,
                    metadata=metadata or {},
                    tags=tags or []
                )
                
                self.active_executions[execution_id] = execution
                
                # 触发回调
                asyncio.create_task(self._trigger_execution_callbacks('start', execution))
                
                logger.debug(f"开始监控执行: {execution_id}")
                return True
                
        except Exception as e:
            logger.error(f"开始执行监控失败: {e}")
            return False
    
    def update_execution(self, execution_id: str, status: ExecutionState,
                        result: Optional[Any] = None, error: Optional[str] = None,
                        error_trace: Optional[str] = None,
                        resource_usage: Optional[Dict[str, float]] = None) -> bool:
        """更新执行状态"""
        try:
            with self._lock:
                if execution_id not in self.active_executions:
                    logger.warning(f"执行ID {execution_id} 不存在")
                    return False
                
                execution = self.active_executions[execution_id]
                execution.status = status
                
                if result is not None:
                    execution.result = result
                
                if error is not None:
                    execution.error = error
                
                if error_trace is not None:
                    execution.error_trace = error_trace
                
                if resource_usage:
                    execution.resource_usage.update(resource_usage)
                
                # 触发回调
                asyncio.create_task(self._trigger_execution_callbacks('update', execution))
                
                return True
                
        except Exception as e:
            logger.error(f"更新执行状态失败: {e}")
            return False
    
    def complete_execution(self, execution_id: str, status: ExecutionState,
                          result: Optional[Any] = None, error: Optional[str] = None,
                          error_trace: Optional[str] = None,
                          resource_usage: Optional[Dict[str, float]] = None) -> bool:
        """完成执行监控"""
        try:
            with self._lock:
                if execution_id not in self.active_executions:
                    logger.warning(f"执行ID {execution_id} 不存在")
                    return False
                
                execution = self.active_executions[execution_id]
                execution.end_time = time.time()
                execution.duration = execution.end_time - execution.start_time
                execution.status = status
                
                if result is not None:
                    execution.result = result
                
                if error is not None:
                    execution.error = error
                
                if error_trace is not None:
                    execution.error_trace = error_trace
                
                if resource_usage:
                    execution.resource_usage.update(resource_usage)
                
                # 移动到历史记录
                self.execution_history.append(execution)
                self.executions_by_type[execution.execution_type].append(execution)
                self.executions_by_task_type[execution.task_type].append(execution)
                
                # 更新指标
                self._update_metrics(execution)
                
                # 检测异常
                asyncio.create_task(self._detect_anomalies(execution))
                
                # 从活跃执行中移除
                del self.active_executions[execution_id]
                
                # 触发回调
                asyncio.create_task(self._trigger_execution_callbacks('complete', execution))
                
                logger.debug(f"完成执行监控: {execution_id}, 状态: {status.value}")
                return True
                
        except Exception as e:
            logger.error(f"完成执行监控失败: {e}")
            return False
    
    def _update_metrics(self, execution: ExecutionRecord):
        """更新指标"""
        try:
            # 更新总体指标
            self.overall_metrics.update(execution)
            
            # 更新类型指标
            self.metrics_by_type[execution.execution_type].update(execution)
            
            # 更新任务类型指标
            self.metrics_by_task_type[execution.task_type].update(execution)
            
            # 更新错误模式统计
            if execution.error:
                self.error_patterns[execution.error] += 1
            
            # 检测慢执行
            if execution.duration and execution.duration > 60:  # 超过1分钟
                self.slow_executions.append(execution.execution_id)
                if len(self.slow_executions) > 100:  # 保持最近100个
                    self.slow_executions.pop(0)
            
        except Exception as e:
            logger.error(f"更新指标失败: {e}")
    
    async def _detect_anomalies(self, execution: ExecutionRecord):
        """检测异常"""
        try:
            # 检测错误率异常
            if self.overall_metrics.error_rate > self.alert_thresholds['error_rate']:
                await self._trigger_alert('high_error_rate', {
                    'error_rate': self.overall_metrics.error_rate,
                    'threshold': self.alert_thresholds['error_rate']
                })
            
            # 检测超时率异常
            timeout_rate = (self.overall_metrics.timeout_executions / 
                          max(1, self.overall_metrics.total_executions)) * 100
            if timeout_rate > self.alert_thresholds['timeout_rate']:
                await self._trigger_alert('high_timeout_rate', {
                    'timeout_rate': timeout_rate,
                    'threshold': self.alert_thresholds['timeout_rate']
                })
            
            # 检测慢执行
            if execution.duration and execution.duration > 300:  # 超过5分钟
                await self._trigger_alert('slow_execution', {
                    'execution_id': execution.execution_id,
                    'duration': execution.duration,
                    'task_type': execution.task_type
                })
            
            # 检测重复错误
            if execution.error and self.error_patterns[execution.error] > 5:
                await self._trigger_alert('recurring_error', {
                    'error': execution.error,
                    'count': self.error_patterns[execution.error]
                })
                
        except Exception as e:
            logger.error(f"检测异常失败: {e}")
    
    async def _trigger_alert(self, alert_type: str, data: Dict[str, Any]):
        """触发告警"""
        try:
            alert_data = {
                'alert_type': alert_type,
                'timestamp': time.time(),
                'data': data
            }
            
            for callback in self.alert_callbacks:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert_data)
                else:
                    callback(alert_data)
            
            logger.warning(f"执行监控告警: {alert_type}, 数据: {data}")
            
        except Exception as e:
            logger.error(f"触发告警失败: {e}")
    
    async def _trigger_execution_callbacks(self, event_type: str, execution: ExecutionRecord):
        """触发执行回调"""
        try:
            for callback in self.execution_callbacks:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event_type, execution)
                else:
                    callback(event_type, execution)
        except Exception as e:
            logger.error(f"触发执行回调失败: {e}")
    
    async def _cleanup_loop(self):
        """清理循环"""
        while self._running:
            try:
                await asyncio.sleep(300)  # 每5分钟清理一次
                
                # 清理过期的慢执行记录
                current_time = time.time()
                self.slow_executions = [
                    exec_id for exec_id in self.slow_executions
                    if exec_id in [e.execution_id for e in self.execution_history]
                ]
                
                # 清理错误模式统计（保留最近的错误）
                if len(self.error_patterns) > 100:
                    sorted_errors = sorted(
                        self.error_patterns.items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )[:50]
                    self.error_patterns = dict(sorted_errors)
                
                # 触发指标回调
                await self._trigger_metrics_callbacks()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"清理循环异常: {e}")
    
    async def _trigger_metrics_callbacks(self):
        """触发指标回调"""
        try:
            metrics_data = {
                'overall': self.overall_metrics,
                'by_type': dict(self.metrics_by_type),
                'by_task_type': dict(self.metrics_by_task_type),
                'timestamp': time.time()
            }
            
            for callback in self.metrics_callbacks:
                if asyncio.iscoroutinefunction(callback):
                    await callback(metrics_data)
                else:
                    callback(metrics_data)
                    
        except Exception as e:
            logger.error(f"触发指标回调失败: {e}")
    
    def get_execution_status(self, execution_id: str) -> Optional[ExecutionState]:
        """获取执行状态"""
        try:
            if execution_id in self.active_executions:
                return self.active_executions[execution_id].status
            else:
                # 在历史记录中查找
                for execution in self.execution_history:
                    if execution.execution_id == execution_id:
                        return execution.status
            return None
        except Exception as e:
            logger.error(f"获取执行状态失败: {e}")
            return None
    
    def get_execution_details(self, execution_id: str) -> Optional[ExecutionRecord]:
        """获取执行详情"""
        try:
            # 查找活跃执行
            if execution_id in self.active_executions:
                return self.active_executions[execution_id]
            
            # 在历史记录中查找
            for execution in self.execution_history:
                if execution.execution_id == execution_id:
                    return execution
            
            return None
        except Exception as e:
            logger.error(f"获取执行详情失败: {e}")
            return None
    
    def get_active_executions(self, execution_type: Optional[ExecutionType] = None,
                            task_type: Optional[str] = None) -> List[ExecutionRecord]:
        """获取活跃执行"""
        try:
            executions = list(self.active_executions.values())
            
            if execution_type:
                executions = [e for e in executions if e.execution_type == execution_type]
            
            if task_type:
                executions = [e for e in executions if e.task_type == task_type]
            
            return executions
        except Exception as e:
            logger.error(f"获取活跃执行失败: {e}")
            return []
    
    def get_execution_history(self, hours: int = 24, 
                            execution_type: Optional[ExecutionType] = None,
                            task_type: Optional[str] = None,
                            status: Optional[ExecutionState] = None) -> List[ExecutionRecord]:
        """获取执行历史"""
        try:
            cutoff_time = time.time() - (hours * 3600)
            executions = [e for e in self.execution_history if e.start_time >= cutoff_time]
            
            if execution_type:
                executions = [e for e in executions if e.execution_type == execution_type]
            
            if task_type:
                executions = [e for e in executions if e.task_type == task_type]
            
            if status:
                executions = [e for e in executions if e.status == status]
            
            return executions
        except Exception as e:
            logger.error(f"获取执行历史失败: {e}")
            return []
    
    def get_overall_metrics(self) -> ExecutionMetrics:
        """获取总体指标"""
        return self.overall_metrics
    
    def get_metrics_by_type(self, execution_type: ExecutionType) -> Optional[ExecutionMetrics]:
        """获取按类型分类的指标"""
        return self.metrics_by_type.get(execution_type)
    
    def get_metrics_by_task_type(self, task_type: str) -> Optional[ExecutionMetrics]:
        """获取按任务类型分类的指标"""
        return self.metrics_by_task_type.get(task_type)
    
    def get_error_patterns(self, limit: int = 20) -> List[Tuple[str, int]]:
        """获取错误模式"""
        try:
            sorted_errors = sorted(
                self.error_patterns.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            return sorted_errors[:limit]
        except Exception as e:
            logger.error(f"获取错误模式失败: {e}")
            return []
    
    def get_slow_executions(self, limit: int = 50) -> List[str]:
        """获取慢执行列表"""
        return self.slow_executions[-limit:]
    
    def add_execution_callback(self, callback: Callable):
        """添加执行回调"""
        self.execution_callbacks.append(callback)
    
    def add_alert_callback(self, callback: Callable):
        """添加告警回调"""
        self.alert_callbacks.append(callback)
    
    def add_metrics_callback(self, callback: Callable):
        """添加指标回调"""
        self.metrics_callbacks.append(callback)
    
    def set_alert_thresholds(self, thresholds: Dict[str, float]):
        """设置告警阈值"""
        self.alert_thresholds.update(thresholds)
        logger.info(f"告警阈值已更新: {self.alert_thresholds}")
    
    def get_monitor_stats(self) -> Dict[str, Any]:
        """获取监控统计"""
        try:
            return {
                'active_executions': len(self.active_executions),
                'total_history': len(self.execution_history),
                'monitoring_enabled': self.monitoring_enabled,
                'overall_metrics': {
                    'total_executions': self.overall_metrics.total_executions,
                    'success_rate': self.overall_metrics.success_rate,
                    'error_rate': self.overall_metrics.error_rate,
                    'average_duration': self.overall_metrics.average_duration
                },
                'error_patterns_count': len(self.error_patterns),
                'slow_executions_count': len(self.slow_executions),
                'execution_types': list(self.metrics_by_type.keys()),
                'task_types': list(self.metrics_by_task_type.keys())
            }
        except Exception as e:
            logger.error(f"获取监控统计失败: {e}")
            return {}