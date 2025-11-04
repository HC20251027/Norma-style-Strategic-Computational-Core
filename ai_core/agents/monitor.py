"""
任务监控器

提供任务状态跟踪、进度监控、性能指标收集等功能
"""

import asyncio
import time
import psutil
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json

from .task_models import Task, TaskStatus, TaskPriority, TaskRegistry


@dataclass
class MonitorConfig:
    """监控配置"""
    update_interval: float = 1.0          # 更新间隔（秒）
    metrics_retention: int = 3600         # 指标保留时间（秒）
    enable_system_metrics: bool = True    # 是否启用系统指标
    enable_alerts: bool = True           # 是否启用告警
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "cpu_usage": 80.0,
        "memory_usage": 85.0,
        "disk_usage": 90.0,
        "task_failure_rate": 10.0,
        "avg_task_duration": 300.0
    })


@dataclass
class SystemMetrics:
    """系统指标"""
    timestamp: datetime = field(default_factory=datetime.now)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used: int = 0
    memory_available: int = 0
    disk_percent: float = 0.0
    disk_used: int = 0
    disk_total: int = 0
    network_io: Dict[str, int] = field(default_factory=dict)
    process_count: int = 0
    thread_count: int = 0


@dataclass
class TaskMetrics:
    """任务指标"""
    task_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    duration: Optional[float] = None
    retry_count: int = 0
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    error_count: int = 0


@dataclass
class Alert:
    """告警"""
    id: str
    level: str  # INFO, WARNING, ERROR, CRITICAL
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class TaskMonitor:
    """任务监控器"""
    
    def __init__(self, config: MonitorConfig = None):
        self.config = config or MonitorConfig()
        self.logger = logging.getLogger(__name__)
        
        # 任务注册表
        self.task_registry: Optional[TaskRegistry] = None
        
        # 指标存储
        self.system_metrics_history: deque = deque(maxlen=self.config.metrics_retention)
        self.task_metrics_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.config.metrics_retention)
        )
        
        # 告警管理
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        # 回调函数
        self.status_change_callbacks: List[Callable] = []
        self.progress_update_callbacks: List[Callable] = []
        self.alert_callbacks: List[Callable] = []
        
        # 控制任务
        self.monitor_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        # 统计信息
        self.stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "cancelled_tasks": 0,
            "avg_duration": 0.0,
            "throughput": 0.0,
            "error_rate": 0.0
        }
    
    def set_task_registry(self, registry: TaskRegistry):
        """设置任务注册表"""
        self.task_registry = registry
    
    async def start(self):
        """启动监控器"""
        if self.is_running:
            return
        
        self.is_running = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        self.logger.info("任务监控器已启动")
    
    async def stop(self):
        """停止监控器"""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("任务监控器已停止")
    
    async def track_task_status(self, task: Task, old_status: TaskStatus, new_status: TaskStatus):
        """跟踪任务状态变化"""
        # 记录指标
        await self._record_task_metrics(task)
        
        # 检查状态变化告警
        await self._check_status_alerts(task, old_status, new_status)
        
        # 调用状态变化回调
        await self._notify_status_change_callbacks(task, old_status, new_status)
        
        # 更新统计信息
        await self._update_stats(task, new_status)
    
    async def track_task_progress(self, task: Task, progress: float):
        """跟踪任务进度"""
        # 记录指标
        task.metrics.progress = progress
        await self._record_task_metrics(task)
        
        # 调用进度更新回调
        await self._notify_progress_update_callbacks(task, progress)
        
        # 检查进度告警
        await self._check_progress_alerts(task, progress)
    
    async def track_task_error(self, task: Task, error: str):
        """跟踪任务错误"""
        # 记录错误指标
        task.metrics.retry_count += 1
        
        # 检查错误告警
        await self._check_error_alerts(task, error)
        
        # 记录告警
        alert = Alert(
            id=f"task_error_{task.id}_{int(time.time())}",
            level="ERROR",
            message=f"任务 {task.name} 发生错误: {error}",
            metadata={"task_id": task.id, "error": error}
        )
        await self._add_alert(alert)
    
    async def get_system_metrics(self) -> Optional[SystemMetrics]:
        """获取当前系统指标"""
        if not self.config.enable_system_metrics:
            return None
        
        try:
            # 获取系统指标
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            process_count = len(psutil.pids())
            
            # 获取线程数（所有进程的总线程数）
            thread_count = 0
            for proc in psutil.process_iter(['num_threads']):
                try:
                    thread_count += proc.info['num_threads'] or 0
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used=memory.used,
                memory_available=memory.available,
                disk_percent=disk.percent,
                disk_used=disk.used,
                disk_total=disk.total,
                network_io={
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv
                },
                process_count=process_count,
                thread_count=thread_count
            )
        except Exception as e:
            self.logger.error(f"获取系统指标失败: {str(e)}")
            return None
    
    async def get_task_metrics(self, task_id: str) -> List[TaskMetrics]:
        """获取任务指标历史"""
        return list(self.task_metrics_history.get(task_id, []))
    
    async def get_recent_alerts(self, hours: int = 24) -> List[Alert]:
        """获取最近的告警"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alert_history if alert.timestamp > cutoff_time]
    
    async def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警"""
        return [alert for alert in self.active_alerts.values() if not alert.resolved]
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """解决告警"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            self.logger.info(f"告警已解决: {alert_id}")
            return True
        return False
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        # 计算任务吞吐量（每分钟完成的任务数）
        recent_completed = await self._get_recent_completed_tasks(60)  # 最近1分钟
        throughput = len(recent_completed) / 60.0  # 任务/秒
        
        # 计算错误率
        recent_tasks = await self._get_recent_tasks(60)
        if recent_tasks:
            failed_count = sum(1 for task in recent_tasks if task.status == TaskStatus.FAILED)
            error_rate = (failed_count / len(recent_tasks)) * 100
        else:
            error_rate = 0.0
        
        # 计算平均执行时间
        completed_with_duration = [
            task for task in recent_completed 
            if task.metrics.duration is not None
        ]
        if completed_with_duration:
            avg_duration = sum(task.metrics.duration for task in completed_with_duration) / len(completed_with_duration)
        else:
            avg_duration = 0.0
        
        return {
            "total_tasks": self.stats["total_tasks"],
            "completed_tasks": self.stats["completed_tasks"],
            "failed_tasks": self.stats["failed_tasks"],
            "cancelled_tasks": self.stats["cancelled_tasks"],
            "throughput": throughput,
            "error_rate": error_rate,
            "avg_duration": avg_duration,
            "active_alerts": len(await self.get_active_alerts())
        }
    
    async def export_metrics(self, file_path: str, format: str = "json") -> bool:
        """导出指标数据"""
        try:
            metrics_data = {
                "system_metrics": [
                    {
                        "timestamp": m.timestamp.isoformat(),
                        "cpu_percent": m.cpu_percent,
                        "memory_percent": m.memory_percent,
                        "disk_percent": m.disk_percent,
                        "process_count": m.process_count,
                        "thread_count": m.thread_count
                    }
                    for m in self.system_metrics_history
                ],
                "task_metrics": {
                    task_id: [
                        {
                            "timestamp": m.timestamp.isoformat(),
                            "status": m.status.value,
                            "progress": m.progress,
                            "duration": m.duration,
                            "retry_count": m.retry_count
                        }
                        for m in metrics
                    ]
                    for task_id, metrics in self.task_metrics_history.items()
                },
                "alerts": [
                    {
                        "id": alert.id,
                        "level": alert.level,
                        "message": alert.message,
                        "timestamp": alert.timestamp.isoformat(),
                        "resolved": alert.resolved,
                        "metadata": alert.metadata
                    }
                    for alert in self.alert_history
                ],
                "stats": self.stats
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(metrics_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"指标数据已导出到 {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"导出指标数据失败: {str(e)}")
            return False
    
    def add_status_change_callback(self, callback: Callable):
        """添加状态变化回调"""
        self.status_change_callbacks.append(callback)
    
    def add_progress_update_callback(self, callback: Callable):
        """添加进度更新回调"""
        self.progress_update_callbacks.append(callback)
    
    def add_alert_callback(self, callback: Callable):
        """添加告警回调"""
        self.alert_callbacks.append(callback)
    
    async def _monitor_loop(self):
        """监控主循环"""
        while self.is_running:
            try:
                # 收集系统指标
                if self.config.enable_system_metrics:
                    metrics = await self.get_system_metrics()
                    if metrics:
                        self.system_metrics_history.append(metrics)
                        await self._check_system_alerts(metrics)
                
                # 清理过期的指标数据
                await self._cleanup_old_metrics()
                
                # 等待下次更新
                await asyncio.sleep(self.config.update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"监控循环异常: {str(e)}")
                await asyncio.sleep(5)
    
    async def _record_task_metrics(self, task: Task):
        """记录任务指标"""
        metrics = TaskMetrics(
            task_id=task.id,
            status=task.status,
            progress=task.metrics.progress,
            duration=task.metrics.duration,
            retry_count=task.metrics.retry_count,
            memory_usage=task.metrics.memory_usage,
            cpu_usage=task.metrics.cpu_usage
        )
        
        self.task_metrics_history[task.id].append(metrics)
    
    async def _check_status_alerts(self, task: Task, old_status: TaskStatus, new_status: TaskStatus):
        """检查状态变化告警"""
        # 任务失败告警
        if new_status == TaskStatus.FAILED:
            alert = Alert(
                id=f"task_failed_{task.id}_{int(time.time())}",
                level="ERROR",
                message=f"任务 {task.name} 执行失败",
                metadata={"task_id": task.id, "old_status": old_status.value}
            )
            await self._add_alert(alert)
        
        # 任务超时告警
        elif new_status == TaskStatus.TIMEOUT:
            alert = Alert(
                id=f"task_timeout_{task.id}_{int(time.time())}",
                level="WARNING",
                message=f"任务 {task.name} 执行超时",
                metadata={"task_id": task.id, "timeout": task.config.timeout}
            )
            await self._add_alert(alert)
    
    async def _check_progress_alerts(self, task: Task, progress: float):
        """检查进度告警"""
        # 进度停滞告警（如果进度长时间没有变化）
        # 这里简化处理，实际需要更复杂的逻辑
        pass
    
    async def _check_error_alerts(self, task: Task, error: str):
        """检查错误告警"""
        # 多次重试告警
        if task.metrics.retry_count >= task.config.retry_count:
            alert = Alert(
                id=f"task_retry_limit_{task.id}_{int(time.time())}",
                level="WARNING",
                message=f"任务 {task.name} 重试次数已达上限",
                metadata={"task_id": task.id, "retry_count": task.metrics.retry_count}
            )
            await self._add_alert(alert)
    
    async def _check_system_alerts(self, metrics: SystemMetrics):
        """检查系统告警"""
        thresholds = self.config.alert_thresholds
        
        # CPU使用率告警
        if metrics.cpu_percent > thresholds.get("cpu_usage", 80):
            alert = Alert(
                id=f"high_cpu_{int(time.time())}",
                level="WARNING",
                message=f"CPU使用率过高: {metrics.cpu_percent:.1f}%",
                metadata={"cpu_percent": metrics.cpu_percent}
            )
            await self._add_alert(alert)
        
        # 内存使用率告警
        if metrics.memory_percent > thresholds.get("memory_usage", 85):
            alert = Alert(
                id=f"high_memory_{int(time.time())}",
                level="WARNING",
                message=f"内存使用率过高: {metrics.memory_percent:.1f}%",
                metadata={"memory_percent": metrics.memory_percent}
            )
            await self._add_alert(alert)
        
        # 磁盘使用率告警
        if metrics.disk_percent > thresholds.get("disk_usage", 90):
            alert = Alert(
                id=f"high_disk_{int(time.time())}",
                level="CRITICAL",
                message=f"磁盘使用率过高: {metrics.disk_percent:.1f}%",
                metadata={"disk_percent": metrics.disk_percent}
            )
            await self._add_alert(alert)
    
    async def _add_alert(self, alert: Alert):
        """添加告警"""
        self.active_alerts[alert.id] = alert
        self.alert_history.append(alert)
        
        # 调用告警回调
        await self._notify_alert_callbacks(alert)
        
        self.logger.warning(f"告警: {alert.message}")
    
    async def _notify_status_change_callbacks(self, task: Task, old_status: TaskStatus, new_status: TaskStatus):
        """通知状态变化回调"""
        for callback in self.status_change_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(task, old_status, new_status)
                else:
                    callback(task, old_status, new_status)
            except Exception as e:
                self.logger.error(f"状态变化回调异常: {str(e)}")
    
    async def _notify_progress_update_callbacks(self, task: Task, progress: float):
        """通知进度更新回调"""
        for callback in self.progress_update_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(task, progress)
                else:
                    callback(task, progress)
            except Exception as e:
                self.logger.error(f"进度更新回调异常: {str(e)}")
    
    async def _notify_alert_callbacks(self, alert: Alert):
        """通知告警回调"""
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                self.logger.error(f"告警回调异常: {str(e)}")
    
    async def _update_stats(self, task: Task, new_status: TaskStatus):
        """更新统计信息"""
        self.stats["total_tasks"] += 1
        
        if new_status == TaskStatus.COMPLETED:
            self.stats["completed_tasks"] += 1
        elif new_status == TaskStatus.FAILED:
            self.stats["failed_tasks"] += 1
        elif new_status == TaskStatus.CANCELLED:
            self.stats["cancelled_tasks"] += 1
        
        # 更新错误率
        total_finished = (self.stats["completed_tasks"] + 
                         self.stats["failed_tasks"] + 
                         self.stats["cancelled_tasks"])
        if total_finished > 0:
            self.stats["error_rate"] = (self.stats["failed_tasks"] / total_finished) * 100
    
    async def _cleanup_old_metrics(self):
        """清理过期的指标数据"""
        cutoff_time = datetime.now() - timedelta(seconds=self.config.metrics_retention)
        
        # 清理系统指标
        while (self.system_metrics_history and 
               self.system_metrics_history[0].timestamp < cutoff_time):
            self.system_metrics_history.popleft()
        
        # 清理任务指标
        for task_id in list(self.task_metrics_history.keys()):
            metrics_queue = self.task_metrics_history[task_id]
            while (metrics_queue and 
                   metrics_queue[0].timestamp < cutoff_time):
                metrics_queue.popleft()
            
            # 如果没有指标了，删除该任务的历史记录
            if not metrics_queue:
                del self.task_metrics_history[task_id]
    
    async def _get_recent_completed_tasks(self, minutes: int) -> List[Task]:
        """获取最近完成的任务"""
        if not self.task_registry:
            return []
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        all_tasks = await self.task_registry.list_tasks()
        
        return [
            task for task in all_tasks
            if (task.status == TaskStatus.COMPLETED and 
                task.metrics.completed_at and 
                task.metrics.completed_at > cutoff_time)
        ]
    
    async def _get_recent_tasks(self, minutes: int) -> List[Task]:
        """获取最近的任务"""
        if not self.task_registry:
            return []
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        all_tasks = await self.task_registry.list_tasks()
        
        return [
            task for task in all_tasks
            if task.created_at > cutoff_time
        ]


class MetricsCollector:
    """指标收集器"""
    
    def __init__(self, monitor: TaskMonitor):
        self.monitor = monitor
        self.logger = logging.getLogger(__name__)
    
    async def collect_task_performance_metrics(self, task: Task) -> Dict[str, Any]:
        """收集任务性能指标"""
        metrics = {
            "task_id": task.id,
            "name": task.name,
            "status": task.status.value,
            "progress": task.metrics.progress,
            "duration": task.metrics.duration,
            "retry_count": task.metrics.retry_count,
            "created_at": task.created_at.isoformat(),
            "started_at": task.metrics.started_at.isoformat() if task.metrics.started_at else None,
            "completed_at": task.metrics.completed_at.isoformat() if task.metrics.completed_at else None
        }
        
        return metrics
    
    async def collect_system_performance_metrics(self) -> Dict[str, Any]:
        """收集系统性能指标"""
        system_metrics = await self.monitor.get_system_metrics()
        if not system_metrics:
            return {}
        
        return {
            "cpu_percent": system_metrics.cpu_percent,
            "memory_percent": system_metrics.memory_percent,
            "memory_used_gb": system_metrics.memory_used / (1024**3),
            "memory_available_gb": system_metrics.memory_available / (1024**3),
            "disk_percent": system_metrics.disk_percent,
            "disk_used_gb": system_metrics.disk_used / (1024**3),
            "disk_total_gb": system_metrics.disk_total / (1024**3),
            "process_count": system_metrics.process_count,
            "thread_count": system_metrics.thread_count,
            "timestamp": system_metrics.timestamp.isoformat()
        }