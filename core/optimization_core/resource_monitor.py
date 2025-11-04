"""
资源监控器

提供系统资源监控、告警和趋势分析功能
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import psutil
import json
from collections import deque, defaultdict
import statistics

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """指标类型"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertSeverity(Enum):
    """告警严重级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """告警状态"""
    ACTIVE = "active"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class Metric:
    """指标"""
    name: str
    value: float
    metric_type: MetricType
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    unit: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'value': self.value,
            'type': self.metric_type.value,
            'timestamp': self.timestamp,
            'labels': self.labels,
            'unit': self.unit
        }


@dataclass
class AlertRule:
    """告警规则"""
    rule_id: str
    name: str
    metric_name: str
    condition: str  # >, <, >=, <=, ==, !=
    threshold: float
    severity: AlertSeverity
    duration: float = 60.0  # 持续时间（秒）
    enabled: bool = True
    description: str = ""
    created_at: float = field(default_factory=time.time)


@dataclass
class Alert:
    """告警"""
    alert_id: str
    rule_id: str
    metric_name: str
    current_value: float
    threshold: float
    severity: AlertSeverity
    status: AlertStatus = AlertStatus.ACTIVE
    triggered_at: float = field(default_factory=time.time)
    resolved_at: Optional[float] = None
    message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'alert_id': self.alert_id,
            'rule_id': self.rule_id,
            'metric_name': self.metric_name,
            'current_value': self.current_value,
            'threshold': self.threshold,
            'severity': self.severity.value,
            'status': self.status.value,
            'triggered_at': self.triggered_at,
            'resolved_at': self.resolved_at,
            'message': self.message
        }


@dataclass
class ResourceSnapshot:
    """资源快照"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used: float
    memory_available: float
    disk_usage: Dict[str, float]
    network_io: Dict[str, float]
    process_count: int
    load_average: List[float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_used': self.memory_used,
            'memory_available': self.memory_available,
            'disk_usage': self.disk_usage,
            'network_io': self.network_io,
            'process_count': self.process_count,
            'load_average': self.load_average
        }


class ResourceMonitor:
    """资源监控器"""
    
    def __init__(self, collection_interval: float = 10.0, retention_hours: int = 24):
        self.collection_interval = collection_interval
        self.retention_hours = retention_hours
        
        # 指标存储
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.current_metrics: Dict[str, Metric] = {}
        
        # 告警管理
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        # 资源快照
        self.resource_snapshots: deque = deque(maxlen=retention_hours * 360 // collection_interval)
        
        # 监控状态
        self.is_monitoring = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # 回调函数
        self.metric_callbacks: List[Callable] = []
        self.alert_callbacks: List[Callable] = []
        
        # 系统指标收集器
        self.system_collectors = {
            'cpu_percent': self._collect_cpu_metrics,
            'memory': self._collect_memory_metrics,
            'disk': self._collect_disk_metrics,
            'network': self._collect_network_metrics,
            'process': self._collect_process_metrics
        }
    
    async def start(self):
        """启动资源监控"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info(f"资源监控器已启动，收集间隔: {self.collection_interval}秒")
    
    async def stop(self):
        """停止资源监控"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("资源监控器已停止")
    
    async def _monitoring_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                start_time = time.time()
                
                # 收集系统指标
                await self._collect_system_metrics()
                
                # 收集自定义指标
                await self._collect_custom_metrics()
                
                # 检查告警规则
                await self._check_alert_rules()
                
                # 计算资源快照
                await self._create_resource_snapshot()
                
                # 触发回调
                await self._trigger_metric_callbacks()
                
                # 计算睡眠时间
                elapsed = time.time() - start_time
                sleep_time = max(0, self.collection_interval - elapsed)
                
                await asyncio.sleep(sleep_time)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"监控循环异常: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _collect_system_metrics(self):
        """收集系统指标"""
        try:
            # CPU指标
            cpu_percent = psutil.cpu_percent(interval=1)
            await self._record_metric('system.cpu.percent', cpu_percent, MetricType.GAUGE)
            
            # 内存指标
            memory = psutil.virtual_memory()
            await self._record_metric('system.memory.percent', memory.percent, MetricType.GAUGE)
            await self._record_metric('system.memory.used_gb', memory.used / (1024**3), MetricType.GAUGE)
            await self._record_metric('system.memory.available_gb', memory.available / (1024**3), MetricType.GAUGE)
            
            # 磁盘指标
            disk_usage = psutil.disk_usage('/')
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            await self._record_metric('system.disk.percent', disk_percent, MetricType.GAUGE)
            await self._record_metric('system.disk.used_gb', disk_usage.used / (1024**3), MetricType.GAUGE)
            await self._record_metric('system.disk.free_gb', disk_usage.free / (1024**3), MetricType.GAUGE)
            
            # 网络指标
            network_io = psutil.net_io_counters()
            await self._record_metric('system.network.bytes_sent', network_io.bytes_sent, MetricType.COUNTER)
            await self._record_metric('system.network.bytes_recv', network_io.bytes_recv, MetricType.COUNTER)
            await self._record_metric('system.network.packets_sent', network_io.packets_sent, MetricType.COUNTER)
            await self._record_metric('system.network.packets_recv', network_io.packets_recv, MetricType.COUNTER)
            
            # 进程指标
            process_count = len(psutil.pids())
            await self._record_metric('system.process.count', process_count, MetricType.GAUGE)
            
            # 负载平均
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
            await self._record_metric('system.load.1min', load_avg[0], MetricType.GAUGE)
            await self._record_metric('system.load.5min', load_avg[1], MetricType.GAUGE)
            await self._record_metric('system.load.15min', load_avg[2], MetricType.GAUGE)
            
        except Exception as e:
            logger.error(f"收集系统指标失败: {e}")
    
    async def _collect_custom_metrics(self):
        """收集自定义指标"""
        try:
            for collector_name, collector_func in self.system_collectors.items():
                if collector_name not in ['cpu_percent', 'memory', 'disk', 'network', 'process']:
                    await collector_func()
        except Exception as e:
            logger.error(f"收集自定义指标失败: {e}")
    
    async def _collect_cpu_metrics(self):
        """收集CPU指标"""
        # CPU使用率已在上方收集
        pass
    
    async def _collect_memory_metrics(self):
        """收集内存指标"""
        # 内存指标已在上方收集
        pass
    
    async def _collect_disk_metrics(self):
        """收集磁盘指标"""
        # 磁盘指标已在上方收集
        pass
    
    async def _collect_network_metrics(self):
        """收集网络指标"""
        # 网络指标已在上方收集
        pass
    
    async def _collect_process_metrics(self):
        """收集进程指标"""
        # 进程指标已在上方收集
        pass
    
    async def _record_metric(self, name: str, value: float, 
                           metric_type: MetricType = MetricType.GAUGE,
                           labels: Optional[Dict[str, str]] = None,
                           unit: str = ""):
        """记录指标"""
        try:
            metric = Metric(
                name=name,
                value=value,
                metric_type=metric_type,
                timestamp=time.time(),
                labels=labels or {},
                unit=unit
            )
            
            self.metrics[name].append(metric)
            self.current_metrics[name] = metric
            
        except Exception as e:
            logger.error(f"记录指标失败: {e}")
    
    async def _create_resource_snapshot(self):
        """创建资源快照"""
        try:
            snapshot = ResourceSnapshot(
                timestamp=time.time(),
                cpu_percent=self.current_metrics.get('system.cpu.percent', Metric('', 0, MetricType.GAUGE, 0)).value,
                memory_percent=self.current_metrics.get('system.memory.percent', Metric('', 0, MetricType.GAUGE, 0)).value,
                memory_used=self.current_metrics.get('system.memory.used_gb', Metric('', 0, MetricType.GAUGE, 0)).value,
                memory_available=self.current_metrics.get('system.memory.available_gb', Metric('', 0, MetricType.GAUGE, 0)).value,
                disk_usage={
                    'percent': self.current_metrics.get('system.disk.percent', Metric('', 0, MetricType.GAUGE, 0)).value,
                    'used_gb': self.current_metrics.get('system.disk.used_gb', Metric('', 0, MetricType.GAUGE, 0)).value,
                    'free_gb': self.current_metrics.get('system.disk.free_gb', Metric('', 0, MetricType.GAUGE, 0)).value
                },
                network_io={
                    'bytes_sent': self.current_metrics.get('system.network.bytes_sent', Metric('', 0, MetricType.GAUGE, 0)).value,
                    'bytes_recv': self.current_metrics.get('system.network.bytes_recv', Metric('', 0, MetricType.GAUGE, 0)).value
                },
                process_count=int(self.current_metrics.get('system.process.count', Metric('', 0, MetricType.GAUGE, 0)).value),
                load_average=[
                    self.current_metrics.get('system.load.1min', Metric('', 0, MetricType.GAUGE, 0)).value,
                    self.current_metrics.get('system.load.5min', Metric('', 0, MetricType.GAUGE, 0)).value,
                    self.current_metrics.get('system.load.15min', Metric('', 0, MetricType.GAUGE, 0)).value
                ]
            )
            
            self.resource_snapshots.append(snapshot)
            
        except Exception as e:
            logger.error(f"创建资源快照失败: {e}")
    
    async def _check_alert_rules(self):
        """检查告警规则"""
        try:
            current_time = time.time()
            
            for rule_id, rule in self.alert_rules.items():
                if not rule.enabled:
                    continue
                
                # 获取指标值
                metric = self.current_metrics.get(rule.metric_name)
                if not metric:
                    continue
                
                current_value = metric.value
                
                # 检查条件
                condition_met = self._evaluate_condition(current_value, rule.condition, rule.threshold)
                
                if condition_met:
                    # 检查是否已经触发告警
                    existing_alert = self.active_alerts.get(rule_id)
                    
                    if not existing_alert:
                        # 创建新告警
                        alert = Alert(
                            alert_id=f"alert_{rule_id}_{int(current_time)}",
                            rule_id=rule_id,
                            metric_name=rule.metric_name,
                            current_value=current_value,
                            threshold=rule.threshold,
                            severity=rule.severity,
                            message=f"{rule.name}: {rule.metric_name} {rule.condition} {rule.threshold} (当前值: {current_value})"
                        )
                        
                        self.active_alerts[rule_id] = alert
                        await self._trigger_alert_callbacks(alert)
                        
                        logger.warning(f"告警触发: {alert.message}")
                    
                    elif existing_alert.status == AlertStatus.RESOLVED:
                        # 重新激活告警
                        existing_alert.status = AlertStatus.ACTIVE
                        existing_alert.triggered_at = current_time
                        await self._trigger_alert_callbacks(existing_alert)
                
                else:
                    # 检查是否需要解决告警
                    existing_alert = self.active_alerts.get(rule_id)
                    if existing_alert and existing_alert.status == AlertStatus.ACTIVE:
                        existing_alert.status = AlertStatus.RESOLVED
                        existing_alert.resolved_at = current_time
                        self.alert_history.append(existing_alert)
                        del self.active_alerts[rule_id]
                        
                        await self._trigger_alert_callbacks(existing_alert)
                        
                        logger.info(f"告警解决: {rule.name}")
                        
        except Exception as e:
            logger.error(f"检查告警规则失败: {e}")
    
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """评估条件"""
        try:
            if condition == ">":
                return value > threshold
            elif condition == "<":
                return value < threshold
            elif condition == ">=":
                return value >= threshold
            elif condition == "<=":
                return value <= threshold
            elif condition == "==":
                return abs(value - threshold) < 1e-6
            elif condition == "!=":
                return abs(value - threshold) >= 1e-6
            else:
                logger.warning(f"未知的条件操作符: {condition}")
                return False
        except Exception as e:
            logger.error(f"评估条件失败: {e}")
            return False
    
    async def _trigger_metric_callbacks(self):
        """触发指标回调"""
        try:
            for callback in self.metric_callbacks:
                if asyncio.iscoroutinefunction(callback):
                    await callback(self.current_metrics)
                else:
                    callback(self.current_metrics)
        except Exception as e:
            logger.error(f"触发指标回调失败: {e}")
    
    async def _trigger_alert_callbacks(self, alert: Alert):
        """触发告警回调"""
        try:
            for callback in self.alert_callbacks:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
        except Exception as e:
            logger.error(f"触发告警回调失败: {e}")
    
    def add_metric(self, name: str, value: float, 
                  metric_type: MetricType = MetricType.GAUGE,
                  labels: Optional[Dict[str, str]] = None,
                  unit: str = ""):
        """添加自定义指标"""
        asyncio.create_task(self._record_metric(name, value, metric_type, labels, unit))
    
    def add_alert_rule(self, rule: AlertRule):
        """添加告警规则"""
        self.alert_rules[rule.rule_id] = rule
        logger.info(f"添加告警规则: {rule.name}")
    
    def remove_alert_rule(self, rule_id: str) -> bool:
        """移除告警规则"""
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            logger.info(f"移除告警规则: {rule_id}")
            return True
        return False
    
    def add_metric_callback(self, callback: Callable):
        """添加指标回调"""
        self.metric_callbacks.append(callback)
    
    def add_alert_callback(self, callback: Callable):
        """添加告警回调"""
        self.alert_callbacks.append(callback)
    
    def get_current_metrics(self) -> Dict[str, Metric]:
        """获取当前指标"""
        return self.current_metrics.copy()
    
    def get_metric_history(self, metric_name: str, hours: int = 1) -> List[Metric]:
        """获取指标历史"""
        try:
            cutoff_time = time.time() - (hours * 3600)
            metrics = self.metrics.get(metric_name, [])
            return [m for m in metrics if m.timestamp >= cutoff_time]
        except Exception as e:
            logger.error(f"获取指标历史失败: {e}")
            return []
    
    def get_resource_snapshots(self, hours: int = 1) -> List[ResourceSnapshot]:
        """获取资源快照"""
        try:
            cutoff_time = time.time() - (hours * 3600)
            return [s for s in self.resource_snapshots if s.timestamp >= cutoff_time]
        except Exception as e:
            logger.error(f"获取资源快照失败: {e}")
            return []
    
    def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """获取告警历史"""
        try:
            cutoff_time = time.time() - (hours * 3600)
            return [a for a in self.alert_history if a.triggered_at >= cutoff_time]
        except Exception as e:
            logger.error(f"获取告警历史失败: {e}")
            return []
    
    def get_resource_trends(self, metric_name: str, hours: int = 1) -> Dict[str, Any]:
        """获取资源趋势"""
        try:
            history = self.get_metric_history(metric_name, hours)
            
            if not history:
                return {}
            
            values = [m.value for m in history]
            
            return {
                'metric_name': metric_name,
                'data_points': len(values),
                'min_value': min(values),
                'max_value': max(values),
                'avg_value': statistics.mean(values),
                'median_value': statistics.median(values),
                'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
                'trend': 'increasing' if values[-1] > values[0] else 'decreasing',
                'latest_value': values[-1] if values else 0
            }
            
        except Exception as e:
            logger.error(f"获取资源趋势失败: {e}")
            return {}
    
    def get_monitor_stats(self) -> Dict[str, Any]:
        """获取监控统计"""
        try:
            return {
                'is_monitoring': self.is_monitoring,
                'collection_interval': self.collection_interval,
                'total_metrics': len(self.metrics),
                'active_metrics': len(self.current_metrics),
                'active_alerts': len(self.active_alerts),
                'alert_rules': len(self.alert_rules),
                'resource_snapshots': len(self.resource_snapshots),
                'monitoring_duration': (
                    time.time() - self.resource_snapshots[0].timestamp 
                    if self.resource_snapshots else 0
                )
            }
        except Exception as e:
            logger.error(f"获取监控统计失败: {e}")
            return {}