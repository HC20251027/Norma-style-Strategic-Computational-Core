"""
性能监控和优化
负责监控任务执行性能、收集指标、提供优化建议
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
import time
import threading
import psutil
import json
import statistics
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from abc import ABC, abstractmethod

# 配置日志
logger = logging.getLogger(__name__)


class MetricType(Enum):
    """指标类型"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"


class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Metric:
    """性能指标"""
    name: str
    value: float
    metric_type: MetricType
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    unit: str = ""
    description: str = ""
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'name': self.name,
            'value': self.value,
            'type': self.metric_type.value,
            'timestamp': self.timestamp,
            'labels': self.labels,
            'unit': self.unit,
            'description': self.description
        }


@dataclass
class Alert:
    """性能告警"""
    id: str
    name: str
    level: AlertLevel
    message: str
    metric_name: str
    threshold: float
    current_value: float
    timestamp: float
    resolved: bool = False
    resolved_at: Optional[float] = None
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'id': self.id,
            'name': self.name,
            'level': self.level.value,
            'message': self.message,
            'metric_name': self.metric_name,
            'threshold': self.threshold,
            'current_value': self.current_value,
            'timestamp': self.timestamp,
            'resolved': self.resolved,
            'resolved_at': self.resolved_at,
            'metadata': self.metadata
        }


@dataclass
class PerformanceSnapshot:
    """性能快照"""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    active_tasks: int
    completed_tasks: int
    failed_tasks: int
    avg_response_time: float
    throughput: float
    error_rate: float


class MetricsCollector:
    """指标收集器"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        self._lock = threading.RLock()
        
        logger.info("指标收集器初始化完成")
    
    def record_metric(self, metric: Metric):
        """记录指标"""
        with self._lock:
            self.metrics[metric.name].append(metric)
            
            # 更新计数器
            if metric.metric_type == MetricType.COUNTER:
                self.counters[metric.name] += metric.value
            elif metric.metric_type == MetricType.GAUGE:
                self.gauges[metric.name] = metric.value
            
            logger.debug(f"指标已记录: {metric.name} = {metric.value}")
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None):
        """增加计数器"""
        metric = Metric(
            name=name,
            value=value,
            metric_type=MetricType.COUNTER,
            timestamp=time.time(),
            labels=labels or {}
        )
        self.record_metric(metric)
    
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """设置仪表盘值"""
        metric = Metric(
            name=name,
            value=value,
            metric_type=MetricType.GAUGE,
            timestamp=time.time(),
            labels=labels or {}
        )
        self.record_metric(metric)
    
    def record_timer(self, name: str, duration: float, labels: Dict[str, str] = None):
        """记录计时器"""
        metric = Metric(
            name=name,
            value=duration,
            metric_type=MetricType.TIMER,
            timestamp=time.time(),
            labels=labels or {}
        )
        self.record_metric(metric)
    
    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """记录直方图"""
        metric = Metric(
            name=name,
            value=value,
            metric_type=MetricType.HISTOGRAM,
            timestamp=time.time(),
            labels=labels or {}
        )
        self.record_metric(metric)
    
    def get_metric_history(self, name: str, limit: int = None) -> List[Metric]:
        """获取指标历史"""
        with self._lock:
            history = list(self.metrics[name])
            if limit:
                history = history[-limit:]
            return history
    
    def get_current_value(self, name: str) -> Optional[float]:
        """获取当前值"""
        with self._lock:
            if name in self.gauges:
                return self.gauges[name]
            elif name in self.counters:
                return self.counters[name]
            elif self.metrics[name]:
                return self.metrics[name][-1].value
            return None
    
    def get_statistics(self, name: str) -> Optional[dict]:
        """获取指标统计信息"""
        with self._lock:
            history = list(self.metrics[name])
            if not history:
                return None
            
            values = [m.value for m in history]
            
            return {
                'count': len(values),
                'min': min(values),
                'max': max(values),
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'std': statistics.stdev(values) if len(values) > 1 else 0,
                'p95': np.percentile(values, 95) if len(values) > 1 else values[0],
                'p99': np.percentile(values, 99) if len(values) > 1 else values[0],
                'latest': values[-1],
                'first': values[0]
            }
    
    def export_metrics(self, filepath: str):
        """导出指标"""
        export_data = {
            'export_time': time.time(),
            'counters': dict(self.counters),
            'gauges': dict(self.gauges),
            'metrics': {}
        }
        
        for name, history in self.metrics.items():
            export_data['metrics'][name] = [m.to_dict() for m in history]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"指标已导出到: {filepath}")


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(
        self,
        collection_interval: float = 5.0,
        alert_threshold_cpu: float = 80.0,
        alert_threshold_memory: float = 85.0,
        alert_threshold_response_time: float = 5.0
    ):
        self.collection_interval = collection_interval
        self.alert_thresholds = {
            'cpu': alert_threshold_cpu,
            'memory': alert_threshold_memory,
            'response_time': alert_threshold_response_time
        }
        
        self.metrics_collector = MetricsCollector()
        self.alerts: Dict[str, Alert] = {}
        self.snapshots: deque = deque(maxlen=1000)
        
        self.running = False
        self._monitoring_task: Optional[asyncio.Task] = None
        self._lock = threading.RLock()
        
        # 性能统计
        self.performance_stats = {
            'total_measurements': 0,
            'avg_cpu_usage': 0.0,
            'avg_memory_usage': 0.0,
            'peak_cpu_usage': 0.0,
            'peak_memory_usage': 0.0,
            'total_alerts': 0,
            'active_alerts': 0
        }
        
        # 回调函数
        self.alert_callbacks: List[Callable] = []
        
        logger.info("性能监控器初始化完成")
    
    async def start_monitoring(self):
        """启动监控"""
        if self.running:
            logger.warning("性能监控已在运行")
            return
        
        self.running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("性能监控已启动")
    
    async def stop_monitoring(self):
        """停止监控"""
        if not self.running:
            return
        
        self.running = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("性能监控已停止")
    
    async def _monitoring_loop(self):
        """监控循环"""
        while self.running:
            try:
                # 收集系统指标
                await self._collect_system_metrics()
                
                # 检查告警
                await self._check_alerts()
                
                # 创建性能快照
                await self._create_snapshot()
                
                # 更新统计信息
                self._update_performance_stats()
                
                await asyncio.sleep(self.collection_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"性能监控错误: {e}")
                await asyncio.sleep(1)
    
    async def _collect_system_metrics(self):
        """收集系统指标"""
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=0.1)
        self.metrics_collector.set_gauge('system.cpu.usage', cpu_percent)
        
        # 内存使用率
        memory = psutil.virtual_memory()
        self.metrics_collector.set_gauge('system.memory.usage', memory.percent)
        self.metrics_collector.set_gauge('system.memory.available', memory.available / (1024**3))
        
        # 磁盘使用率
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        self.metrics_collector.set_gauge('system.disk.usage', disk_percent)
        
        # 网络IO
        net_io = psutil.net_io_counters()
        self.metrics_collector.set_gauge('system.network.bytes_sent', net_io.bytes_sent)
        self.metrics_collector.set_gauge('system.network.bytes_recv', net_io.bytes_recv)
        
        # 进程信息
        process = psutil.Process()
        self.metrics_collector.set_gauge('process.cpu.usage', process.cpu_percent())
        self.metrics_collector.set_gauge('process.memory.usage', process.memory_percent())
        self.metrics_collector.set_gauge('process.threads', process.num_threads())
        
        self.performance_stats['total_measurements'] += 1
    
    async def _check_alerts(self):
        """检查告警条件"""
        current_time = time.time()
        
        # 检查CPU使用率
        cpu_usage = self.metrics_collector.get_current_value('system.cpu.usage')
        if cpu_usage and cpu_usage > self.alert_thresholds['cpu']:
            await self._create_alert(
                'high_cpu_usage',
                AlertLevel.WARNING,
                f"CPU使用率过高: {cpu_usage:.1f}%",
                'system.cpu.usage',
                self.alert_thresholds['cpu'],
                cpu_usage
            )
        
        # 检查内存使用率
        memory_usage = self.metrics_collector.get_current_value('system.memory.usage')
        if memory_usage and memory_usage > self.alert_thresholds['memory']:
            await self._create_alert(
                'high_memory_usage',
                AlertLevel.WARNING,
                f"内存使用率过高: {memory_usage:.1f}%",
                'system.memory.usage',
                self.alert_thresholds['memory'],
                memory_usage
            )
        
        # 检查响应时间
        response_time = self.metrics_collector.get_current_value('task.response_time')
        if response_time and response_time > self.alert_thresholds['response_time']:
            await self._create_alert(
                'high_response_time',
                AlertLevel.WARNING,
                f"响应时间过长: {response_time:.2f}s",
                'task.response_time',
                self.alert_thresholds['response_time'],
                response_time
            )
    
    async def _create_alert(
        self,
        name: str,
        level: AlertLevel,
        message: str,
        metric_name: str,
        threshold: float,
        current_value: float
    ):
        """创建告警"""
        alert_id = f"{name}_{int(time.time())}"
        
        alert = Alert(
            id=alert_id,
            name=name,
            level=level,
            message=message,
            metric_name=metric_name,
            threshold=threshold,
            current_value=current_value,
            timestamp=time.time()
        )
        
        with self._lock:
            # 检查是否已存在相同告警
            existing_alert = None
            for existing in self.alerts.values():
                if (existing.name == name and 
                    not existing.resolved and
                    current_time - existing.timestamp < 300):  # 5分钟内不重复告警
                    existing_alert = existing
                    break
            
            if not existing_alert:
                self.alerts[alert_id] = alert
                self.performance_stats['total_alerts'] += 1
                self.performance_stats['active_alerts'] += 1
                
                logger.warning(f"性能告警: {message}")
                
                # 调用告警回调
                for callback in self.alert_callbacks:
                    try:
                        await callback(alert)
                    except Exception as e:
                        logger.error(f"告警回调错误: {e}")
    
    async def _create_snapshot(self):
        """创建性能快照"""
        cpu_usage = self.metrics_collector.get_current_value('system.cpu.usage') or 0
        memory_usage = self.metrics_collector.get_current_value('system.memory.usage') or 0
        disk_usage = self.metrics_collector.get_current_value('system.disk.usage') or 0
        
        net_io = psutil.net_io_counters()
        network_io = {
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv
        }
        
        # 获取任务统计
        task_stats = self._get_task_statistics()
        
        snapshot = PerformanceSnapshot(
            timestamp=time.time(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            network_io=network_io,
            active_tasks=task_stats.get('active', 0),
            completed_tasks=task_stats.get('completed', 0),
            failed_tasks=task_stats.get('failed', 0),
            avg_response_time=task_stats.get('avg_response_time', 0),
            throughput=task_stats.get('throughput', 0),
            error_rate=task_stats.get('error_rate', 0)
        )
        
        self.snapshots.append(snapshot)
    
    def _get_task_statistics(self) -> dict:
        """获取任务统计信息"""
        # 这里应该从任务调度器获取实际统计
        # 暂时返回模拟数据
        return {
            'active': 0,
            'completed': 0,
            'failed': 0,
            'avg_response_time': 0,
            'throughput': 0,
            'error_rate': 0
        }
    
    def _update_performance_stats(self):
        """更新性能统计"""
        cpu_usage = self.metrics_collector.get_current_value('system.cpu.usage') or 0
        memory_usage = self.metrics_collector.get_current_value('system.memory.usage') or 0
        
        # 更新平均值
        total_measurements = self.performance_stats['total_measurements']
        if total_measurements > 0:
            self.performance_stats['avg_cpu_usage'] = (
                (self.performance_stats['avg_cpu_usage'] * (total_measurements - 1) + cpu_usage) /
                total_measurements
            )
            self.performance_stats['avg_memory_usage'] = (
                (self.performance_stats['avg_memory_usage'] * (total_measurements - 1) + memory_usage) /
                total_measurements
            )
        
        # 更新峰值
        self.performance_stats['peak_cpu_usage'] = max(
            self.performance_stats['peak_cpu_usage'], cpu_usage
        )
        self.performance_stats['peak_memory_usage'] = max(
            self.performance_stats['peak_memory_usage'], memory_usage
        )
        
        # 更新活跃告警数
        self.performance_stats['active_alerts'] = len([
            alert for alert in self.alerts.values() if not alert.resolved
        ])
    
    def add_alert_callback(self, callback: Callable):
        """添加告警回调"""
        self.alert_callbacks.append(callback)
    
    def resolve_alert(self, alert_id: str) -> bool:
        """解决告警"""
        with self._lock:
            if alert_id in self.alerts:
                alert = self.alerts[alert_id]
                alert.resolved = True
                alert.resolved_at = time.time()
                
                self.performance_stats['active_alerts'] = max(
                    0, self.performance_stats['active_alerts'] - 1
                )
                
                logger.info(f"告警已解决: {alert.name}")
                return True
        return False
    
    def get_alerts(self, resolved: bool = None) -> List[Alert]:
        """获取告警列表"""
        with self._lock:
            alerts = list(self.alerts.values())
            if resolved is not None:
                alerts = [a for a in alerts if a.resolved == resolved]
            return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    def get_performance_snapshot(self, limit: int = 10) -> List[PerformanceSnapshot]:
        """获取性能快照"""
        return list(self.snapshots)[-limit:]
    
    def get_performance_statistics(self) -> dict:
        """获取性能统计"""
        return self.performance_stats.copy()
    
    def get_optimization_recommendations(self) -> List[dict]:
        """获取优化建议"""
        recommendations = []
        
        # 基于CPU使用率的建议
        avg_cpu = self.performance_stats['avg_cpu_usage']
        if avg_cpu > 70:
            recommendations.append({
                'type': 'cpu_optimization',
                'priority': 'high',
                'title': 'CPU使用率优化',
                'description': f'平均CPU使用率为 {avg_cpu:.1f}%，建议优化任务分配或增加工作线程',
                'actions': [
                    '检查是否有CPU密集型任务',
                    '考虑增加工作线程数量',
                    '优化算法复杂度',
                    '使用更高效的编程语言或库'
                ]
            })
        
        # 基于内存使用率的建议
        avg_memory = self.performance_stats['avg_memory_usage']
        if avg_memory > 80:
            recommendations.append({
                'type': 'memory_optimization',
                'priority': 'high',
                'title': '内存使用优化',
                'description': f'平均内存使用率为 {avg_memory:.1f}%，建议优化内存使用',
                'actions': [
                    '检查内存泄漏',
                    '优化数据结构',
                    '增加内存限制',
                    '使用内存池技术'
                ]
            })
        
        # 基于响应时间的建议
        response_time_stats = self.metrics_collector.get_statistics('task.response_time')
        if response_time_stats and response_time_stats['p95'] > 5.0:
            recommendations.append({
                'type': 'response_time_optimization',
                'priority': 'medium',
                'title': '响应时间优化',
                'description': f'95%的任务响应时间超过 {response_time_stats["p95"]:.2f}s',
                'actions': [
                    '分析慢任务的原因',
                    '优化数据库查询',
                    '增加缓存机制',
                    '使用异步处理'
                ]
            })
        
        # 基于错误率的建议
        error_rate = self.performance_stats.get('error_rate', 0)
        if error_rate > 0.05:  # 5%
            recommendations.append({
                'type': 'reliability_optimization',
                'priority': 'high',
                'title': '可靠性优化',
                'description': f'错误率为 {error_rate * 100:.1f}%，需要提高系统稳定性',
                'actions': [
                    '分析错误原因',
                    '增加重试机制',
                    '改进错误处理',
                    '增加监控和告警'
                ]
            })
        
        return recommendations
    
    def export_performance_report(self, filepath: str):
        """导出性能报告"""
        report = {
            'report_time': time.time(),
            'performance_statistics': self.get_performance_statistics(),
            'recent_alerts': [alert.to_dict() for alert in self.get_alerts()[:10]],
            'optimization_recommendations': self.get_optimization_recommendations(),
            'system_metrics': {},
            'task_metrics': {}
        }
        
        # 添加系统指标
        for metric_name in ['system.cpu.usage', 'system.memory.usage', 'system.disk.usage']:
            stats = self.metrics_collector.get_statistics(metric_name)
            if stats:
                report['system_metrics'][metric_name] = stats
        
        # 添加任务指标
        for metric_name in ['task.response_time', 'task.throughput', 'task.error_rate']:
            stats = self.metrics_collector.get_statistics(metric_name)
            if stats:
                report['task_metrics'][metric_name] = stats
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"性能报告已导出到: {filepath}")


# 性能分析工具
class PerformanceAnalyzer:
    """性能分析器"""
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
    
    def analyze_bottlenecks(self) -> List[dict]:
        """分析性能瓶颈"""
        bottlenecks = []
        
        # 分析CPU瓶颈
        cpu_stats = self.monitor.metrics_collector.get_statistics('system.cpu.usage')
        if cpu_stats and cpu_stats['mean'] > 80:
            bottlenecks.append({
                'type': 'cpu_bottleneck',
                'severity': 'high',
                'description': f'CPU使用率持续过高，平均 {cpu_stats["mean"]:.1f}%',
                'impact': '任务执行延迟增加',
                'suggestions': [
                    '增加CPU核心数量',
                    '优化算法效率',
                    '分散计算负载'
                ]
            })
        
        # 分析内存瓶颈
        memory_stats = self.monitor.metrics_collector.get_statistics('system.memory.usage')
        if memory_stats and memory_stats['mean'] > 85:
            bottlenecks.append({
                'type': 'memory_bottleneck',
                'severity': 'high',
                'description': f'内存使用率过高，平均 {memory_stats["mean"]:.1f}%',
                'impact': '可能导致系统不稳定',
                'suggestions': [
                    '优化内存使用',
                    '增加内存容量',
                    '实施内存管理策略'
                ]
            })
        
        # 分析响应时间瓶颈
        response_stats = self.monitor.metrics_collector.get_statistics('task.response_time')
        if response_stats and response_stats['p95'] > 10:
            bottlenecks.append({
                'type': 'response_time_bottleneck',
                'severity': 'medium',
                'description': f'响应时间过长，95%分位数 {response_stats["p95"]:.2f}s',
                'impact': '用户体验下降',
                'suggestions': [
                    '优化慢查询',
                    '增加缓存',
                    '使用CDN'
                ]
            })
        
        return bottlenecks
    
    def predict_resource_needs(self, time_horizon: int = 3600) -> dict:
        """预测资源需求"""
        # 基于历史数据预测未来的资源需求
        predictions = {}
        
        # CPU需求预测
        cpu_history = self.monitor.metrics_collector.get_metric_history('system.cpu.usage')
        if len(cpu_history) > 10:
            recent_cpu = [m.value for m in cpu_history[-10:]]
            trend = np.polyfit(range(len(recent_cpu)), recent_cpu, 1)[0]
            predicted_cpu = recent_cpu[-1] + trend * (time_horizon / self.monitor.collection_interval)
            predictions['cpu'] = {
                'current': recent_cpu[-1],
                'predicted': max(0, min(100, predicted_cpu)),
                'confidence': 'medium'
            }
        
        # 内存需求预测
        memory_history = self.monitor.metrics_collector.get_metric_history('system.memory.usage')
        if len(memory_history) > 10:
            recent_memory = [m.value for m in memory_history[-10:]]
            trend = np.polyfit(range(len(recent_memory)), recent_memory, 1)[0]
            predicted_memory = recent_memory[-1] + trend * (time_horizon / self.monitor.collection_interval)
            predictions['memory'] = {
                'current': recent_memory[-1],
                'predicted': max(0, min(100, predicted_memory)),
                'confidence': 'medium'
            }
        
        return predictions


# 示例用法
async def example_usage():
    """示例用法"""
    # 创建性能监控器
    monitor = PerformanceMonitor(
        collection_interval=2.0,
        alert_threshold_cpu=75.0,
        alert_threshold_memory=80.0
    )
    
    # 添加告警回调
    async def alert_handler(alert: Alert):
        print(f"告警: {alert.message}")
    
    monitor.add_alert_callback(alert_handler)
    
    # 启动监控
    await monitor.start_monitoring()
    
    # 模拟任务执行和指标记录
    for i in range(20):
        # 记录任务响应时间
        response_time = np.random.normal(2.0, 0.5)
        monitor.metrics_collector.record_timer('task.response_time', response_time)
        
        # 记录任务吞吐量
        monitor.metrics_collector.increment_counter('task.throughput', np.random.poisson(5))
        
        # 记录错误
        if np.random.random() < 0.1:  # 10%错误率
            monitor.metrics_collector.increment_counter('task.errors')
        
        await asyncio.sleep(1)
    
    # 获取性能分析
    analyzer = PerformanceAnalyzer(monitor)
    bottlenecks = analyzer.analyze_bottlenecks()
    predictions = analyzer.predict_resource_needs()
    
    print("性能瓶颈分析:")
    for bottleneck in bottlenecks:
        print(f"- {bottleneck['description']}")
    
    print("\n资源需求预测:")
    for resource, prediction in predictions.items():
        print(f"- {resource}: 当前 {prediction['current']:.1f}%, 预测 {prediction['predicted']:.1f}%")
    
    print("\n优化建议:")
    recommendations = monitor.get_optimization_recommendations()
    for rec in recommendations:
        print(f"- {rec['title']}: {rec['description']}")
    
    # 导出报告
    monitor.export_performance_report("performance_report.json")
    
    # 停止监控
    await monitor.stop_monitoring()
    
    print("性能监控示例完成")


if __name__ == "__main__":
    asyncio.run(example_usage())