"""
指标收集器

提供统一的指标收集、转换和分发功能
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import json
import psutil
from collections import defaultdict, deque
import threading

logger = logging.getLogger(__name__)


class CollectorType(Enum):
    """收集器类型"""
    SYSTEM = "system"
    APPLICATION = "application"
    CUSTOM = "custom"
    PUSH = "push"
    PULL = "pull"


class MetricFormat(Enum):
    """指标格式"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class CollectorConfig:
    """收集器配置"""
    collector_id: str
    name: str
    collector_type: CollectorType
    enabled: bool = True
    interval: float = 10.0
    timeout: float = 5.0
    retry_count: int = 3
    batch_size: int = 100
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CollectedMetric:
    """收集的指标"""
    name: str
    value: Union[int, float]
    metric_type: MetricFormat
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'value': self.value,
            'metric_type': self.metric_type.value,
            'timestamp': self.timestamp,
            'labels': self.labels,
            'source': self.source,
            'metadata': self.metadata
        }


@dataclass
class CollectorResult:
    """收集器结果"""
    collector_id: str
    success: bool
    metrics_count: int
    execution_time: float
    error: Optional[str] = None
    metrics: List[CollectedMetric] = field(default_factory=list)


class MetricsCollector:
    """指标收集器"""
    
    def __init__(self, max_collectors: int = 50):
        self.max_collectors = max_collectors
        
        # 收集器管理
        self.collectors: Dict[str, CollectorConfig] = {}
        self.collector_instances: Dict[str, Callable] = {}
        
        # 指标存储
        self.collected_metrics: deque = deque(maxlen=10000)
        self.metrics_by_source: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=5000)
        )
        
        # 收集统计
        self.collection_stats = {
            'total_collections': 0,
            'successful_collections': 0,
            'failed_collections': 0,
            'total_metrics': 0,
            'average_collection_time': 0.0,
            'metrics_per_second': 0.0
        }
        
        # 任务管理
        self._running = False
        self._collection_tasks: Dict[str, asyncio.Task] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        self._lock = threading.RLock()
        
        # 回调函数
        self.metric_callbacks: List[Callable] = []
        self.collection_callbacks: List[Callable] = []
        self.error_callbacks: List[Callable] = []
        
        # 内置收集器
        self._register_builtin_collectors()
    
    def _register_builtin_collectors(self):
        """注册内置收集器"""
        try:
            # 系统指标收集器
            self.collector_instances['system_metrics'] = self._collect_system_metrics
            
            # 应用指标收集器
            self.collector_instances['app_metrics'] = self._collect_application_metrics
            
            # 自定义指标收集器
            self.collector_instances['custom_metrics'] = self._collect_custom_metrics
            
        except Exception as e:
            logger.error(f"注册内置收集器失败: {e}")
    
    async def start(self):
        """启动指标收集器"""
        if self._running:
            return
        
        self._running = True
        
        # 启动启用的收集器
        for collector_id, config in self.collectors.items():
            if config.enabled:
                await self._start_collector(collector_id)
        
        # 启动清理任务
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("指标收集器已启动")
    
    async def stop(self):
        """停止指标收集器"""
        if not self._running:
            return
        
        self._running = False
        
        # 停止所有收集任务
        for task in self._collection_tasks.values():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # 停止清理任务
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("指标收集器已停止")
    
    def register_collector(self, config: CollectorConfig, 
                          collector_func: Optional[Callable] = None) -> bool:
        """注册收集器"""
        try:
            with self._lock:
                if len(self.collectors) >= self.max_collectors:
                    logger.warning(f"收集器数量已达上限 {self.max_collectors}")
                    return False
                
                self.collectors[config.collector_id] = config
                
                # 如果提供了自定义收集器函数
                if collector_func:
                    self.collector_instances[config.collector_id] = collector_func
                elif config.collector_id not in self.collector_instances:
                    logger.warning(f"收集器 {config.collector_id} 没有对应的实现函数")
                
                logger.info(f"收集器 {config.collector_id} 注册成功")
                return True
                
        except Exception as e:
            logger.error(f"注册收集器失败: {e}")
            return False
    
    def unregister_collector(self, collector_id: str) -> bool:
        """注销收集器"""
        try:
            with self._lock:
                if collector_id in self.collectors:
                    # 停止收集任务
                    if collector_id in self._collection_tasks:
                        self._collection_tasks[collector_id].cancel()
                        del self._collection_tasks[collector_id]
                    
                    # 移除收集器
                    del self.collectors[collector_id]
                    
                    # 移除收集器实例
                    if collector_id in self.collector_instances:
                        del self.collector_instances[collector_id]
                    
                    logger.info(f"收集器 {collector_id} 注销成功")
                    return True
                
                return False
                
        except Exception as e:
            logger.error(f"注销收集器失败: {e}")
            return False
    
    async def _start_collector(self, collector_id: str):
        """启动单个收集器"""
        try:
            if collector_id not in self.collectors:
                return
            
            config = self.collectors[collector_id]
            if not config.enabled:
                return
            
            # 创建收集任务
            task = asyncio.create_task(self._collector_loop(collector_id))
            self._collection_tasks[collector_id] = task
            
            logger.info(f"收集器 {collector_id} 已启动")
            
        except Exception as e:
            logger.error(f"启动收集器 {collector_id} 失败: {e}")
    
    async def _collector_loop(self, collector_id: str):
        """收集器循环"""
        try:
            config = self.collectors[collector_id]
            collector_func = self.collector_instances.get(collector_id)
            
            if not collector_func:
                logger.error(f"收集器 {collector_id} 没有实现函数")
                return
            
            while self._running and config.enabled:
                try:
                    start_time = time.time()
                    
                    # 执行收集
                    result = await self._execute_collector(collector_id, collector_func, config)
                    
                    execution_time = time.time() - start_time
                    
                    # 更新统计
                    self._update_collection_stats(result, execution_time)
                    
                    # 存储指标
                    if result.success and result.metrics:
                        await self._store_metrics(result.metrics)
                    
                    # 触发回调
                    await self._trigger_collection_callbacks(result)
                    
                    # 等待下次收集
                    await asyncio.sleep(config.interval)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"收集器 {collector_id} 执行异常: {e}")
                    
                    # 触发错误回调
                    await self._trigger_error_callbacks(collector_id, str(e))
                    
                    # 等待后重试
                    await asyncio.sleep(config.interval)
                    
        except Exception as e:
            logger.error(f"收集器循环异常: {e}")
    
    async def _execute_collector(self, collector_id: str, collector_func: Callable,
                               config: CollectorConfig) -> CollectorResult:
        """执行收集器"""
        try:
            start_time = time.time()
            
            # 执行收集函数
            if asyncio.iscoroutinefunction(collector_func):
                metrics = await collector_func(config)
            else:
                metrics = collector_func(config)
            
            # 确保返回的是列表
            if not isinstance(metrics, list):
                metrics = []
            
            execution_time = time.time() - start_time
            
            return CollectorResult(
                collector_id=collector_id,
                success=True,
                metrics_count=len(metrics),
                execution_time=execution_time,
                metrics=metrics
            )
            
        except Exception as e:
            return CollectorResult(
                collector_id=collector_id,
                success=False,
                metrics_count=0,
                execution_time=time.time() - start_time,
                error=str(e)
            )
    
    async def _collect_system_metrics(self, config: CollectorConfig) -> List[CollectedMetric]:
        """收集系统指标"""
        metrics = []
        
        try:
            # CPU指标
            cpu_percent = psutil.cpu_percent(interval=1)
            metrics.append(CollectedMetric(
                name="system.cpu.percent",
                value=cpu_percent,
                metric_type=MetricFormat.GAUGE,
                timestamp=time.time(),
                source="system_metrics",
                labels={"core": "total"}
            ))
            
            # 内存指标
            memory = psutil.virtual_memory()
            metrics.append(CollectedMetric(
                name="system.memory.percent",
                value=memory.percent,
                metric_type=MetricFormat.GAUGE,
                timestamp=time.time(),
                source="system_metrics"
            ))
            
            metrics.append(CollectedMetric(
                name="system.memory.used_bytes",
                value=memory.used,
                metric_type=MetricFormat.GAUGE,
                timestamp=time.time(),
                source="system_metrics"
            ))
            
            metrics.append(CollectedMetric(
                name="system.memory.available_bytes",
                value=memory.available,
                metric_type=MetricFormat.GAUGE,
                timestamp=time.time(),
                source="system_metrics"
            ))
            
            # 磁盘指标
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            metrics.append(CollectedMetric(
                name="system.disk.percent",
                value=disk_percent,
                metric_type=MetricFormat.GAUGE,
                timestamp=time.time(),
                source="system_metrics"
            ))
            
            # 网络指标
            network = psutil.net_io_counters()
            metrics.append(CollectedMetric(
                name="system.network.bytes_sent",
                value=network.bytes_sent,
                metric_type=MetricFormat.COUNTER,
                timestamp=time.time(),
                source="system_metrics"
            ))
            
            metrics.append(CollectedMetric(
                name="system.network.bytes_recv",
                value=network.bytes_recv,
                metric_type=MetricFormat.COUNTER,
                timestamp=time.time(),
                source="system_metrics"
            ))
            
            # 进程指标
            process_count = len(psutil.pids())
            metrics.append(CollectedMetric(
                name="system.process.count",
                value=process_count,
                metric_type=MetricFormat.GAUGE,
                timestamp=time.time(),
                source="system_metrics"
            ))
            
        except Exception as e:
            logger.error(f"收集系统指标失败: {e}")
        
        return metrics
    
    async def _collect_application_metrics(self, config: CollectorConfig) -> List[CollectedMetric]:
        """收集应用指标"""
        metrics = []
        
        try:
            # 这里可以添加应用特定的指标收集逻辑
            # 例如：请求数、响应时间、错误率等
            
            # 示例：模拟应用指标
            import random
            metrics.append(CollectedMetric(
                name="app.requests.total",
                value=random.randint(100, 1000),
                metric_type=MetricFormat.COUNTER,
                timestamp=time.time(),
                source="app_metrics"
            ))
            
            metrics.append(CollectedMetric(
                name="app.response_time.avg",
                value=random.uniform(50, 200),
                metric_type=MetricFormat.GAUGE,
                timestamp=time.time(),
                source="app_metrics"
            ))
            
            metrics.append(CollectedMetric(
                name="app.error.rate",
                value=random.uniform(0, 5),
                metric_type=MetricFormat.GAUGE,
                timestamp=time.time(),
                source="app_metrics"
            ))
            
        except Exception as e:
            logger.error(f"收集应用指标失败: {e}")
        
        return metrics
    
    async def _collect_custom_metrics(self, config: CollectorConfig) -> List[CollectedMetric]:
        """收集自定义指标"""
        # 这个方法可以被用户重写或通过register_collector注册自定义实现
        return []
    
    async def _store_metrics(self, metrics: List[CollectedMetric]):
        """存储指标"""
        try:
            with self._lock:
                for metric in metrics:
                    self.collected_metrics.append(metric)
                    
                    # 按源分组存储
                    self.metrics_by_source[metric.source].append(metric)
                
                self.collection_stats['total_metrics'] += len(metrics)
                
        except Exception as e:
            logger.error(f"存储指标失败: {e}")
    
    def _update_collection_stats(self, result: CollectorResult, execution_time: float):
        """更新收集统计"""
        try:
            self.collection_stats['total_collections'] += 1
            
            if result.success:
                self.collection_stats['successful_collections'] += 1
            else:
                self.collection_stats['failed_collections'] += 1
            
            # 更新平均收集时间
            total_collections = self.collection_stats['total_collections']
            self.collection_stats['average_collection_time'] = (
                (self.collection_stats['average_collection_time'] * 
                 (total_collections - 1) + execution_time) / total_collections
            )
            
        except Exception as e:
            logger.error(f"更新收集统计失败: {e}")
    
    async def _trigger_metric_callbacks(self, metrics: List[CollectedMetric]):
        """触发指标回调"""
        try:
            for callback in self.metric_callbacks:
                if asyncio.iscoroutinefunction(callback):
                    await callback(metrics)
                else:
                    callback(metrics)
        except Exception as e:
            logger.error(f"触发指标回调失败: {e}")
    
    async def _trigger_collection_callbacks(self, result: CollectorResult):
        """触发收集回调"""
        try:
            for callback in self.collection_callbacks:
                if asyncio.iscoroutinefunction(callback):
                    await callback(result)
                else:
                    callback(result)
        except Exception as e:
            logger.error(f"触发收集回调失败: {e}")
    
    async def _trigger_error_callbacks(self, collector_id: str, error: str):
        """触发错误回调"""
        try:
            error_data = {
                'collector_id': collector_id,
                'error': error,
                'timestamp': time.time()
            }
            
            for callback in self.error_callbacks:
                if asyncio.iscoroutinefunction(callback):
                    await callback(error_data)
                else:
                    callback(error_data)
        except Exception as e:
            logger.error(f"触发错误回调失败: {e}")
    
    async def _cleanup_loop(self):
        """清理循环"""
        while self._running:
            try:
                await asyncio.sleep(300)  # 每5分钟清理一次
                
                # 清理过期的指标数据
                current_time = time.time()
                cutoff_time = current_time - (24 * 3600)  # 保留24小时数据
                
                # 清理主指标队列
                while (self.collected_metrics and 
                       self.collected_metrics[0].timestamp < cutoff_time):
                    self.collected_metrics.popleft()
                
                # 清理按源分组的指标队列
                for source_queue in self.metrics_by_source.values():
                    while (source_queue and 
                           source_queue[0].timestamp < cutoff_time):
                        source_queue.popleft()
                
                # 更新吞吐量统计
                recent_metrics = [
                    m for m in self.collected_metrics 
                    if m.timestamp >= current_time - 60
                ]
                self.collection_stats['metrics_per_second'] = len(recent_metrics) / 60.0
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"清理循环异常: {e}")
    
    async def collect_now(self, collector_id: str) -> Optional[CollectorResult]:
        """立即收集指定收集器的指标"""
        try:
            if collector_id not in self.collectors:
                return None
            
            config = self.collectors[collector_id]
            collector_func = self.collector_instances.get(collector_id)
            
            if not collector_func:
                return None
            
            return await self._execute_collector(collector_id, collector_func, config)
            
        except Exception as e:
            logger.error(f"立即收集失败: {e}")
            return None
    
    def add_metric(self, name: str, value: Union[int, float], 
                  metric_type: MetricFormat = MetricFormat.GAUGE,
                  labels: Optional[Dict[str, str]] = None,
                  source: str = "custom"):
        """手动添加指标"""
        try:
            metric = CollectedMetric(
                name=name,
                value=value,
                metric_type=metric_type,
                timestamp=time.time(),
                labels=labels or {},
                source=source
            )
            
            # 直接存储
            self.collected_metrics.append(metric)
            self.metrics_by_source[source].append(metric)
            
            # 触发回调
            asyncio.create_task(self._trigger_metric_callbacks([metric]))
            
        except Exception as e:
            logger.error(f"添加指标失败: {e}")
    
    def add_metric_callback(self, callback: Callable):
        """添加指标回调"""
        self.metric_callbacks.append(callback)
    
    def add_collection_callback(self, callback: Callable):
        """添加收集回调"""
        self.collection_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable):
        """添加错误回调"""
        self.error_callbacks.append(callback)
    
    def get_collectors(self) -> Dict[str, CollectorConfig]:
        """获取所有收集器"""
        return self.collectors.copy()
    
    def get_collector_config(self, collector_id: str) -> Optional[CollectorConfig]:
        """获取收集器配置"""
        return self.collectors.get(collector_id)
    
    def get_metrics(self, source: Optional[str] = None, hours: int = 1) -> List[CollectedMetric]:
        """获取指标"""
        try:
            cutoff_time = time.time() - (hours * 3600)
            
            if source:
                metrics_queue = self.metrics_by_source.get(source, deque())
                return [m for m in metrics_queue if m.timestamp >= cutoff_time]
            else:
                return [m for m in self.collected_metrics if m.timestamp >= cutoff_time]
                
        except Exception as e:
            logger.error(f"获取指标失败: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """获取收集统计"""
        return {
            'total_collectors': len(self.collectors),
            'active_collectors': len([c for c in self.collectors.values() if c.enabled]),
            'total_metrics': len(self.collected_metrics),
            'sources': list(self.metrics_by_source.keys()),
            'collection_stats': self.collection_stats.copy()
        }
    
    def export_metrics(self, source: Optional[str] = None, hours: int = 1, 
                      format_type: str = "json") -> Optional[str]:
        """导出指标"""
        try:
            metrics = self.get_metrics(source, hours)
            
            if format_type == "json":
                data = [m.to_dict() for m in metrics]
                return json.dumps(data, indent=2)
            else:
                # 可以扩展其他格式支持
                return None
                
        except Exception as e:
            logger.error(f"导出指标失败: {e}")
            return None