"""
性能统计组件

提供性能指标收集、统计分析和趋势分析功能
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import statistics
import json
from collections import defaultdict, deque
import threading

logger = logging.getLogger(__name__)


class MetricCategory(Enum):
    """指标类别"""
    SYSTEM = "system"
    APPLICATION = "application"
    BUSINESS = "business"
    CUSTOM = "custom"


class AggregationType(Enum):
    """聚合类型"""
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    PERCENTILE = "percentile"


@dataclass
class PerformanceMetric:
    """性能指标"""
    name: str
    category: MetricCategory
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'category': self.category.value,
            'value': self.value,
            'timestamp': self.timestamp,
            'tags': self.tags,
            'unit': self.unit,
            'metadata': self.metadata
        }


@dataclass
class MetricAggregation:
    """指标聚合"""
    metric_name: str
    aggregation_type: AggregationType
    value: float
    timestamp: float
    sample_count: int
    period: float  # 聚合周期（秒）
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'metric_name': self.metric_name,
            'aggregation_type': self.aggregation_type.value,
            'value': self.value,
            'timestamp': self.timestamp,
            'sample_count': self.sample_count,
            'period': self.period
        }


@dataclass
class PerformanceReport:
    """性能报告"""
    report_id: str
    generated_at: float
    time_range: Tuple[float, float]  # (start_time, end_time)
    metrics_summary: Dict[str, Any]
    trends: Dict[str, Any]
    anomalies: List[Dict[str, Any]]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'report_id': self.report_id,
            'generated_at': self.generated_at,
            'time_range': self.time_range,
            'metrics_summary': self.metrics_summary,
            'trends': self.trends,
            'anomalies': self.anomalies,
            'recommendations': self.recommendations
        }


class PerformanceStats:
    """性能统计组件"""
    
    def __init__(self, retention_hours: int = 168, aggregation_interval: int = 60):
        self.retention_hours = retention_hours
        self.aggregation_interval = aggregation_interval
        
        # 指标存储
        self.raw_metrics: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=retention_hours * 360)
        )
        self.aggregated_metrics: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=retention_hours * 60)
        )
        
        # 统计配置
        self.percentiles = [50, 75, 90, 95, 99]
        self.alert_thresholds: Dict[str, Dict[str, float]] = {}
        
        # 分析结果
        self.trend_analysis: Dict[str, Dict[str, Any]] = {}
        self.anomaly_detection_results: List[Dict[str, Any]] = []
        
        # 回调函数
        self.metric_callbacks: List[Callable] = []
        self.aggregation_callbacks: List[Callable] = []
        self.alert_callbacks: List[Callable] = []
        
        # 任务管理
        self._aggregation_task: Optional[asyncio.Task] = None
        self._analysis_task: Optional[asyncio.Task] = None
        self._running = False
        self._lock = threading.RLock()
    
    async def start(self):
        """启动性能统计"""
        if self._running:
            return
        
        self._running = True
        
        # 启动聚合任务
        self._aggregation_task = asyncio.create_task(self._aggregation_loop())
        
        # 启动分析任务
        self._analysis_task = asyncio.create_task(self._analysis_loop())
        
        logger.info("性能统计组件已启动")
    
    async def stop(self):
        """停止性能统计"""
        if not self._running:
            return
        
        self._running = False
        
        # 取消任务
        if self._aggregation_task:
            self._aggregation_task.cancel()
            try:
                await self._aggregation_task
            except asyncio.CancelledError:
                pass
        
        if self._analysis_task:
            self._analysis_task.cancel()
            try:
                await self._analysis_task
            except asyncio.CancelledError:
                pass
        
        logger.info("性能统计组件已停止")
    
    async def record_metric(self, metric: PerformanceMetric):
        """记录指标"""
        try:
            with self._lock:
                self.raw_metrics[metric.name].append(metric)
                
                # 触发指标回调
                await self._trigger_metric_callbacks(metric)
                
        except Exception as e:
            logger.error(f"记录指标失败: {e}")
    
    async def record_metrics(self, metrics: List[PerformanceMetric]):
        """批量记录指标"""
        try:
            with self._lock:
                for metric in metrics:
                    self.raw_metrics[metric.name].append(metric)
                
                # 触发批量指标回调
                await self._trigger_batch_metric_callbacks(metrics)
                
        except Exception as e:
            logger.error(f"批量记录指标失败: {e}")
    
    async def _aggregation_loop(self):
        """聚合循环"""
        while self._running:
            try:
                await asyncio.sleep(self.aggregation_interval)
                
                # 对每个指标进行聚合
                current_time = time.time()
                
                for metric_name, metrics_deque in self.raw_metrics.items():
                    try:
                        await self._aggregate_metric(metric_name, metrics_deque, current_time)
                    except Exception as e:
                        logger.error(f"聚合指标 {metric_name} 失败: {e}")
                
                # 触发聚合回调
                await self._trigger_aggregation_callbacks()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"聚合循环异常: {e}")
    
    async def _aggregate_metric(self, metric_name: str, metrics_deque: deque, 
                              current_time: float):
        """聚合单个指标"""
        try:
            if not metrics_deque:
                return
            
            # 获取时间窗口内的指标
            window_start = current_time - self.aggregation_interval
            window_metrics = [
                m for m in metrics_deque 
                if m.timestamp >= window_start
            ]
            
            if not window_metrics:
                return
            
            # 计算各种聚合值
            values = [m.value for m in window_metrics]
            
            # 基础聚合
            aggregations = [
                MetricAggregation(
                    metric_name=metric_name,
                    aggregation_type=AggregationType.SUM,
                    value=sum(values),
                    timestamp=current_time,
                    sample_count=len(values),
                    period=self.aggregation_interval
                ),
                MetricAggregation(
                    metric_name=metric_name,
                    aggregation_type=AggregationType.AVG,
                    value=statistics.mean(values),
                    timestamp=current_time,
                    sample_count=len(values),
                    period=self.aggregation_interval
                ),
                MetricAggregation(
                    metric_name=metric_name,
                    aggregation_type=AggregationType.MIN,
                    value=min(values),
                    timestamp=current_time,
                    sample_count=len(values),
                    period=self.aggregation_interval
                ),
                MetricAggregation(
                    metric_name=metric_name,
                    aggregation_type=AggregationType.MAX,
                    value=max(values),
                    timestamp=current_time,
                    sample_count=len(values),
                    period=self.aggregation_interval
                ),
                MetricAggregation(
                    metric_name=metric_name,
                    aggregation_type=AggregationType.COUNT,
                    value=len(values),
                    timestamp=current_time,
                    sample_count=len(values),
                    period=self.aggregation_interval
                )
            ]
            
            # 计算百分位数
            for percentile in self.percentiles:
                percentile_value = statistics.quantiles(values, n=100)[percentile - 1]
                aggregations.append(
                    MetricAggregation(
                        metric_name=metric_name,
                        aggregation_type=AggregationType.PERCENTILE,
                        value=percentile_value,
                        timestamp=current_time,
                        sample_count=len(values),
                        period=self.aggregation_interval
                    )
                )
            
            # 存储聚合结果
            for aggregation in aggregations:
                self.aggregated_metrics[metric_name].append(aggregation)
            
        except Exception as e:
            logger.error(f"聚合指标 {metric_name} 失败: {e}")
    
    async def _analysis_loop(self):
        """分析循环"""
        while self._running:
            try:
                await asyncio.sleep(300)  # 每5分钟分析一次
                
                # 趋势分析
                await self._perform_trend_analysis()
                
                # 异常检测
                await self._detect_anomalies()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"分析循环异常: {e}")
    
    async def _perform_trend_analysis(self):
        """执行趋势分析"""
        try:
            current_time = time.time()
            
            for metric_name, metrics_deque in self.raw_metrics.items():
                try:
                    # 获取最近1小时的数据
                    hour_ago = current_time - 3600
                    recent_metrics = [
                        m for m in metrics_deque 
                        if m.timestamp >= hour_ago
                    ]
                    
                    if len(recent_metrics) < 10:  # 数据点太少
                        continue
                    
                    values = [m.value for m in recent_metrics]
                    
                    # 计算趋势
                    trend = self._calculate_trend(values)
                    
                    # 计算变化率
                    if len(values) >= 2:
                        change_rate = ((values[-1] - values[0]) / values[0]) * 100
                    else:
                        change_rate = 0
                    
                    # 存储趋势分析结果
                    self.trend_analysis[metric_name] = {
                        'trend': trend,
                        'change_rate': change_rate,
                        'data_points': len(values),
                        'latest_value': values[-1],
                        'average_value': statistics.mean(values),
                        'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
                        'analysis_time': current_time
                    }
                    
                except Exception as e:
                    logger.error(f"趋势分析 {metric_name} 失败: {e}")
                    
        except Exception as e:
            logger.error(f"执行趋势分析失败: {e}")
    
    def _calculate_trend(self, values: List[float]) -> str:
        """计算趋势"""
        try:
            if len(values) < 3:
                return "stable"
            
            # 使用简单线性回归计算趋势
            n = len(values)
            x_sum = sum(range(n))
            y_sum = sum(values)
            xy_sum = sum(i * values[i] for i in range(n))
            x2_sum = sum(i * i for i in range(n))
            
            # 计算斜率
            slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)
            
            if slope > 0.01:
                return "increasing"
            elif slope < -0.01:
                return "decreasing"
            else:
                return "stable"
                
        except Exception as e:
            logger.error(f"计算趋势失败: {e}")
            return "unknown"
    
    async def _detect_anomalies(self):
        """检测异常"""
        try:
            current_time = time.time()
            
            for metric_name, trend_data in self.trend_analysis.items():
                try:
                    # 获取指标阈值
                    if metric_name not in self.alert_thresholds:
                        continue
                    
                    thresholds = self.alert_thresholds[metric_name]
                    latest_value = trend_data.get('latest_value', 0)
                    
                    anomalies = []
                    
                    # 检查异常值
                    if 'max_threshold' in thresholds and latest_value > thresholds['max_threshold']:
                        anomalies.append({
                            'type': 'high_value',
                            'metric': metric_name,
                            'value': latest_value,
                            'threshold': thresholds['max_threshold'],
                            'severity': 'warning'
                        })
                    
                    if 'min_threshold' in thresholds and latest_value < thresholds['min_threshold']:
                        anomalies.append({
                            'type': 'low_value',
                            'metric': metric_name,
                            'value': latest_value,
                            'threshold': thresholds['min_threshold'],
                            'severity': 'warning'
                        })
                    
                    # 检查趋势异常
                    trend = trend_data.get('trend', 'stable')
                    if trend == 'increasing' and 'max_change_rate' in thresholds:
                        change_rate = abs(trend_data.get('change_rate', 0))
                        if change_rate > thresholds['max_change_rate']:
                            anomalies.append({
                                'type': 'rapid_increase',
                                'metric': metric_name,
                                'change_rate': change_rate,
                                'threshold': thresholds['max_change_rate'],
                                'severity': 'error'
                            })
                    
                    # 存储异常
                    for anomaly in anomalies:
                        anomaly['detected_at'] = current_time
                        self.anomaly_detection_results.append(anomaly)
                        
                        # 触发告警回调
                        await self._trigger_alert_callbacks(anomaly)
                    
                except Exception as e:
                    logger.error(f"检测指标 {metric_name} 异常失败: {e}")
                    
        except Exception as e:
            logger.error(f"执行异常检测失败: {e}")
    
    async def _trigger_metric_callbacks(self, metric: PerformanceMetric):
        """触发指标回调"""
        try:
            for callback in self.metric_callbacks:
                if asyncio.iscoroutinefunction(callback):
                    await callback(metric)
                else:
                    callback(metric)
        except Exception as e:
            logger.error(f"触发指标回调失败: {e}")
    
    async def _trigger_batch_metric_callbacks(self, metrics: List[PerformanceMetric]):
        """触发批量指标回调"""
        try:
            for callback in self.metric_callbacks:
                if asyncio.iscoroutinefunction(callback):
                    await callback(metrics)
                else:
                    callback(metrics)
        except Exception as e:
            logger.error(f"触发批量指标回调失败: {e}")
    
    async def _trigger_aggregation_callbacks(self):
        """触发聚合回调"""
        try:
            for callback in self.aggregation_callbacks:
                if asyncio.iscoroutinefunction(callback):
                    await callback(self.aggregated_metrics)
                else:
                    callback(self.aggregated_metrics)
        except Exception as e:
            logger.error(f"触发聚合回调失败: {e}")
    
    async def _trigger_alert_callbacks(self, anomaly: Dict[str, Any]):
        """触发告警回调"""
        try:
            for callback in self.alert_callbacks:
                if asyncio.iscoroutinefunction(callback):
                    await callback(anomaly)
                else:
                    callback(anomaly)
        except Exception as e:
            logger.error(f"触发告警回调失败: {e}")
    
    def set_alert_threshold(self, metric_name: str, thresholds: Dict[str, float]):
        """设置告警阈值"""
        self.alert_thresholds[metric_name] = thresholds
        logger.info(f"设置指标 {metric_name} 告警阈值: {thresholds}")
    
    def add_metric_callback(self, callback: Callable):
        """添加指标回调"""
        self.metric_callbacks.append(callback)
    
    def add_aggregation_callback(self, callback: Callable):
        """添加聚合回调"""
        self.aggregation_callbacks.append(callback)
    
    def add_alert_callback(self, callback: Callable):
        """添加告警回调"""
        self.alert_callbacks.append(callback)
    
    def get_metric_values(self, metric_name: str, hours: int = 1) -> List[PerformanceMetric]:
        """获取指标值"""
        try:
            cutoff_time = time.time() - (hours * 3600)
            metrics = self.raw_metrics.get(metric_name, [])
            return [m for m in metrics if m.timestamp >= cutoff_time]
        except Exception as e:
            logger.error(f"获取指标值失败: {e}")
            return []
    
    def get_aggregated_values(self, metric_name: str, hours: int = 1) -> List[MetricAggregation]:
        """获取聚合值"""
        try:
            cutoff_time = time.time() - (hours * 3600)
            aggregations = self.aggregated_metrics.get(metric_name, [])
            return [a for a in aggregations if a.timestamp >= cutoff_time]
        except Exception as e:
            logger.error(f"获取聚合值失败: {e}")
            return []
    
    def get_trend_analysis(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """获取趋势分析"""
        return self.trend_analysis.get(metric_name)
    
    def get_all_trends(self) -> Dict[str, Dict[str, Any]]:
        """获取所有趋势"""
        return self.trend_analysis.copy()
    
    def get_anomalies(self, hours: int = 24) -> List[Dict[str, Any]]:
        """获取异常"""
        try:
            cutoff_time = time.time() - (hours * 3600)
            return [a for a in self.anomaly_detection_results if a.get('detected_at', 0) >= cutoff_time]
        except Exception as e:
            logger.error(f"获取异常失败: {e}")
            return []
    
    def generate_performance_report(self, hours: int = 24) -> PerformanceReport:
        """生成性能报告"""
        try:
            report_id = f"report_{int(time.time())}"
            current_time = time.time()
            start_time = current_time - (hours * 3600)
            
            # 收集指标摘要
            metrics_summary = {}
            for metric_name in self.raw_metrics.keys():
                metrics = self.get_metric_values(metric_name, hours)
                if metrics:
                    values = [m.value for m in metrics]
                    metrics_summary[metric_name] = {
                        'count': len(values),
                        'min': min(values),
                        'max': max(values),
                        'avg': statistics.mean(values),
                        'latest': values[-1] if values else 0
                    }
            
            # 收集趋势信息
            trends = {}
            for metric_name, trend_data in self.trend_analysis.items():
                if trend_data.get('analysis_time', 0) >= start_time:
                    trends[metric_name] = trend_data
            
            # 收集异常信息
            anomalies = self.get_anomalies(hours)
            
            # 生成建议
            recommendations = self._generate_recommendations(metrics_summary, trends, anomalies)
            
            return PerformanceReport(
                report_id=report_id,
                generated_at=current_time,
                time_range=(start_time, current_time),
                metrics_summary=metrics_summary,
                trends=trends,
                anomalies=anomalies,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"生成性能报告失败: {e}")
            return PerformanceReport(
                report_id="error",
                generated_at=time.time(),
                time_range=(0, 0),
                metrics_summary={},
                trends={},
                anomalies=[],
                recommendations=[f"报告生成失败: {str(e)}"]
            )
    
    def _generate_recommendations(self, metrics_summary: Dict[str, Any], 
                                trends: Dict[str, Any], 
                                anomalies: List[Dict[str, Any]]) -> List[str]:
        """生成建议"""
        recommendations = []
        
        try:
            # 基于异常生成建议
            for anomaly in anomalies:
                if anomaly.get('type') == 'high_value':
                    recommendations.append(f"指标 {anomaly.get('metric')} 值过高，建议检查相关配置")
                elif anomaly.get('type') == 'rapid_increase':
                    recommendations.append(f"指标 {anomaly.get('metric')} 快速增长，建议调查原因")
            
            # 基于趋势生成建议
            for metric_name, trend_data in trends.items():
                trend = trend_data.get('trend')
                if trend == 'increasing':
                    change_rate = abs(trend_data.get('change_rate', 0))
                    if change_rate > 50:
                        recommendations.append(f"指标 {metric_name} 呈上升趋势，增长率 {change_rate:.1f}%，建议监控")
                elif trend == 'decreasing':
                    recommendations.append(f"指标 {metric_name} 呈下降趋势，建议检查相关服务")
            
            # 基于性能指标生成建议
            for metric_name, summary in metrics_summary.items():
                avg_value = summary.get('avg', 0)
                if avg_value > 1000:
                    recommendations.append(f"指标 {metric_name} 平均值较高 ({avg_value:.2f})，建议优化")
            
        except Exception as e:
            logger.error(f"生成建议失败: {e}")
            recommendations.append(f"建议生成过程中出现错误: {str(e)}")
        
        return recommendations
    
    def export_metrics(self, metric_name: str, hours: int = 24, 
                      format_type: str = "json") -> Optional[str]:
        """导出指标"""
        try:
            metrics = self.get_metric_values(metric_name, hours)
            
            if format_type == "json":
                data = [m.to_dict() for m in metrics]
                return json.dumps(data, indent=2)
            else:
                # 可以扩展其他格式支持
                return None
                
        except Exception as e:
            logger.error(f"导出指标失败: {e}")
            return None
    
    def get_statistics_summary(self) -> Dict[str, Any]:
        """获取统计摘要"""
        try:
            return {
                'total_metrics': len(self.raw_metrics),
                'total_aggregations': sum(len(deque) for deque in self.aggregated_metrics.values()),
                'trend_analyses': len(self.trend_analysis),
                'detected_anomalies': len(self.anomaly_detection_results),
                'retention_hours': self.retention_hours,
                'aggregation_interval': self.aggregation_interval,
                'percentiles': self.percentiles
            }
        except Exception as e:
            logger.error(f"获取统计摘要失败: {e}")
            return {}