"""
实时性能指标收集和分析模块
负责收集、处理和分析系统性能数据
"""

import asyncio
import json
import time
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
import logging
from collections import deque, defaultdict
import threading
import psutil
import numpy as np
from pathlib import Path

class MetricCategory(Enum):
    """指标分类"""
    SYSTEM = "system"
    APPLICATION = "application"
    NETWORK = "network"
    DATABASE = "database"
    AGENT = "agent"
    USER = "user"

class AnalysisType(Enum):
    """分析类型"""
    REAL_TIME = "real_time"
    TREND = "trend"
    ANOMALY = "anomaly"
    CORRELATION = "correlation"
    FORECAST = "forecast"

@dataclass
class PerformanceMetric:
    """性能指标数据结构"""
    name: str
    category: MetricCategory
    value: float
    unit: str
    timestamp: float
    tags: Dict[str, str] = None
    metadata: Dict[str, Any] = None

@dataclass
class AnalysisResult:
    """分析结果"""
    metric_name: str
    analysis_type: AnalysisType
    result: Dict[str, Any]
    confidence: float
    timestamp: float
    recommendations: List[str] = None

class MetricsCollector:
    """指标收集器"""
    
    def __init__(self, db_path: str = "performance_metrics.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.is_collecting = False
        
        # 实时缓冲区
        self.real_time_buffer = defaultdict(lambda: deque(maxlen=1000))
        self.aggregation_buffer = defaultdict(list)
        
        # 收集器配置
        self.config = {
            "collection_interval": 1,  # 秒
            "aggregation_interval": 60,  # 秒
            "buffer_size": 10000,
            "anomaly_threshold": 2.0,  # 标准差倍数
            "trend_window": 300  # 趋势分析窗口（秒）
        }
        
        # 回调函数
        self.metric_callbacks: List[Callable] = []
        self.analysis_callbacks: List[Callable] = []
        
        # 初始化数据库
        self._init_database()
        
        # 线程锁
        self.lock = threading.Lock()
    
    def _init_database(self):
        """初始化数据库"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    value REAL NOT NULL,
                    unit TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    tags TEXT,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    analysis_type TEXT NOT NULL,
                    result TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    recommendations TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS aggregated_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    period_start REAL NOT NULL,
                    period_end REAL NOT NULL,
                    count INTEGER NOT NULL,
                    min_value REAL,
                    max_value REAL,
                    avg_value REAL,
                    median_value REAL,
                    std_dev REAL,
                    p95_value REAL,
                    p99_value REAL,
                    unit TEXT NOT NULL
                )
            """)
            
            # 创建索引
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_analysis_timestamp ON analysis_results(timestamp)")
    
    async def start_collection(self):
        """启动指标收集"""
        if self.is_collecting:
            return
        
        self.is_collecting = True
        self.logger.info("启动性能指标收集器")
        
        # 启动收集任务
        tasks = [
            asyncio.create_task(self._collect_system_metrics()),
            asyncio.create_task(self._collect_application_metrics()),
            asyncio.create_task(self._aggregate_metrics()),
            asyncio.create_task(self._perform_analysis()),
            asyncio.create_task(self._cleanup_old_data())
        ]
        
        await asyncio.gather(*tasks)
    
    async def stop_collection(self):
        """停止指标收集"""
        self.is_collecting = False
        self.logger.info("停止性能指标收集器")
    
    async def _collect_system_metrics(self):
        """收集系统指标"""
        while self.is_collecting:
            try:
                timestamp = time.time()
                
                # CPU指标
                cpu_percent = psutil.cpu_percent(interval=0.1)
                await self._record_metric(PerformanceMetric(
                    name="cpu.usage.percent",
                    category=MetricCategory.SYSTEM,
                    value=cpu_percent,
                    unit="percent",
                    timestamp=timestamp,
                    tags={"core": "all"}
                ))
                
                # 内存指标
                memory = psutil.virtual_memory()
                await self._record_metric(PerformanceMetric(
                    name="memory.usage.percent",
                    category=MetricCategory.SYSTEM,
                    value=memory.percent,
                    unit="percent",
                    timestamp=timestamp
                ))
                
                await self._record_metric(PerformanceMetric(
                    name="memory.available.bytes",
                    category=MetricCategory.SYSTEM,
                    value=memory.available,
                    unit="bytes",
                    timestamp=timestamp
                ))
                
                # 磁盘指标
                disk = psutil.disk_usage('/')
                disk_percent = (disk.used / disk.total) * 100
                await self._record_metric(PerformanceMetric(
                    name="disk.usage.percent",
                    category=MetricCategory.SYSTEM,
                    value=disk_percent,
                    unit="percent",
                    timestamp=timestamp
                ))
                
                # 网络指标
                network = psutil.net_io_counters()
                await self._record_metric(PerformanceMetric(
                    name="network.bytes_sent",
                    category=MetricCategory.NETWORK,
                    value=network.bytes_sent,
                    unit="bytes",
                    timestamp=timestamp
                ))
                
                await self._record_metric(PerformanceMetric(
                    name="network.bytes_recv",
                    category=MetricCategory.NETWORK,
                    value=network.bytes_recv,
                    unit="bytes",
                    timestamp=timestamp
                ))
                
                # 进程指标
                processes = len(psutil.pids())
                await self._record_metric(PerformanceMetric(
                    name="process.count",
                    category=MetricCategory.SYSTEM,
                    value=processes,
                    unit="count",
                    timestamp=timestamp
                ))
                
                await asyncio.sleep(self.config["collection_interval"])
                
            except Exception as e:
                self.logger.error(f"收集系统指标时出错: {e}")
                await asyncio.sleep(5)
    
    async def _collect_application_metrics(self):
        """收集应用指标"""
        # 这里应该集成实际的诺玛Agent指标
        # 目前使用模拟数据
        
        while self.is_collecting:
            try:
                timestamp = time.time()
                
                # 模拟响应时间
                response_time = self._generate_mock_response_time()
                await self._record_metric(PerformanceMetric(
                    name="app.response_time.avg",
                    category=MetricCategory.APPLICATION,
                    value=response_time,
                    unit="seconds",
                    timestamp=timestamp,
                    tags={"service": "norma_agent"}
                ))
                
                # 模拟请求数
                request_count = self._generate_mock_request_count()
                await self._record_metric(PerformanceMetric(
                    name="app.requests.count",
                    category=MetricCategory.APPLICATION,
                    value=request_count,
                    unit="count",
                    timestamp=timestamp,
                    tags={"service": "norma_agent"}
                ))
                
                # 模拟错误率
                error_rate = self._generate_mock_error_rate()
                await self._record_metric(PerformanceMetric(
                    name="app.error_rate.percent",
                    category=MetricCategory.APPLICATION,
                    value=error_rate,
                    unit="percent",
                    timestamp=timestamp,
                    tags={"service": "norma_agent"}
                ))
                
                # 模拟Agent性能指标
                agent_latency = self._generate_mock_agent_latency()
                await self._record_metric(PerformanceMetric(
                    name="agent.latency.avg",
                    category=MetricCategory.AGENT,
                    value=agent_latency,
                    unit="milliseconds",
                    timestamp=timestamp,
                    tags={"agent": "norma"}
                ))
                
                agent_accuracy = self._generate_mock_agent_accuracy()
                await self._record_metric(PerformanceMetric(
                    name="agent.accuracy.percent",
                    category=MetricCategory.AGENT,
                    value=agent_accuracy,
                    unit="percent",
                    timestamp=timestamp,
                    tags={"agent": "norma"}
                ))
                
                await asyncio.sleep(self.config["collection_interval"])
                
            except Exception as e:
                self.logger.error(f"收集应用指标时出错: {e}")
                await asyncio.sleep(5)
    
    async def _record_metric(self, metric: PerformanceMetric):
        """记录指标"""
        with self.lock:
            # 添加到实时缓冲区
            self.real_time_buffer[metric.name].append(metric)
            
            # 添加到聚合缓冲区
            self.aggregation_buffer[metric.name].append(metric)
            
            # 保存到数据库
            await self._save_metric_to_db(metric)
            
            # 触发回调
            for callback in self.metric_callbacks:
                try:
                    await callback(metric)
                except Exception as e:
                    self.logger.error(f"指标回调函数执行出错: {e}")
    
    async def _save_metric_to_db(self, metric: PerformanceMetric):
        """保存指标到数据库"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO metrics 
                    (name, category, value, unit, timestamp, tags, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    metric.name,
                    metric.category.value,
                    metric.value,
                    metric.unit,
                    metric.timestamp,
                    json.dumps(metric.tags) if metric.tags else None,
                    json.dumps(metric.metadata) if metric.metadata else None
                ))
        except Exception as e:
            self.logger.error(f"保存指标到数据库时出错: {e}")
    
    async def _aggregate_metrics(self):
        """聚合指标"""
        while self.is_collecting:
            try:
                await asyncio.sleep(self.config["aggregation_interval"])
                
                with self.lock:
                    for metric_name, metrics in self.aggregation_buffer.items():
                        if len(metrics) < 2:
                            continue
                        
                        # 计算聚合统计
                        values = [m.value for m in metrics]
                        period_start = min(m.timestamp for m in metrics)
                        period_end = max(m.timestamp for m in metrics)
                        
                        aggregation = {
                            "count": len(values),
                            "min": min(values),
                            "max": max(values),
                            "avg": statistics.mean(values),
                            "median": statistics.median(values),
                            "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
                            "p95": np.percentile(values, 95) if len(values) > 1 else values[0],
                            "p99": np.percentile(values, 99) if len(values) > 1 else values[0]
                        }
                        
                        # 保存聚合结果
                        await self._save_aggregated_metric(metric_name, period_start, period_end, aggregation)
                        
                        # 清空聚合缓冲区
                        self.aggregation_buffer[metric_name].clear()
                
            except Exception as e:
                self.logger.error(f"聚合指标时出错: {e}")
                await asyncio.sleep(60)
    
    async def _save_aggregated_metric(self, metric_name: str, period_start: float, 
                                    period_end: float, aggregation: Dict):
        """保存聚合指标"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO aggregated_metrics 
                    (metric_name, category, period_start, period_end, count, 
                     min_value, max_value, avg_value, median_value, std_dev, 
                     p95_value, p99_value, unit)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metric_name,
                    "unknown",  # 这里应该根据metric_name推断类别
                    period_start,
                    period_end,
                    aggregation["count"],
                    aggregation["min"],
                    aggregation["max"],
                    aggregation["avg"],
                    aggregation["median"],
                    aggregation["std_dev"],
                    aggregation["p95"],
                    aggregation["p99"],
                    "unknown"  # 这里应该根据metric_name推断单位
                ))
        except Exception as e:
            self.logger.error(f"保存聚合指标时出错: {e}")
    
    async def _perform_analysis(self):
        """执行指标分析"""
        while self.is_collecting:
            try:
                await asyncio.sleep(30)  # 每30秒执行一次分析
                
                # 实时分析
                await self._real_time_analysis()
                
                # 趋势分析
                await self._trend_analysis()
                
                # 异常检测
                await self._anomaly_detection()
                
                # 相关性分析
                await self._correlation_analysis()
                
            except Exception as e:
                self.logger.error(f"执行指标分析时出错: {e}")
                await asyncio.sleep(30)
    
    async def _real_time_analysis(self):
        """实时分析"""
        with self.lock:
            for metric_name, buffer in self.real_time_buffer.items():
                if len(buffer) < 10:
                    continue
                
                recent_values = [m.value for m in list(buffer)[-10:]]
                
                analysis_result = AnalysisResult(
                    metric_name=metric_name,
                    analysis_type=AnalysisType.REAL_TIME,
                    result={
                        "current_value": recent_values[-1],
                        "moving_average": statistics.mean(recent_values),
                        "trend": "increasing" if recent_values[-1] > recent_values[0] else "decreasing",
                        "volatility": statistics.stdev(recent_values) if len(recent_values) > 1 else 0
                    },
                    confidence=0.8,
                    timestamp=time.time()
                )
                
                await self._save_analysis_result(analysis_result)
    
    async def _trend_analysis(self):
        """趋势分析"""
        with self.lock:
            for metric_name, buffer in self.real_time_buffer.items():
                if len(buffer) < 50:
                    continue
                
                # 获取最近的数据点
                recent_metrics = list(buffer)[-50:]
                values = [m.value for m in recent_metrics]
                timestamps = [m.timestamp for m in recent_metrics]
                
                # 简单线性回归
                if len(values) > 2:
                    slope = self._calculate_slope(timestamps, values)
                    
                    analysis_result = AnalysisResult(
                        metric_name=metric_name,
                        analysis_type=AnalysisType.TREND,
                        result={
                            "slope": slope,
                            "trend_direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable",
                            "trend_strength": abs(slope),
                            "forecast_next_hour": values[-1] + slope * 3600
                        },
                        confidence=0.7,
                        timestamp=time.time()
                    )
                    
                    await self._save_analysis_result(analysis_result)
    
    async def _anomaly_detection(self):
        """异常检测"""
        with self.lock:
            for metric_name, buffer in self.real_time_buffer.items():
                if len(buffer) < 30:
                    continue
                
                values = [m.value for m in buffer]
                mean = statistics.mean(values)
                std_dev = statistics.stdev(values) if len(values) > 1 else 0
                
                if std_dev == 0:
                    continue
                
                threshold = self.config["anomaly_threshold"] * std_dev
                latest_value = values[-1]
                
                is_anomaly = abs(latest_value - mean) > threshold
                
                if is_anomaly:
                    analysis_result = AnalysisResult(
                        metric_name=metric_name,
                        analysis_type=AnalysisType.ANOMALY,
                        result={
                            "anomaly_detected": True,
                            "latest_value": latest_value,
                            "expected_range": [mean - threshold, mean + threshold],
                            "deviation_score": abs(latest_value - mean) / std_dev
                        },
                        confidence=0.9,
                        timestamp=time.time(),
                        recommendations=[
                            f"检测到 {metric_name} 异常值",
                            f"当前值: {latest_value:.2f}",
                            f"预期范围: [{mean - threshold:.2f}, {mean + threshold:.2f}]"
                        ]
                    )
                    
                    await self._save_analysis_result(analysis_result)
    
    async def _correlation_analysis(self):
        """相关性分析"""
        # 这里可以实现指标间的相关性分析
        # 简化实现
        pass
    
    async def _save_analysis_result(self, result: AnalysisResult):
        """保存分析结果"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO analysis_results 
                    (metric_name, analysis_type, result, confidence, timestamp, recommendations)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    result.metric_name,
                    result.analysis_type.value,
                    json.dumps(result.result),
                    result.confidence,
                    result.timestamp,
                    json.dumps(result.recommendations) if result.recommendations else None
                ))
            
            # 触发分析回调
            for callback in self.analysis_callbacks:
                try:
                    await callback(result)
                except Exception as e:
                    self.logger.error(f"分析回调函数执行出错: {e}")
                    
        except Exception as e:
            self.logger.error(f"保存分析结果时出错: {e}")
    
    async def _cleanup_old_data(self):
        """清理旧数据"""
        while self.is_collecting:
            try:
                await asyncio.sleep(3600)  # 每小时清理一次
                
                cutoff_time = time.time() - (7 * 24 * 3600)  # 保留7天数据
                
                with sqlite3.connect(self.db_path) as conn:
                    # 清理旧指标
                    conn.execute("DELETE FROM metrics WHERE timestamp < ?", (cutoff_time,))
                    
                    # 清理旧分析结果
                    conn.execute("DELETE FROM analysis_results WHERE timestamp < ?", (cutoff_time,))
                    
                    # 清理旧聚合指标
                    conn.execute("DELETE FROM aggregated_metrics WHERE period_end < ?", (cutoff_time,))
                
                self.logger.info("清理旧性能数据完成")
                
            except Exception as e:
                self.logger.error(f"清理旧数据时出错: {e}")
                await asyncio.sleep(3600)
    
    def _calculate_slope(self, x_values: List[float], y_values: List[float]) -> float:
        """计算线性回归斜率"""
        n = len(x_values)
        if n < 2:
            return 0
        
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x2 = sum(x * x for x in x_values)
        
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0
        
        return (n * sum_xy - sum_x * sum_y) / denominator
    
    def add_metric_callback(self, callback: Callable):
        """添加指标回调函数"""
        self.metric_callbacks.append(callback)
    
    def add_analysis_callback(self, callback: Callable):
        """添加分析回调函数"""
        self.analysis_callbacks.append(callback)
    
    def get_metric_history(self, metric_name: str, hours: int = 24) -> List[PerformanceMetric]:
        """获取指标历史数据"""
        start_time = time.time() - (hours * 3600)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT name, category, value, unit, timestamp, tags, metadata
                FROM metrics 
                WHERE name = ? AND timestamp > ?
                ORDER BY timestamp ASC
            """, (metric_name, start_time))
            
            metrics = []
            for row in cursor.fetchall():
                name, category, value, unit, timestamp, tags, metadata = row
                metrics.append(PerformanceMetric(
                    name=name,
                    category=MetricCategory(category),
                    value=value,
                    unit=unit,
                    timestamp=timestamp,
                    tags=json.loads(tags) if tags else None,
                    metadata=json.loads(metadata) if metadata else None
                ))
            
            return metrics
    
    def get_analysis_results(self, hours: int = 24) -> List[AnalysisResult]:
        """获取分析结果"""
        start_time = time.time() - (hours * 3600)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT metric_name, analysis_type, result, confidence, timestamp, recommendations
                FROM analysis_results 
                WHERE timestamp > ?
                ORDER BY timestamp DESC
            """, (start_time,))
            
            results = []
            for row in cursor.fetchall():
                metric_name, analysis_type, result, confidence, timestamp, recommendations = row
                results.append(AnalysisResult(
                    metric_name=metric_name,
                    analysis_type=AnalysisType(analysis_type),
                    result=json.loads(result),
                    confidence=confidence,
                    timestamp=timestamp,
                    recommendations=json.loads(recommendations) if recommendations else None
                ))
            
            return results
    
    # 模拟数据生成方法
    def _generate_mock_response_time(self) -> float:
        """生成模拟响应时间"""
        import random
        return random.uniform(0.1, 2.0)
    
    def _generate_mock_request_count(self) -> int:
        """生成模拟请求数"""
        import random
        return random.randint(5, 50)
    
    def _generate_mock_error_rate(self) -> float:
        """生成模拟错误率"""
        import random
        return random.uniform(0, 3.0)
    
    def _generate_mock_agent_latency(self) -> float:
        """生成模拟Agent延迟"""
        import random
        return random.uniform(50, 500)
    
    def _generate_mock_agent_accuracy(self) -> float:
        """生成模拟Agent准确率"""
        import random
        return random.uniform(85, 99)

# 使用示例
async def main():
    """主函数示例"""
    collector = MetricsCollector()
    
    # 添加回调函数
    async def metric_callback(metric: PerformanceMetric):
        print(f"新指标: {metric.name} = {metric.value} {metric.unit}")
    
    async def analysis_callback(result: AnalysisResult):
        print(f"分析结果: {result.metric_name} - {result.analysis_type.value}")
    
    collector.add_metric_callback(metric_callback)
    collector.add_analysis_callback(analysis_callback)
    
    try:
        await collector.start_collection()
        
        # 运行一段时间
        await asyncio.sleep(60)
        
    finally:
        await collector.stop_collection()

if __name__ == "__main__":
    asyncio.run(main())