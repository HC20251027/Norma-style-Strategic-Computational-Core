"""
诺玛Agent监控仪表板
实时显示系统性能、状态和关键指标
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import psutil
import sqlite3
import logging
from pathlib import Path

class MetricType(Enum):
    """指标类型枚举"""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DISK_USAGE = "disk_usage"
    NETWORK_IO = "network_io"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    ACTIVE_USERS = "active_users"
    REQUEST_COUNT = "request_count"
    AGENT_STATUS = "agent_status"

@dataclass
class MetricPoint:
    """指标数据点"""
    timestamp: float
    value: float
    metric_type: MetricType
    metadata: Dict[str, Any] = None

@dataclass
class SystemSnapshot:
    """系统状态快照"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    active_processes: int
    load_average: List[float]
    uptime: float

class MonitoringDashboard:
    """诺玛Agent监控仪表板"""
    
    def __init__(self, db_path: str = "monitoring_data.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.metrics_buffer: List[MetricPoint] = []
        self.alerts_buffer: List[Dict] = []
        self.is_monitoring = False
        
        # 初始化数据库
        self._init_database()
        
        # 监控配置
        self.config = {
            "metrics_collection_interval": 5,  # 秒
            "dashboard_refresh_interval": 1,   # 秒
            "alert_thresholds": {
                "cpu_critical": 90,
                "memory_critical": 85,
                "disk_critical": 90,
                "response_time_critical": 5.0,
                "error_rate_critical": 10.0
            },
            "retention_days": 30
        }
    
    def _init_database(self):
        """初始化数据库"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    metric_type TEXT NOT NULL,
                    value REAL NOT NULL,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    resolved BOOLEAN DEFAULT FALSE,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    cpu_percent REAL NOT NULL,
                    memory_percent REAL NOT NULL,
                    disk_percent REAL NOT NULL,
                    network_bytes_sent INTEGER NOT NULL,
                    network_bytes_recv INTEGER NOT NULL,
                    active_processes INTEGER NOT NULL,
                    load_average TEXT NOT NULL,
                    uptime REAL NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_behavior (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    user_id TEXT,
                    action_type TEXT NOT NULL,
                    session_id TEXT,
                    duration REAL,
                    metadata TEXT
                )
            """)
    
    async def start_monitoring(self):
        """启动监控"""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.logger.info("启动诺玛Agent监控仪表板")
        
        # 启动各个监控任务
        tasks = [
            asyncio.create_task(self._collect_system_metrics()),
            asyncio.create_task(self._collect_application_metrics()),
            asyncio.create_task(self._check_alerts()),
            asyncio.create_task(self._cleanup_old_data()),
            asyncio.create_task(self._generate_dashboard_data())
        ]
        
        await asyncio.gather(*tasks)
    
    async def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        self.logger.info("停止诺玛Agent监控仪表板")
    
    async def _collect_system_metrics(self):
        """收集系统指标"""
        prev_network = psutil.net_io_counters()
        
        while self.is_monitoring:
            try:
                # CPU使用率
                cpu_percent = psutil.cpu_percent(interval=1)
                self._add_metric(MetricType.CPU_USAGE, cpu_percent)
                
                # 内存使用率
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                self._add_metric(MetricType.MEMORY_USAGE, memory_percent)
                
                # 磁盘使用率
                disk = psutil.disk_usage('/')
                disk_percent = (disk.used / disk.total) * 100
                self._add_metric(MetricType.DISK_USAGE, disk_percent)
                
                # 网络IO
                current_network = psutil.net_io_counters()
                bytes_sent = current_network.bytes_sent - prev_network.bytes_sent
                bytes_recv = current_network.bytes_recv - prev_network.bytes_recv
                
                self._add_metric(MetricType.NETWORK_IO, bytes_sent + bytes_recv, {
                    "bytes_sent": bytes_sent,
                    "bytes_recv": bytes_recv
                })
                
                prev_network = current_network
                
                # 系统快照
                snapshot = SystemSnapshot(
                    timestamp=time.time(),
                    cpu_percent=cpu_percent,
                    memory_percent=memory_percent,
                    disk_percent=disk_percent,
                    network_bytes_sent=current_network.bytes_sent,
                    network_bytes_recv=current_network.bytes_recv,
                    active_processes=len(psutil.pids()),
                    load_average=psutil.getloadavg(),
                    uptime=time.time() - psutil.boot_time()
                )
                
                self._save_system_snapshot(snapshot)
                
                await asyncio.sleep(self.config["metrics_collection_interval"])
                
            except Exception as e:
                self.logger.error(f"收集系统指标时出错: {e}")
                await asyncio.sleep(5)
    
    async def _collect_application_metrics(self):
        """收集应用指标"""
        # 这里应该集成实际的诺玛Agent指标
        # 目前使用模拟数据
        
        while self.is_monitoring:
            try:
                # 模拟响应时间
                response_time = self._generate_mock_response_time()
                self._add_metric(MetricType.RESPONSE_TIME, response_time)
                
                # 模拟错误率
                error_rate = self._generate_mock_error_rate()
                self._add_metric(MetricType.ERROR_RATE, error_rate)
                
                # 模拟活跃用户数
                active_users = self._generate_mock_active_users()
                self._add_metric(MetricType.ACTIVE_USERS, active_users)
                
                # 模拟请求数
                request_count = self._generate_mock_request_count()
                self._add_metric(MetricType.REQUEST_COUNT, request_count)
                
                # Agent状态
                agent_status = self._generate_mock_agent_status()
                self._add_metric(MetricType.AGENT_STATUS, agent_status)
                
                await asyncio.sleep(self.config["metrics_collection_interval"])
                
            except Exception as e:
                self.logger.error(f"收集应用指标时出错: {e}")
                await asyncio.sleep(5)
    
    def _add_metric(self, metric_type: MetricType, value: float, metadata: Dict = None):
        """添加指标数据点"""
        metric = MetricPoint(
            timestamp=time.time(),
            value=value,
            metric_type=metric_type,
            metadata=metadata
        )
        
        self.metrics_buffer.append(metric)
        
        # 保存到数据库
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO metrics (timestamp, metric_type, value, metadata)
                VALUES (?, ?, ?, ?)
            """, (
                metric.timestamp,
                metric_type.value,
                metric.value,
                json.dumps(metadata) if metadata else None
            ))
    
    def _save_system_snapshot(self, snapshot: SystemSnapshot):
        """保存系统快照"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO system_snapshots 
                (timestamp, cpu_percent, memory_percent, disk_percent, 
                 network_bytes_sent, network_bytes_recv, active_processes, 
                 load_average, uptime)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                snapshot.timestamp,
                snapshot.cpu_percent,
                snapshot.memory_percent,
                snapshot.disk_percent,
                snapshot.network_bytes_sent,
                snapshot.network_bytes_recv,
                snapshot.active_processes,
                json.dumps(snapshot.load_average),
                snapshot.uptime
            ))
    
    async def _check_alerts(self):
        """检查告警条件"""
        while self.is_monitoring:
            try:
                # 获取最新指标
                latest_metrics = self._get_latest_metrics()
                
                # 检查CPU告警
                if "cpu_usage" in latest_metrics:
                    cpu_value = latest_metrics["cpu_usage"]
                    if cpu_value >= self.config["alert_thresholds"]["cpu_critical"]:
                        self._create_alert(
                            "CPU使用率过高",
                            "critical",
                            f"CPU使用率达到 {cpu_value:.1f}%"
                        )
                
                # 检查内存告警
                if "memory_usage" in latest_metrics:
                    memory_value = latest_metrics["memory_usage"]
                    if memory_value >= self.config["alert_thresholds"]["memory_critical"]:
                        self._create_alert(
                            "内存使用率过高",
                            "critical",
                            f"内存使用率达到 {memory_value:.1f}%"
                        )
                
                # 检查响应时间告警
                if "response_time" in latest_metrics:
                    response_time = latest_metrics["response_time"]
                    if response_time >= self.config["alert_thresholds"]["response_time_critical"]:
                        self._create_alert(
                            "响应时间过长",
                            "warning",
                            f"平均响应时间为 {response_time:.2f}秒"
                        )
                
                # 检查错误率告警
                if "error_rate" in latest_metrics:
                    error_rate = latest_metrics["error_rate"]
                    if error_rate >= self.config["alert_thresholds"]["error_rate_critical"]:
                        self._create_alert(
                            "错误率过高",
                            "critical",
                            f"错误率为 {error_rate:.1f}%"
                        )
                
                await asyncio.sleep(30)  # 每30秒检查一次告警
                
            except Exception as e:
                self.logger.error(f"检查告警时出错: {e}")
                await asyncio.sleep(30)
    
    def _create_alert(self, message: str, severity: str, alert_type: str = "system"):
        """创建告警"""
        alert = {
            "timestamp": time.time(),
            "alert_type": alert_type,
            "severity": severity,
            "message": message,
            "resolved": False
        }
        
        self.alerts_buffer.append(alert)
        
        # 保存到数据库
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO alerts (timestamp, alert_type, severity, message, resolved)
                VALUES (?, ?, ?, ?, ?)
            """, (
                alert["timestamp"],
                alert_type,
                severity,
                message,
                False
            ))
        
        self.logger.warning(f"告警: {message}")
    
    def _get_latest_metrics(self) -> Dict[str, float]:
        """获取最新指标"""
        latest_metrics = {}
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT metric_type, value 
                FROM metrics 
                WHERE timestamp > ? 
                ORDER BY timestamp DESC
            """, (time.time() - 300,))  # 最近5分钟
            
            for metric_type, value in cursor.fetchall():
                if metric_type not in latest_metrics:
                    latest_metrics[metric_type] = value
        
        return latest_metrics
    
    async def _cleanup_old_data(self):
        """清理旧数据"""
        while self.is_monitoring:
            try:
                cutoff_time = time.time() - (self.config["retention_days"] * 24 * 3600)
                
                with sqlite3.connect(self.db_path) as conn:
                    # 清理旧指标
                    conn.execute("DELETE FROM metrics WHERE timestamp < ?", (cutoff_time,))
                    
                    # 清理旧告警
                    conn.execute("DELETE FROM alerts WHERE timestamp < ?", (cutoff_time,))
                    
                    # 清理旧系统快照
                    conn.execute("DELETE FROM system_snapshots WHERE timestamp < ?", (cutoff_time,))
                    
                    # 清理旧用户行为数据
                    conn.execute("DELETE FROM user_behavior WHERE timestamp < ?", (cutoff_time,))
                
                self.logger.info("清理旧监控数据完成")
                await asyncio.sleep(3600)  # 每小时清理一次
                
            except Exception as e:
                self.logger.error(f"清理旧数据时出错: {e}")
                await asyncio.sleep(3600)
    
    async def _generate_dashboard_data(self):
        """生成仪表板数据"""
        while self.is_monitoring:
            try:
                dashboard_data = self._build_dashboard_data()
                
                # 保存仪表板数据
                dashboard_file = Path("dashboard_data.json")
                with open(dashboard_file, 'w', encoding='utf-8') as f:
                    json.dump(dashboard_data, f, ensure_ascii=False, indent=2)
                
                await asyncio.sleep(self.config["dashboard_refresh_interval"])
                
            except Exception as e:
                self.logger.error(f"生成仪表板数据时出错: {e}")
                await asyncio.sleep(5)
    
    def _build_dashboard_data(self) -> Dict[str, Any]:
        """构建仪表板数据"""
        now = time.time()
        hour_ago = now - 3600
        day_ago = now - 86400
        
        with sqlite3.connect(self.db_path) as conn:
            # 获取最近1小时的指标
            cursor = conn.execute("""
                SELECT metric_type, AVG(value) as avg_value, MAX(value) as max_value
                FROM metrics 
                WHERE timestamp > ?
                GROUP BY metric_type
            """, (hour_ago,))
            
            recent_metrics = {}
            for metric_type, avg_value, max_value in cursor.fetchall():
                recent_metrics[metric_type] = {
                    "average": avg_value,
                    "maximum": max_value
                }
            
            # 获取活跃告警
            cursor = conn.execute("""
                SELECT alert_type, severity, COUNT(*) as count
                FROM alerts 
                WHERE resolved = FALSE AND timestamp > ?
                GROUP BY alert_type, severity
            """, (day_ago,))
            
            active_alerts = {}
            for alert_type, severity, count in cursor.fetchall():
                key = f"{alert_type}_{severity}"
                active_alerts[key] = count
            
            # 获取系统状态
            cursor = conn.execute("""
                SELECT * FROM system_snapshots 
                WHERE timestamp > ?
                ORDER BY timestamp DESC 
                LIMIT 1
            """, (hour_ago,))
            
            latest_snapshot = cursor.fetchone()
            system_status = {}
            if latest_snapshot:
                system_status = {
                    "cpu_percent": latest_snapshot[2],
                    "memory_percent": latest_snapshot[3],
                    "disk_percent": latest_snapshot[4],
                    "uptime": latest_snapshot[9]
                }
        
        return {
            "timestamp": now,
            "system_status": system_status,
            "recent_metrics": recent_metrics,
            "active_alerts": active_alerts,
            "monitoring_active": self.is_monitoring
        }
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """获取仪表板数据"""
        try:
            dashboard_file = Path("dashboard_data.json")
            if dashboard_file.exists():
                with open(dashboard_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return self._build_dashboard_data()
        except Exception as e:
            self.logger.error(f"获取仪表板数据时出错: {e}")
            return {"error": str(e)}
    
    def get_metrics_history(self, metric_type: str, hours: int = 24) -> List[Dict]:
        """获取指标历史数据"""
        start_time = time.time() - (hours * 3600)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT timestamp, value, metadata
                FROM metrics 
                WHERE metric_type = ? AND timestamp > ?
                ORDER BY timestamp ASC
            """, (metric_type, start_time))
            
            history = []
            for timestamp, value, metadata in cursor.fetchall():
                history.append({
                    "timestamp": timestamp,
                    "value": value,
                    "metadata": json.loads(metadata) if metadata else None
                })
            
            return history
    
    def get_alerts(self, hours: int = 24) -> List[Dict]:
        """获取告警历史"""
        start_time = time.time() - (hours * 3600)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT timestamp, alert_type, severity, message, resolved
                FROM alerts 
                WHERE timestamp > ?
                ORDER BY timestamp DESC
            """, (start_time,))
            
            alerts = []
            for timestamp, alert_type, severity, message, resolved in cursor.fetchall():
                alerts.append({
                    "timestamp": timestamp,
                    "alert_type": alert_type,
                    "severity": severity,
                    "message": message,
                    "resolved": bool(resolved)
                })
            
            return alerts
    
    def resolve_alert(self, alert_id: int):
        """解决告警"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE alerts 
                SET resolved = TRUE 
                WHERE id = ?
            """, (alert_id,))
    
    # 模拟数据生成方法
    def _generate_mock_response_time(self) -> float:
        """生成模拟响应时间"""
        import random
        return random.uniform(0.1, 3.0)
    
    def _generate_mock_error_rate(self) -> float:
        """生成模拟错误率"""
        import random
        return random.uniform(0, 5.0)
    
    def _generate_mock_active_users(self) -> int:
        """生成模拟活跃用户数"""
        import random
        return random.randint(1, 50)
    
    def _generate_mock_request_count(self) -> int:
        """生成模拟请求数"""
        import random
        return random.randint(10, 100)
    
    def _generate_mock_agent_status(self) -> float:
        """生成模拟Agent状态"""
        import random
        return random.uniform(0.8, 1.0)

# 使用示例
async def main():
    """主函数示例"""
    dashboard = MonitoringDashboard()
    
    try:
        await dashboard.start_monitoring()
        
        # 运行一段时间
        await asyncio.sleep(60)
        
    finally:
        await dashboard.stop_monitoring()

if __name__ == "__main__":
    asyncio.run(main())