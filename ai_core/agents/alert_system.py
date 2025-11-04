"""
智能告警和异常检测系统
负责监控指标异常并发送智能告警
"""

import asyncio
import json
import time
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
import logging
from collections import defaultdict, deque
import threading
import smtplib
try:
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
except ImportError:
    # 兼容性处理
    MIMEText = None
    MIMEMultipart = None
import numpy as np
from pathlib import Path

class AlertSeverity(Enum):
    """告警严重级别"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class AlertStatus(Enum):
    """告警状态"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"

class AnomalyType(Enum):
    """异常类型"""
    SPIKE = "spike"
    DROP = "drop"
    TREND_CHANGE = "trend_change"
    PATTERN_BREAK = "pattern_break"
    SEASONAL_ANOMALY = "seasonal_anomaly"
    STATISTICAL = "statistical"

@dataclass
class AlertRule:
    """告警规则"""
    name: str
    metric_name: str
    condition: str  # "greater_than", "less_than", "equals", "outside_range"
    threshold: float
    severity: AlertSeverity
    enabled: bool = True
    duration: int = 0  # 持续时间（秒）
    description: str = ""
    tags: Dict[str, str] = None

@dataclass
class Alert:
    """告警"""
    id: Optional[int]
    rule_name: str
    metric_name: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    value: float
    threshold: float
    timestamp: float
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[float] = None
    resolved_at: Optional[float] = None
    metadata: Dict[str, Any] = None

@dataclass
class AnomalyDetection:
    """异常检测结果"""
    metric_name: str
    anomaly_type: AnomalyType
    severity: AlertSeverity
    score: float
    timestamp: float
    details: Dict[str, Any]
    context: Dict[str, Any] = None

class AlertSystem:
    """智能告警系统"""
    
    def __init__(self, db_path: str = "alert_system.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.is_running = False
        
        # 告警规则
        self.alert_rules: List[AlertRule] = []
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        
        # 异常检测配置
        self.anomaly_config = {
            "window_size": 100,
            "threshold_std": 2.5,
            "trend_threshold": 0.1,
            "seasonal_period": 24 * 3600,  # 24小时
            "min_data_points": 30
        }
        
        # 告警配置
        self.notification_config = {
            "email_enabled": False,
            "smtp_server": "",
            "smtp_port": 587,
            "username": "",
            "password": "",
            "recipients": [],
            "webhook_url": "",
            "slack_webhook": ""
        }
        
        # 线程锁
        self.lock = threading.Lock()
        
        # 回调函数
        self.alert_callbacks: List[Callable] = []
        
        # 初始化数据库
        self._init_database()
        
        # 加载默认规则
        self._load_default_rules()
    
    def _init_database(self):
        """初始化数据库"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alert_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    metric_name TEXT NOT NULL,
                    condition TEXT NOT NULL,
                    threshold REAL NOT NULL,
                    severity TEXT NOT NULL,
                    enabled BOOLEAN DEFAULT TRUE,
                    duration INTEGER DEFAULT 0,
                    description TEXT,
                    tags TEXT,
                    created_at REAL DEFAULT CURRENT_TIMESTAMP,
                    updated_at REAL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule_name TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    status TEXT NOT NULL,
                    message TEXT NOT NULL,
                    value REAL NOT NULL,
                    threshold REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    acknowledged_by TEXT,
                    acknowledged_at REAL,
                    resolved_at REAL,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS anomaly_detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    anomaly_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    score REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    details TEXT NOT NULL,
                    context TEXT
                )
            """)
            
            # 创建索引
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_anomaly_timestamp ON anomaly_detections(timestamp)")
    
    def _load_default_rules(self):
        """加载默认告警规则"""
        default_rules = [
            AlertRule(
                name="high_cpu_usage",
                metric_name="cpu.usage.percent",
                condition="greater_than",
                threshold=85.0,
                severity=AlertSeverity.WARNING,
                duration=60,
                description="CPU使用率过高"
            ),
            AlertRule(
                name="critical_cpu_usage",
                metric_name="cpu.usage.percent",
                condition="greater_than",
                threshold=95.0,
                severity=AlertSeverity.CRITICAL,
                duration=30,
                description="CPU使用率严重过高"
            ),
            AlertRule(
                name="high_memory_usage",
                metric_name="memory.usage.percent",
                condition="greater_than",
                threshold=90.0,
                severity=AlertSeverity.WARNING,
                duration=60,
                description="内存使用率过高"
            ),
            AlertRule(
                name="high_response_time",
                metric_name="app.response_time.avg",
                condition="greater_than",
                threshold=2.0,
                severity=AlertSeverity.WARNING,
                duration=120,
                description="响应时间过长"
            ),
            AlertRule(
                name="high_error_rate",
                metric_name="app.error_rate.percent",
                condition="greater_than",
                threshold=5.0,
                severity=AlertSeverity.CRITICAL,
                duration=60,
                description="错误率过高"
            ),
            AlertRule(
                name="low_agent_accuracy",
                metric_name="agent.accuracy.percent",
                condition="less_than",
                threshold=80.0,
                severity=AlertSeverity.WARNING,
                duration=300,
                description="Agent准确率过低"
            )
        ]
        
        for rule in default_rules:
            self.add_alert_rule(rule)
    
    async def start_monitoring(self):
        """启动告警监控"""
        if self.is_running:
            return
        
        self.is_running = True
        self.logger.info("启动智能告警系统")
        
        # 启动监控任务
        tasks = [
            asyncio.create_task(self._monitor_alert_rules()),
            asyncio.create_task(self._detect_anomalies()),
            asyncio.create_task(self._cleanup_old_alerts()),
            asyncio.create_task(self._send_notifications())
        ]
        
        await asyncio.gather(*tasks)
    
    async def stop_monitoring(self):
        """停止告警监控"""
        self.is_running = False
        self.logger.info("停止智能告警系统")
    
    def add_alert_rule(self, rule: AlertRule):
        """添加告警规则"""
        with self.lock:
            # 检查规则是否已存在
            existing_rule = next((r for r in self.alert_rules if r.name == rule.name), None)
            if existing_rule:
                self.logger.warning(f"告警规则 {rule.name} 已存在，将被更新")
                self.alert_rules.remove(existing_rule)
            
            self.alert_rules.append(rule)
            
            # 保存到数据库
            self._save_alert_rule_to_db(rule)
            
            self.logger.info(f"添加告警规则: {rule.name}")
    
    def remove_alert_rule(self, rule_name: str):
        """删除告警规则"""
        with self.lock:
            rule = next((r for r in self.alert_rules if r.name == rule_name), None)
            if rule:
                self.alert_rules.remove(rule)
                self._delete_alert_rule_from_db(rule_name)
                self.logger.info(f"删除告警规则: {rule_name}")
    
    def _save_alert_rule_to_db(self, rule: AlertRule):
        """保存告警规则到数据库"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO alert_rules 
                (name, metric_name, condition, threshold, severity, enabled, duration, description, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                rule.name,
                rule.metric_name,
                rule.condition,
                rule.threshold,
                rule.severity.value,
                rule.enabled,
                rule.duration,
                rule.description,
                json.dumps(rule.tags) if rule.tags else None
            ))
    
    def _delete_alert_rule_from_db(self, rule_name: str):
        """从数据库删除告警规则"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM alert_rules WHERE name = ?", (rule_name,))
    
    async def _monitor_alert_rules(self):
        """监控告警规则"""
        while self.is_running:
            try:
                # 获取最新的指标数据
                for rule in self.alert_rules:
                    if not rule.enabled:
                        continue
                    
                    await self._check_rule(rule)
                
                await asyncio.sleep(30)  # 每30秒检查一次规则
                
            except Exception as e:
                self.logger.error(f"监控告警规则时出错: {e}")
                await asyncio.sleep(30)
    
    async def _check_rule(self, rule: AlertRule):
        """检查单个告警规则"""
        try:
            # 获取最新的指标值
            latest_value = await self._get_latest_metric_value(rule.metric_name)
            if latest_value is None:
                return
            
            # 检查条件
            condition_met = self._evaluate_condition(rule.condition, latest_value, rule.threshold)
            
            if condition_met:
                # 检查是否已有活跃告警
                alert_key = f"{rule.name}_{rule.metric_name}"
                
                if alert_key not in self.active_alerts:
                    # 创建新告警
                    alert = Alert(
                        id=None,
                        rule_name=rule.name,
                        metric_name=rule.metric_name,
                        severity=rule.severity,
                        status=AlertStatus.ACTIVE,
                        message=f"{rule.description}: {latest_value:.2f} (阈值: {rule.threshold})",
                        value=latest_value,
                        threshold=rule.threshold,
                        timestamp=time.time()
                    )
                    
                    await self._create_alert(alert)
                else:
                    # 更新现有告警的时间戳
                    self.active_alerts[alert_key].timestamp = time.time()
            else:
                # 检查是否需要解决告警
                alert_key = f"{rule.name}_{rule.metric_name}"
                if alert_key in self.active_alerts:
                    await self._resolve_alert(alert_key)
        
        except Exception as e:
            self.logger.error(f"检查规则 {rule.name} 时出错: {e}")
    
    def _evaluate_condition(self, condition: str, value: float, threshold: float) -> bool:
        """评估告警条件"""
        if condition == "greater_than":
            return value > threshold
        elif condition == "less_than":
            return value < threshold
        elif condition == "equals":
            return abs(value - threshold) < 0.001
        elif condition == "outside_range":
            return value < threshold[0] or value > threshold[1]
        else:
            return False
    
    async def _get_latest_metric_value(self, metric_name: str) -> Optional[float]:
        """获取最新指标值"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT value FROM metrics 
                    WHERE name = ? 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                """, (metric_name,))
                
                result = cursor.fetchone()
                return result[0] if result else None
        except Exception as e:
            self.logger.error(f"获取指标 {metric_name} 值时出错: {e}")
            return None
    
    async def _create_alert(self, alert: Alert):
        """创建告警"""
        with self.lock:
            # 保存到数据库
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    INSERT INTO alerts 
                    (rule_name, metric_name, severity, status, message, value, threshold, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    alert.rule_name,
                    alert.metric_name,
                    alert.severity.value,
                    alert.status.value,
                    alert.message,
                    alert.value,
                    alert.threshold,
                    alert.timestamp
                ))
                
                alert.id = cursor.lastrowid
            
            # 添加到活跃告警列表
            alert_key = f"{alert.rule_name}_{alert.metric_name}"
            self.active_alerts[alert_key] = alert
            
            # 添加到历史记录
            self.alert_history.append(alert)
            
            self.logger.warning(f"创建告警: {alert.message}")
            
            # 触发回调
            for callback in self.alert_callbacks:
                try:
                    await callback(alert)
                except Exception as e:
                    self.logger.error(f"告警回调函数执行出错: {e}")
    
    async def _resolve_alert(self, alert_key: str):
        """解决告警"""
        if alert_key not in self.active_alerts:
            return
        
        alert = self.active_alerts[alert_key]
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = time.time()
        
        # 更新数据库
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE alerts 
                SET status = ?, resolved_at = ?
                WHERE id = ?
            """, (AlertStatus.RESOLVED.value, alert.resolved_at, alert.id))
        
        self.logger.info(f"解决告警: {alert.message}")
        
        # 从活跃告警列表中移除
        del self.active_alerts[alert_key]
    
    async def _detect_anomalies(self):
        """异常检测"""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # 每分钟执行一次异常检测
                
                # 获取所有指标名称
                metric_names = await self._get_all_metric_names()
                
                for metric_name in metric_names:
                    await self._detect_metric_anomalies(metric_name)
                
            except Exception as e:
                self.logger.error(f"异常检测时出错: {e}")
                await asyncio.sleep(60)
    
    async def _detect_metric_anomalies(self, metric_name: str):
        """检测单个指标的异常"""
        try:
            # 获取历史数据
            data = await self._get_metric_history(metric_name, hours=24)
            if len(data) < self.anomaly_config["min_data_points"]:
                return
            
            values = [item["value"] for item in data]
            timestamps = [item["timestamp"] for item in data]
            
            # 统计异常检测
            await self._detect_statistical_anomalies(metric_name, values, timestamps)
            
            # 趋势异常检测
            await self._detect_trend_anomalies(metric_name, values, timestamps)
            
            # 模式异常检测
            await self._detect_pattern_anomalies(metric_name, values, timestamps)
            
        except Exception as e:
            self.logger.error(f"检测指标 {metric_name} 异常时出错: {e}")
    
    async def _detect_statistical_anomalies(self, metric_name: str, values: List[float], timestamps: List[float]):
        """统计异常检测"""
        if len(values) < 10:
            return
        
        mean = statistics.mean(values)
        std_dev = statistics.stdev(values)
        
        if std_dev == 0:
            return
        
        threshold = self.anomaly_config["threshold_std"] * std_dev
        
        # 检查最新值
        latest_value = values[-1]
        if abs(latest_value - mean) > threshold:
            score = abs(latest_value - mean) / std_dev
            
            anomaly = AnomalyDetection(
                metric_name=metric_name,
                anomaly_type=AnomalyType.STATISTICAL,
                severity=AlertSeverity.WARNING if score < 3 else AlertSeverity.CRITICAL,
                score=score,
                timestamp=timestamps[-1],
                details={
                    "latest_value": latest_value,
                    "mean": mean,
                    "std_dev": std_dev,
                    "threshold": threshold
                }
            )
            
            await self._save_anomaly_detection(anomaly)
    
    async def _detect_trend_anomalies(self, metric_name: str, values: List[float], timestamps: List[float]):
        """趋势异常检测"""
        if len(values) < 20:
            return
        
        # 计算滑动窗口的趋势
        window_size = min(10, len(values) // 2)
        
        for i in range(window_size, len(values)):
            window_values = values[i-window_size:i]
            
            # 线性回归计算斜率
            slope = self._calculate_slope(list(range(window_size)), window_values)
            
            if abs(slope) > self.anomaly_config["trend_threshold"]:
                anomaly = AnomalyDetection(
                    metric_name=metric_name,
                    anomaly_type=AnomalyType.TREND_CHANGE,
                    severity=AlertSeverity.WARNING,
                    score=abs(slope),
                    timestamp=timestamps[i],
                    details={
                        "slope": slope,
                        "window_size": window_size,
                        "trend_direction": "increasing" if slope > 0 else "decreasing"
                    }
                )
                
                await self._save_anomaly_detection(anomaly)
    
    async def _detect_pattern_anomalies(self, metric_name: str, values: List[float], timestamps: List[float]):
        """模式异常检测"""
        # 简化实现，检查周期性模式破坏
        if len(values) < 48:  # 至少需要2天的数据
            return
        
        # 这里可以实现更复杂的季节性异常检测
        # 目前使用简化版本
        pass
    
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
    
    async def _save_anomaly_detection(self, anomaly: AnomalyDetection):
        """保存异常检测结果"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO anomaly_detections 
                    (metric_name, anomaly_type, severity, score, timestamp, details, context)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    anomaly.metric_name,
                    anomaly.anomaly_type.value,
                    anomaly.severity.value,
                    anomaly.score,
                    anomaly.timestamp,
                    json.dumps(anomaly.details),
                    json.dumps(anomaly.context) if anomaly.context else None
                ))
            
            self.logger.info(f"检测到异常: {anomaly.metric_name} - {anomaly.anomaly_type.value}")
            
        except Exception as e:
            self.logger.error(f"保存异常检测结果时出错: {e}")
    
    async def _get_all_metric_names(self) -> List[str]:
        """获取所有指标名称"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT DISTINCT name FROM metrics 
                    WHERE timestamp > ?
                    ORDER BY name
                """, (time.time() - 3600,))  # 最近1小时
                
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"获取指标名称时出错: {e}")
            return []
    
    async def _get_metric_history(self, metric_name: str, hours: int) -> List[Dict]:
        """获取指标历史数据"""
        start_time = time.time() - (hours * 3600)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT value, timestamp FROM metrics 
                    WHERE name = ? AND timestamp > ?
                    ORDER BY timestamp ASC
                """, (metric_name, start_time))
                
                return [{"value": row[0], "timestamp": row[1]} for row in cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"获取指标历史时出错: {e}")
            return []
    
    async def _cleanup_old_alerts(self):
        """清理旧告警"""
        while self.is_running:
            try:
                await asyncio.sleep(3600)  # 每小时清理一次
                
                cutoff_time = time.time() - (30 * 24 * 3600)  # 保留30天
                
                with sqlite3.connect(self.db_path) as conn:
                    # 清理旧告警
                    conn.execute("DELETE FROM alerts WHERE timestamp < ?", (cutoff_time,))
                    
                    # 清理旧异常检测结果
                    conn.execute("DELETE FROM anomaly_detections WHERE timestamp < ?", (cutoff_time,))
                
                self.logger.info("清理旧告警数据完成")
                
            except Exception as e:
                self.logger.error(f"清理旧告警时出错: {e}")
                await asyncio.sleep(3600)
    
    async def _send_notifications(self):
        """发送通知"""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # 每分钟检查一次需要发送的通知
                
                # 发送邮件通知
                if self.notification_config["email_enabled"]:
                    await self._send_email_notifications()
                
                # 发送webhook通知
                if self.notification_config["webhook_url"]:
                    await self._send_webhook_notifications()
                
            except Exception as e:
                self.logger.error(f"发送通知时出错: {e}")
                await asyncio.sleep(60)
    
    async def _send_email_notifications(self):
        """发送邮件通知"""
        # 简化实现
        # 实际使用时需要配置SMTP服务器
        pass
    
    async def _send_webhook_notifications(self):
        """发送webhook通知"""
        # 简化实现
        # 实际使用时需要配置webhook URL
        pass
    
    def add_alert_callback(self, callback: Callable):
        """添加告警回调函数"""
        self.alert_callbacks.append(callback)
    
    def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """获取告警历史"""
        start_time = time.time() - (hours * 3600)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, rule_name, metric_name, severity, status, message, 
                       value, threshold, timestamp, acknowledged_by, acknowledged_at, 
                       resolved_at, metadata
                FROM alerts 
                WHERE timestamp > ?
                ORDER BY timestamp DESC
            """, (start_time,))
            
            alerts = []
            for row in cursor.fetchall():
                alerts.append(Alert(
                    id=row[0],
                    rule_name=row[1],
                    metric_name=row[2],
                    severity=AlertSeverity(row[3]),
                    status=AlertStatus(row[4]),
                    message=row[5],
                    value=row[6],
                    threshold=row[7],
                    timestamp=row[8],
                    acknowledged_by=row[9],
                    acknowledged_at=row[10],
                    resolved_at=row[11],
                    metadata=json.loads(row[12]) if row[12] else None
                ))
            
            return alerts
    
    def acknowledge_alert(self, alert_id: int, acknowledged_by: str):
        """确认告警"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE alerts 
                SET status = ?, acknowledged_by = ?, acknowledged_at = ?
                WHERE id = ?
            """, (
                AlertStatus.ACKNOWLEDGED.value,
                acknowledged_by,
                time.time(),
                alert_id
            ))
        
        # 更新内存中的告警状态
        for alert in self.active_alerts.values():
            if alert.id == alert_id:
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = time.time()
                break
    
    def resolve_alert(self, alert_id: int):
        """手动解决告警"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE alerts 
                SET status = ?, resolved_at = ?
                WHERE id = ?
            """, (AlertStatus.RESOLVED.value, time.time(), alert_id))
        
        # 从活跃告警列表中移除
        for key, alert in list(self.active_alerts.items()):
            if alert.id == alert_id:
                del self.active_alerts[key]
                break
    
    def get_anomaly_history(self, hours: int = 24) -> List[AnomalyDetection]:
        """获取异常历史"""
        start_time = time.time() - (hours * 3600)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT metric_name, anomaly_type, severity, score, timestamp, details, context
                FROM anomaly_detections 
                WHERE timestamp > ?
                ORDER BY timestamp DESC
            """, (start_time,))
            
            anomalies = []
            for row in cursor.fetchall():
                anomalies.append(AnomalyDetection(
                    metric_name=row[0],
                    anomaly_type=AnomalyType(row[1]),
                    severity=AlertSeverity(row[2]),
                    score=row[3],
                    timestamp=row[4],
                    details=json.loads(row[5]),
                    context=json.loads(row[6]) if row[6] else None
                ))
            
            return anomalies

# 使用示例
async def main():
    """主函数示例"""
    alert_system = AlertSystem()
    
    # 添加告警回调
    async def alert_callback(alert: Alert):
        print(f"告警: {alert.message} (严重级别: {alert.severity.value})")
    
    alert_system.add_alert_callback(alert_callback)
    
    try:
        await alert_system.start_monitoring()
        
        # 运行一段时间
        await asyncio.sleep(60)
        
    finally:
        await alert_system.stop_monitoring()

if __name__ == "__main__":
    asyncio.run(main())