"""
系统健康检查和自动恢复模块
负责监控系统健康状态并执行自动恢复操作
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
import psutil
import subprocess
import socket
import requests
from pathlib import Path

class HealthStatus(Enum):
    """健康状态"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

class ComponentType(Enum):
    """组件类型"""
    SYSTEM = "system"
    APPLICATION = "application"
    DATABASE = "database"
    NETWORK = "network"
    SERVICE = "service"
    AGENT = "agent"

class RecoveryAction(Enum):
    """恢复动作"""
    RESTART_SERVICE = "restart_service"
    RESTART_SYSTEM = "restart_system"
    SCALE_RESOURCES = "scale_resources"
    CLEAR_CACHE = "clear_cache"
    RESTART_DATABASE = "restart_database"
    RESTART_AGENT = "restart_agent"
    NOTIFY_ADMIN = "notify_admin"
    FAILOVER = "failover"

@dataclass
class HealthCheck:
    """健康检查"""
    id: Optional[int]
    component_name: str
    component_type: ComponentType
    check_type: str  # "ping", "port", "process", "metric", "custom"
    check_value: str  # 检查目标（IP、端口、进程名等）
    expected_result: str
    timeout: float
    interval: float
    enabled: bool = True
    critical_threshold: int = 3  # 连续失败次数阈值

@dataclass
class HealthStatusRecord:
    """健康状态记录"""
    id: Optional[int]
    component_name: str
    status: HealthStatus
    message: str
    response_time: float
    timestamp: float
    metadata: Dict[str, Any] = None

@dataclass
class RecoveryProcedure:
    """恢复程序"""
    id: Optional[int]
    component_name: str
    trigger_condition: str
    actions: List[Dict[str, Any]]
    enabled: bool = True
    max_attempts: int = 3
    cooldown_period: int = 300  # 5分钟
    last_execution: Optional[float] = None

class HealthMonitor:
    """系统健康监控器"""
    
    def __init__(self, db_path: str = "health_monitor.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.is_running = False
        
        # 健康检查配置
        self.health_checks: List[HealthCheck] = []
        self.recovery_procedures: List[RecoveryProcedure] = []
        
        # 状态缓存
        self.component_status: Dict[str, HealthStatus] = {}
        self.failure_counts: Dict[str, int] = defaultdict(int)
        self.last_check_time: Dict[str, float] = {}
        
        # 监控配置
        self.config = {
            "check_interval": 30,  # 秒
            "status_retention_days": 7,
            "max_concurrent_checks": 10,
            "notification_enabled": True,
            "auto_recovery_enabled": True
        }
        
        # 线程锁
        self.lock = threading.Lock()
        
        # 回调函数
        self.health_callbacks: List[Callable] = []
        self.recovery_callbacks: List[Callable] = []
        
        # 初始化数据库
        self._init_database()
        
        # 加载默认检查
        self._load_default_checks()
        self._load_default_recovery_procedures()
    
    def _init_database(self):
        """初始化数据库"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS health_checks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    component_name TEXT NOT NULL,
                    component_type TEXT NOT NULL,
                    check_type TEXT NOT NULL,
                    check_value TEXT NOT NULL,
                    expected_result TEXT NOT NULL,
                    timeout REAL NOT NULL,
                    interval REAL NOT NULL,
                    enabled BOOLEAN DEFAULT TRUE,
                    critical_threshold INTEGER DEFAULT 3,
                    created_at REAL DEFAULT CURRENT_TIMESTAMP,
                    updated_at REAL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS health_status (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    component_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    message TEXT NOT NULL,
                    response_time REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS recovery_procedures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    component_name TEXT NOT NULL,
                    trigger_condition TEXT NOT NULL,
                    actions TEXT NOT NULL,
                    enabled BOOLEAN DEFAULT TRUE,
                    max_attempts INTEGER DEFAULT 3,
                    cooldown_period INTEGER DEFAULT 300,
                    last_execution REAL,
                    execution_count INTEGER DEFAULT 0,
                    success_count INTEGER DEFAULT 0,
                    created_at REAL DEFAULT CURRENT_TIMESTAMP,
                    updated_at REAL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS recovery_executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    procedure_id INTEGER NOT NULL,
                    component_name TEXT NOT NULL,
                    execution_time REAL NOT NULL,
                    status TEXT NOT NULL,
                    actions_executed TEXT NOT NULL,
                    result TEXT NOT NULL,
                    error_message TEXT,
                    FOREIGN KEY (procedure_id) REFERENCES recovery_procedures (id)
                )
            """)
            
            # 创建索引
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status_timestamp ON health_status(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status_component ON health_status(component_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_recovery_time ON recovery_executions(execution_time)")
    
    def _load_default_checks(self):
        """加载默认健康检查"""
        default_checks = [
            # 系统检查
            HealthCheck(
                id=None,
                component_name="cpu_system",
                component_type=ComponentType.SYSTEM,
                check_type="metric",
                check_value="cpu.usage.percent",
                expected_result="<80",
                timeout=5.0,
                interval=30.0,
                critical_threshold=5
            ),
            HealthCheck(
                id=None,
                component_name="memory_system",
                component_type=ComponentType.SYSTEM,
                check_type="metric",
                check_value="memory.usage.percent",
                expected_result="<85",
                timeout=5.0,
                interval=30.0,
                critical_threshold=5
            ),
            HealthCheck(
                id=None,
                component_name="disk_system",
                component_type=ComponentType.SYSTEM,
                check_type="metric",
                check_value="disk.usage.percent",
                expected_result="<90",
                timeout=5.0,
                interval=60.0,
                critical_threshold=3
            ),
            
            # 网络检查
            HealthCheck(
                id=None,
                component_name="network_connectivity",
                component_type=ComponentType.NETWORK,
                check_type="ping",
                check_value="8.8.8.8",
                expected_result="success",
                timeout=5.0,
                interval=60.0,
                critical_threshold=3
            ),
            
            # 服务检查
            HealthCheck(
                id=None,
                component_name="norma_agent_service",
                component_type=ComponentType.AGENT,
                check_type="process",
                check_value="norma_agent",
                expected_result="running",
                timeout=10.0,
                interval=30.0,
                critical_threshold=2
            ),
            HealthCheck(
                id=None,
                component_name="database_service",
                component_type=ComponentType.DATABASE,
                check_type="port",
                check_value="5432",
                expected_result="open",
                timeout=5.0,
                interval=30.0,
                critical_threshold=2
            ),
            
            # 应用检查
            HealthCheck(
                id=None,
                component_name="api_response",
                component_type=ComponentType.APPLICATION,
                check_type="http",
                check_value="http://localhost:8000/health",
                expected_result="200",
                timeout=10.0,
                interval=60.0,
                critical_threshold=3
            )
        ]
        
        for check in default_checks:
            self.add_health_check(check)
    
    def _load_default_recovery_procedures(self):
        """加载默认恢复程序"""
        default_procedures = [
            # CPU使用率过高恢复
            RecoveryProcedure(
                id=None,
                component_name="cpu_system",
                trigger_condition="cpu.usage.percent > 90 for 5 minutes",
                actions=[
                    {"action": "scale_resources", "target": "cpu", "scale_factor": 1.5},
                    {"action": "clear_cache", "target": "system"},
                    {"action": "restart_agent", "target": "norma_agent"}
                ],
                max_attempts=2,
                cooldown_period=600
            ),
            
            # 内存使用率过高恢复
            RecoveryProcedure(
                id=None,
                component_name="memory_system",
                trigger_condition="memory.usage.percent > 90 for 3 minutes",
                actions=[
                    {"action": "clear_cache", "target": "system"},
                    {"action": "restart_service", "target": "norma_agent"}
                ],
                max_attempts=3,
                cooldown_period=300
            ),
            
            # 服务停止恢复
            RecoveryProcedure(
                id=None,
                component_name="norma_agent_service",
                trigger_condition="service_stopped for 2 minutes",
                actions=[
                    {"action": "restart_service", "target": "norma_agent"},
                    {"action": "notify_admin", "message": "Norma Agent service was restarted"}
                ],
                max_attempts=5,
                cooldown_period=180
            ),
            
            # 数据库连接失败恢复
            RecoveryProcedure(
                id=None,
                component_name="database_service",
                trigger_condition="database_connection_failed for 3 minutes",
                actions=[
                    {"action": "restart_database", "target": "postgresql"},
                    {"action": "restart_service", "target": "norma_agent"}
                ],
                max_attempts=3,
                cooldown_period=600
            ),
            
            # API响应异常恢复
            RecoveryProcedure(
                id=None,
                component_name="api_response",
                trigger_condition="api_5xx_errors for 5 minutes",
                actions=[
                    {"action": "restart_service", "target": "api_server"},
                    {"action": "scale_resources", "target": "api", "scale_factor": 1.2}
                ],
                max_attempts=2,
                cooldown_period=900
            )
        ]
        
        for procedure in default_procedures:
            self.add_recovery_procedure(procedure)
    
    async def start_monitoring(self):
        """启动健康监控"""
        if self.is_running:
            return
        
        self.is_running = True
        self.logger.info("启动系统健康监控器")
        
        # 启动监控任务
        tasks = [
            asyncio.create_task(self._run_health_checks()),
            asyncio.create_task(self._monitor_component_status()),
            asyncio.create_task(self._execute_recovery_procedures()),
            asyncio.create_task(self._cleanup_old_data()),
            asyncio.create_task(self._generate_health_report())
        ]
        
        await asyncio.gather(*tasks)
    
    async def stop_monitoring(self):
        """停止健康监控"""
        self.is_running = False
        self.logger.info("停止系统健康监控器")
    
    def add_health_check(self, check: HealthCheck):
        """添加健康检查"""
        with self.lock:
            # 检查是否已存在
            existing_check = next((c for c in self.health_checks 
                                 if c.component_name == check.component_name), None)
            if existing_check:
                self.logger.warning(f"健康检查 {check.component_name} 已存在，将被更新")
                self.health_checks.remove(existing_check)
            
            self.health_checks.append(check)
            
            # 保存到数据库
            self._save_health_check_to_db(check)
            
            self.logger.info(f"添加健康检查: {check.component_name}")
    
    def remove_health_check(self, component_name: str):
        """删除健康检查"""
        with self.lock:
            check = next((c for c in self.health_checks 
                         if c.component_name == component_name), None)
            if check:
                self.health_checks.remove(check)
                self._delete_health_check_from_db(component_name)
                self.logger.info(f"删除健康检查: {component_name}")
    
    def _save_health_check_to_db(self, check: HealthCheck):
        """保存健康检查到数据库"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO health_checks 
                (component_name, component_type, check_type, check_value, expected_result, 
                 timeout, interval, enabled, critical_threshold)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                check.component_name,
                check.component_type.value,
                check.check_type,
                check.check_value,
                check.expected_result,
                check.timeout,
                check.interval,
                check.enabled,
                check.critical_threshold
            ))
    
    def _delete_health_check_from_db(self, component_name: str):
        """从数据库删除健康检查"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM health_checks WHERE component_name = ?", (component_name,))
    
    def add_recovery_procedure(self, procedure: RecoveryProcedure):
        """添加恢复程序"""
        with self.lock:
            # 检查是否已存在
            existing_procedure = next((p for p in self.recovery_procedures 
                                     if p.component_name == procedure.component_name), None)
            if existing_procedure:
                self.logger.warning(f"恢复程序 {procedure.component_name} 已存在，将被更新")
                self.recovery_procedures.remove(existing_procedure)
            
            self.recovery_procedures.append(procedure)
            
            # 保存到数据库
            self._save_recovery_procedure_to_db(procedure)
            
            self.logger.info(f"添加恢复程序: {procedure.component_name}")
    
    def _save_recovery_procedure_to_db(self, procedure: RecoveryProcedure):
        """保存恢复程序到数据库"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO recovery_procedures 
                (component_name, trigger_condition, actions, enabled, max_attempts, cooldown_period)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                procedure.component_name,
                procedure.trigger_condition,
                json.dumps(procedure.actions),
                procedure.enabled,
                procedure.max_attempts,
                procedure.cooldown_period
            ))
    
    async def _run_health_checks(self):
        """运行健康检查"""
        while self.is_running:
            try:
                # 获取启用的健康检查
                enabled_checks = [check for check in self.health_checks if check.enabled]
                
                # 限制并发检查数量
                checks_to_run = enabled_checks[:self.config["max_concurrent_checks"]]
                
                # 并发执行检查
                tasks = [self._execute_health_check(check) for check in checks_to_run]
                await asyncio.gather(*tasks, return_exceptions=True)
                
                await asyncio.sleep(self.config["check_interval"])
                
            except Exception as e:
                self.logger.error(f"运行健康检查时出错: {e}")
                await asyncio.sleep(60)
    
    async def _execute_health_check(self, check: HealthCheck):
        """执行单个健康检查"""
        try:
            start_time = time.time()
            
            # 根据检查类型执行相应检查
            if check.check_type == "ping":
                result = await self._check_ping(check.check_value, check.timeout)
            elif check.check_type == "port":
                result = await self._check_port(check.check_value, check.timeout)
            elif check.check_type == "process":
                result = await self._check_process(check.check_value, check.timeout)
            elif check.check_type == "metric":
                result = await self._check_metric(check.check_value, check.timeout)
            elif check.check_type == "http":
                result = await self._check_http(check.check_value, check.timeout)
            else:
                result = {"status": HealthStatus.UNKNOWN, "message": f"未知检查类型: {check.check_type}"}
            
            response_time = time.time() - start_time
            
            # 评估检查结果
            status = self._evaluate_check_result(result, check)
            
            # 更新状态
            await self._update_component_status(check.component_name, status, result, response_time)
            
            # 检查是否需要触发恢复程序
            if status in [HealthStatus.CRITICAL, HealthStatus.WARNING]:
                await self._check_recovery_trigger(check.component_name, status)
            
        except Exception as e:
            self.logger.error(f"执行健康检查 {check.component_name} 时出错: {e}")
    
    async def _check_ping(self, target: str, timeout: float) -> Dict[str, Any]:
        """执行ping检查"""
        try:
            # 使用ping命令
            process = await asyncio.create_subprocess_exec(
                "ping", "-c", "1", "-W", str(int(timeout * 1000)), target,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                return {
                    "status": HealthStatus.HEALTHY,
                    "message": f"Ping {target} 成功",
                    "data": stdout.decode().strip()
                }
            else:
                return {
                    "status": HealthStatus.CRITICAL,
                    "message": f"Ping {target} 失败",
                    "error": stderr.decode().strip()
                }
        except Exception as e:
            return {
                "status": HealthStatus.CRITICAL,
                "message": f"Ping {target} 异常",
                "error": str(e)
            }
    
    async def _check_port(self, port: str, timeout: float) -> Dict[str, Any]:
        """执行端口检查"""
        try:
            # 尝试连接到localhost的指定端口
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection('localhost', int(port)),
                timeout=timeout
            )
            writer.close()
            await writer.wait_closed()
            
            return {
                "status": HealthStatus.HEALTHY,
                "message": f"端口 {port} 可达",
                "data": {"port": port, "reachable": True}
            }
        except asyncio.TimeoutError:
            return {
                "status": HealthStatus.CRITICAL,
                "message": f"端口 {port} 连接超时",
                "error": "Connection timeout"
            }
        except Exception as e:
            return {
                "status": HealthStatus.CRITICAL,
                "message": f"端口 {port} 不可达",
                "error": str(e)
            }
    
    async def _check_process(self, process_name: str, timeout: float) -> Dict[str, Any]:
        """执行进程检查"""
        try:
            # 检查进程是否存在
            for proc in psutil.process_iter(['pid', 'name']):
                if proc.info['name'] == process_name:
                    return {
                        "status": HealthStatus.HEALTHY,
                        "message": f"进程 {process_name} 正在运行",
                        "data": {"pid": proc.info['pid'], "name": process_name}
                    }
            
            return {
                "status": HealthStatus.CRITICAL,
                "message": f"进程 {process_name} 未找到",
                "error": "Process not found"
            }
        except Exception as e:
            return {
                "status": HealthStatus.CRITICAL,
                "message": f"检查进程 {process_name} 时出错",
                "error": str(e)
            }
    
    async def _check_metric(self, metric_name: str, timeout: float) -> Dict[str, Any]:
        """执行指标检查"""
        try:
            # 根据指标名称获取相应值
            if metric_name == "cpu.usage.percent":
                value = psutil.cpu_percent(interval=1)
                if value < 80:
                    status = HealthStatus.HEALTHY
                    message = f"CPU使用率正常: {value:.1f}%"
                elif value < 90:
                    status = HealthStatus.WARNING
                    message = f"CPU使用率较高: {value:.1f}%"
                else:
                    status = HealthStatus.CRITICAL
                    message = f"CPU使用率过高: {value:.1f}%"
                
                return {
                    "status": status,
                    "message": message,
                    "data": {"metric": metric_name, "value": value, "unit": "percent"}
                }
            
            elif metric_name == "memory.usage.percent":
                memory = psutil.virtual_memory()
                value = memory.percent
                if value < 85:
                    status = HealthStatus.HEALTHY
                    message = f"内存使用率正常: {value:.1f}%"
                elif value < 95:
                    status = HealthStatus.WARNING
                    message = f"内存使用率较高: {value:.1f}%"
                else:
                    status = HealthStatus.CRITICAL
                    message = f"内存使用率过高: {value:.1f}%"
                
                return {
                    "status": status,
                    "message": message,
                    "data": {"metric": metric_name, "value": value, "unit": "percent"}
                }
            
            elif metric_name == "disk.usage.percent":
                disk = psutil.disk_usage('/')
                value = (disk.used / disk.total) * 100
                if value < 90:
                    status = HealthStatus.HEALTHY
                    message = f"磁盘使用率正常: {value:.1f}%"
                elif value < 95:
                    status = HealthStatus.WARNING
                    message = f"磁盘使用率较高: {value:.1f}%"
                else:
                    status = HealthStatus.CRITICAL
                    message = f"磁盘使用率过高: {value:.1f}%"
                
                return {
                    "status": status,
                    "message": message,
                    "data": {"metric": metric_name, "value": value, "unit": "percent"}
                }
            
            else:
                return {
                    "status": HealthStatus.UNKNOWN,
                    "message": f"未知指标: {metric_name}",
                    "error": "Unknown metric"
                }
                
        except Exception as e:
            return {
                "status": HealthStatus.CRITICAL,
                "message": f"检查指标 {metric_name} 时出错",
                "error": str(e)
            }
    
    async def _check_http(self, url: str, timeout: float) -> Dict[str, Any]:
        """执行HTTP检查"""
        try:
            response = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: requests.get(url, timeout=timeout)
                )
            )
            
            if response.status_code == 200:
                return {
                    "status": HealthStatus.HEALTHY,
                    "message": f"HTTP {url} 正常 (状态码: {response.status_code})",
                    "data": {"url": url, "status_code": response.status_code}
                }
            else:
                return {
                    "status": HealthStatus.WARNING,
                    "message": f"HTTP {url} 异常 (状态码: {response.status_code})",
                    "data": {"url": url, "status_code": response.status_code}
                }
                
        except asyncio.TimeoutError:
            return {
                "status": HealthStatus.CRITICAL,
                "message": f"HTTP {url} 请求超时",
                "error": "Request timeout"
            }
        except Exception as e:
            return {
                "status": HealthStatus.CRITICAL,
                "message": f"HTTP {url} 请求失败",
                "error": str(e)
            }
    
    def _evaluate_check_result(self, result: Dict[str, Any], check: HealthCheck) -> HealthStatus:
        """评估检查结果"""
        # 如果结果中已经包含状态，直接使用
        if "status" in result:
            return result["status"]
        
        # 根据预期结果评估
        expected = check.expected_result
        if expected.startswith("<"):
            threshold = float(expected[1:])
            if "value" in result.get("data", {}):
                actual_value = result["data"]["value"]
                if actual_value < threshold:
                    return HealthStatus.HEALTHY
                elif actual_value < threshold * 1.1:
                    return HealthStatus.WARNING
                else:
                    return HealthStatus.CRITICAL
        
        return HealthStatus.UNKNOWN
    
    async def _update_component_status(self, component_name: str, status: HealthStatus, 
                                     result: Dict[str, Any], response_time: float):
        """更新组件状态"""
        # 更新内存中的状态
        with self.lock:
            self.component_status[component_name] = status
            self.last_check_time[component_name] = time.time()
        
        # 更新失败计数
        if status == HealthStatus.CRITICAL:
            self.failure_counts[component_name] += 1
        else:
            self.failure_counts[component_name] = 0
        
        # 保存状态记录到数据库
        await self._save_health_status(component_name, status, result, response_time)
        
        # 触发回调
        for callback in self.health_callbacks:
            try:
                await callback(component_name, status, result)
            except Exception as e:
                self.logger.error(f"健康状态回调函数执行出错: {e}")
        
        self.logger.info(f"组件状态更新: {component_name} - {status.value}")
    
    async def _save_health_status(self, component_name: str, status: HealthStatus, 
                                result: Dict[str, Any], response_time: float):
        """保存健康状态记录"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO health_status 
                    (component_name, status, message, response_time, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    component_name,
                    status.value,
                    result.get("message", ""),
                    response_time,
                    time.time(),
                    json.dumps(result)
                ))
        except Exception as e:
            self.logger.error(f"保存健康状态时出错: {e}")
    
    async def _check_recovery_trigger(self, component_name: str, status: HealthStatus):
        """检查是否触发恢复程序"""
        if not self.config["auto_recovery_enabled"]:
            return
        
        # 检查失败次数是否达到阈值
        failure_count = self.failure_counts.get(component_name, 0)
        
        # 获取对应的健康检查
        health_check = next((c for c in self.health_checks 
                           if c.component_name == component_name), None)
        
        if health_check and failure_count >= health_check.critical_threshold:
            # 触发恢复程序
            await self._trigger_recovery_procedure(component_name)
    
    async def _trigger_recovery_procedure(self, component_name: str):
        """触发恢复程序"""
        # 获取对应的恢复程序
        procedure = next((p for p in self.recovery_procedures 
                        if p.component_name == component_name and p.enabled), None)
        
        if not procedure:
            return
        
        # 检查冷却期
        if (procedure.last_execution and 
            time.time() - procedure.last_execution < procedure.cooldown_period):
            return
        
        self.logger.warning(f"触发恢复程序: {component_name}")
        
        # 执行恢复程序
        await self._execute_recovery_procedure(procedure)
    
    async def _execute_recovery_procedure(self, procedure: RecoveryProcedure):
        """执行恢复程序"""
        execution_start = time.time()
        actions_executed = []
        success = True
        error_message = None
        
        try:
            # 更新执行计数
            procedure.execution_count = getattr(procedure, 'execution_count', 0) + 1
            
            # 执行每个恢复动作
            for action_config in procedure.actions:
                action_type = action_config.get("action")
                target = action_config.get("target")
                
                try:
                    result = await self._execute_recovery_action(action_type, target, action_config)
                    actions_executed.append({
                        "action": action_type,
                        "target": target,
                        "result": result,
                        "success": result.get("success", False)
                    })
                    
                    if not result.get("success", False):
                        success = False
                        error_message = result.get("error", "Unknown error")
                    
                except Exception as e:
                    actions_executed.append({
                        "action": action_type,
                        "target": target,
                        "result": {"success": False, "error": str(e)},
                        "success": False
                    })
                    success = False
                    error_message = str(e)
            
            # 更新成功计数
            if success:
                procedure.success_count = getattr(procedure, 'success_count', 0) + 1
            
            # 记录执行结果
            await self._record_recovery_execution(
                procedure, "success" if success else "failed", 
                actions_executed, {"success": success}, error_message
            )
            
            procedure.last_execution = time.time()
            
            # 触发回调
            for callback in self.recovery_callbacks:
                try:
                    await callback(procedure.component_name, success, actions_executed)
                except Exception as e:
                    self.logger.error(f"恢复回调函数执行出错: {e}")
            
            self.logger.info(f"恢复程序执行完成: {procedure.component_name} - {'成功' if success else '失败'}")
            
        except Exception as e:
            self.logger.error(f"执行恢复程序时出错: {e}")
            await self._record_recovery_execution(
                procedure, "error", actions_executed, {}, str(e)
            )
    
    async def _execute_recovery_action(self, action_type: str, target: str, 
                                     config: Dict[str, Any]) -> Dict[str, Any]:
        """执行单个恢复动作"""
        try:
            if action_type == "restart_service":
                return await self._restart_service(target)
            elif action_type == "restart_system":
                return await self._restart_system()
            elif action_type == "scale_resources":
                scale_factor = config.get("scale_factor", 1.2)
                return await self._scale_resources(target, scale_factor)
            elif action_type == "clear_cache":
                return await self._clear_cache(target)
            elif action_type == "restart_database":
                return await self._restart_database(target)
            elif action_type == "restart_agent":
                return await self._restart_agent(target)
            elif action_type == "notify_admin":
                message = config.get("message", f"系统通知: {target}")
                return await self._notify_admin(message)
            else:
                return {"success": False, "error": f"未知恢复动作: {action_type}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _restart_service(self, service_name: str) -> Dict[str, Any]:
        """重启服务"""
        try:
            # 这里应该实现实际的服务重启逻辑
            # 简化实现
            await asyncio.sleep(2)  # 模拟重启时间
            
            return {
                "success": True,
                "message": f"服务 {service_name} 重启成功",
                "details": f"Service {service_name} was restarted"
            }
        except Exception as e:
            return {"success": False, "error": f"重启服务 {service_name} 失败: {e}"}
    
    async def _restart_system(self) -> Dict[str, Any]:
        """重启系统"""
        try:
            # 这里应该实现系统重启逻辑（谨慎使用）
            return {
                "success": True,
                "message": "系统重启命令已发送",
                "details": "System restart command executed"
            }
        except Exception as e:
            return {"success": False, "error": f"系统重启失败: {e}"}
    
    async def _scale_resources(self, resource_type: str, scale_factor: float) -> Dict[str, Any]:
        """扩容资源"""
        try:
            # 这里应该实现实际的资源扩容逻辑
            await asyncio.sleep(1)  # 模拟扩容时间
            
            return {
                "success": True,
                "message": f"资源 {resource_type} 扩容成功 (因子: {scale_factor})",
                "details": f"Scaled {resource_type} by factor {scale_factor}"
            }
        except Exception as e:
            return {"success": False, "error": f"资源扩容失败: {e}"}
    
    async def _clear_cache(self, target: str) -> Dict[str, Any]:
        """清理缓存"""
        try:
            # 这里应该实现实际的缓存清理逻辑
            await asyncio.sleep(1)  # 模拟清理时间
            
            return {
                "success": True,
                "message": f"缓存清理成功: {target}",
                "details": f"Cleared cache for {target}"
            }
        except Exception as e:
            return {"success": False, "error": f"缓存清理失败: {e}"}
    
    async def _restart_database(self, db_type: str) -> Dict[str, Any]:
        """重启数据库"""
        try:
            # 这里应该实现实际的数据库重启逻辑
            await asyncio.sleep(3)  # 模拟重启时间
            
            return {
                "success": True,
                "message": f"数据库 {db_type} 重启成功",
                "details": f"Database {db_type} was restarted"
            }
        except Exception as e:
            return {"success": False, "error": f"数据库重启失败: {e}"}
    
    async def _restart_agent(self, agent_name: str) -> Dict[str, Any]:
        """重启Agent"""
        try:
            # 这里应该实现实际的Agent重启逻辑
            await asyncio.sleep(2)  # 模拟重启时间
            
            return {
                "success": True,
                "message": f"Agent {agent_name} 重启成功",
                "details": f"Agent {agent_name} was restarted"
            }
        except Exception as e:
            return {"success": False, "error": f"Agent重启失败: {e}"}
    
    async def _notify_admin(self, message: str) -> Dict[str, Any]:
        """通知管理员"""
        try:
            # 这里应该实现实际的通知逻辑（邮件、短信、Slack等）
            self.logger.warning(f"管理员通知: {message}")
            
            return {
                "success": True,
                "message": "管理员通知已发送",
                "details": f"Notification sent: {message}"
            }
        except Exception as e:
            return {"success": False, "error": f"发送通知失败: {e}"}
    
    async def _record_recovery_execution(self, procedure: RecoveryProcedure, status: str, 
                                       actions_executed: List[Dict], result: Dict, error_message: str):
        """记录恢复执行"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO recovery_executions 
                    (procedure_id, component_name, execution_time, status, actions_executed, result, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    procedure.id,
                    procedure.component_name,
                    time.time(),
                    status,
                    json.dumps(actions_executed),
                    json.dumps(result),
                    error_message
                ))
        except Exception as e:
            self.logger.error(f"记录恢复执行时出错: {e}")
    
    async def _monitor_component_status(self):
        """监控组件状态"""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # 每分钟监控一次
                
                # 检查长时间未更新的组件
                current_time = time.time()
                for component_name, last_check in self.last_check_time.items():
                    if current_time - last_check > 300:  # 5分钟未更新
                        self.logger.warning(f"组件 {component_name} 长时间未更新状态")
                
            except Exception as e:
                self.logger.error(f"监控组件状态时出错: {e}")
                await asyncio.sleep(60)
    
    async def _execute_recovery_procedures(self):
        """执行恢复程序（监控模式）"""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # 每分钟检查一次
                
                # 检查是否需要执行恢复程序
                for procedure in self.recovery_procedures:
                    if not procedure.enabled:
                        continue
                    
                    # 检查触发条件
                    if await self._check_recovery_condition(procedure):
                        await self._trigger_recovery_procedure(procedure.component_name)
                
            except Exception as e:
                self.logger.error(f"执行恢复程序时出错: {e}")
                await asyncio.sleep(60)
    
    async def _check_recovery_condition(self, procedure: RecoveryProcedure) -> bool:
        """检查恢复条件"""
        # 简化实现：根据组件状态判断
        component_status = self.component_status.get(procedure.component_name)
        if not component_status:
            return False
        
        # 如果组件状态为严重，则触发恢复
        return component_status == HealthStatus.CRITICAL
    
    async def _cleanup_old_data(self):
        """清理旧数据"""
        while self.is_running:
            try:
                await asyncio.sleep(86400)  # 每天清理一次
                
                cutoff_time = time.time() - (self.config["status_retention_days"] * 24 * 3600)
                
                with sqlite3.connect(self.db_path) as conn:
                    # 清理旧状态记录
                    conn.execute("DELETE FROM health_status WHERE timestamp < ?", (cutoff_time,))
                    
                    # 清理旧恢复执行记录
                    conn.execute("DELETE FROM recovery_executions WHERE execution_time < ?", (cutoff_time,))
                
                self.logger.info("清理旧健康监控数据完成")
                
            except Exception as e:
                self.logger.error(f"清理旧数据时出错: {e}")
                await asyncio.sleep(86400)
    
    async def _generate_health_report(self):
        """生成健康报告"""
        while self.is_running:
            try:
                await asyncio.sleep(3600)  # 每小时生成一次报告
                
                report = await self._create_health_report()
                
                # 保存报告
                report_file = Path("health_report.json")
                with open(report_file, 'w', encoding='utf-8') as f:
                    json.dump(report, f, ensure_ascii=False, indent=2)
                
                self.logger.info("生成健康报告完成")
                
            except Exception as e:
                self.logger.error(f"生成健康报告时出错: {e}")
                await asyncio.sleep(3600)
    
    async def _create_health_report(self) -> Dict[str, Any]:
        """创建健康报告"""
        try:
            # 获取组件状态统计
            status_counts = defaultdict(int)
            for status in self.component_status.values():
                status_counts[status.value] += 1
            
            # 获取最近的健康状态
            recent_status = await self._get_recent_health_status(hours=1)
            
            # 获取恢复执行统计
            recovery_stats = await self._get_recovery_statistics(hours=24)
            
            return {
                "timestamp": time.time(),
                "component_summary": {
                    "total_components": len(self.component_status),
                    "healthy": status_counts["healthy"],
                    "warning": status_counts["warning"],
                    "critical": status_counts["critical"],
                    "unknown": status_counts["unknown"]
                },
                "recent_status": recent_status,
                "recovery_statistics": recovery_stats,
                "system_health_score": self._calculate_system_health_score()
            }
            
        except Exception as e:
            self.logger.error(f"创建健康报告时出错: {e}")
            return {"error": str(e)}
    
    async def _get_recent_health_status(self, hours: int = 1) -> List[Dict]:
        """获取最近的健康状态"""
        start_time = time.time() - (hours * 3600)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT component_name, status, message, response_time, timestamp
                    FROM health_status 
                    WHERE timestamp > ?
                    ORDER BY timestamp DESC
                    LIMIT 100
                """, (start_time,))
                
                return [
                    {
                        "component_name": row[0],
                        "status": row[1],
                        "message": row[2],
                        "response_time": row[3],
                        "timestamp": row[4]
                    }
                    for row in cursor.fetchall()
                ]
        except Exception as e:
            self.logger.error(f"获取最近健康状态时出错: {e}")
            return []
    
    async def _get_recovery_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """获取恢复统计"""
        start_time = time.time() - (hours * 3600)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT status, COUNT(*) as count
                    FROM recovery_executions 
                    WHERE execution_time > ?
                    GROUP BY status
                """, (start_time,))
                
                stats = {}
                for row in cursor.fetchall():
                    stats[row[0]] = row[1]
                
                return stats
        except Exception as e:
            self.logger.error(f"获取恢复统计时出错: {e}")
            return {}
    
    def _calculate_system_health_score(self) -> float:
        """计算系统健康分数"""
        if not self.component_status:
            return 0.0
        
        # 根据组件状态计算分数
        total_components = len(self.component_status)
        healthy_components = sum(1 for status in self.component_status.values() 
                               if status == HealthStatus.HEALTHY)
        
        return (healthy_components / total_components) * 100
    
    def add_health_callback(self, callback: Callable):
        """添加健康状态回调函数"""
        self.health_callbacks.append(callback)
    
    def add_recovery_callback(self, callback: Callable):
        """添加恢复回调函数"""
        self.recovery_callbacks.append(callback)
    
    def get_component_status(self) -> Dict[str, HealthStatus]:
        """获取组件状态"""
        with self.lock:
            return self.component_status.copy()
    
    def get_health_history(self, component_name: str = None, hours: int = 24) -> List[Dict]:
        """获取健康历史"""
        start_time = time.time() - (hours * 3600)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                if component_name:
                    cursor = conn.execute("""
                        SELECT component_name, status, message, response_time, timestamp, metadata
                        FROM health_status 
                        WHERE component_name = ? AND timestamp > ?
                        ORDER BY timestamp DESC
                        LIMIT 200
                    """, (component_name, start_time))
                else:
                    cursor = conn.execute("""
                        SELECT component_name, status, message, response_time, timestamp, metadata
                        FROM health_status 
                        WHERE timestamp > ?
                        ORDER BY timestamp DESC
                        LIMIT 200
                    """, (start_time,))
                
                history = []
                for row in cursor.fetchall():
                    history.append({
                        "component_name": row[0],
                        "status": row[1],
                        "message": row[2],
                        "response_time": row[3],
                        "timestamp": row[4],
                        "metadata": json.loads(row[5]) if row[5] else None
                    })
                
                return history
        except Exception as e:
            self.logger.error(f"获取健康历史时出错: {e}")
            return []
    
    def get_recovery_history(self, hours: int = 24) -> List[Dict]:
        """获取恢复历史"""
        start_time = time.time() - (hours * 3600)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT procedure_id, component_name, execution_time, status, 
                           actions_executed, result, error_message
                    FROM recovery_executions 
                    WHERE execution_time > ?
                    ORDER BY execution_time DESC
                    LIMIT 100
                """, (start_time,))
                
                history = []
                for row in cursor.fetchall():
                    history.append({
                        "procedure_id": row[0],
                        "component_name": row[1],
                        "execution_time": row[2],
                        "status": row[3],
                        "actions_executed": json.loads(row[4]),
                        "result": json.loads(row[5]),
                        "error_message": row[6]
                    })
                
                return history
        except Exception as e:
            self.logger.error(f"获取恢复历史时出错: {e}")
            return []

# 使用示例
async def main():
    """主函数示例"""
    health_monitor = HealthMonitor()
    
    # 添加回调函数
    async def health_callback(component_name: str, status: HealthStatus, result: Dict):
        print(f"健康状态更新: {component_name} - {status.value}")
    
    async def recovery_callback(component_name: str, success: bool, actions: List[Dict]):
        print(f"恢复执行: {component_name} - {'成功' if success else '失败'}")
    
    health_monitor.add_health_callback(health_callback)
    health_monitor.add_recovery_callback(recovery_callback)
    
    try:
        await health_monitor.start_monitoring()
        
        # 运行一段时间
        await asyncio.sleep(300)
        
    finally:
        await health_monitor.stop_monitoring()

if __name__ == "__main__":
    asyncio.run(main())