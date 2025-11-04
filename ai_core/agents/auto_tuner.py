"""
自动性能调优和资源管理系统
负责自动优化系统性能和资源分配
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
import numpy as np
from pathlib import Path

class TuningAction(Enum):
    """调优动作类型"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    RESTART_SERVICE = "restart_service"
    ADJUST_THRESHOLDS = "adjust_thresholds"
    OPTIMIZE_CONFIG = "optimize_config"
    CLEANUP_RESOURCES = "cleanup_resources"

class ResourceType(Enum):
    """资源类型"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    DATABASE = "database"
    AGENT_INSTANCES = "agent_instances"

class OptimizationTarget(Enum):
    """优化目标"""
    PERFORMANCE = "performance"
    COST = "cost"
    AVAILABILITY = "availability"
    USER_EXPERIENCE = "user_experience"

@dataclass
class TuningRule:
    """调优规则"""
    name: str
    resource_type: ResourceType
    metric_name: str
    condition: str
    threshold: float
    action: TuningAction
    target_value: float
    enabled: bool = True
    cooldown_period: int = 300  # 冷却期（秒）
    max_actions_per_hour: int = 5
    description: str = ""

@dataclass
class TuningActionRecord:
    """调优动作记录"""
    id: Optional[int]
    rule_name: str
    action_type: TuningAction
    resource_type: ResourceType
    target_value: float
    timestamp: float
    status: str = "pending"  # pending, executing, completed, failed
    result: Dict[str, Any] = None
    execution_time: float = 0
    error_message: str = None

@dataclass
class ResourceRecommendation:
    """资源推荐"""
    resource_type: ResourceType
    current_usage: float
    recommended_usage: float
    confidence: float
    reasoning: str
    actions: List[str]
    estimated_impact: Dict[str, float]

class AutoTuner:
    """自动性能调优器"""
    
    def __init__(self, db_path: str = "auto_tuner.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.is_running = False
        
        # 调优规则
        self.tuning_rules: List[TuningRule] = []
        self.action_history: deque = deque(maxlen=1000)
        self.last_action_time: Dict[str, float] = {}
        self.action_counts: Dict[str, int] = {}
        
        # 性能基线
        self.performance_baseline: Dict[str, float] = {}
        self.baseline_window = 7 * 24 * 3600  # 7天
        
        # 调优配置
        self.config = {
            "tuning_interval": 60,  # 秒
            "evaluation_window": 300,  # 5分钟
            "min_data_points": 50,
            "confidence_threshold": 0.7,
            "max_concurrent_actions": 3,
            "action_timeout": 300  # 5分钟
        }
        
        # 线程锁
        self.lock = threading.Lock()
        
        # 回调函数
        self.tuning_callbacks: List[Callable] = []
        
        # 初始化数据库
        self._init_database()
        
        # 加载默认规则
        self._load_default_rules()
    
    def _init_database(self):
        """初始化数据库"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tuning_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    resource_type TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    condition TEXT NOT NULL,
                    threshold REAL NOT NULL,
                    action TEXT NOT NULL,
                    target_value REAL NOT NULL,
                    enabled BOOLEAN DEFAULT TRUE,
                    cooldown_period INTEGER DEFAULT 300,
                    max_actions_per_hour INTEGER DEFAULT 5,
                    description TEXT,
                    created_at REAL DEFAULT CURRENT_TIMESTAMP,
                    updated_at REAL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tuning_actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule_name TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    resource_type TEXT NOT NULL,
                    target_value REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    status TEXT NOT NULL,
                    result TEXT,
                    execution_time REAL,
                    error_message TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_baselines (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    baseline_value REAL NOT NULL,
                    standard_deviation REAL NOT NULL,
                    sample_count INTEGER NOT NULL,
                    calculated_at REAL NOT NULL,
                    window_start REAL NOT NULL,
                    window_end REAL NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS resource_recommendations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    resource_type TEXT NOT NULL,
                    current_usage REAL NOT NULL,
                    recommended_usage REAL NOT NULL,
                    confidence REAL NOT NULL,
                    reasoning TEXT NOT NULL,
                    actions TEXT NOT NULL,
                    estimated_impact TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    applied BOOLEAN DEFAULT FALSE
                )
            """)
            
            # 创建索引
            conn.execute("CREATE INDEX IF NOT EXISTS idx_actions_timestamp ON tuning_actions(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_baselines_metric ON performance_baselines(metric_name)")
    
    def _load_default_rules(self):
        """加载默认调优规则"""
        default_rules = [
            TuningRule(
                name="high_cpu_scale_up",
                resource_type=ResourceType.CPU,
                metric_name="cpu.usage.percent",
                condition="greater_than",
                threshold=80.0,
                action=TuningAction.SCALE_UP,
                target_value=70.0,
                cooldown_period=600,
                description="CPU使用率过高时增加资源"
            ),
            TuningRule(
                name="low_cpu_scale_down",
                resource_type=ResourceType.CPU,
                metric_name="cpu.usage.percent",
                condition="less_than",
                threshold=30.0,
                action=TuningAction.SCALE_DOWN,
                target_value=50.0,
                cooldown_period=1800,
                description="CPU使用率过低时减少资源"
            ),
            TuningRule(
                name="high_memory_optimize",
                resource_type=ResourceType.MEMORY,
                metric_name="memory.usage.percent",
                condition="greater_than",
                threshold=85.0,
                action=TuningAction.CLEANUP_RESOURCES,
                target_value=75.0,
                cooldown_period=300,
                description="内存使用率过高时清理资源"
            ),
            TuningRule(
                name="slow_response_optimize",
                resource_type=ResourceType.AGENT_INSTANCES,
                metric_name="app.response_time.avg",
                condition="greater_than",
                threshold=2.0,
                action=TuningAction.SCALE_UP,
                target_value=1.0,
                cooldown_period=900,
                description="响应时间过慢时增加Agent实例"
            ),
            TuningRule(
                name="low_accuracy_retrain",
                resource_type=ResourceType.AGENT_INSTANCES,
                metric_name="agent.accuracy.percent",
                condition="less_than",
                threshold=85.0,
                action=TuningAction.OPTIMIZE_CONFIG,
                target_value=90.0,
                cooldown_period=3600,
                description="Agent准确率过低时优化配置"
            )
        ]
        
        for rule in default_rules:
            self.add_tuning_rule(rule)
    
    async def start_tuning(self):
        """启动自动调优"""
        if self.is_running:
            return
        
        self.is_running = True
        self.logger.info("启动自动性能调优器")
        
        # 启动调优任务
        tasks = [
            asyncio.create_task(self._monitor_performance()),
            asyncio.create_task(self._evaluate_tuning_rules()),
            asyncio.create_task(self._execute_tuning_actions()),
            asyncio.create_task(self._update_performance_baselines()),
            asyncio.create_task(self._generate_recommendations()),
            asyncio.create_task(self._cleanup_old_data())
        ]
        
        await asyncio.gather(*tasks)
    
    async def stop_tuning(self):
        """停止自动调优"""
        self.is_running = False
        self.logger.info("停止自动性能调优器")
    
    def add_tuning_rule(self, rule: TuningRule):
        """添加调优规则"""
        with self.lock:
            # 检查规则是否已存在
            existing_rule = next((r for r in self.tuning_rules if r.name == rule.name), None)
            if existing_rule:
                self.logger.warning(f"调优规则 {rule.name} 已存在，将被更新")
                self.tuning_rules.remove(existing_rule)
            
            self.tuning_rules.append(rule)
            
            # 保存到数据库
            self._save_tuning_rule_to_db(rule)
            
            self.logger.info(f"添加调优规则: {rule.name}")
    
    def remove_tuning_rule(self, rule_name: str):
        """删除调优规则"""
        with self.lock:
            rule = next((r for r in self.tuning_rules if r.name == rule_name), None)
            if rule:
                self.tuning_rules.remove(rule)
                self._delete_tuning_rule_from_db(rule_name)
                self.logger.info(f"删除调优规则: {rule_name}")
    
    def _save_tuning_rule_to_db(self, rule: TuningRule):
        """保存调优规则到数据库"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO tuning_rules 
                (name, resource_type, metric_name, condition, threshold, action, 
                 target_value, enabled, cooldown_period, max_actions_per_hour, description)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                rule.name,
                rule.resource_type.value,
                rule.metric_name,
                rule.condition,
                rule.threshold,
                rule.action.value,
                rule.target_value,
                rule.enabled,
                rule.cooldown_period,
                rule.max_actions_per_hour,
                rule.description
            ))
    
    def _delete_tuning_rule_from_db(self, rule_name: str):
        """从数据库删除调优规则"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM tuning_rules WHERE name = ?", (rule_name,))
    
    async def _monitor_performance(self):
        """监控性能指标"""
        while self.is_running:
            try:
                # 获取关键性能指标
                await self._collect_performance_metrics()
                
                await asyncio.sleep(self.config["tuning_interval"])
                
            except Exception as e:
                self.logger.error(f"监控性能时出错: {e}")
                await asyncio.sleep(60)
    
    async def _collect_performance_metrics(self):
        """收集性能指标"""
        try:
            timestamp = time.time()
            
            # CPU指标
            cpu_percent = psutil.cpu_percent(interval=0.1)
            await self._record_performance_metric("cpu.usage.percent", cpu_percent, timestamp)
            
            # 内存指标
            memory = psutil.virtual_memory()
            await self._record_performance_metric("memory.usage.percent", memory.percent, timestamp)
            
            # 磁盘指标
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            await self._record_performance_metric("disk.usage.percent", disk_percent, timestamp)
            
            # 网络指标
            network = psutil.net_io_counters()
            await self._record_performance_metric("network.bytes_total", 
                                                 network.bytes_sent + network.bytes_recv, timestamp)
            
        except Exception as e:
            self.logger.error(f"收集性能指标时出错: {e}")
    
    async def _record_performance_metric(self, name: str, value: float, timestamp: float):
        """记录性能指标"""
        try:
            # 这里应该集成到现有的metrics系统中
            # 目前使用简化的实现
            pass
        except Exception as e:
            self.logger.error(f"记录性能指标 {name} 时出错: {e}")
    
    async def _evaluate_tuning_rules(self):
        """评估调优规则"""
        while self.is_running:
            try:
                await asyncio.sleep(self.config["tuning_interval"])
                
                for rule in self.tuning_rules:
                    if not rule.enabled:
                        continue
                    
                    await self._evaluate_rule(rule)
                
            except Exception as e:
                self.logger.error(f"评估调优规则时出错: {e}")
                await asyncio.sleep(60)
    
    async def _evaluate_rule(self, rule: TuningRule):
        """评估单个调优规则"""
        try:
            # 获取最新指标值
            latest_value = await self._get_latest_metric_value(rule.metric_name)
            if latest_value is None:
                return
            
            # 检查条件
            condition_met = self._evaluate_condition(rule.condition, latest_value, rule.threshold)
            
            if condition_met:
                # 检查冷却期
                if not self._check_cooldown_period(rule.name, rule.cooldown_period):
                    return
                
                # 检查每小时动作限制
                if not self._check_action_limit(rule.name, rule.max_actions_per_hour):
                    return
                
                # 创建调优动作
                await self._create_tuning_action(rule, latest_value)
        
        except Exception as e:
            self.logger.error(f"评估规则 {rule.name} 时出错: {e}")
    
    def _evaluate_condition(self, condition: str, value: float, threshold: float) -> bool:
        """评估调优条件"""
        if condition == "greater_than":
            return value > threshold
        elif condition == "less_than":
            return value < threshold
        elif condition == "equals":
            return abs(value - threshold) < 0.001
        else:
            return False
    
    async def _get_latest_metric_value(self, metric_name: str) -> Optional[float]:
        """获取最新指标值"""
        try:
            # 这里应该从metrics数据库获取
            # 简化实现，返回模拟值
            import random
            return random.uniform(10, 90)
        except Exception as e:
            self.logger.error(f"获取指标 {metric_name} 值时出错: {e}")
            return None
    
    def _check_cooldown_period(self, rule_name: str, cooldown_period: int) -> bool:
        """检查冷却期"""
        last_action = self.last_action_time.get(rule_name, 0)
        return (time.time() - last_action) >= cooldown_period
    
    def _check_action_limit(self, rule_name: str, max_actions: int) -> bool:
        """检查动作限制"""
        current_hour = int(time.time() // 3600)
        action_key = f"{rule_name}_{current_hour}"
        
        count = self.action_counts.get(action_key, 0)
        return count < max_actions
    
    async def _create_tuning_action(self, rule: TuningRule, current_value: float):
        """创建调优动作"""
        action = TuningActionRecord(
            id=None,
            rule_name=rule.name,
            action_type=rule.action,
            resource_type=rule.resource_type,
            target_value=rule.target_value,
            timestamp=time.time()
        )
        
        # 保存到数据库
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO tuning_actions 
                (rule_name, action_type, resource_type, target_value, timestamp, status)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                action.rule_name,
                action.action_type.value,
                action.resource_type.value,
                action.target_value,
                action.timestamp,
                action.status
            ))
            
            action.id = cursor.lastrowid
        
        # 更新计数和时间
        self.last_action_time[rule.name] = time.time()
        
        current_hour = int(time.time() // 3600)
        action_key = f"{rule.name}_{current_hour}"
        self.action_counts[action_key] = self.action_counts.get(action_key, 0) + 1
        
        self.logger.info(f"创建调优动作: {rule.name} - {rule.action.value}")
        
        # 触发回调
        for callback in self.tuning_callbacks:
            try:
                await callback(action)
            except Exception as e:
                self.logger.error(f"调优回调函数执行出错: {e}")
    
    async def _execute_tuning_actions(self):
        """执行调优动作"""
        while self.is_running:
            try:
                # 获取待执行的动作
                pending_actions = await self._get_pending_actions()
                
                # 限制并发执行数量
                actions_to_execute = pending_actions[:self.config["max_concurrent_actions"]]
                
                # 并发执行动作
                tasks = [self._execute_single_action(action) for action in actions_to_execute]
                await asyncio.gather(*tasks, return_exceptions=True)
                
                await asyncio.sleep(30)  # 每30秒检查一次
                
            except Exception as e:
                self.logger.error(f"执行调优动作时出错: {e}")
                await asyncio.sleep(60)
    
    async def _get_pending_actions(self) -> List[TuningAction]:
        """获取待执行的动作"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT id, rule_name, action_type, resource_type, target_value, 
                           timestamp, status, result, execution_time, error_message
                    FROM tuning_actions 
                    WHERE status = 'pending'
                    ORDER BY timestamp ASC
                """)
                
                actions = []
                for row in cursor.fetchall():
                    actions.append(TuningActionRecord(
                        id=row[0],
                        rule_name=row[1],
                        action_type=TuningAction(row[2]),
                        resource_type=ResourceType(row[3]),
                        target_value=row[4],
                        timestamp=row[5],
                        status=row[6],
                        result=json.loads(row[7]) if row[7] else None,
                        execution_time=row[8] if row[8] else 0,
                        error_message=row[9]
                    ))
                
                return actions
        except Exception as e:
            self.logger.error(f"获取待执行动作时出错: {e}")
            return []
    
    async def _execute_single_action(self, action: TuningAction):
        """执行单个调优动作"""
        start_time = time.time()
        
        try:
            # 更新状态为执行中
            await self._update_action_status(action.id, "executing")
            
            # 根据动作类型执行相应操作
            if action.action_type == TuningAction.SCALE_UP:
                result = await self._execute_scale_up(action)
            elif action.action_type == TuningAction.SCALE_DOWN:
                result = await self._execute_scale_down(action)
            elif action.action_type == TuningAction.CLEANUP_RESOURCES:
                result = await self._execute_cleanup_resources(action)
            elif action.action_type == TuningAction.OPTIMIZE_CONFIG:
                result = await self._execute_optimize_config(action)
            elif action.action_type == TuningAction.RESTART_SERVICE:
                result = await self._execute_restart_service(action)
            else:
                result = {"success": False, "message": f"未知的动作类型: {action.action_type}"}
            
            execution_time = time.time() - start_time
            
            # 更新动作状态
            await self._update_action_result(action.id, "completed", result, execution_time)
            
            self.logger.info(f"调优动作执行完成: {action.rule_name} - {action.action_type.value}")
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_message = str(e)
            
            await self._update_action_result(action.id, "failed", None, execution_time, error_message)
            
            self.logger.error(f"调优动作执行失败: {action.rule_name} - {error_message}")
    
    async def _execute_scale_up(self, action: TuningAction) -> Dict[str, Any]:
        """执行扩容操作"""
        # 模拟扩容操作
        await asyncio.sleep(2)  # 模拟执行时间
        
        return {
            "success": True,
            "message": f"成功扩容 {action.resource_type.value}",
            "old_value": action.target_value * 0.8,
            "new_value": action.target_value,
            "actions_taken": ["增加资源配额", "启动新实例"]
        }
    
    async def _execute_scale_down(self, action: TuningAction) -> Dict[str, Any]:
        """执行缩容操作"""
        # 模拟缩容操作
        await asyncio.sleep(1)  # 模拟执行时间
        
        return {
            "success": True,
            "message": f"成功缩容 {action.resource_type.value}",
            "old_value": action.target_value * 1.2,
            "new_value": action.target_value,
            "actions_taken": ["减少资源配额", "停止空闲实例"]
        }
    
    async def _execute_cleanup_resources(self, action: TuningAction) -> Dict[str, Any]:
        """执行资源清理"""
        # 模拟资源清理
        await asyncio.sleep(3)  # 模拟执行时间
        
        return {
            "success": True,
            "message": f"成功清理 {action.resource_type.value} 资源",
            "cleaned_items": ["临时文件", "缓存数据", "无用进程"],
            "space_freed": "1.2GB"
        }
    
    async def _execute_optimize_config(self, action: TuningAction) -> Dict[str, Any]:
        """执行配置优化"""
        # 模拟配置优化
        await asyncio.sleep(5)  # 模拟执行时间
        
        return {
            "success": True,
            "message": f"成功优化 {action.resource_type.value} 配置",
            "optimizations": ["调整线程池大小", "优化内存分配", "更新算法参数"],
            "performance_gain": "15%"
        }
    
    async def _execute_restart_service(self, action: TuningAction) -> Dict[str, Any]:
        """执行服务重启"""
        # 模拟服务重启
        await asyncio.sleep(10)  # 模拟执行时间
        
        return {
            "success": True,
            "message": f"成功重启 {action.resource_type.value} 服务",
            "restart_duration": "8.5秒",
            "services_restarted": ["norma_agent", "database", "cache"]
        }
    
    async def _update_action_status(self, action_id: int, status: str):
        """更新动作状态"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE tuning_actions 
                SET status = ?
                WHERE id = ?
            """, (status, action_id))
    
    async def _update_action_result(self, action_id: int, status: str, result: Dict, 
                                  execution_time: float, error_message: str = None):
        """更新动作结果"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE tuning_actions 
                SET status = ?, result = ?, execution_time = ?, error_message = ?
                WHERE id = ?
            """, (
                status,
                json.dumps(result) if result else None,
                execution_time,
                error_message,
                action_id
            ))
    
    async def _update_performance_baselines(self):
        """更新性能基线"""
        while self.is_running:
            try:
                await asyncio.sleep(3600)  # 每小时更新一次
                
                # 获取所有指标名称
                metric_names = await self._get_all_metric_names()
                
                for metric_name in metric_names:
                    await self._calculate_baseline(metric_name)
                
            except Exception as e:
                self.logger.error(f"更新性能基线时出错: {e}")
                await asyncio.sleep(3600)
    
    async def _calculate_baseline(self, metric_name: str):
        """计算指标基线"""
        try:
            # 获取历史数据
            end_time = time.time()
            start_time = end_time - self.baseline_window
            
            values = await self._get_metric_history_values(metric_name, start_time, end_time)
            
            if len(values) < self.config["min_data_points"]:
                return
            
            # 计算统计值
            mean_value = statistics.mean(values)
            std_dev = statistics.stdev(values) if len(values) > 1 else 0
            
            # 保存基线
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO performance_baselines 
                    (metric_name, baseline_value, standard_deviation, sample_count, 
                     calculated_at, window_start, window_end)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    metric_name,
                    mean_value,
                    std_dev,
                    len(values),
                    time.time(),
                    start_time,
                    end_time
                ))
            
            # 更新内存中的基线
            self.performance_baseline[metric_name] = mean_value
            
            self.logger.info(f"更新性能基线: {metric_name} = {mean_value:.2f}")
            
        except Exception as e:
            self.logger.error(f"计算基线 {metric_name} 时出错: {e}")
    
    async def _get_all_metric_names(self) -> List[str]:
        """获取所有指标名称"""
        # 简化实现
        return [
            "cpu.usage.percent",
            "memory.usage.percent",
            "disk.usage.percent",
            "app.response_time.avg",
            "agent.accuracy.percent"
        ]
    
    async def _get_metric_history_values(self, metric_name: str, start_time: float, end_time: float) -> List[float]:
        """获取指标历史值"""
        # 简化实现，返回模拟数据
        import random
        return [random.uniform(20, 80) for _ in range(100)]
    
    async def _generate_recommendations(self):
        """生成资源推荐"""
        while self.is_running:
            try:
                await asyncio.sleep(1800)  # 每30分钟生成一次推荐
                
                # 为每种资源类型生成推荐
                for resource_type in ResourceType:
                    await self._generate_resource_recommendation(resource_type)
                
            except Exception as e:
                self.logger.error(f"生成资源推荐时出错: {e}")
                await asyncio.sleep(1800)
    
    async def _generate_resource_recommendation(self, resource_type: ResourceType):
        """生成单个资源推荐"""
        try:
            # 获取当前使用情况
            current_usage = await self._get_current_usage(resource_type)
            
            # 基于基线和当前状态生成推荐
            if resource_type in self.performance_baseline:
                baseline = self.performance_baseline[resource_type.value]
                recommended_usage = self._calculate_recommended_usage(current_usage, baseline)
                
                confidence = self._calculate_confidence(current_usage, baseline)
                
                recommendation = ResourceRecommendation(
                    resource_type=resource_type,
                    current_usage=current_usage,
                    recommended_usage=recommended_usage,
                    confidence=confidence,
                    reasoning=self._generate_reasoning(current_usage, recommended_usage, baseline),
                    actions=self._generate_recommended_actions(resource_type, current_usage, recommended_usage),
                    estimated_impact=self._estimate_impact(resource_type, current_usage, recommended_usage)
                )
                
                await self._save_recommendation(recommendation)
                
                self.logger.info(f"生成资源推荐: {resource_type.value}")
        
        except Exception as e:
            self.logger.error(f"生成资源推荐 {resource_type.value} 时出错: {e}")
    
    async def _get_current_usage(self, resource_type: ResourceType) -> float:
        """获取当前资源使用情况"""
        try:
            if resource_type == ResourceType.CPU:
                return psutil.cpu_percent(interval=1)
            elif resource_type == ResourceType.MEMORY:
                return psutil.virtual_memory().percent
            elif resource_type == ResourceType.DISK:
                disk = psutil.disk_usage('/')
                return (disk.used / disk.total) * 100
            else:
                # 其他资源类型使用模拟数据
                import random
                return random.uniform(30, 70)
        except Exception as e:
            self.logger.error(f"获取 {resource_type.value} 使用情况时出错: {e}")
            return 0.0
    
    def _calculate_recommended_usage(self, current_usage: float, baseline: float) -> float:
        """计算推荐使用量"""
        # 简单算法：如果当前使用量偏离基线太多，推荐调整到基线附近
        deviation = abs(current_usage - baseline)
        if deviation > 20:  # 偏离超过20%
            return baseline + (baseline * 0.1)  # 推荐调整到基线的110%
        else:
            return current_usage  # 保持当前状态
    
    def _calculate_confidence(self, current_usage: float, baseline: float) -> float:
        """计算推荐置信度"""
        deviation = abs(current_usage - baseline)
        max_deviation = max(current_usage, baseline, 100)
        
        if max_deviation == 0:
            return 0.5
        
        normalized_deviation = deviation / max_deviation
        return max(0.1, 1.0 - normalized_deviation)
    
    def _generate_reasoning(self, current: float, recommended: float, baseline: float) -> str:
        """生成推荐理由"""
        if current > recommended:
            return f"当前使用率({current:.1f}%)高于推荐值({recommended:.1f}%)，建议优化配置以提高效率"
        elif current < recommended:
            return f"当前使用率({current:.1f}%)低于推荐值({recommended:.1f}%)，可以考虑释放多余资源"
        else:
            return f"当前使用率({current:.1f}%)处于合理范围内，无需调整"
    
    def _generate_recommended_actions(self, resource_type: ResourceType, current: float, recommended: float) -> List[str]:
        """生成推荐动作"""
        actions = []
        
        if current > recommended:
            actions.append(f"优化{resource_type.value}配置")
            actions.append("清理无用资源")
            if resource_type == ResourceType.CPU:
                actions.append("调整并发数")
            elif resource_type == ResourceType.MEMORY:
                actions.append("清理缓存")
        elif current < recommended:
            actions.append(f"减少{resource_type.value}分配")
            actions.append("优化资源利用率")
        
        return actions
    
    def _estimate_impact(self, resource_type: ResourceType, current: float, recommended: float) -> Dict[str, float]:
        """估算影响"""
        impact = {}
        
        if resource_type == ResourceType.CPU:
            impact["性能提升"] = max(0, (current - recommended) * 0.5)
            impact["成本节省"] = max(0, (recommended - current) * 0.3)
        elif resource_type == ResourceType.MEMORY:
            impact["响应速度提升"] = max(0, (current - recommended) * 0.3)
            impact["稳定性改善"] = max(0, (current - recommended) * 0.2)
        else:
            impact["效率提升"] = abs(current - recommended) * 0.1
        
        return impact
    
    async def _save_recommendation(self, recommendation: ResourceRecommendation):
        """保存推荐"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO resource_recommendations 
                    (resource_type, current_usage, recommended_usage, confidence, 
                     reasoning, actions, estimated_impact, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    recommendation.resource_type.value,
                    recommendation.current_usage,
                    recommendation.recommended_usage,
                    recommendation.confidence,
                    recommendation.reasoning,
                    json.dumps(recommendation.actions),
                    json.dumps(recommendation.estimated_impact),
                    time.time()
                ))
        except Exception as e:
            self.logger.error(f"保存推荐时出错: {e}")
    
    async def _cleanup_old_data(self):
        """清理旧数据"""
        while self.is_running:
            try:
                await asyncio.sleep(86400)  # 每天清理一次
                
                cutoff_time = time.time() - (30 * 24 * 3600)  # 保留30天
                
                with sqlite3.connect(self.db_path) as conn:
                    # 清理旧调优动作
                    conn.execute("DELETE FROM tuning_actions WHERE timestamp < ?", (cutoff_time,))
                    
                    # 清理旧基线数据
                    conn.execute("DELETE FROM performance_baselines WHERE calculated_at < ?", (cutoff_time,))
                    
                    # 清理旧推荐
                    conn.execute("DELETE FROM resource_recommendations WHERE timestamp < ?", (cutoff_time,))
                
                self.logger.info("清理旧调优数据完成")
                
            except Exception as e:
                self.logger.error(f"清理旧数据时出错: {e}")
                await asyncio.sleep(86400)
    
    def add_tuning_callback(self, callback: Callable):
        """添加调优回调函数"""
        self.tuning_callbacks.append(callback)
    
    def get_tuning_history(self, hours: int = 24) -> List[TuningAction]:
        """获取调优历史"""
        start_time = time.time() - (hours * 3600)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, rule_name, action_type, resource_type, target_value, 
                       timestamp, status, result, execution_time, error_message
                FROM tuning_actions 
                WHERE timestamp > ?
                ORDER BY timestamp DESC
            """, (start_time,))
            
            actions = []
            for row in cursor.fetchall():
                actions.append(TuningActionRecord(
                    id=row[0],
                    rule_name=row[1],
                    action_type=TuningAction(row[2]),
                    resource_type=ResourceType(row[3]),
                    target_value=row[4],
                    timestamp=row[5],
                    status=row[6],
                    result=json.loads(row[7]) if row[7] else None,
                    execution_time=row[8] if row[8] else 0,
                    error_message=row[9]
                ))
            
            return actions
    
    def get_recommendations(self, hours: int = 24) -> List[ResourceRecommendation]:
        """获取资源推荐"""
        start_time = time.time() - (hours * 3600)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT resource_type, current_usage, recommended_usage, confidence, 
                       reasoning, actions, estimated_impact, timestamp
                FROM resource_recommendations 
                WHERE timestamp > ?
                ORDER BY timestamp DESC
            """, (start_time,))
            
            recommendations = []
            for row in cursor.fetchall():
                recommendations.append(ResourceRecommendation(
                    resource_type=ResourceType(row[0]),
                    current_usage=row[1],
                    recommended_usage=row[2],
                    confidence=row[3],
                    reasoning=row[4],
                    actions=json.loads(row[5]),
                    estimated_impact=json.loads(row[6])
                ))
            
            return recommendations
    
    def get_performance_baselines(self) -> Dict[str, Dict]:
        """获取性能基线"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT metric_name, baseline_value, standard_deviation, sample_count, calculated_at
                FROM performance_baselines 
                WHERE calculated_at > ?
                ORDER BY calculated_at DESC
            """, (time.time() - self.baseline_window,))
            
            baselines = {}
            for row in cursor.fetchall():
                metric_name = row[0]
                if metric_name not in baselines:
                    baselines[metric_name] = {
                        "baseline_value": row[1],
                        "standard_deviation": row[2],
                        "sample_count": row[3],
                        "calculated_at": row[4]
                    }
            
            return baselines

# 使用示例
async def main():
    """主函数示例"""
    auto_tuner = AutoTuner()
    
    # 添加调优回调
    async def tuning_callback(action: TuningAction):
        print(f"调优动作: {action.rule_name} - {action.action_type.value}")
    
    auto_tuner.add_tuning_callback(tuning_callback)
    
    try:
        await auto_tuner.start_tuning()
        
        # 运行一段时间
        await asyncio.sleep(300)
        
    finally:
        await auto_tuner.stop_tuning()

if __name__ == "__main__":
    asyncio.run(main())