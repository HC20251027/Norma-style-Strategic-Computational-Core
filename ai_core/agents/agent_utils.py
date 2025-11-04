"""
Agent工具类
"""
import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import traceback
import uuid


class LogLevel(Enum):
    """日志级别"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AgentHealthCheck:
    """智能体健康检查"""
    agent_id: str
    status: str  # healthy, unhealthy, unknown
    last_check: datetime
    response_time: float
    error_message: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None


class AgentHealthMonitor:
    """智能体健康监控器"""
    
    def __init__(self):
        self.health_checks = {}
        self.check_intervals = {}
        self.logger = logging.getLogger("agent.health.monitor")
    
    def register_agent(self, agent_id: str, check_interval: int = 30):
        """注册智能体健康检查"""
        self.check_intervals[agent_id] = check_interval
        self.logger.info(f"注册智能体健康检查: {agent_id} (间隔: {check_interval}s)")
    
    async def check_agent_health(self, agent_id: str, health_check_func: Callable) -> AgentHealthCheck:
        """检查智能体健康状态"""
        start_time = time.time()
        
        try:
            # 执行健康检查函数
            result = await health_check_func()
            
            response_time = (time.time() - start_time) * 1000  # 转换为毫秒
            
            health_check = AgentHealthCheck(
                agent_id=agent_id,
                status="healthy",
                last_check=datetime.now(),
                response_time=response_time,
                metrics=result if isinstance(result, dict) else {"status": result}
            )
            
            self.health_checks[agent_id] = health_check
            return health_check
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            
            health_check = AgentHealthCheck(
                agent_id=agent_id,
                status="unhealthy",
                last_check=datetime.now(),
                response_time=response_time,
                error_message=str(e)
            )
            
            self.health_checks[agent_id] = health_check
            self.logger.error(f"智能体健康检查失败: {agent_id} - {e}")
            return health_check
    
    def get_agent_health(self, agent_id: str) -> Optional[AgentHealthCheck]:
        """获取智能体健康状态"""
        return self.health_checks.get(agent_id)
    
    def get_all_health_status(self) -> Dict[str, Any]:
        """获取所有智能体健康状态"""
        healthy_count = 0
        unhealthy_count = 0
        unknown_count = 0
        
        for health_check in self.health_checks.values():
            if health_check.status == "healthy":
                healthy_count += 1
            elif health_check.status == "unhealthy":
                unhealthy_count += 1
            else:
                unknown_count += 1
        
        return {
            "total_agents": len(self.health_checks),
            "healthy": healthy_count,
            "unhealthy": unhealthy_count,
            "unknown": unknown_count,
            "health_checks": {
                agent_id: asdict(check) for agent_id, check in self.health_checks.items()
            }
        }


class AgentMetricsCollector:
    """智能体指标收集器"""
    
    def __init__(self):
        self.metrics = {}
        self.metrics_history = {}
        self.logger = logging.getLogger("agent.metrics.collector")
    
    def record_metric(self, agent_id: str, metric_name: str, value: float, 
                     timestamp: datetime = None):
        """记录指标"""
        if timestamp is None:
            timestamp = datetime.now()
        
        if agent_id not in self.metrics:
            self.metrics[agent_id] = {}
        
        if agent_id not in self.metrics_history:
            self.metrics_history[agent_id] = {}
        
        # 当前指标
        self.metrics[agent_id][metric_name] = value
        
        # 历史记录
        if metric_name not in self.metrics_history[agent_id]:
            self.metrics_history[agent_id][metric_name] = []
        
        self.metrics_history[agent_id][metric_name].append({
            "value": value,
            "timestamp": timestamp
        })
        
        # 保持历史记录在合理范围内
        if len(self.metrics_history[agent_id][metric_name]) > 1000:
            self.metrics_history[agent_id][metric_name] = \
                self.metrics_history[agent_id][metric_name][-1000:]
    
    def get_agent_metrics(self, agent_id: str) -> Dict[str, Any]:
        """获取智能体指标"""
        return self.metrics.get(agent_id, {})
    
    def get_metric_history(self, agent_id: str, metric_name: str, 
                          duration: timedelta = None) -> List[Dict[str, Any]]:
        """获取指标历史"""
        if agent_id not in self.metrics_history:
            return []
        
        if metric_name not in self.metrics_history[agent_id]:
            return []
        
        history = self.metrics_history[agent_id][metric_name]
        
        if duration:
            cutoff_time = datetime.now() - duration
            history = [h for h in history if h["timestamp"] >= cutoff_time]
        
        return history
    
    def calculate_metric_stats(self, agent_id: str, metric_name: str, 
                              duration: timedelta = None) -> Dict[str, Any]:
        """计算指标统计"""
        history = self.get_metric_history(agent_id, metric_name, duration)
        
        if not history:
            return {"count": 0}
        
        values = [h["value"] for h in history]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1] if values else None
        }


class AgentEventLogger:
    """智能体事件日志记录器"""
    
    def __init__(self, max_events: int = 10000):
        self.events = []
        self.max_events = max_events
        self.logger = logging.getLogger("agent.event.logger")
    
    def log_event(self, agent_id: str, event_type: str, 
                  data: Dict[str, Any] = None, level: LogLevel = LogLevel.INFO):
        """记录事件"""
        event = {
            "event_id": str(uuid.uuid4()),
            "agent_id": agent_id,
            "event_type": event_type,
            "data": data or {},
            "level": level.value,
            "timestamp": datetime.now()
        }
        
        self.events.append(event)
        
        # 保持事件数量在限制内
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]
        
        # 记录到日志
        log_message = f"Agent Event: {agent_id} - {event_type}"
        if data:
            log_message += f" - {json.dumps(data, default=str)}"
        
        if level == LogLevel.DEBUG:
            self.logger.debug(log_message)
        elif level == LogLevel.INFO:
            self.logger.info(log_message)
        elif level == LogLevel.WARNING:
            self.logger.warning(log_message)
        elif level == LogLevel.ERROR:
            self.logger.error(log_message)
        elif level == LogLevel.CRITICAL:
            self.logger.critical(log_message)
    
    def get_events(self, agent_id: str = None, event_type: str = None, 
                   level: LogLevel = None, limit: int = 100) -> List[Dict[str, Any]]:
        """获取事件"""
        filtered_events = self.events
        
        if agent_id:
            filtered_events = [e for e in filtered_events if e["agent_id"] == agent_id]
        
        if event_type:
            filtered_events = [e for e in filtered_events if e["event_type"] == event_type]
        
        if level:
            filtered_events = [e for e in filtered_events if e["level"] == level.value]
        
        # 返回最近的事件
        return [asdict(e) for e in filtered_events[-limit:]]
    
    def get_event_summary(self) -> Dict[str, Any]:
        """获取事件摘要"""
        agent_counts = {}
        type_counts = {}
        level_counts = {}
        
        for event in self.events:
            # 智能体统计
            agent_id = event["agent_id"]
            agent_counts[agent_id] = agent_counts.get(agent_id, 0) + 1
            
            # 事件类型统计
            event_type = event["event_type"]
            type_counts[event_type] = type_counts.get(event_type, 0) + 1
            
            # 级别统计
            level = event["level"]
            level_counts[level] = level_counts.get(level, 0) + 1
        
        return {
            "total_events": len(self.events),
            "agent_counts": agent_counts,
            "type_counts": type_counts,
            "level_counts": level_counts,
            "time_range": {
                "start": self.events[0]["timestamp"].isoformat() if self.events else None,
                "end": self.events[-1]["timestamp"].isoformat() if self.events else None
            }
        }


class AgentConfigurationManager:
    """智能体配置管理器"""
    
    def __init__(self):
        self.configurations = {}
        self.default_configs = {}
        self.logger = logging.getLogger("agent.config.manager")
    
    def register_default_config(self, agent_type: str, config: Dict[str, Any]):
        """注册默认配置"""
        self.default_configs[agent_type] = config
        self.logger.info(f"注册默认配置: {agent_type}")
    
    def set_agent_config(self, agent_id: str, config: Dict[str, Any]):
        """设置智能体配置"""
        self.configurations[agent_id] = config
        self.logger.info(f"设置智能体配置: {agent_id}")
    
    def get_agent_config(self, agent_id: str) -> Dict[str, Any]:
        """获取智能体配置"""
        return self.configurations.get(agent_id, {})
    
    def get_config_for_agent_type(self, agent_type: str, agent_id: str = None) -> Dict[str, Any]:
        """获取智能体类型配置"""
        config = self.default_configs.get(agent_type, {}).copy()
        
        if agent_id and agent_id in self.configurations:
            # 合并配置，智能体配置优先
            agent_config = self.configurations[agent_id]
            config.update(agent_config)
        
        return config
    
    def update_agent_config(self, agent_id: str, updates: Dict[str, Any]):
        """更新智能体配置"""
        if agent_id not in self.configurations:
            self.configurations[agent_id] = {}
        
        self.configurations[agent_id].update(updates)
        self.logger.info(f"更新智能体配置: {agent_id}")
    
    def get_all_configs(self) -> Dict[str, Any]:
        """获取所有配置"""
        return {
            "default_configs": self.default_configs,
            "agent_configs": self.configurations
        }


class AgentExceptionHandler:
    """智能体异常处理器"""
    
    def __init__(self):
        self.exception_handlers = {}
        self.exception_history = []
        self.logger = logging.getLogger("agent.exception.handler")
    
    def register_handler(self, exception_type: type, handler: Callable):
        """注册异常处理器"""
        self.exception_handlers[exception_type] = handler
        self.logger.info(f"注册异常处理器: {exception_type.__name__}")
    
    async def handle_exception(self, agent_id: str, exception: Exception, 
                              context: Dict[str, Any] = None):
        """处理异常"""
        exception_info = {
            "exception_id": str(uuid.uuid4()),
            "agent_id": agent_id,
            "exception_type": type(exception).__name__,
            "exception_message": str(exception),
            "traceback": traceback.format_exc(),
            "context": context or {},
            "timestamp": datetime.now()
        }
        
        self.exception_history.append(exception_info)
        
        # 查找处理器
        exception_type = type(exception)
        handler = self.exception_handlers.get(exception_type)
        
        if handler:
            try:
                await handler(agent_id, exception, context)
            except Exception as handler_error:
                self.logger.error(f"异常处理器执行失败: {handler_error}")
        else:
            # 默认处理
            self.logger.error(f"未处理的异常 [{agent_id}]: {exception}")
        
        # 保持历史记录在合理范围内
        if len(self.exception_history) > 1000:
            self.exception_history = self.exception_history[-1000:]
    
    def get_exception_history(self, agent_id: str = None, 
                             exception_type: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """获取异常历史"""
        filtered_exceptions = self.exception_history
        
        if agent_id:
            filtered_exceptions = [e for e in filtered_exceptions if e["agent_id"] == agent_id]
        
        if exception_type:
            filtered_exceptions = [e for e in filtered_exceptions if e["exception_type"] == exception_type]
        
        return filtered_exceptions[-limit:]
    
    def get_exception_summary(self) -> Dict[str, Any]:
        """获取异常摘要"""
        agent_counts = {}
        type_counts = {}
        
        for exception_info in self.exception_history:
            # 智能体统计
            agent_id = exception_info["agent_id"]
            agent_counts[agent_id] = agent_counts.get(agent_id, 0) + 1
            
            # 异常类型统计
            exception_type = exception_info["exception_type"]
            type_counts[exception_type] = type_counts.get(exception_type, 0) + 1
        
        return {
            "total_exceptions": len(self.exception_history),
            "agent_counts": agent_counts,
            "type_counts": type_counts,
            "recent_exceptions": self.exception_history[-10:]
        }


class AgentUtility:
    """智能体工具类"""
    
    @staticmethod
    def generate_agent_id(agent_type: str, suffix: str = None) -> str:
        """生成智能体ID"""
        timestamp = int(time.time())
        base_id = f"{agent_type}-{timestamp}"
        if suffix:
            base_id += f"-{suffix}"
        return base_id
    
    @staticmethod
    def validate_agent_config(config: Dict[str, Any], required_fields: List[str]) -> bool:
        """验证智能体配置"""
        for field in required_fields:
            if field not in config:
                return False
        return True
    
    @staticmethod
    def calculate_agent_load(metrics: Dict[str, float]) -> float:
        """计算智能体负载"""
        if not metrics:
            return 0.0
        
        # 简单的负载计算：各项指标的平均值
        weights = {
            "cpu_usage": 0.3,
            "memory_usage": 0.3,
            "active_connections": 0.2,
            "queue_size": 0.2
        }
        
        total_weight = 0
        weighted_sum = 0
        
        for metric, weight in weights.items():
            if metric in metrics:
                weighted_sum += metrics[metric] * weight
                total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    @staticmethod
    def format_agent_status(agent_info: Dict[str, Any]) -> str:
        """格式化智能体状态"""
        status_parts = []
        
        if "agent_id" in agent_info:
            status_parts.append(f"ID: {agent_info['agent_id']}")
        
        if "status" in agent_info:
            status_parts.append(f"状态: {agent_info['status']}")
        
        if "load" in agent_info:
            status_parts.append(f"负载: {agent_info['load']:.2f}")
        
        if "last_heartbeat" in agent_info:
            status_parts.append(f"心跳: {agent_info['last_heartbeat']}")
        
        return " | ".join(status_parts)
    
    @staticmethod
    def create_task_payload(action: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """创建任务负载"""
        return {
            "type": action,
            "data": data or {},
            "timestamp": datetime.now().isoformat()
        }