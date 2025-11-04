#!/usr/bin/env python3
"""
多智能体系统日志工具
提供统一的日志记录和管理功能

作者: 皇
创建时间: 2025-10-31
"""

import logging
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass, asdict
from collections import deque
import sys
import os
from pathlib import Path

from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler


class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogCategory(Enum):
    """日志类别枚举"""
    SYSTEM = "system"
    AGENT = "agent"
    COLLABORATION = "collaboration"
    TASK = "task"
    COMMUNICATION = "communication"
    PERFORMANCE = "performance"
    SECURITY = "security"
    ERROR = "error"


@dataclass
class LogEntry:
    """日志条目"""
    entry_id: str
    timestamp: str
    level: str
    category: str
    logger_name: str
    message: str
    component: str
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    context: Dict[str, Any] = None
    stack_trace: Optional[str] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False)


class MultiAgentLogger:
    """多智能体系统日志器"""
    
    def __init__(self, name: str, component: str = "default"):
        """初始化日志器
        
        Args:
            name: 日志器名称
            component: 组件名称
        """
        self.logger_name = name
        self.component = component
        
        # 创建Python日志器
        self.logger = logging.getLogger(f"multi_agent.{name}")
        
        # 配置日志器
        self._configure_logger()
        
        # 日志缓冲
        self.log_buffer = deque(maxlen=1000)
        
        # 日志统计
        self.log_stats = {
            "total_logs": 0,
            "logs_by_level": {level.value: 0 for level in LogLevel},
            "logs_by_category": {category.value: 0 for category in LogCategory},
            "errors_count": 0,
            "warnings_count": 0
        }
        
        # 事件回调
        self.event_callbacks: Dict[str, List[Callable]] = {}
        
        # 设置全局日志格式
        self._setup_global_format()
    
    def _configure_logger(self):
        """配置日志器"""
        if not self.logger.handlers:
            # 控制台处理器
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            
            # 文件处理器
            log_dir = Path("/workspace/norma_agent_enhanced/logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            
            file_handler = TimedRotatingFileHandler(
                log_dir / f"{self.logger_name}.log",
                when='midnight',
                interval=1,
                backupCount=30,
                encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG)
            
            # 错误文件处理器
            error_handler = RotatingFileHandler(
                log_dir / f"{self.logger_name}_error.log",
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
            error_handler.setLevel(logging.ERROR)
            
            # 添加处理器
            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)
            self.logger.addHandler(error_handler)
            
            # 设置日志级别
            self.logger.setLevel(logging.DEBUG)
            
            # 防止日志传播到根日志器
            self.logger.propagate = False
    
    def _setup_global_format(self):
        """设置全局日志格式"""
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        for handler in self.logger.handlers:
            handler.setFormatter(formatter)
    
    def _create_log_entry(
        self,
        level: LogLevel,
        category: LogCategory,
        message: str,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        stack_trace: Optional[str] = None
    ) -> LogEntry:
        """创建日志条目"""
        entry = LogEntry(
            entry_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            level=level.value,
            category=category.value,
            logger_name=self.logger_name,
            message=message,
            component=self.component,
            agent_id=agent_id,
            session_id=session_id,
            context=context or {},
            stack_trace=stack_trace
        )
        
        # 添加到缓冲区
        self.log_buffer.append(entry)
        
        # 更新统计
        self.log_stats["total_logs"] += 1
        self.log_stats["logs_by_level"][level.value] += 1
        self.log_stats["logs_by_category"][category.value] += 1
        
        if level == LogLevel.ERROR:
            self.log_stats["errors_count"] += 1
        elif level == LogLevel.WARNING:
            self.log_stats["warnings_count"] += 1
        
        return entry
    
    def _log_to_python_logger(self, entry: LogEntry):
        """记录到Python日志器"""
        level_map = {
            LogLevel.DEBUG: self.logger.debug,
            LogLevel.INFO: self.logger.info,
            LogLevel.WARNING: self.logger.warning,
            LogLevel.ERROR: self.logger.error,
            LogLevel.CRITICAL: self.logger.critical
        }
        
        log_level = LogLevel(entry.level)
        log_func = level_map[log_level]
        
        # 构建日志消息
        log_message = f"[{entry.category}] {entry.message}"
        
        if entry.agent_id:
            log_message = f"[Agent:{entry.agent_id}] {log_message}"
        
        if entry.session_id:
            log_message = f"[Session:{entry.session_id}] {log_message}"
        
        if entry.context:
            log_message += f" | Context: {json.dumps(entry.context, ensure_ascii=False)}"
        
        # 记录日志
        if entry.stack_trace:
            log_func(log_message, exc_info=True)
        else:
            log_func(log_message)
    
    def debug(
        self,
        message: str,
        category: LogCategory = LogCategory.SYSTEM,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """记录调试日志"""
        entry = self._create_log_entry(
            level=LogLevel.DEBUG,
            category=category,
            message=message,
            agent_id=agent_id,
            session_id=session_id,
            context=context
        )
        
        self._log_to_python_logger(entry)
        self._emit_event("log.debug", entry.to_dict())
    
    def info(
        self,
        message: str,
        category: LogCategory = LogCategory.SYSTEM,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """记录信息日志"""
        entry = self._create_log_entry(
            level=LogLevel.INFO,
            category=category,
            message=message,
            agent_id=agent_id,
            session_id=session_id,
            context=context
        )
        
        self._log_to_python_logger(entry)
        self._emit_event("log.info", entry.to_dict())
    
    def warning(
        self,
        message: str,
        category: LogCategory = LogCategory.SYSTEM,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """记录警告日志"""
        entry = self._create_log_entry(
            level=LogLevel.WARNING,
            category=category,
            message=message,
            agent_id=agent_id,
            session_id=session_id,
            context=context
        )
        
        self._log_to_python_logger(entry)
        self._emit_event("log.warning", entry.to_dict())
    
    def error(
        self,
        message: str,
        category: LogCategory = LogCategory.ERROR,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None
    ):
        """记录错误日志"""
        stack_trace = None
        if exception:
            stack_trace = str(exception)
        
        entry = self._create_log_entry(
            level=LogLevel.ERROR,
            category=category,
            message=message,
            agent_id=agent_id,
            session_id=session_id,
            context=context,
            stack_trace=stack_trace
        )
        
        self._log_to_python_logger(entry)
        self._emit_event("log.error", entry.to_dict())
    
    def critical(
        self,
        message: str,
        category: LogCategory = LogCategory.ERROR,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None
    ):
        """记录严重错误日志"""
        stack_trace = None
        if exception:
            stack_trace = str(exception)
        
        entry = self._create_log_entry(
            level=LogLevel.CRITICAL,
            category=category,
            message=message,
            agent_id=agent_id,
            session_id=session_id,
            context=context,
            stack_trace=stack_trace
        )
        
        self._log_to_python_logger(entry)
        self._emit_event("log.critical", entry.to_dict())
    
    def log_agent_action(
        self,
        agent_id: str,
        action: str,
        status: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """记录智能体动作"""
        message = f"Agent {agent_id} {action} - Status: {status}"
        
        context = details or {}
        context.update({
            "agent_id": agent_id,
            "action": action,
            "status": status
        })
        
        self.info(
            message=message,
            category=LogCategory.AGENT,
            agent_id=agent_id,
            context=context
        )
    
    def log_collaboration_event(
        self,
        session_id: str,
        event_type: str,
        participants: List[str],
        details: Optional[Dict[str, Any]] = None
    ):
        """记录协作事件"""
        message = f"Collaboration {event_type} - Session: {session_id}"
        
        context = details or {}
        context.update({
            "session_id": session_id,
            "event_type": event_type,
            "participants": participants
        })
        
        self.info(
            message=message,
            category=LogCategory.COLLABORATION,
            session_id=session_id,
            context=context
        )
    
    def log_task_execution(
        self,
        task_id: str,
        agent_id: str,
        status: str,
        duration: Optional[float] = None,
        error: Optional[str] = None
    ):
        """记录任务执行"""
        message = f"Task {task_id} executed by {agent_id} - Status: {status}"
        
        context = {
            "task_id": task_id,
            "agent_id": agent_id,
            "status": status
        }
        
        if duration:
            context["duration"] = duration
        
        if error:
            context["error"] = error
            self.error(
                message=message,
                category=LogCategory.TASK,
                agent_id=agent_id,
                context=context
            )
        else:
            self.info(
                message=message,
                category=LogCategory.TASK,
                agent_id=agent_id,
                context=context
            )
    
    def log_performance_metric(
        self,
        metric_name: str,
        value: float,
        unit: str,
        component: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """记录性能指标"""
        message = f"Performance metric {metric_name}: {value} {unit}"
        
        perf_context = context or {}
        perf_context.update({
            "metric_name": metric_name,
            "value": value,
            "unit": unit,
            "component": component
        })
        
        self.info(
            message=message,
            category=LogCategory.PERFORMANCE,
            context=perf_context
        )
    
    def log_security_event(
        self,
        event_type: str,
        severity: str,
        description: str,
        source: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """记录安全事件"""
        message = f"Security event: {event_type} - {description}"
        
        sec_context = context or {}
        sec_context.update({
            "event_type": event_type,
            "severity": severity,
            "description": description,
            "source": source
        })
        
        # 根据严重级别选择日志级别
        if severity.lower() in ["high", "critical"]:
            self.critical(
                message=message,
                category=LogCategory.SECURITY,
                context=sec_context
            )
        elif severity.lower() == "medium":
            self.warning(
                message=message,
                category=LogCategory.SECURITY,
                context=sec_context
            )
        else:
            self.info(
                message=message,
                category=LogCategory.SECURITY,
                context=sec_context
            )
    
    def get_recent_logs(
        self,
        count: int = 100,
        level: Optional[LogLevel] = None,
        category: Optional[LogCategory] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """获取最近的日志"""
        logs = list(self.log_buffer)
        
        # 过滤日志
        if level:
            logs = [log for log in logs if log.level == level.value]
        
        if category:
            logs = [log for log in logs if log.category == category.value]
        
        if agent_id:
            logs = [log for log in logs if log.agent_id == agent_id]
        
        if session_id:
            logs = [log for log in logs if log.session_id == session_id]
        
        # 按时间倒序排列并限制数量
        logs.sort(key=lambda x: x.timestamp, reverse=True)
        return [log.to_dict() for log in logs[:count]]
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """获取日志统计信息"""
        return {
            "logger_name": self.logger_name,
            "component": self.component,
            "stats": self.log_stats.copy(),
            "buffer_size": len(self.log_buffer),
            "handlers_count": len(self.logger.handlers)
        }
    
    def search_logs(
        self,
        query: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """搜索日志"""
        results = []
        
        for log in self.log_buffer:
            # 时间范围过滤
            log_time = datetime.fromisoformat(log.timestamp)
            if start_time and log_time < start_time:
                continue
            if end_time and log_time > end_time:
                continue
            
            # 文本搜索
            if (query.lower() in log.message.lower() or 
                query.lower() in log.category.lower() or
                (log.agent_id and query.lower() in log.agent_id.lower()) or
                (log.session_id and query.lower() in log.session_id.lower())):
                
                results.append(log.to_dict())
                
                if len(results) >= limit:
                    break
        
        return results
    
    def export_logs(
        self,
        file_path: str,
        format: str = "json",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> bool:
        """导出日志"""
        try:
            logs = []
            
            for log in self.log_buffer:
                # 时间范围过滤
                log_time = datetime.fromisoformat(log.timestamp)
                if start_time and log_time < start_time:
                    continue
                if end_time and log_time > end_time:
                    continue
                
                logs.append(log.to_dict())
            
            # 写入文件
            with open(file_path, 'w', encoding='utf-8') as f:
                if format.lower() == "json":
                    json.dump(logs, f, ensure_ascii=False, indent=2)
                else:
                    # 文本格式
                    for log in logs:
                        f.write(f"{log['timestamp']} [{log['level']}] [{log['category']}] {log['message']}\n")
            
            self.info(f"Logs exported to {file_path}", LogCategory.SYSTEM)
            return True
            
        except Exception as e:
            self.error(f"Failed to export logs: {e}", LogCategory.SYSTEM)
            return False
    
    def clear_logs(self):
        """清空日志缓冲区"""
        self.log_buffer.clear()
        self.info("Log buffer cleared", LogCategory.SYSTEM)
    
    def on_event(self, event_type: str, callback: Callable):
        """注册事件回调"""
        if event_type not in self.event_callbacks:
            self.event_callbacks[event_type] = []
        self.event_callbacks[event_type].append(callback)
    
    def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """发送事件"""
        if event_type in self.event_callbacks:
            for callback in self.event_callbacks[event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        asyncio.create_task(callback(data))
                    else:
                        callback(data)
                except Exception as e:
                    # 避免日志记录本身的错误导致无限循环
                    print(f"Event callback failed: {e}")


# 全局日志管理器
class GlobalLogManager:
    """全局日志管理器"""
    
    def __init__(self):
        self.loggers: Dict[str, MultiAgentLogger] = {}
        self.global_stats = {
            "total_loggers": 0,
            "total_log_entries": 0,
            "system_start_time": datetime.now().isoformat()
        }
    
    def get_logger(self, name: str, component: str = "default") -> MultiAgentLogger:
        """获取日志器"""
        if name not in self.loggers:
            self.loggers[name] = MultiAgentLogger(name, component)
            self.global_stats["total_loggers"] += 1
        
        return self.loggers[name]
    
    def get_all_loggers(self) -> Dict[str, MultiAgentLogger]:
        """获取所有日志器"""
        return self.loggers.copy()
    
    def get_global_statistics(self) -> Dict[str, Any]:
        """获取全局统计"""
        total_entries = sum(logger.log_stats["total_logs"] for logger in self.loggers.values())
        
        return {
            "total_loggers": len(self.loggers),
            "total_log_entries": total_entries,
            "system_start_time": self.global_stats["system_start_time"],
            "uptime_seconds": (datetime.now() - datetime.fromisoformat(self.global_stats["system_start_time"])).total_seconds(),
            "loggers": {
                name: logger.get_log_statistics()
                for name, logger in self.loggers.items()
            }
        }


# 全局实例
_global_log_manager = GlobalLogManager()


def get_logger(name: str, component: str = "default") -> MultiAgentLogger:
    """获取全局日志器实例"""
    return _global_log_manager.get_logger(name, component)


def get_global_log_manager() -> GlobalLogManager:
    """获取全局日志管理器"""
    return _global_log_manager


# 便捷函数
def log_agent_action(agent_id: str, action: str, status: str, details: Optional[Dict[str, Any]] = None):
    """便捷函数：记录智能体动作"""
    logger = get_logger("agent_actions")
    logger.log_agent_action(agent_id, action, status, details)


def log_collaboration_event(session_id: str, event_type: str, participants: List[str], details: Optional[Dict[str, Any]] = None):
    """便捷函数：记录协作事件"""
    logger = get_logger("collaboration_events")
    logger.log_collaboration_event(session_id, event_type, participants, details)


def log_task_execution(task_id: str, agent_id: str, status: str, duration: Optional[float] = None, error: Optional[str] = None):
    """便捷函数：记录任务执行"""
    logger = get_logger("task_execution")
    logger.log_task_execution(task_id, agent_id, status, duration, error)


def log_performance_metric(metric_name: str, value: float, unit: str, component: str, context: Optional[Dict[str, Any]] = None):
    """便捷函数：记录性能指标"""
    logger = get_logger("performance_metrics")
    logger.log_performance_metric(metric_name, value, unit, component, context)


def log_security_event(event_type: str, severity: str, description: str, source: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
    """便捷函数：记录安全事件"""
    logger = get_logger("security_events")
    logger.log_security_event(event_type, severity, description, source, context)