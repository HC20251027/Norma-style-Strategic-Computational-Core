#!/usr/bin/env python3
"""
日志工具模块
提供统一的日志记录和管理功能

作者: 皇
创建时间: 2025-10-31
"""

import logging
import sys
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
from logging.handlers import RotatingFileHandler
import threading

class NormaFormatter(logging.Formatter):
    """诺玛专用日志格式化器"""
    
    def __init__(self):
        super().__init__()
        self.formats = {
            logging.DEBUG: "[{asctime}] {name} DEBUG {message}",
            logging.INFO: "[{asctime}] {name} INFO {message}",
            logging.WARNING: "[{asctime}] {name} WARNING {message}",
            logging.ERROR: "[{asctime}] {name} ERROR {message}",
            logging.CRITICAL: "[{asctime}] {name} CRITICAL {message}"
        }
    
    def format(self, record):
        log_format = self.formats.get(record.levelno, self.formats[logging.INFO])
        formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S", style="{")
        return formatter.format(record)

class NormaLogger:
    """诺玛日志记录器"""
    
    _loggers: Dict[str, 'NormaLogger'] = {}
    _lock = threading.Lock()
    
    def __init__(self, name: str, log_level: str = "INFO", log_file: Optional[str] = None):
        """初始化诺玛日志记录器
        
        Args:
            name: 日志记录器名称
            log_level: 日志级别
            log_file: 日志文件路径
        """
        self.name = name
        self.logger = logging.getLogger(name)
        
        # 防止重复配置
        if self.logger.handlers:
            return
        
        # 设置日志级别
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # 创建格式化器
        formatter = NormaFormatter()
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # 文件处理器
        if log_file:
            self._setup_file_handler(log_file, formatter)
        
        # 防止日志向上传播
        self.logger.propagate = False
    
    def _setup_file_handler(self, log_file: str, formatter: NormaFormatter):
        """设置文件处理器"""
        
        # 确保日志目录存在
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 轮转文件处理器（最大10MB，保留5个文件）
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def debug(self, message: str, **kwargs):
        """调试日志"""
        extra = self._prepare_extra(kwargs)
        self.logger.debug(message, extra=extra)
    
    def info(self, message: str, **kwargs):
        """信息日志"""
        extra = self._prepare_extra(kwargs)
        self.logger.info(message, extra=extra)
    
    def warning(self, message: str, **kwargs):
        """警告日志"""
        extra = self._prepare_extra(kwargs)
        self.logger.warning(message, extra=extra)
    
    def error(self, message: str, **kwargs):
        """错误日志"""
        extra = self._prepare_extra(kwargs)
        self.logger.error(message, extra=extra)
    
    def critical(self, message: str, **kwargs):
        """严重错误日志"""
        extra = self._prepare_extra(kwargs)
        self.logger.critical(message, extra=extra)
    
    def _prepare_extra(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """准备额外信息"""
        extra = {}
        for key, value in kwargs.items():
            if key.startswith('_'):
                extra[key[1:]] = value
            else:
                extra[key] = value
        return extra
    
    def log_performance(self, operation: str, duration: float, **kwargs):
        """记录性能日志"""
        self.info(f"性能监控 - {operation}: {duration:.3f}秒", operation=operation, duration=duration, **kwargs)
    
    def log_api_call(self, method: str, url: str, status_code: int, duration: float, **kwargs):
        """记录API调用日志"""
        level = logging.INFO if 200 <= status_code < 400 else logging.WARNING
        message = f"API调用 - {method} {url} - {status_code} - {duration:.3f}秒"
        self.logger.log(level, message, extra={
            'api_method': method,
            'api_url': url,
            'api_status': status_code,
            'api_duration': duration,
            **kwargs
        })
    
    def log_user_action(self, user_id: str, action: str, **kwargs):
        """记录用户行为日志"""
        self.info(f"用户行为 - {user_id}: {action}", user_id=user_id, action=action, **kwargs)
    
    def log_system_event(self, event_type: str, data: Dict[str, Any]):
        """记录系统事件日志"""
        self.info(f"系统事件 - {event_type}", event_type=event_type, event_data=data)
    
    def log_conversation(self, session_id: str, message: str, sender: str, **kwargs):
        """记录对话日志"""
        self.debug(f"对话 - {session_id} [{sender}]: {message}", 
                  session_id=session_id, sender=sender, **kwargs)
    
    def log_error_with_context(self, error: Exception, context: Dict[str, Any]):
        """记录带上下文的错误日志"""
        self.error(f"错误 - {type(error).__name__}: {str(error)}", 
                  error_type=type(error).__name__,
                  error_message=str(error),
                  error_context=context)
    
    def log_agui_event(self, event_type: str, run_id: str, data: Dict[str, Any]):
        """记录AG-UI事件日志"""
        self.debug(f"AG-UI事件 - {event_type} [{run_id}]", 
                  agui_event_type=event_type,
                  agui_run_id=run_id,
                  agui_data=data)

class LogManager:
    """日志管理器"""
    
    def __init__(self, log_dir: str = "/workspace/norma_agent/logs"):
        """初始化日志管理器
        
        Args:
            log_dir: 日志目录
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.loggers: Dict[str, NormaLogger] = {}
        self.log_stats = {
            "total_logs": 0,
            "error_logs": 0,
            "warning_logs": 0,
            "info_logs": 0,
            "debug_logs": 0
        }
        
        # 创建主日志记录器
        self.main_logger = self.get_logger("norma_agent")
        
        # 启动日志清理任务
        self._start_cleanup_task()
    
    def get_logger(self, name: str, log_level: str = "INFO") -> NormaLogger:
        """获取日志记录器"""
        
        if name not in self.loggers:
            log_file = self.log_dir / f"{name}.log"
            self.loggers[name] = NormaLogger(name, log_level, str(log_file))
        
        return self.loggers[name]
    
    def _start_cleanup_task(self):
        """启动日志清理任务"""
        def cleanup_task():
            asyncio.run(self._cleanup_old_logs())
        
        # 每小时清理一次旧日志
        threading.Timer(3600, cleanup_task).start()
    
    async def _cleanup_old_logs(self):
        """清理旧日志文件"""
        try:
            cutoff_time = datetime.now().timestamp() - (30 * 24 * 3600)  # 30天前
            
            for log_file in self.log_dir.glob("*.log*"):
                if log_file.stat().st_mtime < cutoff_time:
                    log_file.unlink()
                    self.main_logger.info(f"删除旧日志文件: {log_file}")
            
        except Exception as e:
            self.main_logger.error(f"清理日志文件失败: {e}")
    
    def get_log_stats(self) -> Dict[str, Any]:
        """获取日志统计"""
        
        stats = {
            "total_loggers": len(self.loggers),
            "log_directory": str(self.log_dir),
            "log_files": [],
            "timestamp": datetime.now().isoformat()
        }
        
        # 统计日志文件
        for log_file in self.log_dir.glob("*.log"):
            try:
                file_stat = log_file.stat()
                stats["log_files"].append({
                    "name": log_file.name,
                    "size": file_stat.st_size,
                    "modified": datetime.fromtimestamp(file_stat.st_mtime).isoformat()
                })
            except Exception:
                continue
        
        return stats
    
    async def search_logs(
        self,
        logger_name: Optional[str] = None,
        level: Optional[str] = None,
        keyword: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """搜索日志"""
        
        results = []
        
        # 确定要搜索的日志文件
        if logger_name:
            log_files = [self.log_dir / f"{logger_name}.log"]
        else:
            log_files = list(self.log_dir.glob("*.log"))
        
        for log_file in log_files:
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if len(results) >= limit:
                            break
                        
                        # 解析日志行
                        log_entry = self._parse_log_line(line)
                        if not log_entry:
                            continue
                        
                        # 应用过滤器
                        if level and log_entry.get('level') != level:
                            continue
                        
                        if keyword and keyword.lower() not in log_entry.get('message', '').lower():
                            continue
                        
                        if start_time and log_entry.get('timestamp') < start_time.isoformat():
                            continue
                        
                        if end_time and log_entry.get('timestamp') > end_time.isoformat():
                            continue
                        
                        results.append(log_entry)
                        
            except Exception as e:
                self.main_logger.error(f"读取日志文件失败 {log_file}: {e}")
        
        # 按时间排序
        results.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return results[:limit]
    
    def _parse_log_line(self, line: str) -> Optional[Dict[str, Any]]:
        """解析日志行"""
        
        try:
            # 尝试解析JSON格式
            if line.strip().startswith('{'):
                return json.loads(line.strip())
            
            # 解析文本格式 [时间] 名称 级别 消息
            import re
            pattern = r'\[([^\]]+)\]\s+(\w+)\s+(\w+)\s+(.+)'
            match = re.match(pattern, line.strip())
            
            if match:
                timestamp_str, logger_name, level, message = match.groups()
                
                return {
                    "timestamp": timestamp_str,
                    "logger": logger_name,
                    "level": level,
                    "message": message.strip()
                }
            
        except Exception:
            pass
        
        return None
    
    def export_logs(
        self,
        logger_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        format: str = "json"
    ) -> str:
        """导出日志"""
        
        # 获取日志数据
        logs = asyncio.run(self.search_logs(
            logger_name=logger_name,
            start_time=start_time,
            end_time=end_time,
            limit=10000
        ))
        
        if format == "json":
            return json.dumps(logs, ensure_ascii=False, indent=2)
        elif format == "csv":
            import csv
            import io
            
            output = io.StringIO()
            if logs:
                writer = csv.DictWriter(output, fieldnames=logs[0].keys())
                writer.writeheader()
                writer.writerows(logs)
            
            return output.getvalue()
        
        return str(logs)
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        
        return {
            "component": "log_manager",
            "log_directory": str(self.log_dir),
            "active_loggers": len(self.loggers),
            "log_stats": self.log_stats,
            "timestamp": datetime.now().isoformat()
        }

# 全局日志管理器实例
_log_manager: Optional[LogManager] = None

def get_log_manager() -> LogManager:
    """获取全局日志管理器"""
    global _log_manager
    if _log_manager is None:
        _log_manager = LogManager()
    return _log_manager

def get_logger(name: str, log_level: str = "INFO") -> NormaLogger:
    """获取日志记录器的便利函数"""
    return get_log_manager().get_logger(name, log_level)

# 便利函数
def log_performance(operation: str, duration: float, **kwargs):
    """记录性能日志的便利函数"""
    logger = get_logger("performance")
    logger.log_performance(operation, duration, **kwargs)

def log_api_call(method: str, url: str, status_code: int, duration: float, **kwargs):
    """记录API调用日志的便利函数"""
    logger = get_logger("api")
    logger.log_api_call(method, url, status_code, duration, **kwargs)

def log_user_action(user_id: str, action: str, **kwargs):
    """记录用户行为日志的便利函数"""
    logger = get_logger("user_actions")
    logger.log_user_action(user_id, action, **kwargs)

def log_system_event(event_type: str, data: Dict[str, Any]):
    """记录系统事件日志的便利函数"""
    logger = get_logger("system_events")
    logger.log_system_event(event_type, data)

def log_conversation(session_id: str, message: str, sender: str, **kwargs):
    """记录对话日志的便利函数"""
    logger = get_logger("conversations")
    logger.log_conversation(session_id, message, sender, **kwargs)

def log_error_with_context(error: Exception, context: Dict[str, Any]):
    """记录带上下文的错误日志的便利函数"""
    logger = get_logger("errors")
    logger.log_error_with_context(error, context)

def log_agui_event(event_type: str, run_id: str, data: Dict[str, Any]):
    """记录AG-UI事件日志的便利函数"""
    logger = get_logger("agui_events")
    logger.log_agui_event(event_type, run_id, data)