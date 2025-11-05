#!/usr/bin/env python3
"""
AI核心工具模块
提供AI智能体相关的工具函数和辅助类

作者: 皇
创建时间: 2025-11-05
"""

import json
import logging
import os
import time
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from pathlib import Path


class AgentLogger:
    """智能体专用日志器"""
    
    @staticmethod
    def setup_agent_logger(agent_name: str, level: str = "INFO") -> logging.Logger:
        """设置智能体日志器"""
        logger = logging.getLogger(f"agent.{agent_name}")
        logger.setLevel(getattr(logging, level.upper()))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'%(asctime)s - {agent_name} - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger


class AgentConfig:
    """智能体配置管理"""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.config_dir = Path("config/agents")
        self.config_file = self.config_dir / f"{agent_name}.json"
        self._config = {}
        self.load_config()
    
    def load_config(self):
        """加载智能体配置"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self._config = json.load(f)
            else:
                self._config = self._get_default_config()
                self.save_config()
        except Exception as e:
            print(f"智能体配置加载失败 {self.agent_name}: {e}")
            self._config = self._get_default_config()
    
    def save_config(self):
        """保存智能体配置"""
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"智能体配置保存失败 {self.agent_name}: {e}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "name": self.agent_name,
            "version": "1.0.0",
            "status": "active",
            "capabilities": [],
            "performance": {
                "tasks_completed": 0,
                "success_rate": 1.0,
                "avg_response_time": 0.0
            },
            "settings": {
                "max_concurrent_tasks": 1,
                "timeout": 30,
                "retry_count": 3
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def set(self, key: str, value: Any):
        """设置配置值"""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
            
        config[keys[-1]] = value
        self.save_config()
    
    def update_performance(self, success: bool, response_time: float):
        """更新性能统计"""
        perf = self._config.get("performance", {})
        perf["tasks_completed"] = perf.get("tasks_completed", 0) + 1
        
        if success:
            # 简单的成功率计算
            current_rate = perf.get("success_rate", 1.0)
            tasks = perf.get("tasks_completed", 1)
            perf["success_rate"] = (current_rate * (tasks - 1) + 1) / tasks
        else:
            current_rate = perf.get("success_rate", 1.0)
            tasks = perf.get("tasks_completed", 1)
            perf["success_rate"] = (current_rate * (tasks - 1)) / tasks
        
        # 更新平均响应时间
        current_avg = perf.get("avg_response_time", 0.0)
        perf["avg_response_time"] = (current_avg * (tasks - 1) + response_time) / tasks
        
        self._config["performance"] = perf
        self.save_config()


class TaskManager:
    """任务管理器"""
    
    def __init__(self):
        self.tasks = {}
        self.task_counter = 0
    
    def create_task(self, task_type: str, data: Dict[str, Any]) -> str:
        """创建任务"""
        self.task_counter += 1
        task_id = f"task_{self.task_counter}"
        
        self.tasks[task_id] = {
            "id": task_id,
            "type": task_type,
            "data": data,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "started_at": None,
            "completed_at": None,
            "result": None,
            "error": None
        }
        
        return task_id
    
    def start_task(self, task_id: str) -> bool:
        """开始任务"""
        if task_id in self.tasks and self.tasks[task_id]["status"] == "pending":
            self.tasks[task_id]["status"] = "in_progress"
            self.tasks[task_id]["started_at"] = datetime.now().isoformat()
            return True
        return False
    
    def complete_task(self, task_id: str, result: Any = None) -> bool:
        """完成任务"""
        if task_id in self.tasks and self.tasks[task_id]["status"] == "in_progress":
            self.tasks[task_id]["status"] = "completed"
            self.tasks[task_id]["completed_at"] = datetime.now().isoformat()
            self.tasks[task_id]["result"] = result
            return True
        return False
    
    def fail_task(self, task_id: str, error: str) -> bool:
        """任务失败"""
        if task_id in self.tasks and self.tasks[task_id]["status"] == "in_progress":
            self.tasks[task_id]["status"] = "failed"
            self.tasks[task_id]["completed_at"] = datetime.now().isoformat()
            self.tasks[task_id]["error"] = error
            return True
        return False
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务信息"""
        return self.tasks.get(task_id)
    
    def get_tasks_by_status(self, status: str) -> List[Dict[str, Any]]:
        """按状态获取任务"""
        return [task for task in self.tasks.values() if task["status"] == status]
    
    def cleanup_completed_tasks(self, max_age_hours: int = 24):
        """清理已完成的任务"""
        current_time = datetime.now()
        to_remove = []
        
        for task_id, task in self.tasks.items():
            if task["status"] in ["completed", "failed"]:
                completed_time = datetime.fromisoformat(task["completed_at"])
                if (current_time - completed_time).total_seconds() > max_age_hours * 3600:
                    to_remove.append(task_id)
        
        for task_id in to_remove:
            del self.tasks[task_id]


class CacheManager:
    """缓存管理器"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.cache_dir / "index.json"
        self._index = self._load_index()
    
    def _load_index(self) -> Dict[str, Any]:
        """加载缓存索引"""
        try:
            if self.index_file.exists():
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception:
            return {}
    
    def _save_index(self):
        """保存缓存索引"""
        try:
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(self._index, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"缓存索引保存失败: {e}")
    
    def _get_cache_key(self, key: str) -> str:
        """生成缓存键"""
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{cache_key}.json"
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        cache_key = self._get_cache_key(key)
        cache_path = self._get_cache_path(cache_key)
        
        if cache_key in self._index and cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    
                # 检查是否过期
                if "expires_at" in cache_data:
                    if datetime.fromisoformat(cache_data["expires_at"]) < datetime.now():
                        self.delete(key)
                        return None
                
                return cache_data.get("value")
            except Exception:
                return None
        
        return None
    
    def set(self, key: str, value: Any, expire_hours: int = 24):
        """设置缓存"""
        cache_key = self._get_cache_key(key)
        cache_path = self._get_cache_path(cache_key)
        
        cache_data = {
            "value": value,
            "created_at": datetime.now().isoformat(),
            "expires_at": (datetime.now().timestamp() + expire_hours * 3600)
        }
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            self._index[cache_key] = {
                "key": key,
                "created_at": cache_data["created_at"]
            }
            self._save_index()
        except Exception as e:
            print(f"缓存设置失败: {e}")
    
    def delete(self, key: str):
        """删除缓存"""
        cache_key = self._get_cache_key(key)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            if cache_path.exists():
                cache_path.unlink()
            
            if cache_key in self._index:
                del self._index[cache_key]
                self._save_index()
        except Exception as e:
            print(f"缓存删除失败: {e}")
    
    def clear(self):
        """清空缓存"""
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
            
            self._index = {}
            self._save_index()
        except Exception as e:
            print(f"缓存清空失败: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total_files = len(list(self.cache_dir.glob("*.json")))
        total_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.json") if f.is_file())
        
        return {
            "total_files": total_files,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2)
        }


class MessageBus:
    """消息总线"""
    
    def __init__(self):
        self.subscribers = {}
        self.message_history = []
        self.max_history = 1000
    
    def subscribe(self, topic: str, callback):
        """订阅主题"""
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(callback)
    
    def unsubscribe(self, topic: str, callback):
        """取消订阅"""
        if topic in self.subscribers:
            try:
                self.subscribers[topic].remove(callback)
            except ValueError:
                pass
    
    def publish(self, topic: str, message: Any):
        """发布消息"""
        message_data = {
            "topic": topic,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        
        # 添加到历史记录
        self.message_history.append(message_data)
        if len(self.message_history) > self.max_history:
            self.message_history.pop(0)
        
        # 通知订阅者
        if topic in self.subscribers:
            for callback in self.subscribers[topic]:
                try:
                    callback(message_data)
                except Exception as e:
                    print(f"消息处理错误: {e}")
    
    def get_history(self, topic: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """获取消息历史"""
        if topic:
            return [msg for msg in self.message_history if msg["topic"] == topic][-limit:]
        return self.message_history[-limit:]


# 全局实例
task_manager = TaskManager()
cache_manager = CacheManager()
message_bus = MessageBus()


# 导出常用工具
__all__ = [
    'AgentLogger',
    'AgentConfig',
    'TaskManager',
    'CacheManager',
    'MessageBus',
    'task_manager',
    'cache_manager',
    'message_bus'
]