#!/usr/bin/env python3
"""
核心工具模块
提供项目通用的工具函数和辅助类

作者: 皇
创建时间: 2025-11-05
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from pathlib import Path


class LoggerConfig:
    """日志配置"""
    
    @staticmethod
    def setup_logger(name: str = "norma_core", level: str = "INFO") -> logging.Logger:
        """设置日志器"""
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config.json"
        self._config = {}
        self.load_config()
    
    def load_config(self):
        """加载配置文件"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self._config = json.load(f)
            else:
                self._config = self._get_default_config()
                self.save_config()
        except Exception as e:
            print(f"配置加载失败: {e}")
            self._config = self._get_default_config()
    
    def save_config(self):
        """保存配置文件"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"配置保存失败: {e}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "system": {
                "name": "诺玛式战略计算中枢",
                "version": "3.1.2",
                "debug": False
            },
            "database": {
                "path": "./data/norma.db",
                "backup_path": "./data/backups/"
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8003,
                "cors_origins": ["*"]
            },
            "logging": {
                "level": "INFO",
                "file": "./logs/norma.log"
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


class FileUtils:
    """文件工具类"""
    
    @staticmethod
    def ensure_directory(path: Union[str, Path]):
        """确保目录存在"""
        Path(path).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def safe_filename(filename: str) -> str:
        """生成安全的文件名"""
        import re
        # 移除或替换不安全字符
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # 限制长度
        if len(filename) > 200:
            name, ext = os.path.splitext(filename)
            filename = name[:200-len(ext)] + ext
        return filename
    
    @staticmethod
    def get_file_size(path: Union[str, Path]) -> int:
        """获取文件大小（字节）"""
        try:
            return os.path.getsize(path)
        except OSError:
            return 0
    
    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """格式化文件大小"""
        if size_bytes == 0:
            return "0B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        import math
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_names[i]}"


class TimeUtils:
    """时间工具类"""
    
    @staticmethod
    def get_current_timestamp() -> float:
        """获取当前时间戳"""
        return time.time()
    
    @staticmethod
    def format_datetime(dt: Optional[datetime] = None, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
        """格式化日期时间"""
        if dt is None:
            dt = datetime.now()
        return dt.strftime(format_str)
    
    @staticmethod
    def parse_datetime(date_str: str, format_str: str = "%Y-%m-%d %H:%M:%S") -> Optional[datetime]:
        """解析日期时间字符串"""
        try:
            return datetime.strptime(date_str, format_str)
        except ValueError:
            return None
    
    @staticmethod
    def time_since(timestamp: float) -> str:
        """获取时间间隔描述"""
        now = time.time()
        diff = now - timestamp
        
        if diff < 60:
            return f"{int(diff)}秒前"
        elif diff < 3600:
            return f"{int(diff/60)}分钟前"
        elif diff < 86400:
            return f"{int(diff/3600)}小时前"
        else:
            return f"{int(diff/86400)}天前"


    @staticmethod
    def calculate_tokens_estimate(text: str) -> int:
        """估算文本的token数量"""
        # 简单的中英文token估算
        # 中文：每个字符约1.5个token
        # 英文：每个词约1.3个token
        import re
        
        # 计算中文字符
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        
        # 计算英文单词
        english_words = len(re.findall(r'\b\w+\b', text))
        
        # 计算其他字符
        other_chars = len(text) - chinese_chars - english_words
        
        # 估算token数
        estimated_tokens = int(chinese_chars * 1.5 + english_words * 1.3 + other_chars * 1.0)
        return max(1, estimated_tokens)


class DataValidator:
    """数据验证器"""
    
    @staticmethod
    def is_valid_email(email: str) -> bool:
        """验证邮箱格式"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    @staticmethod
    def is_valid_url(url: str) -> bool:
        """验证URL格式"""
        import re
        pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        return re.match(pattern, url) is not None
    
    @staticmethod
    def sanitize_html(text: str) -> str:
        """清理HTML标签"""
        import re
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.start_time = time.time()
        self.checkpoints = {}
    
    def checkpoint(self, name: str):
        """添加性能检查点"""
        self.checkpoints[name] = time.time()
    
    def get_elapsed_time(self) -> float:
        """获取总耗时"""
        return time.time() - self.start_time
    
    def get_checkpoint_time(self, name: str) -> Optional[float]:
        """获取检查点耗时"""
        if name in self.checkpoints:
            return time.time() - self.checkpoints[name]
        return None
    
    def print_report(self):
        """打印性能报告"""
        print(f"总耗时: {self.get_elapsed_time():.3f}秒")
        for name, checkpoint_time in self.checkpoints.items():
            elapsed = time.time() - checkpoint_time
            print(f"{name}: {elapsed:.3f}秒")


# 全局实例
config_manager = ConfigManager()
logger = LoggerConfig.setup_logger()


# 导出常用工具
__all__ = [
    'LoggerConfig',
    'ConfigManager', 
    'FileUtils',
    'TimeUtils',
    'DataValidator',
    'PerformanceMonitor',
    'config_manager',
    'logger'
]