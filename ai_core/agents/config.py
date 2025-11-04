"""
配置文件

定义异步任务机制的配置选项
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum

from .scheduler import SchedulingStrategy
from .cache_manager import CacheStrategy
from .monitor import MonitorConfig


class LogLevel(Enum):
    """日志级别"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class AsyncTaskConfig:
    """异步任务机制主配置"""
    
    # 核心设置
    max_concurrent: int = 10                    # 最大并发任务数
    default_timeout: Optional[int] = 3600       # 默认超时时间（秒）
    default_retry_count: int = 3                # 默认重试次数
    default_retry_delay: float = 1.0           # 默认重试延迟（秒）
    
    # 调度设置
    scheduling_strategy: SchedulingStrategy = SchedulingStrategy.PRIORITY
    enable_dependency_resolution: bool = True   # 启用依赖解析
    fair_scheduling: bool = False              # 公平调度
    
    # 缓存设置
    cache_enabled: bool = True                 # 启用缓存
    cache_size: int = 1000                     # 缓存大小
    cache_strategy: CacheStrategy = CacheStrategy.LRU
    cache_ttl: Optional[int] = 3600            # 缓存生存时间（秒）
    cache_persistence: bool = True             # 缓存持久化
    
    # 存储设置
    storage_enabled: bool = True               # 启用存储
    storage_type: str = "sqlite"               # 存储类型: sqlite, json
    storage_path: str = "tasks.db"             # 存储路径
    auto_backup: bool = True                   # 自动备份
    backup_interval: int = 3600                # 备份间隔（秒）
    backup_retention: int = 7                  # 备份保留天数
    cleanup_old_tasks: bool = True             # 清理旧任务
    cleanup_interval: int = 86400              # 清理间隔（秒）
    cleanup_retention_days: int = 30           # 清理保留天数
    
    # 监控设置
    monitoring_enabled: bool = True            # 启用监控
    monitor_config: MonitorConfig = field(default_factory=MonitorConfig)
    
    # 日志设置
    log_level: LogLevel = LogLevel.INFO
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[str] = None            # 日志文件路径
    log_max_size: int = 10 * 1024 * 1024      # 日志文件最大大小（字节）
    log_backup_count: int = 5                 # 日志文件备份数量
    
    # 性能设置
    enable_metrics_collection: bool = True     # 启用指标收集
    metrics_retention: int = 3600              # 指标保留时间（秒）
    enable_performance_optimization: bool = True  # 启用性能优化
    
    # 安全设置
    enable_task_isolation: bool = True         # 启用任务隔离
    max_task_memory_mb: int = 512              # 任务最大内存限制（MB）
    enable_resource_monitoring: bool = True    # 启用资源监控
    
    # 通知设置
    enable_notifications: bool = False         # 启用通知
    notification_channels: List[str] = field(default_factory=list)  # 通知渠道
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "cpu_usage": 80.0,
        "memory_usage": 85.0,
        "disk_usage": 90.0,
        "task_failure_rate": 10.0,
        "avg_task_duration": 300.0,
        "queue_size": 1000
    })
    
    # 扩展设置
    custom_settings: Dict[str, Any] = field(default_factory=dict)  # 自定义设置
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                result[key] = value.value
            elif hasattr(value, 'to_dict'):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AsyncTaskConfig':
        """从字典创建配置"""
        # 处理枚举类型
        if 'log_level' in data and isinstance(data['log_level'], str):
            data['log_level'] = LogLevel(data['log_level'])
        
        if 'scheduling_strategy' in data and isinstance(data['scheduling_strategy'], str):
            data['scheduling_strategy'] = SchedulingStrategy(data['scheduling_strategy'])
        
        if 'cache_strategy' in data and isinstance(data['cache_strategy'], str):
            data['cache_strategy'] = CacheStrategy(data['cache_strategy'])
        
        # 处理MonitorConfig
        if 'monitor_config' in data and isinstance(data['monitor_config'], dict):
            data['monitor_config'] = MonitorConfig(**data['monitor_config'])
        
        return cls(**data)
    
    def validate(self) -> List[str]:
        """验证配置"""
        errors = []
        
        if self.max_concurrent <= 0:
            errors.append("最大并发数必须大于0")
        
        if self.cache_size <= 0:
            errors.append("缓存大小必须大于0")
        
        if self.default_retry_count < 0:
            errors.append("默认重试次数不能为负数")
        
        if self.default_retry_delay < 0:
            errors.append("默认重试延迟不能为负数")
        
        if self.backup_interval <= 0:
            errors.append("备份间隔必须大于0")
        
        if self.cleanup_interval <= 0:
            errors.append("清理间隔必须大于0")
        
        if self.cleanup_retention_days <= 0:
            errors.append("清理保留天数必须大于0")
        
        if self.metrics_retention <= 0:
            errors.append("指标保留时间必须大于0")
        
        if self.max_task_memory_mb <= 0:
            errors.append("任务最大内存限制必须大于0")
        
        # 验证阈值
        for threshold_name, threshold_value in self.alert_thresholds.items():
            if threshold_value < 0:
                errors.append(f"告警阈值 {threshold_name} 不能为负数")
        
        return errors


# 预定义配置模板
class ConfigTemplates:
    """配置模板"""
    
    @staticmethod
    def development() -> AsyncTaskConfig:
        """开发环境配置"""
        return AsyncTaskConfig(
            max_concurrent=5,
            cache_size=100,
            log_level=LogLevel.DEBUG,
            monitoring_enabled=True,
            storage_type="json",  # 开发环境使用简单的JSON存储
            auto_backup=False,
            cleanup_old_tasks=False
        )
    
    @staticmethod
    def testing() -> AsyncTaskConfig:
        """测试环境配置"""
        return AsyncTaskConfig(
            max_concurrent=3,
            cache_size=50,
            log_level=LogLevel.WARNING,
            monitoring_enabled=False,
            storage_type="json",
            auto_backup=False,
            cleanup_old_tasks=False,
            enable_notifications=False
        )
    
    @staticmethod
    def production() -> AsyncTaskConfig:
        """生产环境配置"""
        return AsyncTaskConfig(
            max_concurrent=20,
            cache_size=5000,
            log_level=LogLevel.INFO,
            monitoring_enabled=True,
            storage_type="sqlite",
            auto_backup=True,
            backup_interval=1800,  # 30分钟备份一次
            cleanup_old_tasks=True,
            cleanup_interval=3600,  # 1小时清理一次
            enable_notifications=True,
            notification_channels=["email", "webhook"]
        )
    
    @staticmethod
    def high_performance() -> AsyncTaskConfig:
        """高性能配置"""
        return AsyncTaskConfig(
            max_concurrent=50,
            cache_size=10000,
            cache_strategy=CacheStrategy.LFU,
            monitoring_enabled=True,
            enable_performance_optimization=True,
            enable_metrics_collection=True,
            metrics_retention=7200,  # 2小时
            storage_type="sqlite",
            auto_backup=True,
            backup_interval=900  # 15分钟备份一次
        )
    
    @staticmethod
    def low_resource() -> AsyncTaskConfig:
        """低资源环境配置"""
        return AsyncTaskConfig(
            max_concurrent=3,
            cache_size=100,
            cache_strategy=CacheStrategy.FIFO,
            monitoring_enabled=False,
            storage_type="json",
            auto_backup=False,
            cleanup_old_tasks=True,
            cleanup_interval=7200,  # 2小时清理一次
            max_task_memory_mb=256
        )


# 配置管理器
class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self._config: Optional[AsyncTaskConfig] = None
    
    def load_config(self, config_file: Optional[str] = None) -> AsyncTaskConfig:
        """加载配置"""
        import json
        import os
        
        config_path = config_file or self.config_file
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                self._config = AsyncTaskConfig.from_dict(config_data)
            except Exception as e:
                print(f"加载配置文件失败: {e}")
                self._config = AsyncTaskConfig()
        else:
            self._config = AsyncTaskConfig()
        
        # 验证配置
        errors = self._config.validate()
        if errors:
            print(f"配置验证失败: {errors}")
        
        return self._config
    
    def save_config(self, config: AsyncTaskConfig, config_file: Optional[str] = None):
        """保存配置"""
        import json
        import os
        
        config_path = config_file or self.config_file
        
        if not config_path:
            raise ValueError("配置文件路径未指定")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存配置文件失败: {e}")
    
    def get_config(self) -> AsyncTaskConfig:
        """获取当前配置"""
        if self._config is None:
            self._config = self.load_config()
        return self._config
    
    def update_config(self, **kwargs) -> AsyncTaskConfig:
        """更新配置"""
        if self._config is None:
            self._config = AsyncTaskConfig()
        
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
            else:
                self._config.custom_settings[key] = value
        
        return self._config


# 全局配置实例
default_config_manager = ConfigManager()


def get_default_config() -> AsyncTaskConfig:
    """获取默认配置"""
    return default_config_manager.get_config()


def set_default_config(config: AsyncTaskConfig):
    """设置默认配置"""
    default_config_manager._config = config


def load_config_from_file(config_file: str) -> AsyncTaskConfig:
    """从文件加载配置"""
    manager = ConfigManager(config_file)
    return manager.load_config()


# 环境变量支持
def load_config_from_env() -> AsyncTaskConfig:
    """从环境变量加载配置"""
    import os
    
    config = AsyncTaskConfig()
    
    # 从环境变量更新配置
    if os.getenv('ASYNC_TASK_MAX_CONCURRENT'):
        config.max_concurrent = int(os.getenv('ASYNC_TASK_MAX_CONCURRENT'))
    
    if os.getenv('ASYNC_TASK_CACHE_SIZE'):
        config.cache_size = int(os.getenv('ASYNC_TASK_CACHE_SIZE'))
    
    if os.getenv('ASYNC_TASK_LOG_LEVEL'):
        config.log_level = LogLevel(os.getenv('ASYNC_TASK_LOG_LEVEL'))
    
    if os.getenv('ASYNC_TASK_STORAGE_TYPE'):
        config.storage_type = os.getenv('ASYNC_TASK_STORAGE_TYPE')
    
    if os.getenv('ASYNC_TASK_STORAGE_PATH'):
        config.storage_path = os.getenv('ASYNC_TASK_STORAGE_PATH')
    
    if os.getenv('ASYNC_TASK_MONITORING_ENABLED'):
        config.monitoring_enabled = os.getenv('ASYNC_TASK_MONITORING_ENABLED').lower() == 'true'
    
    return config