"""
配置管理模块
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import yaml


logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    """MCP服务器配置"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    workers: int = 1
    log_level: str = "INFO"
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    rate_limit: int = 100
    request_timeout: int = 300


@dataclass
class ToolRegistryConfig:
    """工具注册中心配置"""
    auto_register: bool = True
    validation_enabled: bool = True
    cache_enabled: bool = True
    cache_ttl: int = 300
    max_tools: int = 1000
    cleanup_interval: int = 3600


@dataclass
class SecurityConfig:
    """安全配置"""
    enabled: bool = True
    default_policy: str = "moderate"
    rate_limiting: bool = True
    max_requests_per_minute: int = 60
    max_failed_attempts: int = 5
    block_duration_minutes: int = 15
    audit_logging: bool = True
    encryption_enabled: bool = False


@dataclass
class DatabaseConfig:
    """数据库配置"""
    type: str = "sqlite"  # sqlite, postgresql, mysql
    host: str = "localhost"
    port: int = 5432
    database: str = "mcp_tools"
    username: str = ""
    password: str = ""
    url: str = ""
    pool_size: int = 10
    echo: bool = False


@dataclass
class CacheConfig:
    """缓存配置"""
    enabled: bool = True
    type: str = "memory"  # memory, redis, memcached
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str = ""
    ttl: int = 300
    max_size: int = 1000


@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    console_output: bool = True


@dataclass
class MCPConfig:
    """MCP系统配置"""
    server: MCPServerConfig = field(default_factory=MCPServerConfig)
    tool_registry: ToolRegistryConfig = field(default_factory=ToolRegistryConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    version: str = "1.0.0"
    environment: str = "development"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "server": self.server.__dict__,
            "tool_registry": self.tool_registry.__dict__,
            "security": self.security.__dict__,
            "database": self.database.__dict__,
            "cache": self.cache.__dict__,
            "logging": self.logging.__dict__,
            "version": self.version,
            "environment": self.environment
        }


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()
        self.config: Optional[MCPConfig] = None
        self._load_config()
    
    def _get_default_config_path(self) -> str:
        """获取默认配置路径"""
        # 尝试从环境变量获取
        env_config = os.getenv("MCP_CONFIG_PATH")
        if env_config:
            return env_config
        
        # 默认路径
        return str(Path(__file__).parent.parent.parent / "config" / "mcp_config.yaml")
    
    def _load_config(self):
        """加载配置"""
        try:
            if os.path.exists(self.config_path):
                logger.info(f"加载配置文件: {self.config_path}")
                self.config = self._load_from_file(self.config_path)
            else:
                logger.info("配置文件不存在，使用默认配置")
                self.config = self._create_default_config()
                self.save_config()
        except Exception as e:
            logger.error(f"加载配置失败: {e}")
            self.config = self._create_default_config()
    
    def _load_from_file(self, config_path: str) -> MCPConfig:
        """从文件加载配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config_data = yaml.safe_load(f)
            else:
                config_data = json.load(f)
        
        return self._dict_to_config(config_data)
    
    def _dict_to_config(self, data: Dict[str, Any]) -> MCPConfig:
        """字典转配置对象"""
        return MCPConfig(
            server=MCPServerConfig(**data.get("server", {})),
            tool_registry=ToolRegistryConfig(**data.get("tool_registry", {})),
            security=SecurityConfig(**data.get("security", {})),
            database=DatabaseConfig(**data.get("database", {})),
            cache=CacheConfig(**data.get("cache", {})),
            logging=LoggingConfig(**data.get("logging", {})),
            version=data.get("version", "1.0.0"),
            environment=data.get("environment", "development")
        )
    
    def _create_default_config(self) -> MCPConfig:
        """创建默认配置"""
        return MCPConfig()
    
    def save_config(self, config_path: Optional[str] = None):
        """保存配置"""
        save_path = config_path or self.config_path
        
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            config_dict = self.config.to_dict()
            
            with open(save_path, 'w', encoding='utf-8') as f:
                if save_path.endswith('.yaml') or save_path.endswith('.yml'):
                    yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
                else:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"配置已保存到: {save_path}")
            
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
    
    def get_config(self) -> MCPConfig:
        """获取配置"""
        if self.config is None:
            self._load_config()
        return self.config
    
    def update_config(self, updates: Dict[str, Any]):
        """更新配置"""
        if self.config is None:
            self.config = self._create_default_config()
        
        # 更新配置
        for key, value in updates.items():
            if hasattr(self.config, key):
                if isinstance(value, dict) and hasattr(getattr(self.config, key), '__dict__'):
                    # 更新嵌套配置对象
                    current_obj = getattr(self.config, key)
                    for sub_key, sub_value in value.items():
                        if hasattr(current_obj, sub_key):
                            setattr(current_obj, sub_key, sub_value)
                else:
                    setattr(self.config, key, value)
        
        logger.info("配置已更新")
    
    def get_server_config(self) -> MCPServerConfig:
        """获取服务器配置"""
        return self.get_config().server
    
    def get_security_config(self) -> SecurityConfig:
        """获取安全配置"""
        return self.get_config().security
    
    def get_database_config(self) -> DatabaseConfig:
        """获取数据库配置"""
        return self.get_config().database
    
    def get_cache_config(self) -> CacheConfig:
        """获取缓存配置"""
        return self.get_config().cache
    
    def get_logging_config(self) -> LoggingConfig:
        """获取日志配置"""
        return self.get_config().logging
    
    def reload_config(self):
        """重新加载配置"""
        self._load_config()
        logger.info("配置已重新加载")
    
    def validate_config(self) -> List[str]:
        """验证配置"""
        errors = []
        
        config = self.get_config()
        
        # 验证服务器配置
        if not isinstance(config.server.port, int) or config.server.port <= 0:
            errors.append("服务器端口必须是正整数")
        
        # 验证安全配置
        if config.security.max_requests_per_minute <= 0:
            errors.append("最大请求频率必须是正数")
        
        # 验证数据库配置
        if config.database.type not in ["sqlite", "postgresql", "mysql"]:
            errors.append("数据库类型必须是 sqlite、postgresql 或 mysql")
        
        # 验证缓存配置
        if config.cache.type not in ["memory", "redis", "memcached"]:
            errors.append("缓存类型必须是 memory、redis 或 memcached")
        
        return errors
    
    def export_config(self, export_path: str, format: str = "yaml"):
        """导出配置"""
        config_dict = self.get_config().to_dict()
        
        with open(export_path, 'w', encoding='utf-8') as f:
            if format.lower() == "yaml":
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
            else:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"配置已导出到: {export_path}")


# 环境变量映射
ENV_MAPPING = {
    "MCP_HOST": ("server", "host"),
    "MCP_PORT": ("server", "port"),
    "MCP_DEBUG": ("server", "debug"),
    "MCP_LOG_LEVEL": ("logging", "level"),
    "MCP_DB_TYPE": ("database", "type"),
    "MCP_DB_HOST": ("database", "host"),
    "MCP_DB_PORT": ("database", "port"),
    "MCP_DB_NAME": ("database", "database"),
    "MCP_DB_USER": ("database", "username"),
    "MCP_DB_PASSWORD": ("database", "password"),
    "MCP_CACHE_TYPE": ("cache", "type"),
    "MCP_CACHE_HOST": ("cache", "host"),
    "MCP_CACHE_PORT": ("cache", "port"),
    "MCP_CACHE_TTL": ("cache", "ttl"),
    "MCP_SECURITY_ENABLED": ("security", "enabled"),
    "MCP_RATE_LIMIT": ("security", "max_requests_per_minute"),
}


def load_config_from_env() -> Dict[str, Any]:
    """从环境变量加载配置"""
    config_updates = {}
    
    for env_var, (section, key) in ENV_MAPPING.items():
        env_value = os.getenv(env_var)
        if env_value is not None:
            if section not in config_updates:
                config_updates[section] = {}
            
            # 类型转换
            if env_value.lower() in ["true", "false"]:
                config_updates[section][key] = env_value.lower() == "true"
            elif env_value.isdigit():
                config_updates[section][key] = int(env_value)
            else:
                config_updates[section][key] = env_value
    
    return config_updates


# 全局配置管理器实例
config_manager = ConfigManager()


def get_config() -> MCPConfig:
    """获取全局配置"""
    return config_manager.get_config()


def update_config(updates: Dict[str, Any]):
    """更新全局配置"""
    # 从环境变量加载配置
    env_updates = load_config_from_env()
    updates.update(env_updates)
    
    # 更新配置
    config_manager.update_config(updates)