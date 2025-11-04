#!/usr/bin/env python3
"""
配置管理模块
提供系统配置加载、保存和验证功能

作者: 皇
创建时间: 2025-10-31
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime

from ..utils.logger import get_logger

@dataclass
class AgentConfig:
    """Agent配置"""
    name: str = "诺玛·劳恩斯"
    version: str = "1.0.0"
    personality: str = "卡塞尔学院主控计算机AI系统"
    max_conversation_length: int = 100
    context_window: int = 50
    response_timeout: int = 30

@dataclass
class FeaturesConfig:
    """功能配置"""
    multimodal: bool = True
    agui_protocol: bool = True
    real_time_events: bool = True
    memory_management: bool = True
    context_awareness: bool = True
    analytics: bool = True
    streaming: bool = True

@dataclass
class PerformanceConfig:
    """性能配置"""
    max_concurrent_sessions: int = 10
    max_concurrent_processing: int = 5
    response_timeout: int = 30
    event_buffer_size: int = 1000
    cache_ttl: int = 1800  # 30分钟

@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    enable_file_logging: bool = True
    log_file_path: str = "/workspace/norma_agent/logs/agent.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    enable_performance_logging: bool = True

@dataclass
class DatabaseConfig:
    """数据库配置"""
    memory_db_path: str = "/workspace/norma_agent/data/memory.db"
    conversation_db_path: str = "/workspace/norma_agent/data/conversations.db"
    backup_enabled: bool = True
    backup_interval: int = 3600  # 1小时
    max_backup_files: int = 24

@dataclass
class UIConfig:
    """用户界面配置"""
    theme: str = "light"
    language: str = "zh-CN"
    show_timestamps: bool = True
    show_sender_avatar: bool = True
    auto_save: bool = True
    streaming_enabled: bool = True
    max_messages_per_page: int = 50

@dataclass
class SecurityConfig:
    """安全配置"""
    enable_authentication: bool = False
    session_timeout: int = 3600  # 1小时
    max_login_attempts: int = 3
    enable_audit_log: bool = True
    data_encryption: bool = False

@dataclass
class NormaConfig:
    """诺玛系统主配置"""
    agent: AgentConfig = None
    features: FeaturesConfig = None
    performance: PerformanceConfig = None
    logging: LoggingConfig = None
    database: DatabaseConfig = None
    ui: UIConfig = None
    security: SecurityConfig = None
    custom: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.agent is None:
            self.agent = AgentConfig()
        if self.features is None:
            self.features = FeaturesConfig()
        if self.performance is None:
            self.performance = PerformanceConfig()
        if self.logging is None:
            self.logging = LoggingConfig()
        if self.database is None:
            self.database = DatabaseConfig()
        if self.ui is None:
            self.ui = UIConfig()
        if self.security is None:
            self.security = SecurityConfig()
        if self.custom is None:
            self.custom = {}

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_dir: str = "/workspace/norma_agent/config"):
        """初始化配置管理器
        
        Args:
            config_dir: 配置目录
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = get_logger("config_manager")
        
        # 配置文件路径
        self.config_file = self.config_dir / "norma_config.json"
        self.backup_dir = self.config_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        
        # 当前配置
        self.config: Optional[NormaConfig] = None
        
        # 配置验证规则
        self.validation_rules = {
            "agent.max_conversation_length": {"min": 10, "max": 1000},
            "agent.context_window": {"min": 10, "max": 200},
            "performance.max_concurrent_sessions": {"min": 1, "max": 100},
            "performance.max_concurrent_processing": {"min": 1, "max": 50},
            "ui.max_messages_per_page": {"min": 10, "max": 200},
            "security.session_timeout": {"min": 300, "max": 86400}  # 5分钟到24小时
        }
    
    def load_config(self, config_file: Optional[str] = None) -> NormaConfig:
        """加载配置"""
        
        file_path = Path(config_file) if config_file else self.config_file
        
        try:
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                self.config = self._dict_to_config(config_data)
                self.logger.info(f"配置已从 {file_path} 加载")
            else:
                # 使用默认配置
                self.config = NormaConfig()
                self.logger.info("使用默认配置")
                
                # 保存默认配置
                self.save_config()
            
            # 验证配置
            validation_errors = self.validate_config(self.config)
            if validation_errors:
                self.logger.warning(f"配置验证警告: {validation_errors}")
            
            return self.config
            
        except Exception as e:
            self.logger.error(f"加载配置失败: {e}")
            # 返回默认配置
            self.config = NormaConfig()
            return self.config
    
    def save_config(self, config_file: Optional[str] = None) -> bool:
        """保存配置"""
        
        if not self.config:
            self.logger.error("没有配置可保存")
            return False
        
        file_path = Path(config_file) if config_file else self.config_file
        
        try:
            # 创建备份
            if file_path.exists():
                self._create_backup(file_path)
            
            # 转换配置为字典
            config_dict = self._config_to_dict(self.config)
            
            # 保存配置
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"配置已保存到 {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存配置失败: {e}")
            return False
    
    def _create_backup(self, file_path: Path) -> None:
        """创建配置备份"""
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.backup_dir / f"{file_path.stem}_{timestamp}.json"
            
            import shutil
            shutil.copy2(file_path, backup_file)
            
            # 清理旧备份
            self._cleanup_old_backups()
            
            self.logger.debug(f"配置备份已创建: {backup_file}")
            
        except Exception as e:
            self.logger.error(f"创建配置备份失败: {e}")
    
    def _cleanup_old_backups(self) -> None:
        """清理旧备份文件"""
        
        try:
            backup_files = list(self.backup_dir.glob("norma_config_*.json"))
            backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # 保留最新的10个备份
            for backup_file in backup_files[10:]:
                backup_file.unlink()
                self.logger.debug(f"删除旧备份: {backup_file}")
                
        except Exception as e:
            self.logger.error(f"清理旧备份失败: {e}")
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> NormaConfig:
        """将字典转换为配置对象"""
        
        return NormaConfig(
            agent=AgentConfig(**config_dict.get("agent", {})),
            features=FeaturesConfig(**config_dict.get("features", {})),
            performance=PerformanceConfig(**config_dict.get("performance", {})),
            logging=LoggingConfig(**config_dict.get("logging", {})),
            database=DatabaseConfig(**config_dict.get("database", {})),
            ui=UIConfig(**config_dict.get("ui", {})),
            security=SecurityConfig(**config_dict.get("security", {})),
            custom=config_dict.get("custom", {})
        )
    
    def _config_to_dict(self, config: NormaConfig) -> Dict[str, Any]:
        """将配置对象转换为字典"""
        
        return {
            "agent": asdict(config.agent),
            "features": asdict(config.features),
            "performance": asdict(config.performance),
            "logging": asdict(config.logging),
            "database": asdict(config.database),
            "ui": asdict(config.ui),
            "security": asdict(config.security),
            "custom": config.custom
        }
    
    def validate_config(self, config: NormaConfig) -> List[str]:
        """验证配置"""
        
        errors = []
        config_dict = self._config_to_dict(config)
        
        # 遍历验证规则
        for key_path, rule in self.validation_rules.items():
            value = self._get_nested_value(config_dict, key_path)
            
            if value is not None:
                if "min" in rule and value < rule["min"]:
                    errors.append(f"{key_path} 值 {value} 小于最小值 {rule['min']}")
                
                if "max" in rule and value > rule["max"]:
                    errors.append(f"{key_path} 值 {value} 大于最大值 {rule['max']}")
        
        # 检查必需的配置项
        required_paths = [
            "agent.name",
            "agent.version",
            "logging.level",
            "database.memory_db_path"
        ]
        
        for path in required_paths:
            value = self._get_nested_value(config_dict, path)
            if not value:
                errors.append(f"必需配置项 {path} 为空")
        
        return errors
    
    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """获取嵌套字典值"""
        
        keys = path.split(".")
        value = data
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """更新配置"""
        
        if not self.config:
            self.logger.error("没有加载的配置")
            return False
        
        try:
            # 深度更新配置
            self._deep_update(self.config, updates)
            
            # 验证更新后的配置
            validation_errors = self.validate_config(self.config)
            if validation_errors:
                self.logger.warning(f"配置更新后验证警告: {validation_errors}")
            
            # 保存更新后的配置
            return self.save_config()
            
        except Exception as e:
            self.logger.error(f"更新配置失败: {e}")
            return False
    
    def _deep_update(self, config_obj: Any, updates: Dict[str, Any]) -> None:
        """深度更新配置对象"""
        
        for key, value in updates.items():
            if hasattr(config_obj, key):
                current_value = getattr(config_obj, key)
                
                if isinstance(value, dict) and isinstance(current_value, object):
                    # 如果是嵌套对象，递归更新
                    if hasattr(current_value, '__dict__'):
                        self._deep_update(current_value, value)
                    else:
                        setattr(config_obj, key, value)
                else:
                    setattr(config_obj, key, value)
            else:
                # 处理自定义配置
                if hasattr(config_obj, 'custom') and isinstance(config_obj.custom, dict):
                    config_obj.custom[key] = value
    
    def reset_to_default(self) -> bool:
        """重置为默认配置"""
        
        try:
            self.config = NormaConfig()
            return self.save_config()
            
        except Exception as e:
            self.logger.error(f"重置配置失败: {e}")
            return False
    
    def export_config(self, format: str = "json", output_file: Optional[str] = None) -> str:
        """导出配置"""
        
        if not self.config:
            return "{}"
        
        config_dict = self._config_to_dict(self.config)
        
        if format == "json":
            output = json.dumps(config_dict, ensure_ascii=False, indent=2)
        elif format == "yaml":
            output = yaml.dump(config_dict, default_flow_style=False, allow_unicode=True)
        else:
            output = str(config_dict)
        
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(output)
                self.logger.info(f"配置已导出到 {output_file}")
            except Exception as e:
                self.logger.error(f"导出配置失败: {e}")
        
        return output
    
    def import_config(self, config_file: str) -> bool:
        """导入配置"""
        
        try:
            file_path = Path(config_file)
            if not file_path.exists():
                self.logger.error(f"配置文件不存在: {config_file}")
                return False
            
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix.lower() == '.yaml':
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
            
            # 验证导入的配置
            temp_config = self._dict_to_config(config_data)
            validation_errors = self.validate_config(temp_config)
            
            if validation_errors:
                self.logger.error(f"导入的配置验证失败: {validation_errors}")
                return False
            
            # 应用配置
            self.config = temp_config
            return self.save_config()
            
        except Exception as e:
            self.logger.error(f"导入配置失败: {e}")
            return False
    
    def get_config_summary(self) -> Dict[str, Any]:
        """获取配置摘要"""
        
        if not self.config:
            return {"status": "not_loaded"}
        
        config_dict = self._config_to_dict(self.config)
        
        return {
            "status": "loaded",
            "agent_name": config_dict["agent"]["name"],
            "version": config_dict["agent"]["version"],
            "features_enabled": sum(1 for v in config_dict["features"].values() if v),
            "max_sessions": config_dict["performance"]["max_concurrent_sessions"],
            "log_level": config_dict["logging"]["level"],
            "ui_theme": config_dict["ui"]["theme"],
            "custom_settings": len(config_dict.get("custom", {})),
            "config_file": str(self.config_file),
            "last_modified": datetime.fromtimestamp(self.config_file.stat().st_mtime).isoformat() if self.config_file.exists() else None
        }
    
    def list_config_files(self) -> List[Dict[str, Any]]:
        """列出配置文件"""
        
        files = []
        
        # 主配置文件
        if self.config_file.exists():
            stat = self.config_file.stat()
            files.append({
                "name": self.config_file.name,
                "path": str(self.config_file),
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "type": "main"
            })
        
        # 备份文件
        for backup_file in self.backup_dir.glob("*.json"):
            stat = backup_file.stat()
            files.append({
                "name": backup_file.name,
                "path": str(backup_file),
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "type": "backup"
            })
        
        return sorted(files, key=lambda x: x["modified"], reverse=True)
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        
        return {
            "component": "config_manager",
            "config_loaded": self.config is not None,
            "config_file": str(self.config_file),
            "config_exists": self.config_file.exists(),
            "backup_count": len(list(self.backup_dir.glob("*.json"))),
            "validation_rules": len(self.validation_rules),
            "timestamp": datetime.now().isoformat()
        }

# 全局配置管理器实例
_config_manager: Optional[ConfigManager] = None

def get_config_manager() -> ConfigManager:
    """获取全局配置管理器"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def load_config(config_file: Optional[str] = None) -> NormaConfig:
    """加载配置的便利函数"""
    return get_config_manager().load_config(config_file)

def save_config(config_file: Optional[str] = None) -> bool:
    """保存配置的便利函数"""
    return get_config_manager().save_config(config_file)

def get_config() -> Optional[NormaConfig]:
    """获取当前配置的便利函数"""
    return get_config_manager().config