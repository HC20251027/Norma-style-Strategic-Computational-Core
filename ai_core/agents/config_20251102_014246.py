"""
LLM多模态核心配置管理
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

class ModelType(Enum):
    """模型类型枚举"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    MULTIMODAL = "multimodal"

class Provider(Enum):
    """LLM提供商枚举"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    COHERE = "cohere"
    LOCAL = "local"
    CUSTOM = "custom"

@dataclass
class ModelConfig:
    """模型配置"""
    name: str
    provider: Provider
    model_type: ModelType
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    supports_streaming: bool = True
    supports_multimodal: bool = False
    max_concurrent_requests: int = 10
    rate_limit: int = 100  # 每分钟请求数
    timeout: int = 30
    priority: int = 1  # 优先级，数字越大优先级越高

@dataclass
class CacheConfig:
    """缓存配置"""
    enabled: bool = True
    max_size: int = 1000
    ttl: int = 3600  # 缓存过期时间（秒）
    strategy: str = "lru"  # lru, fifo, lfu

@dataclass
class LoadBalancerConfig:
    """负载均衡配置"""
    algorithm: str = "round_robin"  # round_robin, weighted_round_robin, least_connections
    health_check_interval: int = 30
    failure_threshold: int = 3
    recovery_threshold: int = 2

@dataclass
class StreamingConfig:
    """流式配置"""
    chunk_size: int = 1024
    buffer_size: int = 4096
    enable_compression: bool = True
    max_connections: int = 100

class LLMConfig:
    """LLM配置管理类"""
    
    def __init__(self):
        self.models: Dict[str, ModelConfig] = {}
        self.cache = CacheConfig()
        self.load_balancer = LoadBalancerConfig()
        self.streaming = StreamingConfig()
        self._load_default_models()
    
    def _load_default_models(self):
        """加载默认模型配置"""
        # OpenAI模型
        self.models["gpt-4"] = ModelConfig(
            name="gpt-4",
            provider=Provider.OPENAI,
            model_type=ModelType.MULTIMODAL,
            max_tokens=8192,
            supports_streaming=True,
            supports_multimodal=True,
            priority=5
        )
        
        self.models["gpt-3.5-turbo"] = ModelConfig(
            name="gpt-3.5-turbo",
            provider=Provider.OPENAI,
            model_type=ModelType.TEXT,
            max_tokens=4096,
            supports_streaming=True,
            priority=4
        )
        
        # Anthropic模型
        self.models["claude-3-opus"] = ModelConfig(
            name="claude-3-opus",
            provider=Provider.ANTHROPIC,
            model_type=ModelType.MULTIMODAL,
            max_tokens=8192,
            supports_streaming=True,
            supports_multimodal=True,
            priority=5
        )
        
        self.models["claude-3-sonnet"] = ModelConfig(
            name="claude-3-sonnet",
            provider=Provider.ANTHROPIC,
            model_type=ModelType.MULTIMODAL,
            max_tokens=4096,
            supports_streaming=True,
            supports_multimodal=True,
            priority=4
        )
        
        # Google模型
        self.models["gemini-pro"] = ModelConfig(
            name="gemini-pro",
            provider=Provider.GOOGLE,
            model_type=ModelType.MULTIMODAL,
            max_tokens=4096,
            supports_streaming=True,
            supports_multimodal=True,
            priority=3
        )
        
        # 本地模型
        self.models["local-llm"] = ModelConfig(
            name="local-llm",
            provider=Provider.LOCAL,
            model_type=ModelType.TEXT,
            max_tokens=2048,
            supports_streaming=True,
            priority=2
        )
    
    def get_model(self, name: str) -> Optional[ModelConfig]:
        """获取模型配置"""
        return self.models.get(name)
    
    def get_models_by_type(self, model_type: ModelType) -> List[ModelConfig]:
        """根据类型获取模型列表"""
        return [model for model in self.models.values() if model.model_type == model_type]
    
    def get_models_by_provider(self, provider: Provider) -> List[ModelConfig]:
        """根据提供商获取模型列表"""
        return [model for model in self.models.values() if model.provider == provider]
    
    def add_model(self, config: ModelConfig):
        """添加模型配置"""
        self.models[config.name] = config
    
    def remove_model(self, name: str):
        """移除模型配置"""
        if name in self.models:
            del self.models[name]
    
    def update_api_key(self, name: str, api_key: str):
        """更新模型API密钥"""
        if name in self.models:
            self.models[name].api_key = api_key
    
    def get_available_models(self, model_type: Optional[ModelType] = None) -> List[str]:
        """获取可用模型列表"""
        if model_type:
            return [model.name for model in self.get_models_by_type(model_type)]
        return list(self.models.keys())

# 全局配置实例
config = LLMConfig()