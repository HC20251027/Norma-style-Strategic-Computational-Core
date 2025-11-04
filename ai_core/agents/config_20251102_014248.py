"""
语音流水线配置文件

包含默认配置和自定义配置选项
"""

from typing import Dict, Any
from .models import PipelineConfig


def get_default_config() -> PipelineConfig:
    """获取默认配置"""
    return PipelineConfig(
        # 语音识别配置
        speech_recognition={
            "engine": "whisper",  # whisper, speech_recognition, azure, mock
            "language": "zh-CN",
            "confidence_threshold": 0.7,
            "sample_rate": 16000,
            "max_audio_duration": 300  # 5分钟
        },
        
        # 文本处理配置
        text_processing={
            "intent_recognition": True,
            "entity_extraction": True,
            "context_awareness": True,
            "max_text_length": 1000,
            "supported_languages": ["zh-CN", "en-US", "ja-JP", "ko-KR"]
        },
        
        # 工具映射配置
        tool_mapping={
            "max_tools_per_request": 5,
            "timeout_seconds": 30,
            "retry_attempts": 3,
            "enable_parallel_execution": True,
            "max_concurrent_tools": 3
        },
        
        # 语音合成配置
        text_to_speech={
            "engine": "pyttsx3",  # pyttsx3, gTTS, azure, espeak, mock
            "voice": "zh",
            "rate": 150,
            "volume": 0.8,
            "output_format": "wav",  # wav, mp3
            "sample_rate": 16000
        },
        
        # 性能配置
        performance={
            "max_concurrent_requests": 10,
            "cache_results": True,
            "enable_monitoring": True,
            "cleanup_interval_hours": 24,
            "max_history_size": 1000
        },
        
        # 错误处理配置
        error_handling={
            "max_retries": 3,
            "retry_delay": 1.0,
            "fallback_enabled": True,
            "enable_error_reporting": True,
            "log_level": "INFO"
        }
    )


def get_development_config() -> PipelineConfig:
    """获取开发环境配置"""
    config = get_default_config()
    
    # 开发环境特定配置
    config.speech_recognition["engine"] = "mock"  # 使用模拟引擎避免依赖
    config.text_to_speech["engine"] = "mock"  # 使用模拟引擎
    config.performance["max_concurrent_requests"] = 3  # 减少并发数
    config.error_handling["log_level"] = "DEBUG"  # 详细日志
    
    return config


def get_production_config() -> PipelineConfig:
    """获取生产环境配置"""
    config = get_default_config()
    
    # 生产环境特定配置
    config.speech_recognition["engine"] = "whisper"  # 使用Whisper
    config.text_to_speech["engine"] = "gTTS"  # 使用Google TTS
    config.performance["max_concurrent_requests"] = 20  # 增加并发数
    config.error_handling["log_level"] = "WARNING"  # 减少日志输出
    
    return config


def get_test_config() -> PipelineConfig:
    """获取测试环境配置"""
    config = get_default_config()
    
    # 测试环境特定配置
    config.speech_recognition["engine"] = "mock"
    config.text_to_speech["engine"] = "mock"
    config.performance["max_concurrent_requests"] = 1
    config.error_handling["max_retries"] = 1  # 减少重试次数
    config.error_handling["fallback_enabled"] = True
    
    return config


def create_custom_config(**kwargs) -> PipelineConfig:
    """创建自定义配置"""
    config = get_default_config()
    
    # 更新配置
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config


# 预定义的工具配置
TOOL_CONFIGS = {
    "basic": {
        "weather": {
            "description": "查询天气信息",
            "parameters": {
                "location": {"type": "str", "required": True},
                "date": {"type": "str", "required": False}
            },
            "keywords": ["天气", "气温", "下雨", "晴天", "温度"]
        },
        "time": {
            "description": "查询时间",
            "parameters": {},
            "keywords": ["时间", "几点", "日期", "今天", "明天"]
        },
        "calculator": {
            "description": "执行计算",
            "parameters": {
                "expression": {"type": "str", "required": True}
            },
            "keywords": ["计算", "等于", "加", "减", "乘", "除"]
        }
    },
    
    "extended": {
        "weather": {
            "description": "查询天气信息",
            "parameters": {
                "location": {"type": "str", "required": True},
                "date": {"type": "str", "required": False}
            },
            "keywords": ["天气", "气温", "下雨", "晴天", "温度"]
        },
        "search": {
            "description": "搜索信息",
            "parameters": {
                "query": {"type": "str", "required": True},
                "max_results": {"type": "int", "required": False}
            },
            "keywords": ["搜索", "查找", "找", "查询"]
        },
        "calculator": {
            "description": "执行计算",
            "parameters": {
                "expression": {"type": "str", "required": True}
            },
            "keywords": ["计算", "等于", "加", "减", "乘", "除"]
        },
        "time": {
            "description": "查询时间",
            "parameters": {},
            "keywords": ["时间", "几点", "日期", "今天", "明天"]
        },
        "system_info": {
            "description": "获取系统信息",
            "parameters": {
                "info_type": {"type": "str", "required": False}
            },
            "keywords": ["系统", "信息", "状态", "监控"]
        },
        "network_check": {
            "description": "网络检查",
            "parameters": {
                "target": {"type": "str", "required": False}
            },
            "keywords": ["网络", "连接", "ping", "检查"]
        },
        "file_operation": {
            "description": "文件操作",
            "parameters": {
                "operation": {"type": "str", "required": True},
                "path": {"type": "str", "required": True}
            },
            "keywords": ["文件", "创建", "删除", "读取", "写入"]
        },
        "translation": {
            "description": "翻译文本",
            "parameters": {
                "text": {"type": "str", "required": True},
                "target_language": {"type": "str", "required": True}
            },
            "keywords": ["翻译", "英文", "中文", "日语", "韩语"]
        }
    }
}


# 语音引擎配置
SPEECH_ENGINE_CONFIGS = {
    "whisper": {
        "model_size": "base",  # tiny, base, small, medium, large
        "device": "cpu",  # cpu, cuda
        "compute_type": "int8"  # int8, float16, float32
    },
    
    "speech_recognition": {
        "service": "google",  # google, sphinx, wit, azure, houndify
        "language": "zh-CN",
        "show_all": False
    },
    
    "azure": {
        "subscription_key": "",
        "region": "eastus",
        "language": "zh-CN",
        "voice": "zh-CN-XiaoxiaoNeural"
    }
}


# 语音合成引擎配置
TTS_ENGINE_CONFIGS = {
    "pyttsx3": {
        "driver_name": "sapi5",  # espeak, nsss, say
        "voice_id": None,
        "rate": 150,
        "volume": 0.8
    },
    
    "gTTS": {
        "language": "zh-cn",
        "slow": False,
        "tld": "com"  # com, com.au, co.uk
    },
    
    "azure": {
        "subscription_key": "",
        "region": "eastus",
        "voice": "zh-CN-XiaoxiaoNeural",
        "style": "general",
        "rate": "0%",
        "volume": "0%"
    },
    
    "espeak": {
        "voice": "zh",
        "speed": 150,
        "amplitude": 200,
        "pitch": 50
    }
}