"""
语音服务配置
"""

# 默认配置
DEFAULT_CONFIG = {
    "asr": {
        "default_engine": "whisper",
        "engines": {
            "whisper": {
                "model_size": "base",
                "device": "cpu",
                "language": "zh-CN"
            },
            "azure_speech": {
                "subscription_key": "your_azure_key",
                "region": "your_region",
                "language": "zh-CN"
            }
        }
    },
    "tts": {
        "default_engine": "edge_tts",
        "engines": {
            "edge_tts": {
                "default_voice": "zh-CN-XiaoxiaoNeural"
            },
            "azure_speech": {
                "subscription_key": "your_azure_key",
                "region": "your_region",
                "default_voice": "zh-CN-XiaoxiaoNeural"
            }
        }
    },
    "emotion": {
        "emotion_configs": {
            "happy": {
                "intensity": 0.8,
                "pitch_shift": 2.0,
                "speed_factor": 1.1,
                "volume_factor": 1.2,
                "breath_pause": 0.1,
                "emphasis_words": ["太好了", "非常", "真的", "太棒了"]
            },
            "sad": {
                "intensity": 0.7,
                "pitch_shift": -3.0,
                "speed_factor": 0.8,
                "volume_factor": 0.7,
                "breath_pause": 0.3,
                "emphasis_words": ["遗憾", "可惜", "难过"]
            }
        }
    },
    "streaming": {
        "chunk_size": 1024,
        "sample_rate": 16000,
        "channels": 1,
        "buffer_size": 30,
        "overlap_size": 512,
        "processing_interval": 0.1,
        "enable_vad": True,
        "vad_threshold": 0.5,
        "max_latency": 0.5
    },
    "correction": {
        "language": "zh-CN",
        "confidence_threshold": 0.7,
        "enable_spelling": True,
        "enable_homophone": True,
        "enable_context": True
    },
    "optimization": {
        "enable_cache": True,
        "cache_size": 1000,
        "cache_ttl": 3600,
        "enable_batch": True,
        "batch_size": 10,
        "batch_timeout": 1.0,
        "enable_parallel": True,
        "max_workers": 4,
        "enable_compression": True,
        "compression_level": 6,
        "adaptive_sampling": True,
        "performance_threshold": 0.8
    }
}

# 语音配置
VOICE_CONFIGS = {
    "zh-CN": {
        "edge_tts": {
            "voices": [
                {"id": "zh-CN-XiaoxiaoNeural", "name": "晓晓", "gender": "Female"},
                {"id": "zh-CN-YunyangNeural", "name": "云扬", "gender": "Male"},
                {"id": "zh-CN-XiaoyiNeural", "name": "晓伊", "gender": "Female"},
                {"id": "zh-CN-YunjianNeural", "name": "云健", "gender": "Male"}
            ]
        },
        "azure_speech": {
            "voices": [
                {"id": "zh-CN-XiaoxiaoNeural", "name": "晓晓", "gender": "Female"},
                {"id": "zh-CN-YunyangNeural", "name": "云扬", "gender": "Male"},
                {"id": "zh-CN-XiaoyiNeural", "name": "晓伊", "gender": "Female"},
                {"id": "zh-CN-YunjianNeural", "name": "云健", "gender": "Male"}
            ]
        }
    },
    "en-US": {
        "edge_tts": {
            "voices": [
                {"id": "en-US-AriaNeural", "name": "Aria", "gender": "Female"},
                {"id": "en-US-GuyNeural", "name": "Guy", "gender": "Male"},
                {"id": "en-US-JennyNeural", "name": "Jenny", "gender": "Female"},
                {"id": "en-US-DavisNeural", "name": "Davis", "gender": "Male"}
            ]
        },
        "azure_speech": {
            "voices": [
                {"id": "en-US-AriaNeural", "name": "Aria", "gender": "Female"},
                {"id": "en-US-GuyNeural", "name": "Guy", "gender": "Male"},
                {"id": "en-US-JennyNeural", "name": "Jenny", "gender": "Female"},
                {"id": "en-US-DavisNeural", "name": "Davis", "gender": "Male"}
            ]
        }
    }
}

# 情感配置
EMOTION_CONFIGS = {
    "default": {
        "happy": {
            "intensity": 0.8,
            "pitch_shift": 2.0,
            "speed_factor": 1.1,
            "volume_factor": 1.2,
            "breath_pause": 0.1,
            "emphasis_words": ["太好了", "非常", "真的", "太棒了"]
        },
        "sad": {
            "intensity": 0.7,
            "pitch_shift": -3.0,
            "speed_factor": 0.8,
            "volume_factor": 0.7,
            "breath_pause": 0.3,
            "emphasis_words": ["遗憾", "可惜", "难过"]
        },
        "angry": {
            "intensity": 0.9,
            "pitch_shift": 1.0,
            "speed_factor": 1.3,
            "volume_factor": 1.5,
            "breath_pause": 0.05,
            "emphasis_words": ["绝对", "必须", "一定", "坚决"]
        },
        "neutral": {
            "intensity": 0.5,
            "pitch_shift": 0.0,
            "speed_factor": 1.0,
            "volume_factor": 1.0,
            "breath_pause": 0.2,
            "emphasis_words": []
        }
    },
    "children": {
        "happy": {
            "intensity": 1.0,
            "pitch_shift": 3.0,
            "speed_factor": 1.2,
            "volume_factor": 1.3,
            "breath_pause": 0.05,
            "emphasis_words": ["哇", "太棒了", "好厉害"]
        },
        "neutral": {
            "intensity": 0.6,
            "pitch_shift": 1.0,
            "speed_factor": 1.1,
            "volume_factor": 1.1,
            "breath_pause": 0.15,
            "emphasis_words": []
        }
    },
    "professional": {
        "neutral": {
            "intensity": 0.4,
            "pitch_shift": -1.0,
            "speed_factor": 0.9,
            "volume_factor": 1.0,
            "breath_pause": 0.3,
            "emphasis_words": ["请", "建议", "可以"]
        },
        "calm": {
            "intensity": 0.3,
            "pitch_shift": -1.0,
            "speed_factor": 0.9,
            "volume_factor": 0.9,
            "breath_pause": 0.4,
            "emphasis_words": ["请", "建议", "可以", "也许"]
        }
    }
}

# 性能配置
PERFORMANCE_CONFIGS = {
    "high_quality": {
        "enable_cache": True,
        "cache_size": 2000,
        "cache_ttl": 7200,
        "enable_batch": False,
        "enable_parallel": True,
        "max_workers": 2,
        "adaptive_sampling": False,
        "performance_threshold": 0.9
    },
    "balanced": {
        "enable_cache": True,
        "cache_size": 1000,
        "cache_ttl": 3600,
        "enable_batch": True,
        "batch_size": 10,
        "enable_parallel": True,
        "max_workers": 4,
        "adaptive_sampling": True,
        "performance_threshold": 0.8
    },
    "high_speed": {
        "enable_cache": True,
        "cache_size": 500,
        "cache_ttl": 1800,
        "enable_batch": True,
        "batch_size": 20,
        "enable_parallel": True,
        "max_workers": 8,
        "adaptive_sampling": True,
        "performance_threshold": 0.7
    }
}


def get_config(config_type: str = "default") -> dict:
    """获取配置"""
    if config_type == "default":
        return DEFAULT_CONFIG.copy()
    elif config_type in PERFORMANCE_CONFIGS:
        base_config = DEFAULT_CONFIG.copy()
        base_config["optimization"].update(PERFORMANCE_CONFIGS[config_type])
        return base_config
    else:
        return DEFAULT_CONFIG.copy()


def get_voice_config(language: str = "zh-CN") -> dict:
    """获取语音配置"""
    return VOICE_CONFIGS.get(language, VOICE_CONFIGS["zh-CN"])


def get_emotion_config(profile: str = "default") -> dict:
    """获取情感配置"""
    return EMOTION_CONFIGS.get(profile, EMOTION_CONFIGS["default"])


# 环境变量配置
import os

def load_config_from_env() -> dict:
    """从环境变量加载配置"""
    config = DEFAULT_CONFIG.copy()
    
    # Azure配置
    if os.getenv("AZURE_SPEECH_KEY"):
        config["asr"]["engines"]["azure_speech"]["subscription_key"] = os.getenv("AZURE_SPEECH_KEY")
        config["tts"]["engines"]["azure_speech"]["subscription_key"] = os.getenv("AZURE_SPEECH_KEY")
    
    if os.getenv("AZURE_SPEECH_REGION"):
        config["asr"]["engines"]["azure_speech"]["region"] = os.getenv("AZURE_SPEECH_REGION")
        config["tts"]["engines"]["azure_speech"]["region"] = os.getenv("AZURE_SPEECH_REGION")
    
    # 性能配置
    if os.getenv("SPEECH_CACHE_SIZE"):
        config["optimization"]["cache_size"] = int(os.getenv("SPEECH_CACHE_SIZE"))
    
    if os.getenv("SPEECH_MAX_WORKERS"):
        config["optimization"]["max_workers"] = int(os.getenv("SPEECH_MAX_WORKERS"))
    
    return config