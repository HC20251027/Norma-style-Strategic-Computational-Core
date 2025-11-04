"""
知识库记忆系统配置文件
"""

from typing import Dict, Any
from .utils.tools import merge_configs

# 默认配置
DEFAULT_CONFIG = {
    # 记忆管理配置
    'memory': {
        'db_path': 'memory.db',
        'max_short_term_memories': 100,
        'long_term_threshold': 24,  # 小时
        'importance_threshold': 2,  # 1-4级别
        'cleanup_interval': 86400,  # 秒 (24小时)
        'max_memories_per_type': 10000
    },
    
    # 知识图谱配置
    'knowledge_graph': {
        'db_path': 'knowledge_graph.db',
        'max_entities': 10000,
        'max_relations': 50000,
        'auto_cleanup': True,
        'cleanup_threshold': 0.3
    },
    
    # 向量数据库配置
    'vector_db': {
        'db_path': 'vector_db.db',
        'dimension': 384,
        'index_type': 'flat',  # flat, ivf, hnsw, pq
        'max_vectors_per_collection': 100000,
        'similarity_metric': 'cosine'
    },
    
    # 语义搜索配置
    'semantic_search': {
        'default_top_k': 10,
        'similarity_threshold': 0.7,
        'max_context_length': 500,
        'cache_size': 1000,
        'enable_fuzzy_search': True,
        'fuzzy_threshold': 0.3
    },
    
    # 推理引擎配置
    'reasoning': {
        'max_reasoning_depth': 3,
        'confidence_threshold': 0.5,
        'cache_size': 500,
        'enable_transitivity': True,
        'enable_inheritance': True,
        'enable_composition': True
    },
    
    # 个性化学习配置
    'learning': {
        'learning_rate': 0.1,
        'adaptation_threshold': 0.7,
        'pattern_window': 100,
        'min_interactions_for_learning': 5,
        'preference_decay': 0.95,
        'enable_behavior_analysis': True
    },
    
    # 知识库管理配置
    'knowledge_base': {
        'confidence_threshold': 0.6,
        'max_knowledge_items': 100000,
        'auto_cleanup': True,
        'cleanup_older_than_days': 30,
        'min_confidence_for_cleanup': 0.3,
        'batch_size': 100
    },
    
    # 系统配置
    'system': {
        'log_level': 'INFO',
        'log_file': 'knowledge_system.log',
        'max_concurrent_operations': 10,
        'operation_timeout': 300,  # 秒
        'enable_metrics': True,
        'metrics_interval': 3600   # 秒
    },
    
    # 性能配置
    'performance': {
        'enable_caching': True,
        'cache_ttl': 3600,  # 秒
        'batch_processing': True,
        'batch_size': 50,
        'enable_compression': False,
        'max_memory_usage': 1024  # MB
    },
    
    # 安全配置
    'security': {
        'max_content_length': 10000,
        'allowed_content_types': ['text', 'json', 'html'],
        'enable_input_validation': True,
        'sanitize_inputs': True,
        'rate_limiting': {
            'enabled': False,
            'requests_per_minute': 60,
            'burst_size': 10
        }
    }
}

# 开发环境配置
DEV_CONFIG = merge_configs(DEFAULT_CONFIG, {
    'memory': {
        'max_short_term_memories': 50,
        'long_term_threshold': 1  # 1小时用于测试
    },
    'system': {
        'log_level': 'DEBUG',
        'enable_metrics': True
    },
    'performance': {
        'cache_ttl': 300,  # 5分钟
        'batch_size': 10
    }
})

# 生产环境配置
PROD_CONFIG = merge_configs(DEFAULT_CONFIG, {
    'memory': {
        'max_short_term_memories': 500,
        'long_term_threshold': 48
    },
    'vector_db': {
        'index_type': 'hnsw',  # 更快的搜索
        'dimension': 768  # 更精确的向量
    },
    'system': {
        'log_level': 'WARNING',
        'enable_metrics': True,
        'max_concurrent_operations': 20
    },
    'performance': {
        'enable_caching': True,
        'cache_ttl': 7200,  # 2小时
        'batch_size': 200
    },
    'security': {
        'rate_limiting': {
            'enabled': True,
            'requests_per_minute': 100,
            'burst_size': 20
        }
    }
})

# 测试环境配置
TEST_CONFIG = merge_configs(DEFAULT_CONFIG, {
    'memory': {
        'db_path': ':memory:',  # 内存数据库
        'max_short_term_memories': 10,
        'long_term_threshold': 0.1  # 6分钟
    },
    'knowledge_graph': {
        'db_path': ':memory:'
    },
    'vector_db': {
        'db_path': ':memory:',
        'dimension': 128  # 更小的向量用于测试
    },
    'system': {
        'log_level': 'ERROR',  # 减少测试输出
        'enable_metrics': False
    },
    'performance': {
        'enable_caching': False
    }
})


def get_config(environment: str = 'default') -> Dict[str, Any]:
    """
    获取配置
    
    Args:
        environment: 环境名称 ('default', 'dev', 'prod', 'test')
    
    Returns:
        配置字典
    """
    
    configs = {
        'default': DEFAULT_CONFIG,
        'dev': DEV_CONFIG,
        'prod': PROD_CONFIG,
        'test': TEST_CONFIG
    }
    
    return configs.get(environment, DEFAULT_CONFIG)


def validate_config(config: Dict[str, Any]) -> bool:
    """
    验证配置的有效性
    """
    
    required_sections = [
        'memory', 'knowledge_graph', 'vector_db', 
        'semantic_search', 'reasoning', 'learning',
        'knowledge_base', 'system'
    ]
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"配置缺少必需的部分: {section}")
    
    # 验证数值范围
    if config['memory']['max_short_term_memories'] < 1:
        raise ValueError("max_short_term_memories 必须大于0")
    
    if not (0.1 <= config['learning']['learning_rate'] <= 1.0):
        raise ValueError("learning_rate 必须在0.1-1.0之间")
    
    if not (0.1 <= config['semantic_search']['similarity_threshold'] <= 1.0):
        raise ValueError("similarity_threshold 必须在0.1-1.0之间")
    
    return True


def create_custom_config(**kwargs) -> Dict[str, Any]:
    """
    创建自定义配置
    """
    
    config = DEFAULT_CONFIG.copy()
    
    # 递归更新配置
    def deep_update(base_dict: Dict, update_dict: Dict):
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    deep_update(config, kwargs)
    
    # 验证配置
    validate_config(config)
    
    return config


# 配置模板
CONFIG_TEMPLATES = {
    'minimal': {
        'memory': {
            'db_path': 'memory.db',
            'max_short_term_memories': 50
        },
        'vector_db': {
            'db_path': 'vector_db.db',
            'dimension': 128
        },
        'system': {
            'log_level': 'INFO'
        }
    },
    
    'performance': {
        'vector_db': {
            'index_type': 'hnsw',
            'dimension': 768
        },
        'semantic_search': {
            'default_top_k': 20,
            'cache_size': 2000
        },
        'performance': {
            'enable_caching': True,
            'cache_ttl': 7200,
            'batch_size': 200
        }
    },
    
    'research': {
        'memory': {
            'max_short_term_memories': 1000,
            'long_term_threshold': 168  # 一周
        },
        'knowledge_graph': {
            'max_entities': 50000,
            'max_relations': 200000
        },
        'learning': {
            'pattern_window': 1000,
            'min_interactions_for_learning': 10
        }
    }
}


def get_config_template(template_name: str) -> Dict[str, Any]:
    """
    获取配置模板
    """
    
    if template_name not in CONFIG_TEMPLATES:
        raise ValueError(f"未知的配置模板: {template_name}")
    
    template = CONFIG_TEMPLATES[template_name]
    return merge_configs(DEFAULT_CONFIG, template)


# 环境变量支持
import os

def load_config_from_env() -> Dict[str, Any]:
    """
    从环境变量加载配置覆盖
    """
    
    config_overrides = {}
    
    # 数据库路径
    if os.getenv('KNOWLEDGE_DB_PATH'):
        config_overrides.setdefault('memory', {})['db_path'] = os.getenv('KNOWLEDGE_DB_PATH')
        config_overrides.setdefault('knowledge_graph', {})['db_path'] = os.getenv('KNOWLEDGE_DB_PATH')
        config_overrides.setdefault('vector_db', {})['db_path'] = os.getenv('KNOWLEDGE_DB_PATH')
    
    # 日志级别
    if os.getenv('LOG_LEVEL'):
        config_overrides.setdefault('system', {})['log_level'] = os.getenv('LOG_LEVEL')
    
    # 向量维度
    if os.getenv('VECTOR_DIMENSION'):
        dimension = int(os.getenv('VECTOR_DIMENSION'))
        config_overrides.setdefault('vector_db', {})['dimension'] = dimension
    
    # 学习率
    if os.getenv('LEARNING_RATE'):
        learning_rate = float(os.getenv('LEARNING_RATE'))
        config_overrides.setdefault('learning', {})['learning_rate'] = learning_rate
    
    return config_overrides


def get_merged_config(environment: str = 'default', env_overrides: bool = True) -> Dict[str, Any]:
    """
    获取合并后的配置
    
    Args:
        environment: 环境名称
        env_overrides: 是否应用环境变量覆盖
    
    Returns:
        合并后的配置字典
    """
    
    # 获取基础配置
    config = get_config(environment)
    
    # 应用环境变量覆盖
    if env_overrides:
        env_overrides_dict = load_config_from_env()
        if env_overrides_dict:
            config = merge_configs(config, env_overrides_dict)
    
    # 验证最终配置
    validate_config(config)
    
    return config