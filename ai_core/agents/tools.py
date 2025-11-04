"""
工具函数模块
提供知识库记忆系统的通用工具函数
"""

import asyncio
import numpy as np
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import hashlib
import pickle
import logging


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """设置日志配置"""
    
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    return logging.getLogger(__name__)


def generate_embedding(text: str, dimension: int = 384) -> List[float]:
    """
    生成文本嵌入向量
    简化实现，实际应用中应使用预训练模型
    """
    
    # 清理文本
    text = text.lower().strip()
    
    # 创建向量
    vector = np.zeros(dimension)
    
    # 简单的词袋模型
    words = text.split()
    
    for i, word in enumerate(words):
        # 使用哈希将词映射到向量位置
        hash_value = int(hashlib.md5(word.encode()).hexdigest(), 16)
        position = hash_value % dimension
        
        # 词频加权
        word_freq = words.count(word)
        vector[position] += word_freq
    
    # 归一化向量
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    
    return vector.tolist()


def calculate_similarity(
    vector1: Union[List[float], np.ndarray],
    vector2: Union[List[float], np.ndarray],
    metric: str = "cosine"
) -> float:
    """
    计算两个向量的相似度
    """
    
    # 确保输入是numpy数组
    if isinstance(vector1, list):
        vector1 = np.array(vector1, dtype=np.float32)
    if isinstance(vector2, list):
        vector2 = np.array(vector2, dtype=np.float32)
    
    if metric == "cosine":
        # 余弦相似度
        dot_product = np.dot(vector1, vector2)
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
    
    elif metric == "euclidean":
        # 欧几里得距离
        distance = np.linalg.norm(vector1 - vector2)
        # 转换为相似度分数
        similarity = 1 / (1 + distance)
        return float(similarity)
    
    elif metric == "manhattan":
        # 曼哈顿距离
        distance = np.sum(np.abs(vector1 - vector2))
        similarity = 1 / (1 + distance)
        return float(similarity)
    
    else:
        raise ValueError(f"不支持的相似度度量: {metric}")


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    提取关键词
    """
    
    # 停用词列表
    stop_words = {
        '的', '了', '是', '在', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这',
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
    }
    
    # 清理文本
    import re
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    words = re.findall(r'\b\w+\b', text)
    
    # 过滤停用词和短词
    keywords = [
        word for word in words 
        if len(word) > 2 and word not in stop_words
    ]
    
    # 统计词频
    from collections import Counter
    word_counts = Counter(keywords)
    
    # 返回最高频的关键词
    return [word for word, count in word_counts.most_common(max_keywords)]


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    计算两个文本的相似度
    """
    
    # 提取关键词
    keywords1 = set(extract_keywords(text1))
    keywords2 = set(extract_keywords(text2))
    
    if not keywords1 and not keywords2:
        return 1.0
    if not keywords1 or not keywords2:
        return 0.0
    
    # 计算Jaccard相似度
    intersection = keywords1.intersection(keywords2)
    union = keywords1.union(keywords2)
    
    jaccard_similarity = len(intersection) / len(union)
    
    return jaccard_similarity


def normalize_score(score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """
    归一化分数到指定范围
    """
    
    return max(min_val, min(max_val, score))


def calculate_confidence(
    evidence_count: int,
    consistency_score: float,
    source_reliability: float = 0.8
) -> float:
    """
    计算置信度分数
    """
    
    # 基于证据数量的置信度
    evidence_factor = min(1.0, evidence_count / 5.0)  # 5个证据达到满分
    
    # 综合置信度
    confidence = (evidence_factor * 0.4 + consistency_score * 0.4 + source_reliability * 0.2)
    
    return normalize_score(confidence)


def format_timestamp(timestamp: datetime) -> str:
    """
    格式化时间戳
    """
    
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def parse_timestamp(timestamp_str: str) -> datetime:
    """
    解析时间戳字符串
    """
    
    return datetime.fromisoformat(timestamp_str)


def create_unique_id(prefix: str, content: str) -> str:
    """
    创建唯一ID
    """
    
    timestamp = datetime.now().isoformat()
    combined_content = f"{prefix}:{content}:{timestamp}"
    
    return hashlib.md5(combined_content.encode()).hexdigest()


def batch_process(items: List[Any], process_func, batch_size: int = 100) -> List[Any]:
    """
    批量处理项目
    """
    
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = asyncio.run(process_func(batch))
        results.extend(batch_results)
    
    return results


async def async_batch_process(items: List[Any], process_func, batch_size: int = 100) -> List[Any]:
    """
    异步批量处理项目
    """
    
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = await process_func(batch)
        results.extend(batch_results)
    
    return results


def measure_time(func):
    """
    装饰器：测量函数执行时间
    """
    
    async def async_wrapper(*args, **kwargs):
        start_time = datetime.now()
        result = await func(*args, **kwargs)
        end_time = datetime.now()
        
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"函数 {func.__name__} 执行时间: {execution_time:.3f} 秒")
        
        return result
    
    def sync_wrapper(*args, **kwargs):
        start_time = datetime.now()
        result = func(*args, **kwargs)
        end_time = datetime.now()
        
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"函数 {func.__name__} 执行时间: {execution_time:.3f} 秒")
        
        return result
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


def validate_embedding(vector: List[float], expected_dimension: int = 384) -> bool:
    """
    验证嵌入向量
    """
    
    if not isinstance(vector, list):
        return False
    
    if len(vector) != expected_dimension:
        return False
    
    try:
        vector_array = np.array(vector, dtype=np.float32)
        return not np.isnan(vector_array).any()
    except:
        return False


def merge_metadata(*metadata_dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并元数据字典
    """
    
    merged = {}
    
    for metadata in metadata_dicts:
        if isinstance(metadata, dict):
            merged.update(metadata)
    
    return merged


def filter_by_confidence(items: List[Dict[str, Any]], min_confidence: float) -> List[Dict[str, Any]]:
    """
    按置信度过滤项目
    """
    
    return [
        item for item in items 
        if item.get('confidence', 0.0) >= min_confidence
    ]


def sort_by_relevance(items: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """
    按相关性排序项目
    """
    
    def relevance_score(item):
        content = item.get('content', '')
        return calculate_text_similarity(query, content)
    
    return sorted(items, key=relevance_score, reverse=True)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    将文本分块
    """
    
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # 尝试在句子边界处分割
        if end < len(text):
            for i in range(end, max(start, end - 100), -1):
                if text[i] in '.!?。！？':
                    end = i + 1
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # 设置下一个块的起始位置（包含重叠）
        start = max(start + 1, end - overlap)
    
    return chunks


def calculate_reading_time(text: str, words_per_minute: int = 200) -> float:
    """
    计算阅读时间（分钟）
    """
    
    word_count = len(text.split())
    return word_count / words_per_minute


def generate_summary(text: str, max_length: int = 200) -> str:
    """
    生成文本摘要（简化实现）
    """
    
    sentences = text.split('.')
    
    if len(sentences) <= 3:
        return text[:max_length] + "..." if len(text) > max_length else text
    
    # 选择前几句作为摘要
    summary_sentences = sentences[:2]
    summary = '.'.join(summary_sentences)
    
    if len(summary) > max_length:
        summary = summary[:max_length] + "..."
    
    return summary


def log_performance_metrics(metrics: Dict[str, Any], logger_instance: logging.Logger = None):
    """
    记录性能指标
    """
    
    if logger_instance is None:
        logger_instance = logger
    
    logger_instance.info("性能指标:")
    for key, value in metrics.items():
        if isinstance(value, float):
            logger_instance.info(f"  {key}: {value:.3f}")
        else:
            logger_instance.info(f"  {key}: {value}")


def create_progress_bar(current: int, total: int, width: int = 50) -> str:
    """
    创建进度条
    """
    
    if total == 0:
        return "[" + "=" * width + "] 0%"
    
    progress = current / total
    filled_width = int(progress * width)
    
    bar = "[" + "=" * filled_width + " " * (width - filled_width) + "]"
    percentage = int(progress * 100)
    
    return f"{bar} {percentage}%"


def validate_config(config: Dict[str, Any], required_keys: List[str]) -> bool:
    """
    验证配置字典
    """
    
    for key in required_keys:
        if key not in config:
            logger.error(f"缺少必需的配置项: {key}")
            return False
    
    return True


def safe_get_nested_value(data: Dict[str, Any], path: str, default: Any = None) -> Any:
    """
    安全获取嵌套字典值
    """
    
    keys = path.split('.')
    current = data
    
    try:
        for key in keys:
            current = current[key]
        return current
    except (KeyError, TypeError):
        return default


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并配置字典
    """
    
    merged = {}
    
    for config in configs:
        if isinstance(config, dict):
            merged.update(config)
    
    return merged