"""
响应缓存系统
支持多层缓存、智能过期和缓存命中优化
"""

import asyncio
import json
import time
from typing import Any, Dict, Optional, Union, List
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import OrderedDict
import hashlib
import logging

from ..utils import generate_cache_key

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    ttl: Optional[float] = None
    size: int = 0
    metadata: Dict[str, Any] = None
    
    def is_expired(self, current_time: float = None) -> bool:
        """检查是否过期"""
        if current_time is None:
            current_time = time.time()
        
        if self.ttl is None:
            return False
        
        return (current_time - self.created_at) > self.ttl
    
    def access(self):
        """访问记录"""
        self.last_accessed = time.time()
        self.access_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

class LRUCache:
    """LRU缓存实现"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.total_size = 0
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        if key in self.cache:
            entry = self.cache[key]
            if not entry.is_expired():
                # 移动到末尾（最近使用）
                self.cache.move_to_end(key)
                entry.access()
                return entry.value
            else:
                # 过期则删除
                self._remove(key)
        return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None, metadata: Dict[str, Any] = None):
        """设置缓存值"""
        current_time = time.time()
        
        # 如果键已存在，更新值
        if key in self.cache:
            self._remove(key)
        
        # 创建新条目
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=current_time,
            last_accessed=current_time,
            ttl=ttl,
            metadata=metadata or {}
        )
        
        # 计算大小
        try:
            entry.size = len(json.dumps(value, default=str))
        except:
            entry.size = len(str(value))
        
        # 检查是否超过最大大小
        if entry.size > self._calculate_available_space():
            self._evict_lru()
        
        # 添加到缓存
        self.cache[key] = entry
        self.total_size += entry.size
    
    def _remove(self, key: str):
        """移除缓存条目"""
        if key in self.cache:
            entry = self.cache[key]
            self.total_size -= entry.size
            del self.cache[key]
    
    def _evict_lru(self):
        """移除最近最少使用的条目"""
        if self.cache:
            oldest_key = next(iter(self.cache))
            self._remove(oldest_key)
    
    def _calculate_available_space(self) -> int:
        """计算可用空间"""
        return max(0, self.max_size - self.total_size)
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.total_size = 0
    
    def size(self) -> int:
        """获取缓存大小"""
        return len(self.cache)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total_access = sum(entry.access_count for entry in self.cache.values())
        expired_count = sum(1 for entry in self.cache.values() if entry.is_expired())
        
        return {
            'size': self.size(),
            'total_size': self.total_size,
            'max_size': self.max_size,
            'total_access': total_access,
            'expired_count': expired_count,
            'utilization': self.total_size / self.max_size if self.max_size > 0 else 0
        }

class ResponseCache:
    """多级响应缓存系统"""
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        # 缓存配置
        self.enabled = config_dict.get('enabled', True) if config_dict else True
        self.max_size = config_dict.get('max_size', 1000) if config_dict else 1000
        self.ttl = config_dict.get('ttl', 3600) if config_dict else 3600
        self.strategy = config_dict.get('strategy', 'lru') if config_dict else 'lru'
        
        # 缓存存储
        self.l1_cache = LRUCache(self.max_size)  # L1缓存（内存）
        self.l2_cache: Dict[str, Any] = {}  # L2缓存（持久化）
        
        # 统计信息
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0
        }
        
        # 缓存键前缀
        self.key_prefix = "llm_cache:"
        
        logger.info(f"响应缓存初始化完成 - 策略: {self.strategy}, 最大大小: {self.max_size}, TTL: {self.ttl}s")
    
    def _normalize_key(self, key: str) -> str:
        """标准化缓存键"""
        return f"{self.key_prefix}{key}"
    
    def _generate_key(self, request_data: Dict[str, Any]) -> str:
        """生成缓存键"""
        return generate_cache_key(request_data)
    
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        if not self.enabled:
            return None
        
        self.stats['total_requests'] += 1
        normalized_key = self._normalize_key(key)
        
        # 尝试L1缓存
        value = self.l1_cache.get(normalized_key)
        if value is not None:
            self.stats['hits'] += 1
            logger.debug(f"L1缓存命中: {key}")
            return value
        
        # 尝试L2缓存
        if normalized_key in self.l2_cache:
            entry_data = self.l2_cache[normalized_key]
            entry = CacheEntry(**entry_data)
            
            if not entry.is_expired():
                # L2缓存命中，移动到L1
                self.l1_cache.put(normalized_key, entry.value, entry.ttl, entry.metadata)
                self.stats['hits'] += 1
                logger.debug(f"L2缓存命中: {key}")
                return entry.value
            else:
                # 过期则删除
                del self.l2_cache[normalized_key]
        
        # 缓存未命中
        self.stats['misses'] += 1
        logger.debug(f"缓存未命中: {key}")
        return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """设置缓存值"""
        if not self.enabled:
            return False
        
        normalized_key = self._normalize_key(key)
        actual_ttl = ttl or self.ttl
        
        # 设置L1缓存
        self.l1_cache.put(normalized_key, value, actual_ttl, metadata)
        
        # 设置L2缓存（持久化）
        try:
            entry = CacheEntry(
                key=normalized_key,
                value=value,
                created_at=time.time(),
                last_accessed=time.time(),
                ttl=actual_ttl,
                metadata=metadata or {}
            )
            self.l2_cache[normalized_key] = entry.to_dict()
        except Exception as e:
            logger.warning(f"L2缓存设置失败: {e}")
        
        logger.debug(f"缓存设置完成: {key}")
        return True
    
    async def delete(self, key: str) -> bool:
        """删除缓存值"""
        if not self.enabled:
            return False
        
        normalized_key = self._normalize_key(key)
        
        # 从L1缓存删除
        if normalized_key in self.l1_cache.cache:
            self.l1_cache._remove(normalized_key)
        
        # 从L2缓存删除
        if normalized_key in self.l2_cache:
            del self.l2_cache[normalized_key]
        
        return True
    
    async def clear(self):
        """清空所有缓存"""
        self.l1_cache.clear()
        self.l2_cache.clear()
        logger.info("缓存已清空")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        l1_stats = self.l1_cache.get_stats()
        
        return {
            'enabled': self.enabled,
            'strategy': self.strategy,
            'l1_cache': l1_stats,
            'l2_cache': {
                'size': len(self.l2_cache),
                'keys': list(self.l2_cache.keys())
            },
            'global_stats': self.stats.copy(),
            'hit_rate': (
                self.stats['hits'] / self.stats['total_requests'] 
                if self.stats['total_requests'] > 0 else 0
            )
        }
    
    async def cleanup_expired(self):
        """清理过期缓存"""
        current_time = time.time()
        expired_keys = []
        
        # 清理L1缓存
        for key, entry in list(self.l1_cache.cache.items()):
            if entry.is_expired(current_time):
                expired_keys.append(key)
        
        for key in expired_keys:
            self.l1_cache._remove(key)
        
        # 清理L2缓存
        expired_l2_keys = []
        for key, entry_data in list(self.l2_cache.items()):
            entry = CacheEntry(**entry_data)
            if entry.is_expired(current_time):
                expired_l2_keys.append(key)
        
        for key in expired_l2_keys:
            del self.l2_cache[key]
        
        if expired_keys or expired_l2_keys:
            logger.info(f"清理过期缓存: L1={len(expired_keys)}, L2={len(expired_l2_keys)}")
    
    def enable(self):
        """启用缓存"""
        self.enabled = True
        logger.info("缓存已启用")
    
    def disable(self):
        """禁用缓存"""
        self.enabled = False
        logger.info("缓存已禁用")
    
    async def warm_up(self, cache_data: List[Dict[str, Any]]):
        """预热缓存"""
        logger.info(f"开始预热缓存，共{len(cache_data)}条数据")
        
        for data in cache_data:
            try:
                key = data.get('key')
                value = data.get('value')
                ttl = data.get('ttl')
                metadata = data.get('metadata')
                
                if key and value is not None:
                    await self.set(key, value, ttl, metadata)
            except Exception as e:
                logger.warning(f"预热缓存失败: {e}")
        
        logger.info("缓存预热完成")
    
    def get_cache_key_for_request(
        self,
        model_name: str,
        prompt: str,
        parameters: Dict[str, Any],
        media_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """为请求生成缓存键"""
        request_data = {
            'model': model_name,
            'prompt': prompt,
            'parameters': parameters,
            'media': media_data
        }
        
        return self._generate_key(request_data)
    
    async def batch_get(self, keys: List[str]) -> Dict[str, Any]:
        """批量获取缓存"""
        results = {}
        
        for key in keys:
            value = await self.get(key)
            if value is not None:
                results[key] = value
        
        return results
    
    async def batch_set(self, items: List[Dict[str, Any]]) -> int:
        """批量设置缓存"""
        success_count = 0
        
        for item in items:
            try:
                key = item.get('key')
                value = item.get('value')
                ttl = item.get('ttl')
                metadata = item.get('metadata')
                
                if key and value is not None:
                    if await self.set(key, value, ttl, metadata):
                        success_count += 1
            except Exception as e:
                logger.warning(f"批量设置缓存失败: {e}")
        
        return success_count