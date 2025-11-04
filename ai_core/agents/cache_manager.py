"""
缓存管理器

负责任务结果的缓存、过期管理、缓存策略等功能
"""

import asyncio
import hashlib
import json
import pickle
import time
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, Union, Callable
from functools import wraps
import logging
from enum import Enum

from .task_models import Task, TaskResult


class CacheStrategy(Enum):
    """缓存策略"""
    NO_CACHE = "no_cache"           # 不缓存
    LRU = "lru"                     # LRU缓存
    TTL = "ttl"                     # TTL缓存
    LFU = "lfu"                     # LFU缓存
    FIFO = "fifo"                   # FIFO缓存


class CacheEntry:
    """缓存条目"""
    
    def __init__(self, key: str, value: Any, ttl: Optional[int] = None):
        self.key = key
        self.value = value
        self.created_at = time.time()
        self.last_accessed = time.time()
        self.access_count = 0
        self.ttl = ttl
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def access(self):
        """访问缓存"""
        self.last_accessed = time.time()
        self.access_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "key": self.key,
            "value": self.value,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "ttl": self.ttl
        }


class CacheManager:
    """缓存管理器"""
    
    def __init__(
        self, 
        max_size: int = 1000,
        strategy: CacheStrategy = CacheStrategy.LRU,
        default_ttl: Optional[int] = None,
        enable_persistence: bool = False,
        persist_path: str = "task_cache.json"
    ):
        self.max_size = max_size
        self.strategy = strategy
        self.default_ttl = default_ttl
        self.enable_persistence = enable_persistence
        self.persist_path = persist_path
        
        self.logger = logging.getLogger(__name__)
        
        # 缓存存储
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: list = []  # 用于LRU
        
        # 统计信息
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        
        # 锁
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        async with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._misses += 1
                return None
            
            if entry.is_expired():
                await self._remove(key)
                self._misses += 1
                return None
            
            # 更新访问信息
            entry.access()
            if self.strategy == CacheStrategy.LRU:
                self._update_lru_order(key)
            
            self._hits += 1
            return entry.value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存"""
        async with self._lock:
            # 检查是否需要驱逐
            if key not in self._cache and len(self._cache) >= self.max_size:
                await self._evict()
            
            # 创建缓存条目
            entry_ttl = ttl or self.default_ttl
            entry = CacheEntry(key, value, entry_ttl)
            
            # 添加到缓存
            self._cache[key] = entry
            
            # 更新访问顺序
            if self.strategy == CacheStrategy.LRU:
                self._access_order.append(key)
            
            return True
    
    async def delete(self, key: str) -> bool:
        """删除缓存"""
        async with self._lock:
            return await self._remove(key)
    
    async def clear(self):
        """清空缓存"""
        async with self._lock:
            self._cache.clear()
            self._access_order.clear()
    
    async def has(self, key: str) -> bool:
        """检查缓存是否存在"""
        async with self._lock:
            entry = self._cache.get(key)
            if entry and not entry.is_expired():
                return True
            if entry and entry.is_expired():
                await self._remove(key)
            return False
    
    async def get_or_set(self, key: str, factory: Callable, ttl: Optional[int] = None) -> Any:
        """获取或设置缓存"""
        value = await self.get(key)
        if value is not None:
            return value
        
        # 生成值
        if asyncio.iscoroutinefunction(factory):
            value = await factory()
        else:
            value = factory()
        
        # 设置缓存
        await self.set(key, value, ttl)
        return value
    
    def generate_cache_key(self, func: Callable, args: tuple, kwargs: Dict[str, Any]) -> str:
        """生成缓存键"""
        # 序列化函数和参数
        key_data = {
            "func": func.__name__,
            "module": func.__module__,
            "args": args,
            "kwargs": sorted(kwargs.items())
        }
        
        # 生成哈希
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def cached(
        self, 
        key_prefix: str = "",
        ttl: Optional[int] = None,
        key_func: Optional[Callable] = None
    ):
        """缓存装饰器"""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                # 生成缓存键
                if key_func:
                    cache_key = key_func(func, args, kwargs)
                else:
                    cache_key = self.generate_cache_key(func, args, kwargs)
                
                if key_prefix:
                    cache_key = f"{key_prefix}:{cache_key}"
                
                # 尝试从缓存获取
                result = await self.get(cache_key)
                if result is not None:
                    return result
                
                # 执行函数并缓存结果
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                await self.set(cache_key, result, ttl)
                return result
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # 同步版本的装饰器
                cache_key = self.generate_cache_key(func, args, kwargs)
                if key_prefix:
                    cache_key = f"{key_prefix}:{cache_key}"
                
                # 这里需要同步版本的缓存逻辑
                # 简化处理，实际应该实现同步版本
                return func(*args, **kwargs)
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        async with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0
            
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "size": len(self._cache),
                "max_size": self.max_size,
                "strategy": self.strategy.value,
                "evictions": self._evictions
            }
    
    async def cleanup_expired(self) -> int:
        """清理过期缓存"""
        async with self._lock:
            expired_keys = []
            for key, entry in self._cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                await self._remove(key)
            
            return len(expired_keys)
    
    async def save_to_disk(self):
        """保存缓存到磁盘"""
        if not self.enable_persistence:
            return
        
        try:
            async with self._lock:
                cache_data = {}
                for key, entry in self._cache.items():
                    if not entry.is_expired():
                        cache_data[key] = entry.to_dict()
            
            with open(self.persist_path, 'w') as f:
                json.dump(cache_data, f, default=str)
                
            self.logger.info(f"缓存已保存到 {self.persist_path}")
            
        except Exception as e:
            self.logger.error(f"保存缓存失败: {str(e)}")
    
    async def load_from_disk(self):
        """从磁盘加载缓存"""
        if not self.enable_persistence:
            return
        
        try:
            with open(self.persist_path, 'r') as f:
                cache_data = json.load(f)
            
            async with self._lock:
                for key, data in cache_data.items():
                    entry = CacheEntry(
                        key=key,
                        value=data["value"],
                        ttl=data.get("ttl")
                    )
                    entry.created_at = data["created_at"]
                    entry.last_accessed = data["last_accessed"]
                    entry.access_count = data["access_count"]
                    
                    # 只加载未过期的缓存
                    if not entry.is_expired():
                        self._cache[key] = entry
                        if self.strategy == CacheStrategy.LRU:
                            self._access_order.append(key)
            
            self.logger.info(f"已从 {self.persist_path} 加载缓存")
            
        except FileNotFoundError:
            self.logger.info("缓存文件不存在，跳过加载")
        except Exception as e:
            self.logger.error(f"加载缓存失败: {str(e)}")
    
    async def _remove(self, key: str) -> bool:
        """移除缓存条目"""
        if key in self._cache:
            del self._cache[key]
            
            # 从访问顺序中移除
            if self.strategy == CacheStrategy.LRU and key in self._access_order:
                self._access_order.remove(key)
            
            return True
        return False
    
    async def _evict(self):
        """驱逐缓存条目"""
        if not self._cache:
            return
        
        if self.strategy == CacheStrategy.LRU:
            await self._evict_lru()
        elif self.strategy == CacheStrategy.LFU:
            await self._evict_lfu()
        elif self.strategy == CacheStrategy.FIFO:
            await self._evict_fifo()
        elif self.strategy == CacheStrategy.TTL:
            await self._evict_ttl()
    
    async def _evict_lru(self):
        """LRU驱逐"""
        if self._access_order:
            key_to_evict = self._access_order.pop(0)
            await self._remove(key_to_evict)
            self._evictions += 1
    
    async def _evict_lfu(self):
        """LFU驱逐"""
        if self._cache:
            # 找到访问次数最少的条目
            min_access = min(entry.access_count for entry in self._cache.values())
            lfu_keys = [key for key, entry in self._cache.items() 
                       if entry.access_count == min_access]
            
            # 驱逐最早访问的
            key_to_evict = min(lfu_keys, key=lambda k: self._cache[k].last_accessed)
            await self._remove(key_to_evict)
            self._evictions += 1
    
    async def _evict_fifo(self):
        """FIFO驱逐"""
        if self._cache:
            # 找到最早创建的条目
            oldest_key = min(self._cache.keys(), 
                           key=lambda k: self._cache[k].created_at)
            await self._remove(oldest_key)
            self._evictions += 1
    
    async def _evict_ttl(self):
        """TTL驱逐"""
        # 驱逐最早过期的条目
        expired_entries = [(key, entry) for key, entry in self._cache.items() 
                          if entry.is_expired()]
        
        if expired_entries:
            # 按创建时间排序，驱逐最早的
            expired_entries.sort(key=lambda x: x[1].created_at)
            key_to_evict = expired_entries[0][0]
            await self._remove(key_to_evict)
            self._evictions += 1
    
    def _update_lru_order(self, key: str):
        """更新LRU访问顺序"""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)


class TaskResultCache:
    """任务结果专用缓存"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.logger = logging.getLogger(__name__)
    
    async def cache_task_result(self, task: Task, result: TaskResult):
        """缓存任务结果"""
        cache_key = f"task_result:{task.id}"
        await self.cache_manager.set(cache_key, result.to_dict())
    
    async def get_cached_result(self, task_id: str) -> Optional[TaskResult]:
        """获取缓存的任务结果"""
        cache_key = f"task_result:{task_id}"
        result_data = await self.cache_manager.get(cache_key)
        
        if result_data:
            # 重建TaskResult对象
            return TaskResult(
                task_id=result_data["task_id"],
                status=TaskStatus(result_data["status"]),
                result=result_data["result"],
                error=result_data["error"],
                traceback=result_data.get("traceback"),
                cached=True
            )
        
        return None
    
    async def is_result_cached(self, task_id: str) -> bool:
        """检查结果是否已缓存"""
        cache_key = f"task_result:{task_id}"
        return await self.cache_manager.has(cache_key)
    
    async def invalidate_task_cache(self, task_id: str):
        """使任务缓存失效"""
        cache_key = f"task_result:{task_id}"
        await self.cache_manager.delete(cache_key)


# 导入TaskStatus
from .task_models import TaskStatus