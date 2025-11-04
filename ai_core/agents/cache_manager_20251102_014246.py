"""
工具调用缓存和性能优化系统
实现智能缓存机制和性能监控优化
"""

import asyncio
import hashlib
import json
import logging
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import pickle
import gzip
import sqlite3
import os
import psutil
import threading
from collections import defaultdict, OrderedDict
import time

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """缓存策略枚举"""
    LRU = "lru"  # 最近最少使用
    LFU = "lfu"  # 最少使用频率
    TTL = "ttl"  # 时间过期
    ADAPTIVE = "adaptive"  # 自适应


class CompressionType(Enum):
    """压缩类型枚举"""
    NONE = "none"
    GZIP = "gzip"
    ZLIB = "zlib"


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    size_bytes: int = 0
    compression_type: CompressionType = CompressionType.NONE
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """性能指标"""
    tool_name: str
    total_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_execution_time: float = 0.0
    avg_execution_time: float = 0.0
    error_count: int = 0
    success_rate: float = 0.0
    throughput: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


class IntelligentCache:
    """智能缓存系统"""
    
    def __init__(self, 
                 max_size: int = 1000,
                 max_memory_mb: int = 100,
                 default_ttl: int = 3600,
                 strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
                 enable_compression: bool = True):
        
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.strategy = strategy
        self.enable_compression = enable_compression
        
        # 内存缓存
        self._memory_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._access_order: List[str] = []  # 用于LRU跟踪
        
        # 统计信息
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'memory_usage': 0,
            'total_size': 0
        }
        
        # 线程锁
        self._lock = threading.RLock()
        
        # 持久化缓存 (SQLite)
        self._setup_persistent_cache()
        
        # 性能监控
        self._performance_monitor = PerformanceMonitor()
    
    def _setup_persistent_cache(self):
        """设置持久化缓存"""
        cache_dir = "/tmp/intelligent_tools_cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        self._db_path = os.path.join(cache_dir, "cache.db")
        
        # 创建数据库表
        with sqlite3.connect(self._db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    created_at TIMESTAMP,
                    last_accessed TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    size_bytes INTEGER,
                    compression_type TEXT,
                    metadata TEXT
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_last_accessed 
                ON cache_entries(last_accessed)
            ''')
    
    def _generate_cache_key(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """生成缓存键"""
        # 创建参数的有序表示
        param_str = json.dumps(parameters, sort_keys=True, default=str)
        key_data = f"{tool_name}:{param_str}"
        
        # 生成哈希
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]
    
    def _serialize_value(self, value: Any, compression: bool = True) -> Tuple[bytes, CompressionType]:
        """序列化值"""
        # 序列化为pickle
        serialized = pickle.dumps(value)
        
        if not compression or not self.enable_compression:
            return serialized, CompressionType.NONE
        
        # 尝试压缩
        try:
            compressed = gzip.compress(serialized)
            # 如果压缩效果不好，使用原始数据
            if len(compressed) < len(serialized) * 0.9:
                return compressed, CompressionType.GZIP
        except:
            pass
        
        return serialized, CompressionType.NONE
    
    def _deserialize_value(self, data: bytes, compression_type: CompressionType) -> Any:
        """反序列化值"""
        try:
            if compression_type == CompressionType.GZIP:
                data = gzip.decompress(data)
            
            return pickle.loads(data)
        except Exception as e:
            logger.error(f"反序列化失败: {e}")
            return None
    
    def get(self, tool_name: str, parameters: Dict[str, Any]) -> Optional[Any]:
        """获取缓存值"""
        cache_key = self._generate_cache_key(tool_name, parameters)
        
        with self._lock:
            # 先检查内存缓存
            if cache_key in self._memory_cache:
                entry = self._memory_cache[cache_key]
                
                # 检查TTL
                if self._is_expired(entry):
                    self._remove_entry(cache_key)
                    self._stats['misses'] += 1
                    return None
                
                # 更新访问信息
                entry.last_accessed = datetime.now()
                entry.access_count += 1
                
                # 更新访问顺序 (LRU)
                if self.strategy == CacheStrategy.LRU:
                    self._update_access_order(cache_key)
                
                self._stats['hits'] += 1
                self._performance_monitor.record_cache_hit(tool_name)
                
                return entry.value
            
            # 检查持久化缓存
            persistent_value = self._get_from_persistent(cache_key)
            if persistent_value is not None:
                # 加载到内存缓存
                self._add_to_memory_cache(cache_key, persistent_value)
                self._stats['hits'] += 1
                self._performance_monitor.record_cache_hit(tool_name)
                return persistent_value.value
            
            self._stats['misses'] += 1
            self._performance_monitor.record_cache_miss(tool_name)
            return None
    
    def put(self, tool_name: str, parameters: Dict[str, Any], value: Any, ttl: Optional[int] = None) -> bool:
        """存储缓存值"""
        cache_key = self._generate_cache_key(tool_name, parameters)
        ttl = ttl or self.default_ttl
        
        # 序列化值
        serialized_data, compression_type = self._serialize_value(value)
        size_bytes = len(serialized_data)
        
        # 检查大小限制
        if size_bytes > self.max_memory_bytes // 4:  # 单个条目不超过总内存的1/4
            logger.warning(f"缓存条目过大，跳过: {size_bytes} bytes")
            return False
        
        # 创建缓存条目
        entry = CacheEntry(
            key=cache_key,
            value=value,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            size_bytes=size_bytes,
            compression_type=compression_type,
            metadata={'ttl': ttl, 'tool_name': tool_name}
        )
        
        with self._lock:
            # 检查是否需要清理
            self._evict_if_necessary()
            
            # 添加到内存缓存
            self._memory_cache[cache_key] = entry
            self._update_access_order(cache_key)
            
            # 更新统计
            self._stats['total_size'] += size_bytes
            self._stats['memory_usage'] = self._calculate_memory_usage()
            
            # 保存到持久化缓存
            self._save_to_persistent(cache_key, entry, serialized_data)
            
            self._performance_monitor.record_cache_write(tool_name, size_bytes)
            
            return True
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """检查是否过期"""
        ttl = entry.metadata.get('ttl', self.default_ttl)
        return datetime.now() - entry.created_at > timedelta(seconds=ttl)
    
    def _evict_if_necessary(self):
        """必要时清理缓存"""
        # 检查大小限制
        if len(self._memory_cache) >= self.max_size:
            self._evict_entries(1)
        
        # 检查内存限制
        if self._calculate_memory_usage() > self.max_memory_bytes:
            self._evict_by_strategy()
    
    def _evict_entries(self, count: int):
        """清理指定数量的条目"""
        with self._lock:
            keys_to_remove = []
            
            if self.strategy == CacheStrategy.LRU:
                # LRU: 清理最久未访问的
                keys_to_remove = self._access_order[:count]
            elif self.strategy == CacheStrategy.LFU:
                # LFU: 清理访问频率最低的
                sorted_entries = sorted(
                    self._memory_cache.items(),
                    key=lambda x: x[1].access_count
                )
                keys_to_remove = [key for key, _ in sorted_entries[:count]]
            else:
                # 默认: 清理最久未访问的
                keys_to_remove = list(self._memory_cache.keys())[:count]
            
            for key in keys_to_remove:
                self._remove_entry(key)
            
            self._stats['evictions'] += count
    
    def _evict_by_strategy(self):
        """根据策略清理缓存"""
        if self.strategy == CacheStrategy.ADAPTIVE:
            # 自适应策略: 组合多种方法
            self._evict_expired()
            if self._calculate_memory_usage() > self.max_memory_bytes * 0.8:
                self._evict_entries(len(self._memory_cache) // 4)
        else:
            self._evict_entries(len(self._memory_cache) // 4)
    
    def _evict_expired(self):
        """清理过期条目"""
        expired_keys = [
            key for key, entry in self._memory_cache.items()
            if self._is_expired(entry)
        ]
        
        for key in expired_keys:
            self._remove_entry(key)
    
    def _remove_entry(self, cache_key: str):
        """移除缓存条目"""
        if cache_key in self._memory_cache:
            entry = self._memory_cache.pop(cache_key)
            self._stats['total_size'] -= entry.size_bytes
            self._stats['memory_usage'] = self._calculate_memory_usage()
            
            # 从访问顺序中移除
            if cache_key in self._access_order:
                self._access_order.remove(cache_key)
            
            # 从持久化缓存中移除
            self._remove_from_persistent(cache_key)
    
    def _update_access_order(self, cache_key: str):
        """更新访问顺序"""
        if cache_key in self._access_order:
            self._access_order.remove(cache_key)
        self._access_order.append(cache_key)
        
        # 保持访问顺序在合理范围内
        if len(self._access_order) > self.max_size * 2:
            self._access_order = self._access_order[-self.max_size:]
    
    def _calculate_memory_usage(self) -> int:
        """计算内存使用量"""
        return sum(entry.size_bytes for entry in self._memory_cache.values())
    
    def _add_to_memory_cache(self, cache_key: str, entry: CacheEntry):
        """添加到内存缓存"""
        self._memory_cache[cache_key] = entry
        self._update_access_order(cache_key)
    
    def _get_from_persistent(self, cache_key: str) -> Optional[CacheEntry]:
        """从持久化缓存获取"""
        try:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.execute(
                    'SELECT * FROM cache_entries WHERE key = ?',
                    (cache_key,)
                )
                row = cursor.fetchone()
                
                if row:
                    # 检查是否过期
                    last_accessed = datetime.fromisoformat(row[3])
                    if datetime.now() - last_accessed > timedelta(seconds=self.default_ttl):
                        conn.execute('DELETE FROM cache_entries WHERE key = ?', (cache_key,))
                        return None
                    
                    # 反序列化值
                    value = self._deserialize_value(row[1], CompressionType(row[5]))
                    
                    if value is not None:
                        return CacheEntry(
                            key=row[0],
                            value=value,
                            created_at=datetime.fromisoformat(row[2]),
                            last_accessed=last_accessed,
                            access_count=row[4],
                            size_bytes=row[6],
                            compression_type=CompressionType(row[5]),
                            metadata=json.loads(row[7]) if row[7] else {}
                        )
        except Exception as e:
            logger.error(f"从持久化缓存获取失败: {e}")
        
        return None
    
    def _save_to_persistent(self, cache_key: str, entry: CacheEntry, serialized_data: bytes):
        """保存到持久化缓存"""
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO cache_entries 
                    (key, value, created_at, last_accessed, access_count, size_bytes, compression_type, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    cache_key,
                    serialized_data,
                    entry.created_at.isoformat(),
                    entry.last_accessed.isoformat(),
                    entry.access_count,
                    entry.size_bytes,
                    entry.compression_type.value,
                    json.dumps(entry.metadata)
                ))
        except Exception as e:
            logger.error(f"保存到持久化缓存失败: {e}")
    
    def _remove_from_persistent(self, cache_key: str):
        """从持久化缓存移除"""
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute('DELETE FROM cache_entries WHERE key = ?', (cache_key,))
        except Exception as e:
            logger.error(f"从持久化缓存移除失败: {e}")
    
    def clear(self):
        """清空缓存"""
        with self._lock:
            self._memory_cache.clear()
            self._access_order.clear()
            
            # 清空持久化缓存
            try:
                with sqlite3.connect(self._db_path) as conn:
                    conn.execute('DELETE FROM cache_entries')
            except Exception as e:
                logger.error(f"清空持久化缓存失败: {e}")
            
            # 重置统计
            self._stats = {
                'hits': 0,
                'misses': 0,
                'evictions': 0,
                'memory_usage': 0,
                'total_size': 0
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total_requests = self._stats['hits'] + self._stats['misses']
        hit_rate = self._stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            'hits': self._stats['hits'],
            'misses': self._stats['misses'],
            'hit_rate': hit_rate,
            'evictions': self._stats['evictions'],
            'memory_usage_mb': self._stats['memory_usage'] / (1024 * 1024),
            'total_size_mb': self._stats['total_size'] / (1024 * 1024),
            'entries_count': len(self._memory_cache),
            'strategy': self.strategy.value
        }


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics: Dict[str, PerformanceMetrics] = {}
        self._lock = threading.RLock()
        self._start_time = time.time()
    
    def record_tool_execution(self, tool_name: str, execution_time: float, success: bool):
        """记录工具执行"""
        with self._lock:
            if tool_name not in self.metrics:
                self.metrics[tool_name] = PerformanceMetrics(tool_name=tool_name)
            
            metrics = self.metrics[tool_name]
            metrics.total_calls += 1
            metrics.total_execution_time += execution_time
            metrics.avg_execution_time = metrics.total_execution_time / metrics.total_calls
            
            if success:
                metrics.success_rate = (metrics.success_rate * (metrics.total_calls - 1) + 1) / metrics.total_calls
            else:
                metrics.error_count += 1
                metrics.success_rate = (metrics.success_rate * (metrics.total_calls - 1)) / metrics.total_calls
            
            # 计算吞吐量 (每分钟调用次数)
            elapsed_minutes = (time.time() - self._start_time) / 60
            metrics.throughput = metrics.total_calls / elapsed_minutes if elapsed_minutes > 0 else 0
            
            metrics.last_updated = datetime.now()
    
    def record_cache_hit(self, tool_name: str):
        """记录缓存命中"""
        with self._lock:
            if tool_name not in self.metrics:
                self.metrics[tool_name] = PerformanceMetrics(tool_name=tool_name)
            self.metrics[tool_name].cache_hits += 1
    
    def record_cache_miss(self, tool_name: str):
        """记录缓存未命中"""
        with self._lock:
            if tool_name not in self.metrics:
                self.metrics[tool_name] = PerformanceMetrics(tool_name=tool_name)
            self.metrics[tool_name].cache_misses += 1
    
    def record_cache_write(self, tool_name: str, size_bytes: int):
        """记录缓存写入"""
        # 可以在这里添加缓存写入统计
        pass
    
    def get_metrics(self, tool_name: Optional[str] = None) -> Dict[str, Any]:
        """获取性能指标"""
        with self._lock:
            if tool_name:
                if tool_name in self.metrics:
                    return {
                        tool_name: {
                            'total_calls': self.metrics[tool_name].total_calls,
                            'cache_hits': self.metrics[tool_name].cache_hits,
                            'cache_misses': self.metrics[tool_name].cache_misses,
                            'avg_execution_time': self.metrics[tool_name].avg_execution_time,
                            'success_rate': self.metrics[tool_name].success_rate,
                            'throughput': self.metrics[tool_name].throughput
                        }
                    }
                else:
                    return {}
            else:
                return {
                    name: {
                        'total_calls': metrics.total_calls,
                        'cache_hits': metrics.cache_hits,
                        'cache_misses': metrics.cache_misses,
                        'avg_execution_time': metrics.avg_execution_time,
                        'success_rate': metrics.success_rate,
                        'throughput': metrics.throughput
                    }
                    for name, metrics in self.metrics.items()
                }
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """获取系统指标"""
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'process_count': len(psutil.pids()),
            'timestamp': datetime.now().isoformat()
        }


class CacheManager:
    """缓存管理器"""
    
    def __init__(self, cache_config: Optional[Dict[str, Any]] = None):
        config = cache_config or {}
        
        self.cache = IntelligentCache(
            max_size=config.get('max_size', 1000),
            max_memory_mb=config.get('max_memory_mb', 100),
            default_ttl=config.get('default_ttl', 3600),
            strategy=CacheStrategy(config.get('strategy', 'adaptive')),
            enable_compression=config.get('enable_compression', True)
        )
        
        self.performance_monitor = PerformanceMonitor()
        self._optimization_enabled = True
    
    def get_cached_result(self, tool_name: str, parameters: Dict[str, Any]) -> Tuple[Optional[Any], bool]:
        """获取缓存结果 (返回结果和是否来自缓存)"""
        result = self.cache.get(tool_name, parameters)
        is_cached = result is not None
        
        if is_cached:
            self.performance_monitor.record_cache_hit(tool_name)
        else:
            self.performance_monitor.record_cache_miss(tool_name)
        
        return result, is_cached
    
    def cache_result(self, tool_name: str, parameters: Dict[str, Any], result: Any, ttl: Optional[int] = None):
        """缓存结果"""
        success = self.cache.put(tool_name, parameters, result, ttl)
        if success:
            self.performance_monitor.record_cache_write(tool_name, 0)  # 简化统计
    
    def optimize_cache(self):
        """优化缓存"""
        if not self._optimization_enabled:
            return
        
        # 清理过期条目
        self.cache._evict_expired()
        
        # 记录优化统计
        stats = self.cache.get_stats()
        logger.info(f"缓存优化完成: {stats}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        return {
            'cache_stats': self.cache.get_stats(),
            'tool_metrics': self.performance_monitor.get_metrics(),
            'system_metrics': self.performance_monitor.get_system_metrics()
        }
    
    def export_performance_data(self, filepath: str):
        """导出性能数据"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'cache_stats': self.cache.get_stats(),
            'tool_metrics': self.performance_monitor.get_metrics(),
            'system_metrics': self.performance_monitor.get_system_metrics()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"性能数据已导出到: {filepath}")