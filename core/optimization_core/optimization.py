"""
语音处理性能优化模块
提供缓存、负载均衡、并发处理等优化功能
"""

import asyncio
import logging
import time
import hashlib
import pickle
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import numpy as np
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import weakref

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """优化策略"""
    CACHE = "cache"
    BATCH = "batch"
    PARALLEL = "parallel"
    COMPRESSION = "compression"
    ADAPTIVE = "adaptive"


@dataclass
class PerformanceMetrics:
    """性能指标"""
    latency: float
    throughput: float
    accuracy: float
    memory_usage: float
    cpu_usage: float
    cache_hit_rate: float
    error_rate: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class OptimizationConfig:
    """优化配置"""
    enable_cache: bool = True
    cache_size: int = 1000
    cache_ttl: int = 3600  # 秒
    enable_batch: bool = True
    batch_size: int = 10
    batch_timeout: float = 1.0  # 秒
    enable_parallel: bool = True
    max_workers: int = 4
    enable_compression: bool = True
    compression_level: int = 6
    adaptive_sampling: bool = True
    performance_threshold: float = 0.8


class AudioCache:
    """音频数据缓存"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.access_times: Dict[str, float] = {}
        self.lock = threading.RLock()
        self.cleanup_interval = 300  # 5分钟清理一次
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """启动清理任务"""
        def cleanup():
            while True:
                time.sleep(self.cleanup_interval)
                self._cleanup_expired()
        
        cleanup_thread = threading.Thread(target=cleanup, daemon=True)
        cleanup_thread.start()
    
    def _cleanup_expired(self):
        """清理过期缓存"""
        current_time = time.time()
        expired_keys = []
        
        with self.lock:
            for key, (value, timestamp) in self.cache.items():
                if current_time - timestamp > self.ttl:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]
        
        if expired_keys:
            logger.info(f"清理了 {len(expired_keys)} 个过期缓存项")
    
    def _generate_key(self, data: Any, **kwargs) -> str:
        """生成缓存键"""
        # 创建唯一键
        key_data = {
            "data": data,
            "params": kwargs
        }
        key_str = str(key_data)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, data: Any, **kwargs) -> Optional[Any]:
        """获取缓存"""
        key = self._generate_key(data, **kwargs)
        
        with self.lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                current_time = time.time()
                
                # 检查是否过期
                if current_time - timestamp <= self.ttl:
                    self.access_times[key] = current_time
                    return value
                else:
                    # 过期删除
                    del self.cache[key]
                    if key in self.access_times:
                        del self.access_times[key]
        
        return None
    
    def set(self, data: Any, value: Any, **kwargs):
        """设置缓存"""
        key = self._generate_key(data, **kwargs)
        
        with self.lock:
            # 检查缓存大小
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            current_time = time.time()
            self.cache[key] = (value, current_time)
            self.access_times[key] = current_time
    
    def _evict_lru(self):
        """LRU淘汰"""
        if not self.access_times:
            return
        
        # 找到最少使用的键
        lru_key = min(self.access_times.items(), key=lambda x: x[1])[0]
        
        # 删除
        if lru_key in self.cache:
            del self.cache[lru_key]
        del self.access_times[lru_key]
    
    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        with self.lock:
            current_time = time.time()
            valid_items = sum(
                1 for _, timestamp in self.cache.values()
                if current_time - timestamp <= self.ttl
            )
            
            return {
                "total_items": len(self.cache),
                "valid_items": valid_items,
                "expired_items": len(self.cache) - valid_items,
                "max_size": self.max_size,
                "utilization": len(self.cache) / self.max_size
            }


class BatchProcessor:
    """批处理器"""
    
    def __init__(self, batch_size: int = 10, timeout: float = 1.0):
        self.batch_size = batch_size
        self.timeout = timeout
        self.pending_batches: List[List] = []
        self.batch_queue = asyncio.Queue()
        self.processing = False
        self.lock = asyncio.Lock()
    
    async def add_item(self, item: Any) -> asyncio.Future:
        """添加处理项"""
        future = asyncio.Future()
        
        async with self.lock:
            if not self.pending_batches:
                self.pending_batches.append([])
            
            batch = self.pending_batches[0]
            batch.append((item, future))
            
            # 检查是否达到批处理大小
            if len(batch) >= self.batch_size:
                await self._process_batch(batch)
                self.pending_batches[0] = []
            else:
                # 启动超时处理
                asyncio.create_task(self._schedule_timeout(batch))
        
        return future
    
    async def _schedule_timeout(self, batch: List):
        """安排超时处理"""
        await asyncio.sleep(self.timeout)
        
        async with self.lock:
            if batch in self.pending_batches and batch:
                await self._process_batch(batch)
                if batch in self.pending_batches:
                    self.pending_batches.remove(batch)
    
    async def _process_batch(self, batch: List):
        """处理批次"""
        if not batch:
            return
        
        items = [item for item, _ in batch]
        futures = [future for _, future in batch]
        
        try:
            # 这里应该调用实际的批处理函数
            results = await self._process_batch_items(items)
            
            # 设置结果
            for future, result in zip(futures, results):
                if not future.done():
                    future.set_result(result)
        
        except Exception as e:
            # 设置错误
            for future in futures:
                if not future.done():
                    future.set_exception(e)
    
    async def _process_batch_items(self, items: List) -> List:
        """实际处理批次项目"""
        # 模拟批处理
        await asyncio.sleep(0.01)  # 模拟处理时间
        
        # 返回处理结果
        return [{"result": f"processed_{i}"} for i in range(len(items))]


class AdaptiveSampler:
    """自适应采样器"""
    
    def __init__(self, initial_rate: float = 1.0, min_rate: float = 0.1, max_rate: float = 1.0):
        self.current_rate = initial_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.performance_history = deque(maxlen=100)
        self.last_adjustment = time.time()
        self.adjustment_interval = 5.0  # 5秒调整一次
    
    def should_sample(self) -> bool:
        """判断是否应该采样"""
        return np.random.random() < self.current_rate
    
    def record_performance(self, metrics: PerformanceMetrics):
        """记录性能指标"""
        self.performance_history.append(metrics)
        
        current_time = time.time()
        if current_time - self.last_adjustment >= self.adjustment_interval:
            self._adjust_sampling_rate()
            self.last_adjustment = current_time
    
    def _adjust_sampling_rate(self):
        """调整采样率"""
        if len(self.performance_history) < 10:
            return
        
        # 计算性能趋势
        recent_metrics = list(self.performance_history)[-10:]
        latencies = [m.latency for m in recent_metrics]
        throughputs = [m.throughput for m in recent_metrics]
        
        # 计算性能变化
        latency_trend = np.mean(latencies[-5:]) - np.mean(latencies[:5])
        throughput_trend = np.mean(throughputs[-5:]) - np.mean(throughputs[:5])
        
        # 调整采样率
        if latency_trend > 0.1 or throughput_trend < -0.1:
            # 性能下降，降低采样率
            self.current_rate = max(
                self.min_rate, 
                self.current_rate * 0.9
            )
        elif latency_trend < -0.05 and throughput_trend > 0.05:
            # 性能提升，可以增加采样率
            self.current_rate = min(
                self.max_rate,
                self.current_rate * 1.1
            )
        
        logger.info(f"调整采样率到: {self.current_rate:.2f}")


class ResourceMonitor:
    """资源监控器"""
    
    def __init__(self):
        self.cpu_history = deque(maxlen=60)  # 1分钟历史
        self.memory_history = deque(maxlen=60)
        self.last_check = time.time()
        self.check_interval = 1.0  # 1秒检查一次
    
    def get_current_metrics(self) -> Tuple[float, float]:
        """获取当前资源使用情况"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        current_time = time.time()
        if current_time - self.last_check >= self.check_interval:
            self.cpu_history.append(cpu_percent)
            self.memory_history.append(memory_percent)
            self.last_check = current_time
        
        return cpu_percent, memory_percent
    
    def get_average_metrics(self, window_size: int = 10) -> Tuple[float, float]:
        """获取平均资源使用情况"""
        if len(self.cpu_history) == 0:
            return 0.0, 0.0
        
        recent_cpu = list(self.cpu_history)[-window_size:]
        recent_memory = list(self.memory_history)[-window_size:]
        
        return np.mean(recent_cpu), np.mean(recent_memory)
    
    def should_throttle(self, threshold: float = 80.0) -> bool:
        """判断是否应该限流"""
        avg_cpu, avg_memory = self.get_average_metrics()
        return avg_cpu > threshold or avg_memory > threshold


class PerformanceOptimizer:
    """性能优化器"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.cache = AudioCache(config.cache_size, config.cache_ttl) if config.enable_cache else None
        self.batch_processor = BatchProcessor(config.batch_size, config.batch_timeout) if config.enable_batch else None
        self.adaptive_sampler = AdaptiveSampler() if config.adaptive_sampling else None
        self.resource_monitor = ResourceMonitor()
        
        # 执行器
        self.thread_executor = ThreadPoolExecutor(max_workers=config.max_workers) if config.enable_parallel else None
        self.process_executor = ProcessPoolExecutor(max_workers=config.max_workers) if config.enable_parallel else None
        
        # 性能统计
        self.metrics_history = deque(maxlen=1000)
        self.call_times = defaultdict(list)
        
        # 优化策略
        self.strategies = {
            OptimizationStrategy.CACHE: self._optimize_with_cache,
            OptimizationStrategy.BATCH: self._optimize_with_batch,
            OptimizationStrategy.PARALLEL: self._optimize_with_parallel,
            OptimizationStrategy.COMPRESSION: self._optimize_with_compression,
            OptimizationStrategy.ADAPTIVE: self._optimize_with_adaptive
        }
    
    async def optimize_asr(
        self,
        audio_data: bytes,
        asr_func: Callable,
        **kwargs
    ) -> Any:
        """优化ASR处理"""
        start_time = time.time()
        
        # 检查缓存
        if self.cache:
            cached_result = self.cache.get(audio_data, **kwargs)
            if cached_result is not None:
                logger.debug("ASR缓存命中")
                return cached_result
        
        # 检查是否需要限流
        if self.resource_monitor.should_throttle():
            logger.warning("资源使用过高，实施限流")
            await asyncio.sleep(0.1)
        
        # 执行ASR
        try:
            if self.config.enable_parallel and self.thread_executor:
                result = await asyncio.get_event_loop().run_in_executor(
                    self.thread_executor, asr_func, audio_data, **kwargs
                )
            else:
                result = await asyncio.to_thread(asr_func, audio_data, **kwargs)
            
            # 缓存结果
            if self.cache:
                self.cache.set(audio_data, result, **kwargs)
            
            # 记录性能指标
            latency = time.time() - start_time
            await self._record_metrics("asr", latency, True)
            
            return result
            
        except Exception as e:
            await self._record_metrics("asr", time.time() - start_time, False)
            raise
    
    async def optimize_tts(
        self,
        text: str,
        tts_func: Callable,
        **kwargs
    ) -> Any:
        """优化TTS处理"""
        start_time = time.time()
        
        # 检查缓存
        if self.cache:
            cached_result = self.cache.get(text, **kwargs)
            if cached_result is not None:
                logger.debug("TTS缓存命中")
                return cached_result
        
        # 检查资源使用
        if self.resource_monitor.should_throttle():
            await asyncio.sleep(0.1)
        
        # 执行TTS
        try:
            if self.config.enable_parallel and self.thread_executor:
                result = await asyncio.get_event_loop().run_in_executor(
                    self.thread_executor, tts_func, text, **kwargs
                )
            else:
                result = await asyncio.to_thread(tts_func, text, **kwargs)
            
            # 缓存结果
            if self.cache:
                self.cache.set(text, result, **kwargs)
            
            # 记录性能指标
            latency = time.time() - start_time
            await self._record_metrics("tts", latency, True)
            
            return result
            
        except Exception as e:
            await self._record_metrics("tts", time.time() - start_time, False)
            raise
    
    async def batch_process(
        self,
        items: List[Any],
        process_func: Callable,
        batch_size: Optional[int] = None
    ) -> List[Any]:
        """批处理"""
        if not self.config.enable_batch or not self.batch_processor:
            # 逐个处理
            results = []
            for item in items:
                try:
                    result = await process_func(item)
                    results.append(result)
                except Exception as e:
                    logger.error(f"批处理项目失败: {e}")
                    results.append(None)
            return results
        
        # 批量处理
        batch_size = batch_size or self.config.batch_size
        results = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_tasks = []
            
            for item in batch:
                task = self.batch_processor.add_item(item)
                batch_tasks.append(task)
            
            # 等待批次完成
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            results.extend(batch_results)
        
        return results
    
    def _optimize_with_cache(self, func: Callable, *args, **kwargs) -> Callable:
        """缓存优化"""
        if not self.cache:
            return func
        
        async def cached_func(*args, **kwargs):
            cache_key = self._generate_cache_key(args, kwargs)
            result = self.cache.get(cache_key)
            if result is not None:
                return result
            
            result = await func(*args, **kwargs)
            self.cache.set(cache_key, result)
            return result
        
        return cached_func
    
    def _optimize_with_batch(self, func: Callable, *args, **kwargs) -> Callable:
        """批处理优化"""
        if not self.batch_processor:
            return func
        
        async def batched_func(*args, **kwargs):
            return await self.batch_processor.add_item((func, args, kwargs))
        
        return batched_func
    
    def _optimize_with_parallel(self, func: Callable, *args, **kwargs) -> Callable:
        """并行优化"""
        if not self.thread_executor:
            return func
        
        async def parallel_func(*args, **kwargs):
            return await asyncio.get_event_loop().run_in_executor(
                self.thread_executor, func, *args, **kwargs
            )
        
        return parallel_func
    
    def _optimize_with_compression(self, data: bytes) -> bytes:
        """压缩优化"""
        if not self.config.enable_compression:
            return data
        
        try:
            import gzip
            return gzip.compress(data, self.config.compression_level)
        except ImportError:
            logger.warning("gzip不可用，跳过压缩")
            return data
    
    def _optimize_with_adaptive(self, func: Callable, *args, **kwargs) -> Callable:
        """自适应优化"""
        if not self.adaptive_sampler:
            return func
        
        async def adaptive_func(*args, **kwargs):
            # 根据采样率决定是否执行
            if not self.adaptive_sampler.should_sample():
                # 返回默认结果或跳过
                return None
            
            result = await func(*args, **kwargs)
            
            # 记录性能
            metrics = PerformanceMetrics(
                latency=0.0,  # 这里应该记录实际延迟
                throughput=0.0,
                accuracy=0.0,
                memory_usage=0.0,
                cpu_usage=0.0,
                cache_hit_rate=0.0,
                error_rate=0.0
            )
            self.adaptive_sampler.record_performance(metrics)
            
            return result
        
        return adaptive_func
    
    def _generate_cache_key(self, args: tuple, kwargs: dict) -> str:
        """生成缓存键"""
        key_data = {
            "args": args,
            "kwargs": kwargs
        }
        key_str = str(key_data)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    async def _record_metrics(self, operation: str, latency: float, success: bool):
        """记录性能指标"""
        cpu_usage, memory_usage = self.resource_monitor.get_current_metrics()
        
        metrics = PerformanceMetrics(
            latency=latency,
            throughput=1.0 / latency if latency > 0 else 0.0,
            accuracy=1.0 if success else 0.0,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            cache_hit_rate=0.8 if self.cache else 0.0,  # 简化计算
            error_rate=0.0 if success else 1.0
        )
        
        self.metrics_history.append(metrics)
        self.call_times[operation].append(time.time())
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        if not self.metrics_history:
            return {"message": "暂无性能数据"}
        
        recent_metrics = list(self.metrics_history)[-100:]  # 最近100个
        
        return {
            "avg_latency": np.mean([m.latency for m in recent_metrics]),
            "avg_throughput": np.mean([m.throughput for m in recent_metrics]),
            "avg_accuracy": np.mean([m.accuracy for m in recent_metrics]),
            "avg_memory_usage": np.mean([m.memory_usage for m in recent_metrics]),
            "avg_cpu_usage": np.mean([m.cpu_usage for m in recent_metrics]),
            "total_operations": len(self.metrics_history),
            "cache_stats": self.cache.get_stats() if self.cache else None,
            "resource_usage": {
                "current_cpu": psutil.cpu_percent(),
                "current_memory": psutil.virtual_memory().percent
            }
        }
    
    def optimize_config(self, target_latency: float = 1.0) -> OptimizationConfig:
        """根据性能目标优化配置"""
        current_metrics = self.get_performance_summary()
        
        # 基于当前性能调整配置
        if current_metrics.get("avg_latency", 0) > target_latency:
            # 延迟过高，增加缓存和批处理
            new_config = OptimizationConfig(
                enable_cache=True,
                cache_size=min(self.config.cache_size * 2, 5000),
                enable_batch=True,
                batch_size=min(self.config.batch_size * 2, 50),
                enable_parallel=True,
                max_workers=min(self.config.max_workers * 2, 8),
                adaptive_sampling=True,
                performance_threshold=self.config.performance_threshold
            )
        else:
            # 性能良好，可以减少资源使用
            new_config = OptimizationConfig(
                enable_cache=True,
                cache_size=max(self.config.cache_size // 2, 100),
                enable_batch=True,
                batch_size=max(self.config.batch_size // 2, 5),
                enable_parallel=self.config.max_workers > 1,
                max_workers=max(self.config.max_workers // 2, 1),
                adaptive_sampling=True,
                performance_threshold=self.config.performance_threshold
            )
        
        logger.info(f"优化配置: {new_config}")
        return new_config
    
    def cleanup(self):
        """清理资源"""
        if self.cache:
            self.cache.clear()
        
        if self.thread_executor:
            self.thread_executor.shutdown(wait=True)
        
        if self.process_executor:
            self.process_executor.shutdown(wait=True)
        
        logger.info("性能优化器已清理")


class OptimizationManager:
    """优化管理器"""
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.optimizer = PerformanceOptimizer(self.config)
        self.is_running = False
    
    def optimize(self, operation: str, data: Any, func: Callable, **kwargs) -> Any:
        """执行优化操作"""
        if operation == "asr":
            return asyncio.run(self.optimizer.optimize_asr(data, func, **kwargs))
        elif operation == "tts":
            return asyncio.run(self.optimizer.optimize_tts(data, func, **kwargs))
        else:
            return func(data, **kwargs)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        return self.optimizer.get_performance_summary()
    
    def optimize_configuration(self, target_latency: float = 1.0) -> OptimizationConfig:
        """优化配置"""
        return self.optimizer.optimize_config(target_latency)
    
    def start_monitoring(self):
        """开始监控"""
        self.is_running = True
        logger.info("优化管理器开始监控")
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_running = False
        self.optimizer.cleanup()
        logger.info("优化管理器停止监控")


class OptimizationMode(Enum):
    """优化模式"""
    PERFORMANCE = "performance"  # 性能优先
    ACCURACY = "accuracy"        # 准确度优先
    BALANCED = "balanced"        # 平衡模式
    ADAPTIVE = "adaptive"        # 自适应模式