"""
非阻塞等待系统异步结果管理器
"""

import asyncio
import json
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging

from .config import NonBlockingConfig
from .event_system import event_system, EventType


class ResultStatus(Enum):
    """结果状态"""
    PENDING = "pending"
    AVAILABLE = "available"
    ERROR = "error"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


@dataclass
class AsyncResult:
    """异步结果"""
    id: str
    task_id: str
    status: ResultStatus
    created_at: datetime
    result: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    expires_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        if self.expires_at:
            data['expires_at'] = self.expires_at.isoformat()
        return data
    
    @property
    def is_ready(self) -> bool:
        """结果是否就绪"""
        return self.status == ResultStatus.AVAILABLE
    
    @property
    def is_error(self) -> bool:
        """是否出错"""
        return self.status == ResultStatus.ERROR
    
    @property
    def is_expired(self) -> bool:
        """是否过期"""
        if not self.expires_at:
            return False
        return datetime.now(timezone.utc) > self.expires_at
    
    @property
    def age(self) -> float:
        """结果存在时长（秒）"""
        return (datetime.now(timezone.utc) - self.created_at).total_seconds()


class ResultStorage:
    """结果存储"""
    
    def __init__(self, config: NonBlockingConfig):
        self.config = config
        self.results: Dict[str, AsyncResult] = {}
        self.result_queue: List[str] = []  # 用于LRU清理
        self.access_times: Dict[str, datetime] = {}
    
    def store(self, result: AsyncResult) -> None:
        """存储结果"""
        self.results[result.id] = result
        self.result_queue.append(result.id)
        self.access_times[result.id] = datetime.now(timezone.utc)
        
        # 清理过期结果
        self._cleanup_expired()
        
        # 如果缓存已满，删除最旧的条目
        if len(self.results) > self.config.max_cache_size:
            self._evict_oldest()
    
    def get(self, result_id: str) -> Optional[AsyncResult]:
        """获取结果"""
        result = self.results.get(result_id)
        if result and not result.is_expired:
            # 更新访问时间
            self.access_times[result_id] = datetime.now(timezone.utc)
            return result
        elif result and result.is_expired:
            # 移除过期结果
            self.delete(result_id)
        return None
    
    def delete(self, result_id: str) -> bool:
        """删除结果"""
        if result_id in self.results:
            del self.results[result_id]
            self.result_queue = [rid for rid in self.result_queue if rid != result_id]
            self.access_times.pop(result_id, None)
            return True
        return False
    
    def _cleanup_expired(self) -> None:
        """清理过期结果"""
        expired_ids = [
            result_id for result_id, result in self.results.items()
            if result.is_expired
        ]
        
        for result_id in expired_ids:
            self.delete(result_id)
    
    def _evict_oldest(self) -> None:
        """清理最旧的条目"""
        # 按访问时间排序
        sorted_results = sorted(
            self.access_times.items(),
            key=lambda x: x[1]
        )
        
        # 删除最旧的条目直到缓存大小合适
        while len(self.results) > self.config.max_cache_size and sorted_results:
            oldest_id, _ = sorted_results.pop(0)
            self.delete(oldest_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取存储统计"""
        total = len(self.results)
        ready = sum(1 for r in self.results.values() if r.is_ready)
        error = sum(1 for r in self.results.values() if r.is_error)
        expired = sum(1 for r in self.results.values() if r.is_expired)
        
        return {
            "total_results": total,
            "ready_results": ready,
            "error_results": error,
            "expired_results": expired,
            "cache_utilization": total / self.config.max_cache_size if self.config.max_cache_size > 0 else 0
        }


class AsyncResultManager:
    """异步结果管理器"""
    
    def __init__(self, config: Optional[NonBlockingConfig] = None):
        self.config = config or NonBlockingConfig()
        self.storage = ResultStorage(self.config)
        self.waiters: Dict[str, List[asyncio.Future]] = {}
        self.result_callbacks: Dict[str, List[Callable]] = {}
        self.cleanup_task: Optional[asyncio.Task] = None
        self.running = False
        
    async def start(self) -> None:
        """启动结果管理器"""
        if self.running:
            return
        
        self.running = True
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        logging.info("AsyncResultManager started")
    
    async def stop(self) -> None:
        """停止结果管理器"""
        if not self.running:
            return
        
        self.running = False
        
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        # 取消所有等待者
        for waiters in self.waiters.values():
            for waiter in waiters:
                if not waiter.done():
                    waiter.cancel()
        
        logging.info("AsyncResultManager stopped")
    
    async def create_result(self, task_id: str, expires_in: Optional[int] = None) -> str:
        """创建异步结果"""
        result_id = str(uuid.uuid4())
        
        expires_at = None
        if expires_in:
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
        
        result = AsyncResult(
            id=result_id,
            task_id=task_id,
            status=ResultStatus.PENDING,
            created_at=datetime.now(timezone.utc),
            expires_at=expires_at
        )
        
        self.storage.store(result)
        
        # 发送事件
        await event_system.emit_ux_feedback(
            "result_created",
            f"结果对象已创建: {result_id}",
            {"result_id": result_id, "task_id": task_id}
        )
        
        return result_id
    
    async def set_result(self, result_id: str, result: Any, metadata: Dict[str, Any] = None) -> bool:
        """设置结果"""
        stored_result = self.storage.get(result_id)
        if not stored_result:
            return False
        
        stored_result.status = ResultStatus.AVAILABLE
        stored_result.result = result
        stored_result.metadata = metadata or {}
        
        # 通知等待者
        await self._notify_waiters(result_id, stored_result)
        
        # 调用回调函数
        await self._invoke_callbacks(result_id, stored_result)
        
        # 发送事件
        await event_system.emit_ux_feedback(
            "result_ready",
            f"结果已就绪: {result_id}",
            {"result_id": result_id, "task_id": stored_result.task_id}
        )
        
        return True
    
    async def set_error(self, result_id: str, error: str) -> bool:
        """设置错误"""
        stored_result = self.storage.get(result_id)
        if not stored_result:
            return False
        
        stored_result.status = ResultStatus.ERROR
        stored_result.error = error
        
        # 通知等待者
        await self._notify_waiters(result_id, stored_result)
        
        # 调用回调函数
        await self._invoke_callbacks(result_id, stored_result)
        
        # 发送事件
        await event_system.emit_ux_feedback(
            "result_error",
            f"结果出错: {error}",
            {"result_id": result_id, "task_id": stored_result.task_id, "error": error}
        )
        
        return True
    
    async def get_result(self, result_id: str, timeout: Optional[float] = None) -> AsyncResult:
        """获取结果（阻塞等待）"""
        stored_result = self.storage.get(result_id)
        if not stored_result:
            raise ValueError(f"Result {result_id} not found")
        
        # 如果结果已经就绪，直接返回
        if stored_result.is_ready or stored_result.is_error:
            return stored_result
        
        # 等待结果
        return await self._wait_for_result(result_id, timeout)
    
    async def _wait_for_result(self, result_id: str, timeout: Optional[float]) -> AsyncResult:
        """等待结果"""
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        
        # 添加到等待者列表
        if result_id not in self.waiters:
            self.waiters[result_id] = []
        self.waiters[result_id].append(future)
        
        try:
            if timeout:
                result = await asyncio.wait_for(future, timeout=timeout)
            else:
                result = await future
            return result
        except asyncio.TimeoutError:
            # 移除这个future
            if result_id in self.waiters:
                self.waiters[result_id] = [f for f in self.waiters[result_id] if f != future]
            raise asyncio.TimeoutError(f"Timeout waiting for result {result_id}")
        finally:
            # 清理空的等待者列表
            if result_id in self.waiters and not self.waiters[result_id]:
                del self.waiters[result_id]
    
    async def _notify_waiters(self, result_id: str, result: AsyncResult) -> None:
        """通知等待者"""
        waiters = self.waiters.get(result_id, [])
        for waiter in waiters:
            if not waiter.done():
                waiter.set_result(result)
    
    async def _invoke_callbacks(self, result_id: str, result: AsyncResult) -> None:
        """调用回调函数"""
        callbacks = self.result_callbacks.get(result_id, [])
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(result)
                else:
                    callback(result)
            except Exception as e:
                logging.error(f"Result callback error: {e}")
    
    def add_callback(self, result_id: str, callback: Callable[[AsyncResult], None]) -> None:
        """添加结果回调"""
        if result_id not in self.result_callbacks:
            self.result_callbacks[result_id] = []
        self.result_callbacks[result_id].append(callback)
    
    def remove_callback(self, result_id: str, callback: Callable) -> bool:
        """移除回调"""
        if result_id not in self.result_callbacks:
            return False
        
        try:
            self.result_callbacks[result_id].remove(callback)
            return True
        except ValueError:
            return False
    
    async def poll_result(self, result_id: str) -> Optional[AsyncResult]:
        """轮询结果（非阻塞）"""
        return self.storage.get(result_id)
    
    def cancel_result(self, result_id: str) -> bool:
        """取消结果"""
        stored_result = self.storage.get(result_id)
        if not stored_result:
            return False
        
        stored_result.status = ResultStatus.CANCELLED
        
        # 通知等待者
        asyncio.create_task(self._notify_waiters(result_id, stored_result))
        
        return True
    
    def get_result_sync(self, result_id: str) -> Optional[AsyncResult]:
        """同步获取结果"""
        return self.storage.get(result_id)
    
    async def cleanup_old_results(self) -> int:
        """清理旧结果"""
        old_count = 0
        
        # 清理过期的结果
        expired_ids = [
            result_id for result_id, result in self.storage.results.items()
            if result.is_expired
        ]
        
        for result_id in expired_ids:
            if self.storage.delete(result_id):
                old_count += 1
        
        # 清理过旧的PENDING结果（超过1小时）
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=1)
        pending_ids = [
            result_id for result_id, result in self.storage.results.items()
            if result.status == ResultStatus.PENDING and result.created_at < cutoff_time
        ]
        
        for result_id in pending_ids:
            if self.storage.delete(result_id):
                old_count += 1
        
        if old_count > 0:
            logging.info(f"Cleaned up {old_count} old results")
        
        return old_count
    
    async def _cleanup_loop(self) -> None:
        """清理循环"""
        while self.running:
            try:
                await asyncio.sleep(300)  # 5分钟清理一次
                await self.cleanup_old_results()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Cleanup loop error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        storage_stats = self.storage.get_stats()
        
        return {
            **storage_stats,
            "active_waiters": sum(len(waiters) for waiters in self.waiters.values()),
            "registered_callbacks": sum(len(callbacks) for callbacks in self.result_callbacks.values())
        }