"""
非阻塞等待系统实时状态管理器
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging

from .config import NonBlockingConfig
from .event_system import event_system, EventType
from .task_manager import TaskManager, TaskStatus, Task


@dataclass
class StatusUpdate:
    """状态更新"""
    task_id: str
    old_status: str
    new_status: str
    timestamp: datetime
    message: str
    metadata: Dict[str, Any] = None
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class ConnectionManager:
    """连接管理器"""
    
    def __init__(self):
        self.connections: Dict[str, Any] = {}  # connection_id -> connection
        self.task_subscriptions: Dict[str, Set[str]] = defaultdict(set)  # task_id -> set of connection_ids
        self.user_subscriptions: Dict[str, Set[str]] = defaultdict(set)  # user_id -> set of connection_ids
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}  # connection_id -> metadata
    
    def add_connection(self, connection_id: str, connection: Any, metadata: Dict[str, Any] = None) -> None:
        """添加连接"""
        self.connections[connection_id] = connection
        self.connection_metadata[connection_id] = metadata or {}
    
    def remove_connection(self, connection_id: str) -> None:
        """移除连接"""
        self.connections.pop(connection_id, None)
        self.connection_metadata.pop(connection_id, None)
        
        # 从所有订阅中移除
        for task_id in list(self.task_subscriptions.keys()):
            self.task_subscriptions[task_id].discard(connection_id)
            if not self.task_subscriptions[task_id]:
                del self.task_subscriptions[task_id]
        
        for user_id in list(self.user_subscriptions.keys()):
            self.user_subscriptions[user_id].discard(connection_id)
            if not self.user_subscriptions[user_id]:
                del self.user_subscriptions[user_id]
    
    def subscribe_to_task(self, connection_id: str, task_id: str) -> None:
        """订阅任务"""
        self.task_subscriptions[task_id].add(connection_id)
    
    def unsubscribe_from_task(self, connection_id: str, task_id: str) -> None:
        """取消订阅任务"""
        self.task_subscriptions[task_id].discard(connection_id)
    
    def subscribe_to_user(self, connection_id: str, user_id: str) -> None:
        """订阅用户"""
        self.user_subscriptions[user_id].add(connection_id)
    
    def unsubscribe_from_user(self, connection_id: str, user_id: str) -> None:
        """取消订阅用户"""
        self.user_subscriptions[user_id].discard(connection_id)
    
    def get_task_subscribers(self, task_id: str) -> Set[str]:
        """获取任务的订阅者"""
        return self.task_subscriptions.get(task_id, set()).copy()
    
    def get_user_subscribers(self, user_id: str) -> Set[str]:
        """获取用户的订阅者"""
        return self.user_subscriptions.get(user_id, set()).copy()
    
    def get_connection_count(self) -> int:
        """获取连接数"""
        return len(self.connections)


class RealtimeStatusManager:
    """实时状态管理器"""
    
    def __init__(self, task_manager: TaskManager, config: Optional[NonBlockingConfig] = None):
        self.task_manager = task_manager
        self.config = config or NonBlockingConfig()
        self.connection_manager = ConnectionManager()
        self.status_history: Dict[str, List[StatusUpdate]] = defaultdict(list)
        self.status_listeners: Dict[str, List[Callable]] = defaultdict(list)
        self.running = False
        self.status_check_task: Optional[asyncio.Task] = None
        
    async def start(self) -> None:
        """启动状态管理器"""
        if self.running:
            return
        
        self.running = True
        self.status_check_task = asyncio.create_task(self._status_check_loop())
        logging.info("RealtimeStatusManager started")
    
    async def stop(self) -> None:
        """停止状态管理器"""
        if not self.running:
            return
        
        self.running = False
        
        if self.status_check_task:
            self.status_check_task.cancel()
            try:
                await self.status_check_task
            except asyncio.CancelledError:
                pass
        
        logging.info("RealtimeStatusManager stopped")
    
    def add_connection(self, connection_id: str, connection: Any, metadata: Dict[str, Any] = None) -> None:
        """添加连接"""
        self.connection_manager.add_connection(connection_id, connection, metadata)
    
    def remove_connection(self, connection_id: str) -> None:
        """移除连接"""
        self.connection_manager.remove_connection(connection_id)
    
    def subscribe_to_task(self, connection_id: str, task_id: str) -> None:
        """订阅任务状态"""
        self.connection_manager.subscribe_to_task(connection_id, task_id)
        
        # 发送当前状态
        task = self.task_manager.get_task(task_id)
        if task:
            asyncio.create_task(self._send_current_status(connection_id, task))
    
    def unsubscribe_from_task(self, connection_id: str, task_id: str) -> None:
        """取消订阅任务"""
        self.connection_manager.unsubscribe_from_task(connection_id, task_id)
    
    def subscribe_to_user(self, connection_id: str, user_id: str) -> None:
        """订阅用户的所有任务"""
        self.connection_manager.subscribe_to_user(connection_id, user_id)
        
        # 发送用户的所有任务状态
        tasks = self.task_manager.get_tasks_by_correlation(user_id)
        for task in tasks:
            asyncio.create_task(self._send_current_status(connection_id, task))
    
    def unsubscribe_from_user(self, connection_id: str, user_id: str) -> None:
        """取消订阅用户"""
        self.connection_manager.unsubscribe_from_user(connection_id, user_id)
    
    async def broadcast_status_update(self, task_id: str, old_status: str, new_status: str, message: str = None, metadata: Dict[str, Any] = None, correlation_id: str = None) -> None:
        """广播状态更新"""
        update = StatusUpdate(
            task_id=task_id,
            old_status=old_status,
            new_status=new_status,
            timestamp=datetime.now(timezone.utc),
            message=message or self._get_status_message(new_status),
            metadata=metadata or {},
            correlation_id=correlation_id
        )
        
        # 保存到历史记录
        self.status_history[task_id].append(update)
        if len(self.status_history[task_id]) > 100:
            self.status_history[task_id].pop(0)
        
        # 获取订阅者
        subscribers = self.connection_manager.get_task_subscribers(task_id)
        
        # 广播给所有订阅者
        for connection_id in subscribers:
            await self._send_to_connection(connection_id, update)
        
        # 调用状态监听器
        for listener in self.status_listeners.get(task_id, []):
            try:
                if asyncio.iscoroutinefunction(listener):
                    await listener(update)
                else:
                    listener(update)
            except Exception as e:
                logging.error(f"Status listener error: {e}")
        
        # 发送事件
        await event_system.emit_status_change(task_id, old_status, new_status, correlation_id)
    
    async def _send_current_status(self, connection_id: str, task: Task) -> None:
        """发送当前状态"""
        update = StatusUpdate(
            task_id=task.id,
            old_status="unknown",
            new_status=task.status.value,
            timestamp=datetime.now(timezone.utc),
            message=self._get_status_message(task.status.value),
            metadata={
                "progress": task.progress,
                "queue_position": self.task_manager.get_queue_position(task.id),
                "estimated_duration": task.duration
            },
            correlation_id=task.correlation_id
        )
        
        await self._send_to_connection(connection_id, update)
    
    async def _send_to_connection(self, connection_id: str, update: StatusUpdate) -> None:
        """发送到连接"""
        connection = self.connection_manager.connections.get(connection_id)
        if not connection:
            return
        
        try:
            # 根据连接类型发送消息
            if hasattr(connection, 'send_text'):
                # WebSocket连接
                await connection.send_text(json.dumps(update.to_dict(), ensure_ascii=False))
            elif hasattr(connection, 'write_message'):
                # WebSocket连接（同步版本）
                connection.write_message(json.dumps(update.to_dict(), ensure_ascii=False))
            elif hasattr(connection, 'send'):
                # 其他类型的连接
                await connection.send(update.to_dict())
            else:
                logging.warning(f"Unknown connection type for {connection_id}")
        except Exception as e:
            logging.error(f"Error sending to connection {connection_id}: {e}")
            # 连接可能已断开，清理连接
            self.connection_manager.remove_connection(connection_id)
    
    def _get_status_message(self, status: str) -> str:
        """获取状态消息"""
        messages = {
            "pending": "等待处理",
            "queued": "排队中",
            "running": "正在执行",
            "completed": "已完成",
            "failed": "执行失败",
            "cancelled": "已取消",
            "timeout": "执行超时"
        }
        return messages.get(status, "未知状态")
    
    async def _status_check_loop(self) -> None:
        """状态检查循环"""
        while self.running:
            try:
                await asyncio.sleep(1.0)  # 每秒检查一次
                
                # 检查运行中的任务状态
                running_tasks = self.task_manager.get_tasks_by_status(TaskStatus.RUNNING)
                for task in running_tasks:
                    # 这里可以添加额外的状态检查逻辑
                    # 比如检查任务是否真的在运行，是否有死锁等
                    pass
                
                # 清理过期的状态历史
                await self._cleanup_old_status_history()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Status check loop error: {e}")
    
    async def _cleanup_old_status_history(self) -> None:
        """清理过期的状态历史"""
        cutoff_time = datetime.now(timezone.utc).timestamp() - 3600  # 1小时前
        
        for task_id in list(self.status_history.keys()):
            history = self.status_history[task_id]
            # 保留最近1小时的状态更新
            recent_updates = [
                update for update in history
                if update.timestamp.timestamp() > cutoff_time
            ]
            
            if recent_updates:
                self.status_history[task_id] = recent_updates
            else:
                del self.status_history[task_id]
    
    def add_status_listener(self, task_id: str, listener: Callable[[StatusUpdate], None]) -> None:
        """添加状态监听器"""
        self.status_listeners[task_id].append(listener)
    
    def remove_status_listener(self, task_id: str, listener: Callable) -> bool:
        """移除状态监听器"""
        if task_id in self.status_listeners:
            try:
                self.status_listeners[task_id].remove(listener)
                return True
            except ValueError:
                pass
        return False
    
    def get_status_history(self, task_id: str, limit: int = 50) -> List[StatusUpdate]:
        """获取状态历史"""
        return self.status_history.get(task_id, [])[-limit:]
    
    def get_queue_info(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取队列信息"""
        task = self.task_manager.get_task(task_id)
        if not task:
            return None
        
        queue_position = self.task_manager.get_queue_position(task_id)
        
        return {
            "task_id": task_id,
            "current_status": task.status.value,
            "queue_position": queue_position,
            "estimated_wait_time": self._estimate_wait_time(queue_position),
            "total_queued": len(self.task_manager.get_tasks_by_status(TaskStatus.PENDING))
        }
    
    def _estimate_wait_time(self, queue_position: Optional[int]) -> Optional[float]:
        """估算等待时间"""
        if queue_position is None:
            return None
        
        # 简单的等待时间估算：队列位置 * 平均任务执行时间
        # 这里可以基于历史数据来优化估算
        avg_task_duration = 30.0  # 假设平均任务执行30秒
        return queue_position * avg_task_duration
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """获取连接统计"""
        total_connections = self.connection_manager.get_connection_count()
        
        task_subscriptions = sum(
            len(subscribers) for subscribers in self.connection_manager.task_subscriptions.values()
        )
        
        user_subscriptions = sum(
            len(subscribers) for subscribers in self.connection_manager.user_subscriptions.values()
        )
        
        return {
            "total_connections": total_connections,
            "task_subscriptions": task_subscriptions,
            "user_subscriptions": user_subscriptions,
            "tracked_tasks": len(self.status_history)
        }