#!/usr/bin/env python3
"""
实时状态更新和事件通信系统
负责处理系统内部事件、状态更新和实时通信

作者: 皇
创建时间: 2025-10-31
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Set
from enum import Enum
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import weakref

from ..utils.logger import NormaLogger

class EventPriority(Enum):
    """事件优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

class EventStatus(Enum):
    """事件状态"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class SystemEvent:
    """系统事件类"""
    id: str
    type: str
    source: str
    data: Dict[str, Any]
    priority: EventPriority = EventPriority.NORMAL
    status: EventStatus = EventStatus.PENDING
    timestamp: datetime = None
    correlation_id: Optional[str] = None
    parent_id: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "type": self.type,
            "source": self.source,
            "data": self.data,
            "priority": self.priority.value,
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "parent_id": self.parent_id
        }
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(',', ':'))

class EventFilter:
    """事件过滤器"""
    
    def __init__(self):
        self.type_filters: Set[str] = set()
        self.source_filters: Set[str] = set()
        self.priority_filters: Set[EventPriority] = set()
        self.status_filters: Set[EventStatus] = set()
        self.custom_filters: List[Callable[[SystemEvent], bool]] = []
    
    def add_type_filter(self, event_type: str) -> None:
        """添加事件类型过滤器"""
        self.type_filters.add(event_type)
    
    def add_source_filter(self, source: str) -> None:
        """添加事件源过滤器"""
        self.source_filters.add(source)
    
    def add_priority_filter(self, priority: EventPriority) -> None:
        """添加优先级过滤器"""
        self.priority_filters.add(priority)
    
    def add_status_filter(self, status: EventStatus) -> None:
        """添加状态过滤器"""
        self.status_filters.add(status)
    
    def add_custom_filter(self, filter_func: Callable[[SystemEvent], bool]) -> None:
        """添加自定义过滤器"""
        self.custom_filters.append(filter_func)
    
    def matches(self, event: SystemEvent) -> bool:
        """检查事件是否匹配过滤器"""
        # 类型检查
        if self.type_filters and event.type not in self.type_filters:
            return False
        
        # 源检查
        if self.source_filters and event.source not in self.source_filters:
            return False
        
        # 优先级检查
        if self.priority_filters and event.priority not in self.priority_filters:
            return False
        
        # 状态检查
        if self.status_filters and event.status not in self.status_filters:
            return False
        
        # 自定义过滤器检查
        for custom_filter in self.custom_filters:
            if not custom_filter(event):
                return False
        
        return True

class EventListener:
    """事件监听器"""
    
    def __init__(
        self,
        listener_id: str,
        event_types: List[str],
        handler: Callable,
        filter_obj: Optional[EventFilter] = None,
        priority: EventPriority = EventPriority.NORMAL
    ):
        self.listener_id = listener_id
        self.event_types = set(event_types)
        self.handler = handler
        self.filter_obj = filter_obj
        self.priority = priority
        self.call_count = 0
        self.last_called = None
        self.is_active = True
    
    def can_handle(self, event: SystemEvent) -> bool:
        """检查是否可以处理事件"""
        if not self.is_active:
            return False
        
        # 检查事件类型
        if self.event_types and event.type not in self.event_types:
            return False
        
        # 检查过滤器
        if self.filter_obj and not self.filter_obj.matches(event):
            return False
        
        return True
    
    async def handle_event(self, event: SystemEvent) -> bool:
        """处理事件"""
        try:
            if not self.can_handle(event):
                return False
            
            # 更新统计
            self.call_count += 1
            self.last_called = datetime.now()
            
            # 调用处理器
            if asyncio.iscoroutinefunction(self.handler):
                await self.handler(event)
            else:
                self.handler(event)
            
            return True
            
        except Exception as e:
            # 这里应该记录错误，但避免循环依赖
            print(f"事件处理器错误 {self.listener_id}: {e}")
            return False

class EventQueue:
    """事件队列"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self.priority_queues: Dict[EventPriority, asyncio.Queue] = {
            priority: asyncio.Queue(maxsize=max_size) for priority in EventPriority
        }
        self.use_priority = False
    
    async def put(self, event: SystemEvent) -> bool:
        """添加事件到队列"""
        try:
            if self.use_priority:
                priority_queue = self.priority_queues[event.priority]
                await priority_queue.put(event)
            else:
                await self.queue.put(event)
            return True
        except asyncio.QueueFull:
            return False
    
    async def get(self) -> SystemEvent:
        """从队列获取事件"""
        if self.use_priority:
            # 按优先级顺序检查队列
            for priority in EventPriority:
                priority_queue = self.priority_queues[priority]
                try:
                    return priority_queue.get_nowait()
                except asyncio.QueueEmpty:
                    continue
            # 如果所有优先级队列都为空，等待
            return await self.queue.get()
        else:
            return await self.queue.get()
    
    def size(self) -> int:
        """获取队列大小"""
        if self.use_priority:
            return sum(q.qsize() for q in self.priority_queues.values())
        else:
            return self.queue.qsize()
    
    def empty(self) -> bool:
        """检查队列是否为空"""
        if self.use_priority:
            return all(q.empty() for q in self.priority_queues.values())
        else:
            return self.queue.empty()

class EventRouter:
    """事件路由器"""
    
    def __init__(self):
        self.routes: Dict[str, List[EventListener]] = defaultdict(list)
        self.wildcard_listeners: List[EventListener] = []
    
    def add_route(self, event_type: str, listener: EventListener) -> None:
        """添加路由"""
        self.routes[event_type].append(listener)
    
    def add_wildcard_listener(self, listener: EventListener) -> None:
        """添加通配符监听器"""
        self.wildcard_listeners.append(listener)
    
    async def route_event(self, event: SystemEvent) -> List[bool]:
        """路由事件到相应监听器"""
        results = []
        
        # 路由到特定类型监听器
        if event.type in self.routes:
            for listener in self.routes[event.type]:
                result = await listener.handle_event(event)
                results.append(result)
        
        # 路由到通配符监听器
        for listener in self.wildcard_listeners:
            if listener.can_handle(event):
                result = await listener.handle_event(event)
                results.append(result)
        
        return results

class EventSystem:
    """事件系统主类"""
    
    def __init__(self):
        self.logger = NormaLogger("event_system")
        
        # 核心组件
        self.event_queue = EventQueue()
        self.event_router = EventRouter()
        self.event_history: deque = deque(maxlen=10000)
        
        # 监听器管理
        self.listeners: Dict[str, EventListener] = {}
        self.listener_counter = 0
        
        # 状态管理
        self.is_initialized = False
        self.is_running = False
        self.is_processing = False
        
        # 统计信息
        self.stats = {
            "total_events": 0,
            "processed_events": 0,
            "failed_events": 0,
            "active_listeners": 0,
            "queue_size": 0
        }
        
        # 事件处理任务
        self.processing_task: Optional[asyncio.Task] = None
    
    async def initialize(self) -> bool:
        """初始化事件系统"""
        try:
            self.logger.info("初始化事件系统...")
            
            # 设置默认监听器
            await self._setup_default_listeners()
            
            self.is_initialized = True
            self.logger.info("事件系统初始化完成")
            return True
            
        except Exception as e:
            self.logger.error(f"事件系统初始化失败: {e}")
            return False
    
    async def _setup_default_listeners(self) -> None:
        """设置默认监听器"""
        
        # 系统状态监听器
        await self.add_event_listener(
            event_types=["system.*"],
            handler=self._handle_system_events,
            description="系统事件处理器"
        )
        
        # 对话事件监听器
        await self.add_event_listener(
            event_types=["conversation.*", "message.*"],
            handler=self._handle_conversation_events,
            description="对话事件处理器"
        )
        
        # 错误事件监听器
        await self.add_event_listener(
            event_types=["*.error", "*.failed"],
            handler=self._handle_error_events,
            description="错误事件处理器"
        )
    
    async def start(self) -> bool:
        """启动事件系统"""
        if not self.is_initialized:
            self.logger.error("事件系统尚未初始化")
            return False
        
        try:
            self.logger.info("启动事件系统...")
            
            # 启动事件处理循环
            self.processing_task = asyncio.create_task(self._event_processing_loop())
            
            self.is_running = True
            self.logger.info("事件系统启动完成")
            return True
            
        except Exception as e:
            self.logger.error(f"事件系统启动失败: {e}")
            return False
    
    async def stop(self) -> bool:
        """停止事件系统"""
        if not self.is_running:
            return True
        
        try:
            self.logger.info("停止事件系统...")
            
            # 停止事件处理
            self.is_running = False
            
            # 取消处理任务
            if self.processing_task:
                self.processing_task.cancel()
                try:
                    await self.processing_task
                except asyncio.CancelledError:
                    pass
            
            self.logger.info("事件系统已停止")
            return True
            
        except Exception as e:
            self.logger.error(f"事件系统停止失败: {e}")
            return False
    
    async def publish_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        source: str = "system",
        priority: EventPriority = EventPriority.NORMAL,
        correlation_id: Optional[str] = None
    ) -> str:
        """发布事件"""
        
        event = SystemEvent(
            id=str(uuid.uuid4()),
            type=event_type,
            source=source,
            data=data,
            priority=priority,
            correlation_id=correlation_id
        )
        
        # 添加到历史
        self.event_history.append(event)
        
        # 更新统计
        self.stats["total_events"] += 1
        
        # 添加到队列
        await self.event_queue.put(event)
        
        self.logger.debug(f"发布事件: {event_type}")
        return event.id
    
    async def add_event_listener(
        self,
        event_types: List[str],
        handler: Callable,
        description: str = "",
        filter_obj: Optional[EventFilter] = None,
        priority: EventPriority = EventPriority.NORMAL
    ) -> str:
        """添加事件监听器"""
        
        self.listener_counter += 1
        listener_id = f"listener_{self.listener_counter}"
        
        listener = EventListener(
            listener_id=listener_id,
            event_types=event_types,
            handler=handler,
            filter_obj=filter_obj,
            priority=priority
        )
        
        self.listeners[listener_id] = listener
        self.stats["active_listeners"] += 1
        
        # 注册到路由器
        for event_type in event_types:
            if "*" in event_type:
                self.event_router.add_wildcard_listener(listener)
            else:
                self.event_router.add_route(event_type, listener)
        
        self.logger.info(f"添加事件监听器: {listener_id} - {description}")
        return listener_id
    
    async def remove_event_listener(self, listener_id: str) -> bool:
        """移除事件监听器"""
        
        if listener_id not in self.listeners:
            return False
        
        listener = self.listeners[listener_id]
        listener.is_active = False
        
        del self.listeners[listener_id]
        self.stats["active_listeners"] -= 1
        
        self.logger.info(f"移除事件监听器: {listener_id}")
        return True
    
    async def _event_processing_loop(self) -> None:
        """事件处理循环"""
        
        while self.is_running:
            try:
                # 从队列获取事件
                event = await self.event_queue.get()
                
                # 更新状态
                event.status = EventStatus.PROCESSING
                self.is_processing = True
                
                # 路由事件
                results = await self.event_router.route_event(event)
                
                # 更新事件状态
                if any(results):
                    event.status = EventStatus.COMPLETED
                    self.stats["processed_events"] += 1
                else:
                    event.status = EventStatus.FAILED
                    self.stats["failed_events"] += 1
                
                self.is_processing = False
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"事件处理错误: {e}")
                self.is_processing = False
    
    # 默认事件处理器
    async def _handle_system_events(self, event: SystemEvent) -> None:
        """处理系统事件"""
        self.logger.debug(f"系统事件: {event.type} - {event.data}")
    
    async def _handle_conversation_events(self, event: SystemEvent) -> None:
        """处理对话事件"""
        self.logger.debug(f"对话事件: {event.type} - {event.data}")
    
    async def _handle_error_events(self, event: SystemEvent) -> None:
        """处理错误事件"""
        self.logger.error(f"错误事件: {event.type} - {event.data}")
    
    async def get_event_history(
        self,
        event_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """获取事件历史"""
        
        events = list(self.event_history)
        
        # 按类型过滤
        if event_type:
            events = [e for e in events if e.type == event_type]
        
        # 限制数量
        events = events[-limit:]
        
        return [event.to_dict() for event in events]
    
    async def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        
        return {
            "component": "event_system",
            "status": "running" if self.is_running else "stopped",
            "initialized": self.is_initialized,
            "processing": self.is_processing,
            "stats": self.stats.copy(),
            "queue_size": self.event_queue.size(),
            "active_listeners": len(self.listeners),
            "timestamp": datetime.now().isoformat()
        }
    
    async def clear_history(self) -> int:
        """清空事件历史"""
        
        cleared_count = len(self.event_history)
        self.event_history.clear()
        
        self.logger.info(f"清空事件历史: {cleared_count} 条")
        return cleared_count
    
    async def reset_stats(self) -> None:
        """重置统计信息"""
        
        self.stats = {
            "total_events": 0,
            "processed_events": 0,
            "failed_events": 0,
            "active_listeners": 0,
            "queue_size": 0
        }
        
        self.logger.info("重置事件系统统计")
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        
        return {
            "component": "event_system",
            "status": "healthy" if self.is_running else "stopped",
            "initialized": self.is_initialized,
            "running": self.is_running,
            "processing": self.is_processing,
            "queue_size": self.event_queue.size(),
            "history_size": len(self.event_history),
            "active_listeners": len(self.listeners),
            "stats": self.stats.copy()
        }
    
    # 便利方法
    async def emit_system_status(self, status_data: Dict[str, Any]) -> None:
        """发射系统状态事件"""
        await self.publish_event(
            "system.status",
            status_data,
            source="event_system"
        )
    
    async def emit_conversation_start(self, session_id: str, user_id: str = None) -> None:
        """发射对话开始事件"""
        await self.publish_event(
            "conversation.start",
            {
                "session_id": session_id,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            },
            source="conversation_engine"
        )
    
    async def emit_conversation_end(self, session_id: str, message_count: int) -> None:
        """发射对话结束事件"""
        await self.publish_event(
            "conversation.end",
            {
                "session_id": session_id,
                "message_count": message_count,
                "timestamp": datetime.now().isoformat()
            },
            source="conversation_engine"
        )
    
    async def emit_error(self, error_type: str, error_message: str, context: Dict[str, Any] = None) -> None:
        """发射错误事件"""
        await self.publish_event(
            f"{error_type}.error",
            {
                "message": error_message,
                "context": context or {},
                "timestamp": datetime.now().isoformat()
            },
            source="system",
            priority=EventPriority.HIGH
        )