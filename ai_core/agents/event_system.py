"""
非阻塞等待系统事件系统
"""

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Callable, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict
import logging


class EventType(Enum):
    """事件类型"""
    # 任务相关事件
    TASK_CREATED = "task_created"
    TASK_STARTED = "task_started"
    TASK_PROGRESS = "task_progress"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_CANCELLED = "task_cancelled"
    TASK_TIMEOUT = "task_timeout"
    
    # 进度相关事件
    PROGRESS_UPDATE = "progress_update"
    PROGRESS_PREDICTION = "progress_prediction"
    PROGRESS_ESTIMATE = "progress_estimate"
    
    # 状态相关事件
    STATUS_CHANGE = "status_change"
    STATUS_QUEUE_POSITION = "status_queue_position"
    STATUS_WAITING = "status_waiting"
    STATUS_PROCESSING = "status_processing"
    
    # 用户体验事件
    UX_FEEDBACK = "ux_feedback"
    UX_NOTIFICATION = "ux_notification"
    UX_WARNING = "ux_warning"
    
    # 系统事件
    SYSTEM_ERROR = "system_error"
    SYSTEM_WARNING = "system_warning"
    SYSTEM_INFO = "system_info"
    
    # WebSocket事件
    WS_CONNECTED = "ws_connected"
    WS_DISCONNECTED = "ws_disconnected"
    WS_MESSAGE = "ws_message"


@dataclass
class Event:
    """事件基类"""
    id: str
    type: EventType
    timestamp: datetime
    source: str
    data: Dict[str, Any]
    correlation_id: Optional[str] = None
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        result['type'] = self.type.value
        result['timestamp'] = self.timestamp.isoformat()
        return result
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """从字典创建事件"""
        data['type'] = EventType(data['type'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Event':
        """从JSON字符串创建事件"""
        return cls.from_dict(json.loads(json_str))


class EventHandler:
    """事件处理器"""
    
    def __init__(self, func: Callable, event_types: List[EventType] = None):
        self.func = func
        self.event_types = event_types or []
        self.id = str(uuid.uuid4())
    
    async def handle(self, event: Event) -> Any:
        """处理事件"""
        try:
            if asyncio.iscoroutinefunction(self.func):
                return await self.func(event)
            else:
                return self.func(event)
        except Exception as e:
            logging.error(f"Event handler error: {e}")
            raise


class EventSystem:
    """事件系统"""
    
    def __init__(self):
        self.handlers: Dict[EventType, List[EventHandler]] = defaultdict(list)
        self.event_history: List[Event] = []
        self.max_history_size = 1000
        self.correlation_map: Dict[str, List[Event]] = defaultdict(list)
        
    def subscribe(self, event_type: EventType, handler: Callable) -> str:
        """订阅事件"""
        event_handler = EventHandler(handler, [event_type])
        self.handlers[event_type].append(event_handler)
        return event_handler.id
    
    def subscribe_multiple(self, event_types: List[EventType], handler: Callable) -> str:
        """订阅多个事件类型"""
        event_handler = EventHandler(handler, event_types)
        for event_type in event_types:
            self.handlers[event_type].append(event_handler)
        return event_handler.id
    
    def unsubscribe(self, handler_id: str) -> bool:
        """取消订阅"""
        for event_type, handlers in self.handlers.items():
            for handler in handlers:
                if handler.id == handler_id:
                    handlers.remove(handler)
                    return True
        return False
    
    async def publish(self, event: Event) -> None:
        """发布事件"""
        # 添加到历史记录
        self.event_history.append(event)
        if len(self.event_history) > self.max_history_size:
            self.event_history.pop(0)
        
        # 添加到关联映射
        if event.correlation_id:
            self.correlation_map[event.correlation_id].append(event)
        
        # 异步分发事件
        tasks = []
        for handler in self.handlers.get(event.type, []):
            tasks.append(handler.handle(event))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def create_event(self, 
                    event_type: EventType, 
                    source: str, 
                    data: Dict[str, Any],
                    correlation_id: str = None,
                    parent_id: str = None,
                    metadata: Dict[str, Any] = None) -> Event:
        """创建事件"""
        return Event(
            id=str(uuid.uuid4()),
            type=event_type,
            timestamp=datetime.now(timezone.utc),
            source=source,
            data=data,
            correlation_id=correlation_id,
            parent_id=parent_id,
            metadata=metadata
        )
    
    async def emit_task_created(self, task_id: str, task_data: Dict[str, Any], correlation_id: str = None) -> None:
        """发送任务创建事件"""
        event = self.create_event(
            EventType.TASK_CREATED,
            "task_manager",
            {"task_id": task_id, **task_data},
            correlation_id
        )
        await self.publish(event)
    
    async def emit_task_progress(self, task_id: str, progress: float, message: str = None, correlation_id: str = None) -> None:
        """发送任务进度事件"""
        event = self.create_event(
            EventType.TASK_PROGRESS,
            "task_manager",
            {
                "task_id": task_id,
                "progress": progress,
                "message": message
            },
            correlation_id
        )
        await self.publish(event)
    
    async def emit_task_completed(self, task_id: str, result: Any, correlation_id: str = None) -> None:
        """发送任务完成事件"""
        event = self.create_event(
            EventType.TASK_COMPLETED,
            "task_manager",
            {
                "task_id": task_id,
                "result": result
            },
            correlation_id
        )
        await self.publish(event)
    
    async def emit_task_failed(self, task_id: str, error: str, correlation_id: str = None) -> None:
        """发送任务失败事件"""
        event = self.create_event(
            EventType.TASK_FAILED,
            "task_manager",
            {
                "task_id": task_id,
                "error": error
            },
            correlation_id
        )
        await self.publish(event)
    
    async def emit_status_change(self, task_id: str, old_status: str, new_status: str, correlation_id: str = None) -> None:
        """发送状态变化事件"""
        event = self.create_event(
            EventType.STATUS_CHANGE,
            "status_manager",
            {
                "task_id": task_id,
                "old_status": old_status,
                "new_status": new_status
            },
            correlation_id
        )
        await self.publish(event)
    
    async def emit_ux_feedback(self, feedback_type: str, message: str, data: Dict[str, Any] = None, correlation_id: str = None) -> None:
        """发送用户体验反馈事件"""
        event = self.create_event(
            EventType.UX_FEEDBACK,
            "user_experience",
            {
                "feedback_type": feedback_type,
                "message": message,
                "data": data or {}
            },
            correlation_id
        )
        await self.publish(event)
    
    def get_event_history(self, limit: int = 100) -> List[Event]:
        """获取事件历史"""
        return self.event_history[-limit:]
    
    def get_correlation_events(self, correlation_id: str) -> List[Event]:
        """获取关联事件"""
        return self.correlation_map.get(correlation_id, [])
    
    def get_events_by_type(self, event_type: EventType, limit: int = 100) -> List[Event]:
        """获取指定类型的事件"""
        return [event for event in self.event_history if event.type == event_type][-limit:]
    
    def clear_history(self) -> None:
        """清除历史记录"""
        self.event_history.clear()
        self.correlation_map.clear()


# 全局事件系统实例
event_system = EventSystem()