"""
AG-UI协议实现 - Agent与UI之间的通信协议
"""
import asyncio
import json
import time
import uuid
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import logging


class MessageType(Enum):
    """消息类型枚举"""
    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    NOTIFICATION = "notification"
    HEARTBEAT = "heartbeat"
    ERROR = "error"


class EventType(Enum):
    """事件类型枚举"""
    AGENT_STATUS_CHANGE = "agent_status_change"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    NEW_MESSAGE = "new_message"
    SYSTEM_ALERT = "system_alert"
    DATA_UPDATE = "data_update"
    USER_ACTION = "user_action"


@dataclass
class AGUIMessage:
    """AG-UI消息结构"""
    id: str
    type: MessageType
    timestamp: datetime
    source: str
    target: str
    payload: Dict[str, Any]
    correlation_id: Optional[str] = None
    priority: int = 1  # 1-5, 5为最高优先级
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class GUIEvent:
    """GUI事件结构"""
    id: str
    type: EventType
    timestamp: datetime
    source: str
    data: Dict[str, Any]
    severity: str = "info"  # info, warning, error, critical


class AGUIProtocol:
    """AG-UI协议处理器"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.message_handlers = {}
        self.event_listeners = {}
        self.message_queue = asyncio.Queue()
        self.response_waiters = {}  # correlation_id -> asyncio.Future
        self.logger = logging.getLogger(f"agui.protocol.{agent_id}")
        
    def register_message_handler(self, message_type: MessageType, handler: Callable):
        """注册消息处理器"""
        self.message_handlers[message_type] = handler
        self.logger.info(f"注册消息处理器: {message_type.value}")
    
    def register_event_listener(self, event_type: EventType, listener: Callable):
        """注册事件监听器"""
        if event_type not in self.event_listeners:
            self.event_listeners[event_type] = []
        self.event_listeners[event_type].append(listener)
        self.logger.info(f"注册事件监听器: {event_type.value}")
    
    async def send_message(self, target: str, message_type: MessageType, 
                          payload: Dict[str, Any], priority: int = 1) -> str:
        """发送消息"""
        message = AGUIMessage(
            id=str(uuid.uuid4()),
            type=message_type,
            timestamp=datetime.now(),
            source=self.agent_id,
            target=target,
            payload=payload,
            priority=priority
        )
        
        await self.message_queue.put(message)
        self.logger.debug(f"发送消息: {message.id} -> {target}")
        return message.id
    
    async def send_request(self, target: str, action: str, 
                          data: Dict[str, Any] = None, timeout: float = 30.0) -> Dict[str, Any]:
        """发送请求并等待响应"""
        correlation_id = str(uuid.uuid4())
        
        # 创建等待响应的Future
        response_future = asyncio.Future()
        self.response_waiters[correlation_id] = response_future
        
        try:
            # 发送请求
            await self.send_message(
                target=target,
                message_type=MessageType.REQUEST,
                payload={
                    "action": action,
                    "data": data or {},
                    "correlation_id": correlation_id
                }
            )
            
            # 等待响应
            response = await asyncio.wait_for(response_future, timeout=timeout)
            return response
            
        except asyncio.TimeoutError:
            self.logger.warning(f"请求超时: {correlation_id}")
            return {"status": "error", "message": "请求超时"}
        finally:
            # 清理等待器
            self.response_waiters.pop(correlation_id, None)
    
    async def send_response(self, target: str, correlation_id: str, 
                           status: str, data: Dict[str, Any] = None):
        """发送响应"""
        await self.send_message(
            target=target,
            message_type=MessageType.RESPONSE,
            payload={
                "status": status,
                "data": data or {},
                "correlation_id": correlation_id
            }
        )
    
    async def emit_event(self, event_type: EventType, data: Dict[str, Any], 
                        severity: str = "info"):
        """发射事件"""
        event = GUIEvent(
            id=str(uuid.uuid4()),
            type=event_type,
            timestamp=datetime.now(),
            source=self.agent_id,
            data=data,
            severity=severity
        )
        
        # 通知所有监听器
        listeners = self.event_listeners.get(event_type, [])
        for listener in listeners:
            try:
                await listener(event)
            except Exception as e:
                self.logger.error(f"事件监听器执行失败: {e}")
        
        self.logger.debug(f"发射事件: {event_type.value}")
    
    async def handle_message(self, message: AGUIMessage):
        """处理接收到的消息"""
        try:
            handler = self.message_handlers.get(message.type)
            if handler:
                await handler(message)
            else:
                self.logger.warning(f"未找到消息处理器: {message.type}")
                
        except Exception as e:
            self.logger.error(f"消息处理失败: {e}")
    
    async def handle_response(self, correlation_id: str, response_data: Dict[str, Any]):
        """处理响应"""
        if correlation_id in self.response_waiters:
            future = self.response_waiters[correlation_id]
            if not future.done():
                future.set_result(response_data)
    
    def get_protocol_status(self) -> Dict[str, Any]:
        """获取协议状态"""
        return {
            "agent_id": self.agent_id,
            "message_handlers": len(self.message_handlers),
            "event_listeners": sum(len(listeners) for listeners in self.event_listeners.values()),
            "pending_responses": len(self.response_waiters),
            "queue_size": self.message_queue.qsize()
        }


class RealtimeEventBus:
    """实时事件总线"""
    
    def __init__(self):
        self.subscribers = {}  # event_type -> List[subscriber]
        self.event_history = []
        self.max_history_size = 1000
        self.logger = logging.getLogger("event.bus")
        
    def subscribe(self, event_type: EventType, subscriber: Callable):
        """订阅事件"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(subscriber)
        self.logger.info(f"订阅事件: {event_type.value}")
    
    def unsubscribe(self, event_type: EventType, subscriber: Callable):
        """取消订阅"""
        if event_type in self.subscribers:
            try:
                self.subscribers[event_type].remove(subscriber)
                self.logger.info(f"取消订阅事件: {event_type.value}")
            except ValueError:
                pass
    
    async def publish(self, event: GUIEvent):
        """发布事件"""
        # 添加到历史记录
        self.event_history.append(event)
        if len(self.event_history) > self.max_history_size:
            self.event_history = self.event_history[-self.max_history_size:]
        
        # 通知订阅者
        subscribers = self.subscribers.get(event.type, [])
        for subscriber in subscribers:
            try:
                await subscriber(event)
            except Exception as e:
                self.logger.error(f"事件订阅者执行失败: {e}")
        
        self.logger.debug(f"发布事件: {event.type.value}")
    
    def get_event_history(self, event_type: EventType = None, 
                         limit: int = 100) -> List[Dict[str, Any]]:
        """获取事件历史"""
        history = self.event_history
        
        if event_type:
            history = [e for e in history if e.type == event_type]
        
        return [asdict(event) for event in history[-limit:]]
    
    def get_bus_status(self) -> Dict[str, Any]:
        """获取事件总线状态"""
        return {
            "total_subscribers": sum(len(subs) for subs in self.subscribers.values()),
            "event_types": list(self.subscribers.keys()),
            "history_size": len(self.event_history),
            "max_history_size": self.max_history_size
        }


class WebSocketManager:
    """WebSocket连接管理器"""
    
    def __init__(self):
        self.connections = {}  # connection_id -> connection_info
        self.message_queue = asyncio.Queue()
        self.logger = logging.getLogger("websocket.manager")
        
    async def connect(self, connection_id: str, websocket, user_id: str = None):
        """建立WebSocket连接"""
        self.connections[connection_id] = {
            "websocket": websocket,
            "user_id": user_id,
            "connected_at": datetime.now(),
            "last_activity": datetime.now()
        }
        self.logger.info(f"WebSocket连接建立: {connection_id}")
    
    async def disconnect(self, connection_id: str):
        """断开WebSocket连接"""
        if connection_id in self.connections:
            del self.connections[connection_id]
            self.logger.info(f"WebSocket连接断开: {connection_id}")
    
    async def send_message(self, connection_id: str, message: Dict[str, Any]):
        """发送消息到指定连接"""
        if connection_id in self.connections:
            try:
                connection_info = self.connections[connection_id]
                websocket = connection_info["websocket"]
                
                await websocket.send_text(json.dumps(message))
                connection_info["last_activity"] = datetime.now()
                
            except Exception as e:
                self.logger.error(f"发送WebSocket消息失败: {e}")
                await self.disconnect(connection_id)
    
    async def broadcast(self, message: Dict[str, Any], target_user: str = None):
        """广播消息"""
        disconnected = []
        
        for conn_id, connection_info in self.connections.items():
            # 如果指定了目标用户，只发送给该用户
            if target_user and connection_info.get("user_id") != target_user:
                continue
                
            try:
                websocket = connection_info["websocket"]
                await websocket.send_text(json.dumps(message))
                connection_info["last_activity"] = datetime.now()
                
            except Exception as e:
                self.logger.error(f"广播消息失败: {e}")
                disconnected.append(conn_id)
        
        # 清理断开的连接
        for conn_id in disconnected:
            await self.disconnect(conn_id)
    
    def get_connection_status(self) -> Dict[str, Any]:
        """获取连接状态"""
        return {
            "total_connections": len(self.connections),
            "connections": {
                conn_id: {
                    "user_id": info.get("user_id"),
                    "connected_at": info["connected_at"].isoformat(),
                    "last_activity": info["last_activity"].isoformat()
                }
                for conn_id, info in self.connections.items()
            }
        }