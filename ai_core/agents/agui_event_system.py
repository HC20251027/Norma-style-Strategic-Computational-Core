#!/usr/bin/env python3
"""
诺玛AI系统 - AG-UI事件系统
实现AG-UI标准事件编码器、事件流管理器、WebSocket桥接和事件过滤路由

作者: 皇
创建时间: 2025-10-31
版本: 1.0.0
"""

import asyncio
import json
import uuid
import time
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Callable, Set, Union
from enum import Enum
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import websockets
from fastapi import WebSocket, WebSocketDisconnect

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EventType(Enum):
    """AG-UI事件类型枚举"""
    # 系统事件
    SYSTEM_START = "system.start"
    SYSTEM_STOP = "system.stop"
    SYSTEM_ERROR = "system.error"
    SYSTEM_STATUS = "system.status"
    
    # 用户交互事件
    USER_MESSAGE = "user.message"
    USER_ACTION = "user.action"
    USER_INPUT = "user.input"
    
    # AI响应事件
    AI_RESPONSE = "ai.response"
    AI_THINKING = "ai.thinking"
    AI_PROCESSING = "ai.processing"
    
    # 数据事件
    DATA_UPDATE = "data.update"
    DATA_CREATE = "data.create"
    DATA_DELETE = "data.delete"
    DATA_QUERY = "data.query"
    
    # 血统分析事件
    BLOOD_ANALYSIS_START = "blood.analysis.start"
    BLOOD_ANALYSIS_COMPLETE = "blood.analysis.complete"
    BLOOD_RESULT = "blood.result"
    
    # 安全事件
    SECURITY_SCAN = "security.scan"
    SECURITY_ALERT = "security.alert"
    SECURITY_STATUS = "security.status"
    
    # 网络事件
    NETWORK_CONNECT = "network.connect"
    NETWORK_DISCONNECT = "network.disconnect"
    NETWORK_ERROR = "network.error"
    
    # 知识库事件
    KB_SEARCH = "kb.search"
    KB_RESULT = "kb.result"
    KB_UPDATE = "kb.update"
    
    # 多智能体事件
    AGENT_ACTIVITY = "agent.activity"
    AGENT_COLLABORATION = "agent.collaboration"
    AGENT_STATUS = "agent.status"

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
class AGUIEvent:
    """AG-UI标准事件结构"""
    id: str
    type: EventType
    timestamp: str
    source: str
    target: Optional[str] = None
    priority: EventPriority = EventPriority.NORMAL
    status: EventStatus = EventStatus.PENDING
    data: Dict[str, Any] = None
    metadata: Dict[str, Any] = None
    correlation_id: Optional[str] = None
    parent_id: Optional[str] = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = {}
        if self.metadata is None:
            self.metadata = {}
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = asdict(self)
        result['type'] = self.type.value
        result['priority'] = self.priority.value
        result['status'] = self.status.value
        return result
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(',', ':'))
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AGUIEvent':
        """从字典创建事件"""
        data = data.copy()
        data['type'] = EventType(data['type'])
        data['priority'] = EventPriority(data['priority'])
        data['status'] = EventStatus(data['status'])
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'AGUIEvent':
        """从JSON字符串创建事件"""
        return cls.from_dict(json.loads(json_str))

class EventFilter:
    """事件过滤器"""
    
    def __init__(self):
        self.type_filters: Set[EventType] = set()
        self.source_filters: Set[str] = set()
        self.priority_filters: Set[EventPriority] = set()
        self.status_filters: Set[EventStatus] = set()
        self.custom_filters: List[Callable[[AGUIEvent], bool]] = []
    
    def add_type_filter(self, event_type: EventType):
        """添加事件类型过滤器"""
        self.type_filters.add(event_type)
    
    def add_source_filter(self, source: str):
        """添加事件源过滤器"""
        self.source_filters.add(source)
    
    def add_priority_filter(self, priority: EventPriority):
        """添加优先级过滤器"""
        self.priority_filters.add(priority)
    
    def add_status_filter(self, status: EventStatus):
        """添加状态过滤器"""
        self.status_filters.add(status)
    
    def add_custom_filter(self, filter_func: Callable[[AGUIEvent], bool]):
        """添加自定义过滤器"""
        self.custom_filters.append(filter_func)
    
    def matches(self, event: AGUIEvent) -> bool:
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

class EventRouter:
    """事件路由器"""
    
    def __init__(self):
        self.routes: Dict[str, List[Callable]] = defaultdict(list)
        self.default_handlers: List[Callable] = []
    
    def add_route(self, event_type: EventType, handler: Callable):
        """添加事件路由"""
        self.routes[event_type.value].append(handler)
    
    def add_default_handler(self, handler: Callable):
        """添加默认处理器"""
        self.default_handlers.append(handler)
    
    async def route_event(self, event: AGUIEvent):
        """路由事件到相应处理器"""
        handlers = self.routes.get(event.type.value, []) + self.default_handlers
        
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"事件处理器错误: {e}")

class EventStream:
    """事件流"""
    
    def __init__(self, name: str, max_size: int = 1000):
        self.name = name
        self.max_size = max_size
        self.events: deque = deque(maxlen=max_size)
        self.subscribers: Set[Callable] = set()
        self.filters: List[EventFilter] = []
        self.is_active = True
    
    def add_subscriber(self, subscriber: Callable):
        """添加订阅者"""
        self.subscribers.add(subscriber)
    
    def remove_subscriber(self, subscriber: Callable):
        """移除订阅者"""
        self.subscribers.discard(subscriber)
    
    def add_filter(self, event_filter: EventFilter):
        """添加过滤器"""
        self.filters.append(event_filter)
    
    async def publish(self, event: AGUIEvent):
        """发布事件"""
        if not self.is_active:
            return
        
        # 检查过滤器
        for event_filter in self.filters:
            if not event_filter.matches(event):
                return
        
        self.events.append(event)
        
        # 通知订阅者
        for subscriber in self.subscribers:
            try:
                if asyncio.iscoroutinefunction(subscriber):
                    await subscriber(event)
                else:
                    subscriber(event)
            except Exception as e:
                logger.error(f"事件订阅者错误: {e}")
    
    def get_recent_events(self, count: int = 10) -> List[AGUIEvent]:
        """获取最近的事件"""
        return list(self.events)[-count:]
    
    def clear(self):
        """清空事件流"""
        self.events.clear()

class EventEncoder:
    """AG-UI事件编码器"""
    
    @staticmethod
    def encode_system_event(event_type: EventType, source: str, data: Dict[str, Any] = None) -> AGUIEvent:
        """编码系统事件"""
        return AGUIEvent(
            id=str(uuid.uuid4()),
            type=event_type,
            timestamp=datetime.now(timezone.utc).isoformat(),
            source=source,
            data=data or {},
            metadata={"encoder": "system", "version": "1.0.0"}
        )
    
    @staticmethod
    def encode_user_event(message: str, user_id: str = None, source: str = "user") -> AGUIEvent:
        """编码用户事件"""
        return AGUIEvent(
            id=str(uuid.uuid4()),
            type=EventType.USER_MESSAGE,
            timestamp=datetime.now(timezone.utc).isoformat(),
            source=source,
            target="norma_ai",
            data={
                "message": message,
                "user_id": user_id,
                "message_type": "text"
            },
            metadata={"encoder": "user", "version": "1.0.0"}
        )
    
    @staticmethod
    def encode_ai_response(response: str, query: str = None, source: str = "norma_ai") -> AGUIEvent:
        """编码AI响应事件"""
        return AGUIEvent(
            id=str(uuid.uuid4()),
            type=EventType.AI_RESPONSE,
            timestamp=datetime.now(timezone.utc).isoformat(),
            source=source,
            target="user",
            data={
                "response": response,
                "query": query,
                "response_type": "text"
            },
            metadata={"encoder": "ai", "version": "1.0.0"}
        )
    
    @staticmethod
    def encode_blood_analysis_event(student_name: str, result: Dict[str, Any], source: str = "blood_analyzer") -> AGUIEvent:
        """编码血统分析事件"""
        return AGUIEvent(
            id=str(uuid.uuid4()),
            type=EventType.BLOOD_RESULT,
            timestamp=datetime.now(timezone.utc).isoformat(),
            source=source,
            target="user",
            data={
                "student_name": student_name,
                "analysis_result": result
            },
            metadata={"encoder": "blood", "version": "1.0.0"}
        )
    
    @staticmethod
    def encode_security_event(event_type: EventType, details: Dict[str, Any], source: str = "security_monitor") -> AGUIEvent:
        """编码安全事件"""
        return AGUIEvent(
            id=str(uuid.uuid4()),
            type=event_type,
            timestamp=datetime.now(timezone.utc).isoformat(),
            source=source,
            data=details,
            priority=EventPriority.HIGH,
            metadata={"encoder": "security", "version": "1.0.0"}
        )
    
    @staticmethod
    def encode_network_event(event_type: EventType, connection_info: Dict[str, Any], source: str = "network_monitor") -> AGUIEvent:
        """编码网络事件"""
        return AGUIEvent(
            id=str(uuid.uuid4()),
            type=event_type,
            timestamp=datetime.now(timezone.utc).isoformat(),
            source=source,
            data=connection_info,
            priority=EventPriority.NORMAL,
            metadata={"encoder": "network", "version": "1.0.0"}
        )

class WebSocketBridge:
    """WebSocket桥接器"""
    
    def __init__(self, event_system):
        self.event_system = event_system
        self.websocket_connections: Dict[str, WebSocket] = {}
        self.websocket_streams: Dict[str, EventStream] = {}
    
    async def connect_websocket(self, websocket: WebSocket, connection_id: str, stream_name: str = "default"):
        """连接WebSocket"""
        await websocket.accept()
        self.websocket_connections[connection_id] = websocket
        
        # 创建或获取事件流
        if stream_name not in self.websocket_streams:
            self.websocket_streams[stream_name] = EventStream(stream_name)
        
        stream = self.websocket_streams[stream_name]
        
        # 添加事件处理器
        async def event_handler(event: AGUIEvent):
            try:
                await websocket.send_text(event.to_json())
            except Exception as e:
                logger.error(f"WebSocket发送失败: {e}")
                await self.disconnect_websocket(connection_id)
        
        stream.add_subscriber(event_handler)
        
        logger.info(f"WebSocket连接已建立: {connection_id}")
        
        # 发送连接确认事件
        confirm_event = EventEncoder.encode_system_event(
            EventType.SYSTEM_STATUS,
            "websocket_bridge",
            {"connection_id": connection_id, "status": "connected"}
        )
        await stream.publish(confirm_event)
    
    async def disconnect_websocket(self, connection_id: str):
        """断开WebSocket连接"""
        if connection_id in self.websocket_connections:
            websocket = self.websocket_connections[connection_id]
            try:
                await websocket.close()
            except:
                pass
            del self.websocket_connections[connection_id]
            logger.info(f"WebSocket连接已断开: {connection_id}")
    
    async def send_to_websocket(self, connection_id: str, event: AGUIEvent):
        """向特定WebSocket发送事件"""
        if connection_id in self.websocket_connections:
            websocket = self.websocket_connections[connection_id]
            try:
                await websocket.send_text(event.to_json())
            except Exception as e:
                logger.error(f"向WebSocket发送事件失败: {e}")
                await self.disconnect_websocket(connection_id)
    
    async def broadcast_to_websockets(self, event: AGUIEvent, stream_name: str = "default"):
        """向所有WebSocket广播事件"""
        if stream_name in self.websocket_streams:
            stream = self.websocket_streams[stream_name]
            await stream.publish(event)
    
    async def handle_websocket_message(self, connection_id: str, message: str):
        """处理WebSocket消息"""
        try:
            # 解析消息
            message_data = json.loads(message)
            
            # 如果是事件，直接转发
            if "type" in message_data and "id" in message_data:
                event = AGUIEvent.from_dict(message_data)
                await self.event_system.publish_event(event)
            else:
                # 创建用户消息事件
                user_event = EventEncoder.encode_user_event(
                    message_data.get("message", ""),
                    message_data.get("user_id"),
                    f"websocket_{connection_id}"
                )
                await self.event_system.publish_event(user_event)
                
        except Exception as e:
            logger.error(f"处理WebSocket消息失败: {e}")

class EventStreamManager:
    """事件流管理器"""
    
    def __init__(self):
        self.streams: Dict[str, EventStream] = {}
        self.default_stream = self.create_stream("default")
        self.system_stream = self.create_stream("system")
        self.user_stream = self.create_stream("user")
        self.security_stream = self.create_stream("security")
        self.blood_stream = self.create_stream("blood")
        self.network_stream = self.create_stream("network")
    
    def create_stream(self, name: str, max_size: int = 1000) -> EventStream:
        """创建事件流"""
        stream = EventStream(name, max_size)
        self.streams[name] = stream
        return stream
    
    def get_stream(self, name: str) -> Optional[EventStream]:
        """获取事件流"""
        return self.streams.get(name)
    
    def get_all_streams(self) -> Dict[str, EventStream]:
        """获取所有事件流"""
        return self.streams.copy()
    
    async def publish_to_stream(self, stream_name: str, event: AGUIEvent):
        """向指定流发布事件"""
        stream = self.get_stream(stream_name)
        if stream:
            await stream.publish(event)
    
    async def broadcast_to_all_streams(self, event: AGUIEvent):
        """向所有流广播事件"""
        for stream in self.streams.values():
            await stream.publish(event)

class AGUIEventSystem:
    """AG-UI事件系统主类"""
    
    def __init__(self):
        self.encoder = EventEncoder()
        self.stream_manager = EventStreamManager()
        self.router = EventRouter()
        self.websocket_bridge = WebSocketBridge(self)
        self.event_handlers: Dict[EventType, List[Callable]] = defaultdict(list)
        self.event_history: deque = deque(maxlen=10000)
        self.is_running = False
        
        # 设置默认路由
        self._setup_default_routes()
    
    def _setup_default_routes(self):
        """设置默认路由"""
        # 系统事件路由
        self.router.add_route(EventType.SYSTEM_START, self._handle_system_start)
        self.router.add_route(EventType.SYSTEM_STOP, self._handle_system_stop)
        self.router.add_route(EventType.SYSTEM_ERROR, self._handle_system_error)
        self.router.add_route(EventType.SYSTEM_STATUS, self._handle_system_status)
        
        # 用户事件路由
        self.router.add_route(EventType.USER_MESSAGE, self._handle_user_message)
        self.router.add_route(EventType.USER_ACTION, self._handle_user_action)
        
        # AI响应路由
        self.router.add_route(EventType.AI_RESPONSE, self._handle_ai_response)
        self.router.add_route(EventType.AI_THINKING, self._handle_ai_thinking)
        
        # 安全事件路由
        self.router.add_route(EventType.SECURITY_ALERT, self._handle_security_alert)
        self.router.add_route(EventType.SECURITY_STATUS, self._handle_security_status)
        
        # 血统分析路由
        self.router.add_route(EventType.BLOOD_RESULT, self._handle_blood_result)
        
        # 网络事件路由
        self.router.add_route(EventType.NETWORK_CONNECT, self._handle_network_connect)
        self.router.add_route(EventType.NETWORK_DISCONNECT, self._handle_network_disconnect)
    
    async def start(self):
        """启动事件系统"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # 发送系统启动事件
        start_event = self.encoder.encode_system_event(
            EventType.SYSTEM_START,
            "agui_event_system",
            {"version": "1.0.0", "components": ["encoder", "stream_manager", "router", "websocket_bridge"]}
        )
        await self.publish_event(start_event)
        
        logger.info("AG-UI事件系统已启动")
    
    async def stop(self):
        """停止事件系统"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # 发送系统停止事件
        stop_event = self.encoder.encode_system_event(
            EventType.SYSTEM_STOP,
            "agui_event_system",
            {"timestamp": datetime.now(timezone.utc).isoformat()}
        )
        await self.publish_event(stop_event)
        
        logger.info("AG-UI事件系统已停止")
    
    async def publish_event(self, event: AGUIEvent):
        """发布事件"""
        # 添加到历史记录
        self.event_history.append(event)
        
        # 更新事件状态
        event.status = EventStatus.PROCESSING
        
        # 路由事件
        await self.router.route_event(event)
        
        # 发布到相应的事件流
        await self._publish_to_streams(event)
        
        # 标记为完成
        event.status = EventStatus.COMPLETED
        
        logger.debug(f"事件已发布: {event.type.value} - {event.id}")
    
    async def _publish_to_streams(self, event: AGUIEvent):
        """发布事件到相应的事件流"""
        # 根据事件类型确定目标流
        stream_mapping = {
            EventType.SYSTEM_START: "system",
            EventType.SYSTEM_STOP: "system",
            EventType.SYSTEM_ERROR: "system",
            EventType.SYSTEM_STATUS: "system",
            EventType.USER_MESSAGE: "user",
            EventType.USER_ACTION: "user",
            EventType.AI_RESPONSE: "user",
            EventType.AI_THINKING: "user",
            EventType.SECURITY_ALERT: "security",
            EventType.SECURITY_STATUS: "security",
            EventType.BLOOD_RESULT: "blood",
            EventType.BLOOD_ANALYSIS_COMPLETE: "blood",
            EventType.NETWORK_CONNECT: "network",
            EventType.NETWORK_DISCONNECT: "network",
        }
        
        stream_name = stream_mapping.get(event.type, "default")
        await self.stream_manager.publish_to_stream(stream_name, event)
    
    # 事件处理器方法
    async def _handle_system_start(self, event: AGUIEvent):
        """处理系统启动事件"""
        logger.info(f"系统启动: {event.data}")
    
    async def _handle_system_stop(self, event: AGUIEvent):
        """处理系统停止事件"""
        logger.info(f"系统停止: {event.data}")
    
    async def _handle_system_error(self, event: AGUIEvent):
        """处理系统错误事件"""
        logger.error(f"系统错误: {event.data}")
    
    async def _handle_system_status(self, event: AGUIEvent):
        """处理系统状态事件"""
        logger.debug(f"系统状态: {event.data}")
    
    async def _handle_user_message(self, event: AGUIEvent):
        """处理用户消息事件"""
        logger.info(f"用户消息: {event.data.get('message', '')[:50]}...")
    
    async def _handle_user_action(self, event: AGUIEvent):
        """处理用户动作事件"""
        logger.info(f"用户动作: {event.data}")
    
    async def _handle_ai_response(self, event: AGUIEvent):
        """处理AI响应事件"""
        logger.info(f"AI响应: {event.data.get('response', '')[:50]}...")
    
    async def _handle_ai_thinking(self, event: AGUIEvent):
        """处理AI思考事件"""
        logger.debug(f"AI思考: {event.data}")
    
    async def _handle_security_alert(self, event: AGUIEvent):
        """处理安全警报事件"""
        logger.warning(f"安全警报: {event.data}")
    
    async def _handle_security_status(self, event: AGUIEvent):
        """处理安全状态事件"""
        logger.info(f"安全状态: {event.data}")
    
    async def _handle_blood_result(self, event: AGUIEvent):
        """处理血统分析结果事件"""
        logger.info(f"血统分析结果: {event.data}")
    
    async def _handle_network_connect(self, event: AGUIEvent):
        """处理网络连接事件"""
        logger.info(f"网络连接: {event.data}")
    
    async def _handle_network_disconnect(self, event: AGUIEvent):
        """处理网络断开事件"""
        logger.info(f"网络断开: {event.data}")
    
    # 公共API方法
    def add_event_handler(self, event_type: EventType, handler: Callable):
        """添加事件处理器"""
        self.event_handlers[event_type].append(handler)
    
    def create_filter(self) -> EventFilter:
        """创建事件过滤器"""
        return EventFilter()
    
    def get_event_history(self, count: int = 100) -> List[AGUIEvent]:
        """获取事件历史"""
        return list(self.event_history)[-count:]
    
    def get_stream_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取流统计信息"""
        stats = {}
        for name, stream in self.stream_manager.get_all_streams().items():
            stats[name] = {
                "event_count": len(stream.events),
                "subscriber_count": len(stream.subscribers),
                "filter_count": len(stream.filters),
                "is_active": stream.is_active
            }
        return stats
    
    async def cleanup_streams(self):
        """清理不活跃的事件流"""
        inactive_streams = []
        for name, stream in self.stream_manager.get_all_streams().items():
            if len(stream.subscribers) == 0 and name != "default":
                inactive_streams.append(name)
        
        for stream_name in inactive_streams:
            del self.stream_manager.streams[stream_name]
            if stream_name in self.websocket_bridge.websocket_streams:
                del self.websocket_bridge.websocket_streams[stream_name]
        
        if inactive_streams:
            logger.info(f"已清理不活跃的事件流: {inactive_streams}")

# 全局事件系统实例
event_system = AGUIEventSystem()

# 便利函数
async def publish_system_event(event_type: EventType, data: Dict[str, Any] = None):
    """发布系统事件的便利函数"""
    event = EventEncoder.encode_system_event(event_type, "norma_system", data)
    await event_system.publish_event(event)

async def publish_user_message(message: str, user_id: str = None):
    """发布用户消息的便利函数"""
    event = EventEncoder.encode_user_event(message, user_id)
    await event_system.publish_event(event)

async def publish_ai_response(response: str, query: str = None):
    """发布AI响应的便利函数"""
    event = EventEncoder.encode_ai_response(response, query)
    await event_system.publish_event(event)

async def publish_blood_analysis(student_name: str, result: Dict[str, Any]):
    """发布血统分析结果的便利函数"""
    event = EventEncoder.encode_blood_analysis_event(student_name, result)
    await event_system.publish_event(event)

async def publish_security_event(event_type: EventType, details: Dict[str, Any]):
    """发布安全事件的便利函数"""
    event = EventEncoder.encode_security_event(event_type, details)
    await event_system.publish_event(event)

# WebSocket事件处理器
async def handle_websocket_connection(websocket: WebSocket, connection_id: str, stream_name: str = "default"):
    """处理WebSocket连接的便利函数"""
    await event_system.websocket_bridge.connect_websocket(websocket, connection_id, stream_name)
    
    try:
        while True:
            message = await websocket.receive_text()
            await event_system.websocket_bridge.handle_websocket_message(connection_id, message)
    except WebSocketDisconnect:
        await event_system.websocket_bridge.disconnect_websocket(connection_id)

# 演示和测试函数
async def demo_event_system():
    """演示事件系统功能"""
    print("=== AG-UI事件系统演示 ===")
    
    # 启动事件系统
    await event_system.start()
    
    # 发布各种类型的事件
    await publish_system_event(EventType.SYSTEM_STATUS, {"cpu_usage": "15%", "memory_usage": "42%"})
    
    await publish_user_message("你好，诺玛！", "user001")
    
    await publish_ai_response("你好！我是诺玛·劳恩斯，很高兴为您服务。", "你好，诺玛！")
    
    await publish_blood_analysis("路明非", {
        "bloodline_type": "S级混血种",
        "purity_level": "95.2%",
        "abilities": "黄金瞳、言灵·君焰、时间零",
        "status": "稳定"
    })
    
    await publish_security_event(EventType.SECURITY_STATUS, {
        "firewall_status": "正常",
        "intrusion_detection": "运行中",
        "threat_level": "低"
    })
    
    # 显示事件统计
    stats = event_system.get_stream_stats()
    print("\n事件流统计:")
    for stream_name, stream_stats in stats.items():
        print(f"  {stream_name}: {stream_stats}")
    
    # 显示最近的事件
    recent_events = event_system.get_event_history(5)
    print(f"\n最近 {len(recent_events)} 个事件:")
    for event in recent_events:
        print(f"  {event.timestamp} - {event.type.value} - {event.source}")
    
    # 停止事件系统
    await event_system.stop()
    
    print("\n=== 演示完成 ===")

if __name__ == "__main__":
    asyncio.run(demo_event_system())